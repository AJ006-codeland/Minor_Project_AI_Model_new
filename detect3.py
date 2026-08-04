# ============================================================
# IMPORTS
# ============================================================

import os
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tkinter import Tk
from tkinter.filedialog import askopenfilename
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# ============================================================
# LOAD MODELS  (loaded ONCE — reused for every leaf in the loop)
# ============================================================

yolo_model = YOLO("best.pt")
disease_model = load_model("efficientnetv2_grape_final.keras")
classes = [
    "Black Rot",
    "ESCA",
    "Healthy",
    "Leaf Blight"
]
DISEASE_CONFIDENCE_THRESHOLD = 70
YOLO_CONF = 0.75

# Background fill for the segmented leaf crops: "black" or "white"
MASK_BACKGROUND = "black"
colors = {
    "Healthy": (0, 255, 0),
    "Black Rot": (255, 0, 0),
    "ESCA": (255, 165, 0),
    "Leaf Blight": (255, 255, 0),
    "Unknown Disease": (180, 180, 180)
}

# Folder where each segmented leaf gets saved as its own image
OUTPUT_DIR = "segmented_leaves"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# SELECT IMAGE
# ============================================================

root = Tk()
root.withdraw()
img_path = askopenfilename(
    title="Select a grape plant image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png *.bmp *.jfif *.webp")
    ]
)
if img_path == "":
    print("No image selected.")
    exit()

# ============================================================
# LOAD IMAGE
# ============================================================

original_image = cv2.imread(img_path)
if original_image is None:
    print("Error: Unable to read image.")
    exit()
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
height, width = original_image_rgb.shape[:2]

# ============================================================
# STAGE 1 : YOLOv8 SEGMENTATION -> SAVE EACH LEAF AS ITS OWN IMAGE
# ============================================================

print("\n[Stage 1] Running YOLOv8 segmentation on the input image...")
results = yolo_model.predict(
    source=original_image,
    conf=YOLO_CONF,
    save=False,
    verbose=False
)
result = results[0]
boxes = result.boxes
masks = result.masks  # None if model is detection-only, not -seg
if boxes is None or len(boxes) == 0:
    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")
    print("Status : UNKNOWN PLANT (no leaf detected)")
    print("==============================")
    exit()
masks_np = masks.data.cpu().numpy() if masks is not None else None
if masks_np is None:
    print("WARNING: No masks found — best.pt must be a -seg model for background "
          "removal. Falling back to plain bounding-box crops.")
leaf_image_paths = []
for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if masks_np is not None:
        # Resize this leaf's mask up to the original image resolution
        mask_full = cv2.resize(masks_np[i], (width, height), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_full > 0.5).astype(np.uint8)
        fill_value = 0 if MASK_BACKGROUND == "black" else 255
        masked_full = original_image_rgb.copy()
        masked_full[mask_bin == 0] = fill_value
        crop = masked_full[y1:y2, x1:x2]
    else:
        crop = original_image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    # Save this leaf as its own image file (image1.jpg, image2.jpg, ...)
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    leaf_path = os.path.join(OUTPUT_DIR, f"leaf_{i+1}.jpg")
    cv2.imwrite(leaf_path, crop_bgr)
    leaf_image_paths.append(leaf_path)
print(f"[Stage 1] Segmented {len(leaf_image_paths)} leaf/leaves -> saved in '{OUTPUT_DIR}/'\n")

# ============================================================
# STAGE 2 : LOOP OVER EACH SAVED LEAF IMAGE -> CLASSIFY ONE AT A TIME
# ============================================================

print("[Stage 2] Loading each segmented leaf into the model one by one...\n")
diagnosis_labels = []
for idx, leaf_path in enumerate(leaf_image_paths):

    # -----------------------------
    # Load ONE leaf image
    # -----------------------------
    leaf_bgr = cv2.imread(leaf_path)
    leaf_rgb = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # Prepare for EfficientNet
    # -----------------------------
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # -----------------------------
    # Predict on THIS leaf only
    # -----------------------------
    prediction = disease_model.predict(img, verbose=0)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = classes[predicted_index]
    confidence = prediction[predicted_index] * 100
    if confidence >= DISEASE_CONFIDENCE_THRESHOLD:
        final_label = predicted_class
    else:
        final_label = "Unknown Disease"
    diagnosis_labels.append(final_label)

    # -----------------------------
    # Console output for this leaf
    # -----------------------------
    print(f"========== Leaf {idx+1}/{len(leaf_image_paths)} "
          f"({os.path.basename(leaf_path)}) ==========")
    for j, cls in enumerate(classes):
        print(f"{cls:<15}: {prediction[j]*100:.2f}%")
    print("-------------------------------")
    print(f"Prediction : {final_label}")
    print(f"Confidence : {confidence:.2f}%\n")

    # -----------------------------
    # Display THIS leaf's result
    # (plt.show() blocks here — closing the window moves the loop
    #  on to the next leaf image)
    # -----------------------------
   # Display prediction
    plt.figure(figsize=(5, 5))
    plt.imshow(leaf_rgb)
    plt.axis("off")

    plt.title(
        f"Leaf {idx + 1}/{len(leaf_image_paths)} : {final_label} ({confidence:.1f}%)",
        color="black",          # Title color
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()

# ============================================================
# FINAL OVERALL DIAGNOSIS (after the loop finishes)
# ============================================================

if len(diagnosis_labels) == 0:
    overall_result = "No grape leaf detected"
elif len(set(diagnosis_labels)) == 1:
    overall_result = diagnosis_labels[0]
else:
    counts = {}
    for label in diagnosis_labels:
        counts[label] = counts.get(label, 0) + 1
    overall_result = max(counts, key=counts.get)
print("========================================")
print("FINAL RESULT")
print("========================================")
print(f"Detected Leaves : {len(diagnosis_labels)}")
print(f"Per-Leaf Result : {diagnosis_labels}")
print(f"Overall Diagnosis : {overall_result}")
print("========================================")