import base64
import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
from inference_sdk import InferenceHTTPClient
import shutil
import os

import tensorflow as tf
from tensorflow import keras
from keras.applications.efficientnet_v2 import preprocess_input
import matplotlib.pyplot as plt

# -----------------------------
# Config - Segmentation (Roboflow)
# -----------------------------
API_KEY = "f4pCso2oY1jQ8D3h1oRQ"          # rotate the old one you posted earlier!
WORKSPACE_NAME = "aayushmas-workspace-k2yif"
WORKFLOW_ID = "general-segmentation-api-3"
CLASSES = ["Leaf"]
MIN_AREA_RATIO = 0.050

# -----------------------------
# Config - Classification (EfficientNetV2-S)
# -----------------------------
CLASSIFIER_PATH = "efficientnetv2_grape_final.keras"
CLASSIFIER_IMG_SIZE = 224   # matches how the final model was trained

# Order must match training's alphabetical folder order
CLASS_NAMES = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]
HEALTHY_INDEX = CLASS_NAMES.index("Healthy")

DISEASE_CONFIDENCE_THRESHOLD = 0.40   # below this -> "Unknown Disease" (for a disease-class prediction)
UNKNOWN_PLANT_THRESHOLD = 0.70        # used when there are zero leaf detections

# --- Color-based "is this even a grape leaf" pre-filter ---
# Grape leaves (healthy or diseased) are predominantly green/yellow-green.
# Used as a CONFIRMATORY check even after shape passes — both shape and
# color must agree for a crop to be treated as a grape leaf.
# NOTE: a very clean, uniformly green healthy leaf can score ABOVE 90% green,
# which would fall outside this range and get rejected. Test against real
# Healthy samples and widen GRAPE_COLOR_MATCH_MAX if that happens.
GRAPE_COLOR_MATCH_MIN = 0.70
GRAPE_COLOR_MATCH_MAX = 0.90

# --- Shape-based "is this leaf architecture grape-like" pre-filter ---
# Grape leaves have a palmate, multi-lobed shape with deep sinuses between
# lobes — this shows up as relatively low "solidity" (contour area vs. its
# convex hull area) and multiple significant convexity defects.
# NOTE: this is a coarse heuristic. Other lobed leaves (e.g. maple) can have
# a similar solidity/defect profile, so shape alone won't reliably separate
# grape from every other lobed species — it's a first-pass filter, not a
# trained shape classifier.
GRAPE_SOLIDITY_MIN = 0.55
GRAPE_SOLIDITY_MAX = 0.92
MIN_SIGNIFICANT_DEFECTS = 2

# --- "Better safe than sorry" bias settings ---
# If there's a real disease signal (spots, edge damage, discoloration),
# flag it as diseased even if Healthy technically scored highest.
# Only skip this override when the leaf genuinely shows no disease signal
# (e.g. a partial/cropped leaf with clean tissue only).

DISEASE_OVERRIDE_THRESHOLD = 0.15  # any disease class above this overrides a Healthy call

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

print("Loading classifier model...")
classifier = keras.models.load_model(CLASSIFIER_PATH)
print("Classifier loaded.\n")

# -----------------------------
# Segmentation Helpers
# -----------------------------
def select_image():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a grape plant image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return path


def decode_base64_image(b64_string):
    """Convert base64 string from API response into an OpenCV image."""
    img_bytes = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image


def crop_segmented_leaves(original_image, predictions, output_dir="separated_leaves",
                           min_area_ratio=0.015):
    """
    Crop each detected leaf from the ORIGINAL image using its segmentation mask,
    keeping only leaves whose area is at least `min_area_ratio` of the total image area.
    Saves each surviving crop as a separate file in output_dir.
    Returns list of saved crop file paths (in save order).
    """
    out_folder = Path(output_dir)
    out_folder.mkdir(parents=True, exist_ok=True)

    preds_list = predictions.get("predictions", predictions) if isinstance(predictions, dict) else predictions
    if not isinstance(preds_list, list):
        print("No valid predictions list found — nothing to crop.")
        return []

    img_area = original_image.shape[0] * original_image.shape[1]
    min_pixel_area = img_area * min_area_ratio

    class_counters = {}
    kept, skipped = 0, 0
    saved_paths = []

    for i, pred in enumerate(preds_list):
        cls = pred.get("class", "unknown")
        points = pred.get("points")

        if points:
            poly = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
            area = cv2.contourArea(poly)
        else:
            w, h = pred.get("width"), pred.get("height")
            if w is None or h is None:
                continue
            area = w * h
            poly = None

        if area < min_pixel_area:
            skipped += 1
            continue

        class_counters[cls] = class_counters.get(cls, 0) + 1
        idx = class_counters[cls]

        if poly is not None:
            x, y, w, h = cv2.boundingRect(poly)
            mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            isolated = cv2.bitwise_and(original_image, original_image, mask=mask)
            crop = isolated[y:y + h, x:x + w]
        else:
            cx, cy = pred.get("x"), pred.get("y")
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            crop = original_image[max(0, y1):y2, max(0, x1):x2]

        if crop.size == 0:
            skipped += 1
            continue

        filename = out_folder / f"{cls}_{idx}.jpg"
        cv2.imwrite(str(filename), crop)
        print(f"Saved crop: {filename}  (area={int(area)} px, {area/img_area*100:.2f}% of image)")
        saved_paths.append(str(filename))
        kept += 1

    print(f"\nKept {kept} major leaves, skipped {skipped} small ones.")
    print(f"All leaf crops saved in: {out_folder.resolve()}")
    return saved_paths


# -----------------------------
# Shape Pre-Filter Helper
# -----------------------------
def is_grape_leaf_shape(image_bgr):
    """
    Heuristic check: does this crop's silhouette look like a grape leaf's
    palmate, multi-lobed architecture?

    Works on the masked crop (black background from segmentation) — finds
    the leaf's outer contour and measures solidity + convexity defects.

    Returns (is_grape_shape: bool, solidity: float, significant_defects: int)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, 0

    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)

    if contour_area < 100 or len(contour) < 5:
        return False, 0.0, 0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = (contour_area / hull_area) if hull_area > 0 else 0.0

    # Count convexity defects deep enough to represent a real sinus/notch,
    # scaled relative to the leaf's own size so it works across crop sizes.
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    significant_defects = 0

    if hull_indices is not None and len(hull_indices) > 3:
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
        except cv2.error:
            defects = None

        if defects is not None:
            leaf_scale = np.sqrt(contour_area)
            depth_threshold = leaf_scale * 0.08  # tune if needed
            
            for i in range(defects.shape[0]):
                depth = defects[i, 0, 3] / 256.0
                if depth > depth_threshold:
                    significant_defects += 1

    is_grape_shape = (
        GRAPE_SOLIDITY_MIN <= solidity <= GRAPE_SOLIDITY_MAX
        and significant_defects >= MIN_SIGNIFICANT_DEFECTS
    )

    return is_grape_shape, solidity, significant_defects


# -----------------------------
# Color Pre-Filter Helper
# -----------------------------
def is_grape_leaf_color(image_bgr):
    """
    Heuristic check: does this crop look like a green grape leaf, or does it
    look like something else (autumn foliage, a different plant species, etc.)?

    Works on the masked crop (black background from segmentation), looking
    only at actual leaf pixels (non-black).

    Returns (is_grape_leaf: bool, green_ratio: float, autumn_ratio: float)
    for logging/debugging.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Ignore background: black mask pixels have near-zero value/saturation
    leaf_mask = (v > 20) & (s > 20)
    total_leaf_pixels = int(np.sum(leaf_mask))

    if total_leaf_pixels == 0:
        # Nothing to analyze — let it fall through to normal classification
        return True, 0.0, 0.0

    leaf_hues = h[leaf_mask]

    # OpenCV hue range is 0-179.
    # Green / yellow-green (healthy or diseased grape leaf tissue): ~25-95
    green_mask = (leaf_hues >= 25) & (leaf_hues <= 95)
    # Autumn colors: red/orange/brown - roughly 0-20 and 160-179
    autumn_mask = ((leaf_hues >= 0) & (leaf_hues <= 20)) | (leaf_hues >= 160)

    green_ratio = float(np.sum(green_mask)) / total_leaf_pixels
    autumn_ratio = float(np.sum(autumn_mask)) / total_leaf_pixels

    is_grape_leaf = GRAPE_COLOR_MATCH_MIN <= green_ratio <= GRAPE_COLOR_MATCH_MAX

    return is_grape_leaf, green_ratio, autumn_ratio


# -----------------------------
# Classification Helpers
# -----------------------------
def classify_leaf(image_path):
    """
    Runs the trained classifier on a single cropped leaf image.
    Returns (predicted_label, confidence, full_probability_dict).
    """
    img = cv2.imread(image_path)
    if img is None:
        return "Error reading image", 0.0, {}

    # --- Step 1: Shape check (leaf architecture) ---
    is_grape_shape, solidity, defect_count = is_grape_leaf_shape(img)
    print(f"    [shape check] solidity={solidity:.2f}, defects={defect_count} "
          f"-> {'PASS' if is_grape_shape else 'FAIL'}")

    # --- Step 2: Color check (confirmatory — always runs, even if shape passed) ---
    is_grape_color, green_ratio, autumn_ratio = is_grape_leaf_color(img)
    print(f"    [color check] green={green_ratio*100:.1f}%, autumn={autumn_ratio*100:.1f}% "
          f"-> {'PASS' if is_grape_color else 'FAIL'}")

    if not (is_grape_shape and is_grape_color):
        print(f"    -> Rejected — shape and color must BOTH match to be treated as a grape leaf")
        return "Unknown Plant", 0.0, {
            "Unknown Plant (shape/color check failed)": 1.0
        }

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = classifier.predict(img, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    prob_dict = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

    # Find the strongest disease signal regardless of what the raw argmax says
    disease_probs = {i: p for i, p in enumerate(preds) if i != HEALTHY_INDEX}
    max_disease_idx = max(disease_probs, key=disease_probs.get)
    max_disease_prob = disease_probs[max_disease_idx]
    healthy_prob = float(preds[HEALTHY_INDEX])

    if class_idx == HEALTHY_INDEX:
        if max_disease_prob >= DISEASE_OVERRIDE_THRESHOLD:
            # Real disease signal present (spots/edge damage) — override Healthy
            label = CLASS_NAMES[max_disease_idx]
            confidence = max_disease_prob
        else:
            # No meaningful disease signal — trust Healthy regardless of exact %
            label = "Healthy"
            confidence = healthy_prob
    else:
        # Model's top guess is already a disease class
        if confidence >= DISEASE_CONFIDENCE_THRESHOLD:
            label = CLASS_NAMES[class_idx]
        else:
            label = "Unknown Disease"

    return label, confidence, prob_dict


def display_leaf_result(image_path, label, confidence, prob_dict):
    """Shows the leaf crop with its predicted label and per-class confidence table."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, (ax_img, ax_table) = plt.subplots(1, 2, figsize=(10, 5))

    ax_img.imshow(img_rgb)
    ax_img.set_title(f"{Path(image_path).name}\nPrediction: {label} ({confidence*100:.1f}%)")
    ax_img.axis("off")

    ax_table.axis("off")
    table_text = "\n".join(f"{cls}: {prob*100:.2f}%" for cls, prob in prob_dict.items())
    ax_table.text(0.1, 0.5, table_text, fontsize=12, va="center")
    ax_table.set_title("Class Confidence")

    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():

    if os.path.exists("separated_leaves"):
        shutil.rmtree("separated_leaves")

    print("Opening file explorer — select an image...")
    image_path = select_image()
    if not image_path:
        print("No image selected. Exiting.")
        return

    print(f"Selected: {image_path}")
    print("Sending image to Roboflow API...")

    result = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": image_path},
        parameters={"classes": CLASSES},
        use_cache=True
    )

    output = result[0] if isinstance(result, list) else result

    annotated_b64 = output.get("annotated_image")
    predictions = output.get("predictions")

    if not annotated_b64:
        print("No annotated image returned. Raw response:")
        print(output)
        return

    segmented_image = decode_base64_image(annotated_b64)

    if predictions:
        preds_list = predictions.get("predictions", predictions) if isinstance(predictions, dict) else predictions
        if isinstance(preds_list, list):
            counts = {}
            for p in preds_list:
                cls = p.get("class", "unknown")
                counts[cls] = counts.get(cls, 0) + 1
            for cls, count in counts.items():
                print(f"{cls}: {count}")

    output_path = Path("output_segmented.jpg")
    cv2.imwrite(str(output_path), segmented_image)
    print(f"Saved result to: {output_path.resolve()}")

    original_image = cv2.imread(image_path)

    if not predictions:
        print("\nNo predictions available — treating as UNKNOWN PLANT.")
        return

    crop_paths = crop_segmented_leaves(
        original_image, predictions,
        output_dir="separated_leaves",
        min_area_ratio=MIN_AREA_RATIO
    )

    if not crop_paths:
        print("\nNo leaf crops survived the size filter — treating as UNKNOWN PLANT.")
        return

    # -----------------------------
    # Classify each cropped leaf
    # -----------------------------
    print(f"\nClassifying {len(crop_paths)} leaf crop(s)...\n")

    results_summary = []

    for crop_path in crop_paths:
        label, confidence, prob_dict = classify_leaf(crop_path)
        results_summary.append((crop_path, label, confidence))

        print(f"{Path(crop_path).name}: {label} ({confidence*100:.1f}% confidence)")
        for cls, prob in prob_dict.items():
            print(f"    {cls}: {prob*100:.2f}%")
        print()

        display_leaf_result(crop_path, label, confidence, prob_dict)

    # -----------------------------
    # Overall summary
    # -----------------------------
    print("\n=== SUMMARY ===")
    for crop_path, label, confidence in results_summary:
        print(f"{Path(crop_path).name}: {label} ({confidence*100:.1f}%)")

    cv2.imshow("Segmented Leaves (Roboflow API) - Press any key to close", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()