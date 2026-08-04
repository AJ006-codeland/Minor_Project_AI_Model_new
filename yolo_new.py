from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("best.pt")

results = model.predict(
    source="1_h.jpg",
    conf=0.65,
    save=True
)

# output folder for cropped leaves
os.makedirs("cropped_leaves", exist_ok=True)

for r in results:
    img = r.orig_img  # original image as numpy array (BGR)
 
    if r.masks is None:
        print("No masks found — model must be a segmentation model (best.pt should be -seg).")
        continue

    masks = r.masks.data.cpu().numpy()  # (N, H, W) mask array
    boxes = r.boxes.xyxy.cpu().numpy()  # (N, 4) bounding boxes

    h, w = img.shape[:2]

    for i, (mask, box) in enumerate(zip(masks, boxes)):
        # resize mask to original image size (mask is often at model input res)
        mask_resized = cv2.resize(mask, (w, h))
        mask_bin = (mask_resized > 0.5).astype(np.uint8)

        # apply mask to isolate leaf, black out background
        masked_img = img.copy()
        masked_img[mask_bin == 0] = 0  # set background to black
        # if you want transparent background instead, use the alpha version below

        # crop to bounding box so it's tight around the leaf
        x1, y1, x2, y2 = map(int, box)
        cropped = masked_img[y1:y2, x1:x2]

        out_path = f"cropped_leaves/image{i+1}.jpg"
        cv2.imwrite(out_path, cropped)
        print(f"Saved {out_path}")