import base64
import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
from inference_sdk import InferenceHTTPClient
import shutil

# -----------------------------
# Config
# -----------------------------
API_KEY = "f4pCso2oY1jQ8D3h1oRQ"  # rotate the old one you posted earlier!
WORKSPACE_NAME = "aayushmas-workspace-k2yif"
WORKFLOW_ID = "general-segmentation-api-4"
CLASSES = ["Leaf"]           # set to whatever your workflow expects
MIN_AREA_RATIO = 0.050             # leaf must cover at least this % of image area to keep (tune this)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)


# -----------------------------
# Helpers
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
    """
    out_folder = Path(output_dir)
    out_folder.mkdir(parents=True, exist_ok=True)

    preds_list = predictions.get("predictions", predictions) if isinstance(predictions, dict) else predictions
    if not isinstance(preds_list, list):
        print("No valid predictions list found — nothing to crop.")
        return

    img_area = original_image.shape[0] * original_image.shape[1]
    min_pixel_area = img_area * min_area_ratio

    class_counters = {}
    kept, skipped = 0, 0

    for i, pred in enumerate(preds_list):
        cls = pred.get("class", "unknown")
        points = pred.get("points")

        if points:
            poly = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
            area = cv2.contourArea(poly)  # actual polygon area, not just bbox
        else:
            w, h = pred.get("width"), pred.get("height")
            if w is None or h is None:
                continue
            area = w * h
            poly = None

        # --- Size filter: skip small leaves ---
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
        kept += 1

    print(f"\nKept {kept} major leaves, skipped {skipped} small ones.")
    print(f"All leaf crops saved in: {out_folder.resolve()}")


# -----------------------------
# Main
# -----------------------------
def main():
    
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

    # result is a list with one dict inside
    output = result[0] if isinstance(result, list) else result

    annotated_b64 = output.get("annotated_image")
    predictions = output.get("predictions")

    if not annotated_b64:
        print("No annotated image returned. Raw response:")
        print(output)
        return

    segmented_image = decode_base64_image(annotated_b64)

    # Print detection summary if predictions are present
    if predictions:
        preds_list = predictions.get("predictions", predictions) if isinstance(predictions, dict) else predictions
        if isinstance(preds_list, list):
            counts = {}
            for p in preds_list:
                cls = p.get("class", "unknown")
                counts[cls] = counts.get(cls, 0) + 1
            for cls, count in counts.items():
                print(f"{cls}: {count}")

    # Save the annotated (visualization) image
    output_path = Path("output_segmented.jpg")
    cv2.imwrite(str(output_path), segmented_image)
    print(f"Saved result to: {output_path.resolve()}")

    # Crop and save only MAJOR leaves from the ORIGINAL (clean) image
    original_image = cv2.imread(image_path)
    if predictions:
        crop_segmented_leaves(original_image, predictions,
                               output_dir="separated_leaves",
                               min_area_ratio=MIN_AREA_RATIO)
    else:
        print("No predictions available — skipping leaf cropping.")

    cv2.imshow("Segmented Leaves (Roboflow API) - Press any key to close", segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()