from ultralytics import YOLO
from inference.damage_segmentation import resize_img
import cv2
import os


def detect_car_parts_only(image_paths: list, min_conf: float = 0.25):
    """
    Directly detects car parts in the provided images without looking for damage first.

    Args:
        image_paths (list): List of file paths to the images.
        min_conf (float): Confidence threshold.

    Returns:
        dict: Mapping from image path (str) to a list of detection metadata dicts. Each metadata dict contains:
            {
                'part_name': str,
                'confidence': float,
                'box': [x1, y1, x2, y2]  # coordinates in resized image
            }
        Example:
            {
                '/path/img1.jpg': [
                    {'part_name': 'Bumper', 'confidence': 0.95, 'box': [10,10,100,50]},
                    {'part_name': 'Headlight', 'confidence': 0.80, 'box': [120,20,160,60]}
                ],
                '/path/img2.jpg': []
            }
    """

    # 1. Load ONLY the Parts Model
    # Make sure this points to your parts detection model
    print("Loading Car Parts Model...")
    model = YOLO("models/Parts_Name_Classification.pt")

    # This will hold the final mapping: image_path -> list of metadata dicts
    parts_by_image = {}

    for image_path in image_paths:
        print(f"Processing for parts: {image_path}")
        # List to hold metadata dicts for parts found in THIS specific image
        current_image_parts = []

        # 2. Read and Resize
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Could not read: {image_path}")
            parts_by_image[image_path] = []
            continue

        # We use your existing resize logic here (assuming resize_img is defined in your file)
        img_resized = resize_img(img_bgr, target_size=(640, 640))

        # 3. Run Inference (Directly on the full image)
        # We look for parts immediately
        results = model(img_resized, conf=min_conf, verbose=False)[0]

        # 4. Extract Class Names
        # We don't need to crop or loop through boxes manually just to get names
        if results.boxes:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                part_name = model.names[cls_id]
                # confidence may be stored in box.conf or box.conf[0]
                conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0
                # box.xyxy gives coordinates (tensor-like); convert to list of ints
                try:
                    xyxy = [int(x) for x in box.xyxy[0].tolist()]
                except Exception:
                    # fallback: empty box
                    xyxy = []

                current_image_parts.append({
                    'part_name': part_name,
                    'confidence': conf,
                    'box': xyxy
                })

            print(f"  -> Found parts: {[p['part_name'] for p in current_image_parts]}")
        else:
            print("  -> No parts detected.")

        # 5. Add to mapping
        parts_by_image[image_path] = current_image_parts

    return parts_by_image