from ultralytics import YOLO
from PIL import Image
import supervision as sv
import cv2
import numpy as np


# --- 1. Load Models ---
def load_models():
    """
    Loads both the Damage Detection Model and the Car Part Detection Model.
    """
    print("Loading models...")
    # Your existing damage model
    damage_model = YOLO("models/car_damage_segmentation_model.pt")

    # NEW: Your model that identifies parts (e.g., Bumper, Door, Hood)
    # You must have this file, or use a generic YOLOv8n.pt if just testing
    part_model = YOLO("models/car_parts_segmentation_model.pt")

    print("Models loaded successfully.")
    return damage_model, part_model


# --- 2. Helper: Resize (Your existing function) ---
def resize_img(img, target_size=(640, 640)):
    if isinstance(img, str):
        img = Image.open(img)

    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        resized = img.resize(target_size, Image.LANCZOS)
        resized_np = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)
        return resized_np
    elif isinstance(img, np.ndarray):
        h_t, w_t = target_size[1], target_size[0]
        if img.ndim == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        resized_np = cv2.resize(img_rgb, (w_t, h_t), interpolation=cv2.INTER_LANCZOS4)
        return resized_np
    else:
        raise TypeError("Unsupported image type.")


# --- 3. Main Logic ---
def detect_damage_and_parts(image_paths: list, min_conf: float = 0.25, padding: float = 0.5):
    """
    1. Detects damage in original images.
    2. Crops the damaged area.
    3. Runs the Part Model on the cropped area to identify the part.

    Returns:
        List[List[str]]: 2D array of detected parts per image.
    """
    damage_model, part_model = load_models()

    # This will hold the final 2D array output
    all_images_parts = []

    for image_path in image_paths:
        print(f"Processing: {image_path}")

        # A list to hold parts found in THIS specific image
        current_image_parts = []

        # 1. Read and Resize
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Could not read: {image_path}")
            all_images_parts.append([])  # Append empty list to maintain index
            continue

        img_resized = resize_img(img_bgr, target_size=(640, 640))
        h, w = img_resized.shape[:2]

        # 2. Detect Damage
        results = damage_model(img_resized, conf=min_conf, iou=0.5, imgsz=640, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0:
            print("  No damage found.")
            all_images_parts.append([])
            continue

        # 3. Iterate through found damages
        for idx in range(len(detections)):
            box = detections.xyxy[idx]
            x1, y1, x2, y2 = map(int, box)

            # Apply Padding (Your logic)
            pw = int((x2 - x1) * padding)
            ph = int((y2 - y1) * padding)
            x1_new = max(0, x1 - pw)
            y1_new = max(0, y1 - ph)
            x2_new = min(w, x2 + pw)
            y2_new = min(h, y2 + ph)

            # Crop the damaged area
            damage_crop = img_resized[y1_new:y2_new, x1_new:x2_new]

            # --- NEW STEP: Identify Part in this Crop ---
            # We run the crop through the Part Model
            if damage_crop.size == 0: continue

            part_results = part_model(damage_crop, conf=0.25, verbose=False)[0]

            # If the part model finds something (e.g. "Bumper")
            if len(part_results.boxes) > 0:
                # We take the class with highest confidence
                top_class_id = int(part_results.boxes.cls[0])
                part_name = part_model.names[top_class_id]
                current_image_parts.append(part_name)
                print(f"  -> Found Damage on: {part_name}")
            else:
                # If part model isn't sure, we can label it 'Unknown' or skip
                current_image_parts.append("Unknown Part")

        # Add the list of parts for this image to the master list
        all_images_parts.append(current_image_parts)

    return all_images_parts