from ultralytics import YOLO
import cv2
import os


def detect_car_parts_only(image_paths: list, min_conf: float = 0.25):
    """
    Directly detects car parts in the provided images without looking for damage first.

    Args:
        image_paths (list): List of file paths to the images.
        min_conf (float): Confidence threshold.

    Returns:
        List[List[str]]: 2D array where each inner list contains the parts found in that image.
        Example: [['Bumper', 'Headlight'], ['Door'], []]
    """

    # 1. Load ONLY the Parts Model
    # Make sure this points to your parts detection model
    print("Loading Car Parts Model...")
    model = YOLO("models/Parts_Name_Classification.pt")

    # This will hold the final 2D array output
    all_images_parts = []

    for image_path in image_paths:
        print(f"Processing for parts: {image_path}")

        # List to hold parts found in THIS specific image
        current_image_parts = []

        # 2. Read and Resize
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Could not read: {image_path}")
            all_images_parts.append([])  # Maintain index with empty list
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
                current_image_parts.append(part_name)

            print(f"  -> Found parts: {current_image_parts}")
        else:
            print("  -> No parts detected.")

        # 5. Add to Master List
        all_images_parts.append(current_image_parts)

    return all_images_parts