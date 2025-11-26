from ultralytics import YOLO
from PIL import Image
import supervision as sv
import cv2
import os
import numpy as np
def load_model():
    """
    Loads the pre-trained YOLO model for car damage segmentation.

    Returns:
        model: Loaded YOLO model.
    """
    print("Loading car damage segmentation model...")
    model = YOLO("models/car_damage_segmentation_model.pt")
    print("Model loaded successfully.")
    return model

def resize_img(img, target_size=(640, 640)):
    """
    Resizes the input image to the target size.

    Args:
        img (PIL.Image or np.ndarray or str): Input image or path.
        target_size (tuple): Desired size (width, height).

    Returns:
        resized_img (np.ndarray): Resized image in RGN color.
    """

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
            img_rgb  = img
        resized_np = cv2.resize(img_rgb, (w_t, h_t), interpolation=cv2.INTER_LANCZOS4)
        return resized_np
    else:
        raise TypeError("Unsupported image type for resize_img. Provide path, PIL.Image, or np.ndarray.")

def segment_damage(image_paths :list,min_confidence:float=.2,min_size = 256,padding :float= .5,output_dir:str="damage_segmentation_results"):
    """
    Segments damage areas in a car image.
    Args:
        image_paths (list): List of paths to input images.
        min_confidence (float): Minimum confidence threshold for detections.
        min_size (int): Minimum size for the cropped damage images.
        padding (float): Padding percentage around detected damage boxes.
        output_dir (str): Directory to save the cropped damage images.  
    """
    os.makedirs(output_dir, exist_ok=True)
    model = load_model()
    Damage_Classes = model.names
    for image_path in image_paths:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Could not read image: {image_path}")
            continue


        img_resized = resize_img(img_bgr, target_size=(640, 640))
        h, w = img_resized.shape[:2]

        results = model(img_resized, conf=min_confidence,iou=0.5,imgsz=640)[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections) == 0:
            print(f"No damage detected in {os.path.basename(image_path)}.")
            continue

        for idx in range(len(detections)):
            class_id = int(detections.class_id[idx])
            conf = float(detections.confidence[idx])
            box = detections.xyxy[idx]
            if isinstance(Damage_Classes, dict):
                damage = Damage_Classes.get(class_id, str(class_id))
            else:
                damage = Damage_Classes[class_id]

            x1, y1, x2, y2 = map(int, box)

            pw = int((x2 - x1) * padding)
            ph = int((y2 - y1) * padding)

            x1_new = max(0, x1 - pw)
            y1_new = max(0, y1 - ph)
            x2_new = min(w, x2 + pw)
            y2_new = min(h, y2 + ph)
            if x2_new <= x1_new or y2_new <= y1_new:
                print(f"Skipping invalid crop for {image_path} idx={idx}")
                continue
            crop = cv2.cvtColor(img_resized[y1_new:y2_new, x1_new:x2_new], cv2.COLOR_BGR2RGB)
            crop_h, crop_w = crop.shape[:2]

            size = max(min_size, max(crop_h, crop_w))
            scale = size / max(crop_h, crop_w)
            new_w = max(1, int(crop_w * scale))
            new_h = max(1, int(crop_h * scale))
            crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            
            base = os.path.splitext(os.path.basename(image_path))[0]
            crop_filename = os.path.join(output_dir, f"{base}_{damage}_{conf:.2f}_{idx:04d}.jpg")
            cv2.imwrite(crop_filename, cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
