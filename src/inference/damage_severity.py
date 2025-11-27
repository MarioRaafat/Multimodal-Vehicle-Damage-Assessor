import os
import platform
from pathlib import Path, PosixPath, WindowsPath
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Union, Dict, List, Tuple

# for windows compatibility
if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath


# configs
CURR_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = CURR_DIR.parent.parent / "models" / "damage_severity_level_model.pt"
CLASS_NAMES = ["minor", "moderate", "severe"]


def resize_image(image: np.ndarray, target_size: int = 640) -> np.ndarray:
    height, width = image.shape[:2]

    if height > width:
        scale = target_size / height
    else:
        scale = target_size / width

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return resized_image


# levels are: minor, moderate, severe
class DamageSeverityDetector:
    # use the path of the model or leave None to use default path
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        self.model = YOLO(str(self.model_path))

    def detect_severity_level(
        self, 
        image_source: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640
    ) -> Dict:
        """
        Detect damage severity level in the given image.
        
        Args:
            image_source (Union[str, np.ndarray]): Path to image file or numpy array
            conf_threshold (float): Confidence threshold for detections (0-1)
            iou_threshold (float): IoU threshold for NMS
            imgsz (int): Image size for inference
        
        Returns:
            Dict: Dictionary containing:
                - 'detections': List of detection dictionaries with bbox, confidence, class
                - 'highest_severity': The most severe damage level detected
                - 'annotated_image': Image with drawn bounding boxes and labels
        """
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                raise FileNotFoundError(f"Image not found: {image_source}")
            image = cv2.imread(image_source)
            if image is None:
                raise ValueError(f"Failed to load image: {image_source}")
        else:
            image = image_source.copy()
        
        # Resize image for inference
        image = resize_image(image, target_size=imgsz)

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            verbose=False
        )

        result = results[0]
        detections = []
        class_names = self.model.names
        
        # Extract detection information
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                class_name = class_names[cls_id]
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': class_name,
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                }
                
                detections.append(detection)

        highest_severity = self._determine_highest_severity(result.probs)
        annotated_image = result.plot()
        
        return {
            'detections': detections,
            'highest_severity': highest_severity,
            'annotated_image': annotated_image,
        }
    
    def _determine_highest_severity(self, probs):
        if probs is None:
            return "No damage detected"

        try:
            # support torch tensors and numpy arrays
            if hasattr(probs, 'cpu'):
                arr = probs.cpu().numpy()
            else:
                arr = np.array(probs)

            # flatten to 1D and take argmax
            arr = arr.ravel()
            max_idx = int(np.argmax(arr)) if arr.size > 0 else None
            class_names = self.model.names
            if max_idx is None or max_idx < 0 or max_idx >= len(class_names):
                return "No damage detected"
            return class_names[max_idx]
        except Exception:
            return "No damage detected"
    
    def batch_detect(
        self,
        image_sources: List[Union[str, np.ndarray]],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640
    ) -> Dict[str, Dict]:
        """
        Run detect_severity_level on a batch of image sources and return a mapping
        keyed by the image source (string) to the result dict for that image.

        Returns:
            Dict[str, Dict]: { image_source_str: result_dict }
        """
        results = {}
        for image_source in image_sources:
            key = str(image_source)
            try:
                result = self.detect_severity_level(
                    image_source,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    imgsz=imgsz
                )
                results[key] = result
            except Exception as e:
                print(f"Error processing image {key}: {e}")
                results[key] = {'error': str(e)}

        return results
    

if __name__ == "__main__":
    model_class = DamageSeverityDetector()

    imgs_dir = CURR_DIR.parent.parent / "imgs"
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(imgs_dir.glob(ext)))

    results = model_class.batch_detect([str(img_path) for img_path in image_paths])

    # Display results for each image using the returned dict keyed by image path
    for img_path in image_paths:
        key = str(img_path)
        result = results.get(key, {'error': 'no result'})
        if 'error' in result:
            print(f"\n{img_path.name}: Error - {result['error']}")
            continue

        cv2.imshow(f"Severity Detection - {img_path.name}", result['annotated_image'])

        key_press = cv2.waitKey(0)
        if key_press == ord('q'):
            break

        cv2.destroyAllWindows()

    cv2.destroyAllWindows()