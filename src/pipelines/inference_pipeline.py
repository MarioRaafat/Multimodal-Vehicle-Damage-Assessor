from inference.damage_segmentation import segment_damage
from inference.Parts_name_classification import detect_car_parts_only
from inference.damage_severity import DamageSeverityDetector
from inference.RAG import process_full_case

def inference_pipeline(image_paths :list,car_info: str):
    
    """
    Full inference pipeline that segments damage and classifies part names and classifies serverity of damaged parts.

    Args:
        image_paths (list): List of file paths to the images. 
    Returns:
        pdf report path (str): Path to the generated PDF report.
    """

    # 1. Segment Damage
    print("Starting Damage Segmentation...")
    cropped_image_paths, damage_labels = segment_damage(image_paths, min_confidence=0.25, min_size = 256,padding =.5)

    if not cropped_image_paths:
        print("No damage detected in any images. Exiting pipeline.")
        return None

    # 2. Classify Parts Names
    print("Starting Parts Name Classification...")
    parts_metadata = detect_car_parts_only(cropped_image_paths, min_conf=0.25)

    print("Starting severity damage classification...")
    # 2.5 Classify Damage Severity Levels
    severity_detector = DamageSeverityDetector()
    # run batch detection on the cropped images (paths)
    severity_results = severity_detector.batch_detect(cropped_image_paths, conf_threshold=0.25, iou_threshold=0.45, imgsz=640)
    # 3. Combine Results
    combined_results = []
    for crop_path in cropped_image_paths:
        damage_info = damage_labels.get(crop_path, {})
        parts_info = parts_metadata.get(crop_path, [])
        
        # Get damage_type from damage_info
        damage_type = damage_info.get('damage_type', 'no damage detected')
        
        # Select highest-confidence detected part for this crop
        part_name = None
        if isinstance(parts_info, list) and parts_info:
            # parts_info is a list of dicts with 'part_name' and 'confidence'
            best_part = max(parts_info, key=lambda x: x.get('confidence', 0.0))
            part_name = best_part.get('part_name') or best_part.get('class_name')
        
        if not part_name:
            part_name = 'Unknown Part'
        
        # Only keep the highest severity label for the crop (string) to keep results compact
        severity_info = severity_results.get(crop_path, {})
        highest_severity = None
        if isinstance(severity_info, dict):
            highest_severity = severity_info.get('highest_severity')

        combined_results.append({
            'damage_type': damage_type,
            'part_name': part_name,
            'severity': highest_severity
        })

    # 4. Generate PDF Report using RAG
    pdf_path = process_full_case(car_info, combined_results)
    
    return pdf_path

