"""
Inference Package - Core ML models for damage assessment, parts classification, severity detection, and reporting.
"""

from .damage_segmentation import segment_damage, load_model, resize_img
from .Parts_name_classification import detect_car_parts_only
from .damage_severity import DamageSeverityDetector
from .RAG import (
    process_full_case,
    generate_final_report,
    search_web,
    decide_repair_strategy,
    convert_html_to_pdf,
)

__all__ = [
    "segment_damage",
    "load_model",
    "resize_img",
    "detect_car_parts_only",
    "DamageSeverityDetector",
    "process_full_case",
    "generate_final_report",
    "search_web",
    "decide_repair_strategy",
    "convert_html_to_pdf",
]
