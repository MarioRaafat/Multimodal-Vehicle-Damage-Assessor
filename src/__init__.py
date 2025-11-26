"""
Multimodal Vehicle Damage Assessor - Main Package
Provides integrated damage assessment, parts classification, severity detection, and reporting.
"""

__version__ = "0.1.0"
__author__ = "MarioRaafat"

# Re-export main pipeline for convenient top-level imports
from .pipelines.inference_pipeline import inference_pipeline

# Re-export core inference modules
from .inference.damage_segmentation import segment_damage, load_model, resize_img
from .inference.Parts_name_classification import detect_car_parts_only
from .inference.damage_severity import DamageSeverityDetector
from .inference.RAG import process_full_case, generate_final_report, search_web, decide_repair_strategy

__all__ = [
    "inference_pipeline",
    "segment_damage",
    "load_model",
    "resize_img",
    "detect_car_parts_only",
    "DamageSeverityDetector",
    "process_full_case",
    "generate_final_report",
    "search_web",
    "decide_repair_strategy",
]
