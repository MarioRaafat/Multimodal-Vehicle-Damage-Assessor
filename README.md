# 🚗 Multimodal Vehicle Damage Assessor

An AI-powered vehicle damage assessment system that uses multiple deep learning models to detect, classify, and analyze car damage from images. The system generates comprehensive repair reports with cost estimates using RAG (Retrieval-Augmented Generation) technology.

**SIC Graduation Project**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Models](#-models)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Environment Variables](#-environment-variables)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The Multimodal Vehicle Damage Assessor is a complete end-to-end solution for automated vehicle damage assessment. It combines three specialized YOLO models with a RAG-based report generation system to provide:

1. **Damage Detection & Segmentation** - Identifies and crops damaged areas from vehicle images
2. **Parts Classification** - Determines which car parts are damaged
3. **Severity Assessment** - Classifies damage severity (Minor/Moderate/Severe)
4. **AI-Powered Reports** - Generates detailed repair recommendations with cost estimates

---

## ✨ Features

- 📷 **Multi-Image Upload** - Process multiple vehicle images simultaneously
- 🔍 **Instance Segmentation** - Precise damage area detection using YOLOv8-seg
- 🏷️ **Part Identification** - Automatic classification of 20+ car parts
- 📊 **Severity Grading** - Three-level severity classification system
- 🤖 **AI Report Generation** - Comprehensive reports using Llama 3.3 + Gemini 2.5
- 🌐 **Web Search Integration** - Real-time repair cost data via Serper API
- 📄 **PDF Export** - Download professional damage assessment reports
- 🎨 **Interactive UI** - User-friendly Streamlit web interface

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT WEB APP                             │
│                         (Multi-Image Upload UI)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                             │
│                     (src/pipelines/inference_pipeline.py)               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  DAMAGE DETECTION │   │ PARTS CLASSIFIER  │   │ SEVERITY DETECTOR │
│    (YOLOv8-seg)   │   │    (YOLOv11)      │   │     (YOLOv8)      │
│                   │   │                   │   │                   │
│ • Instance Seg    │   │ • 11  Part Types  │   │ • Minor           │
│ • Damage Cropping │   │ • Confidence      │   │ • Moderate        │
│ • 4 Damage Types  │   │   Scoring         │   │ • Severe          │
└───────────────────┘   └───────────────────┘   └───────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG REPORT GENERATOR                            │
│                                                                         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                │
│  │   GROQ      │     │   SERPER    │     │   GEMINI    │                │
│  │  Llama 3.3  │───▶│  Web Search │───▶│  2.5 Flash  │                │
│  │  (Reasoning)│     │  (Cost Data)│     │  (Reports)  │                │
│  └─────────────┘     └─────────────┘     └─────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────┐
                        │   PDF REPORT      │
                        │   (xhtml2pdf)     │
                        └───────────────────┘
```

---

## 🧠 Models

### 1. Damage Segmentation Model (YOLOv8-seg)

| Property | Value |
|----------|-------|
| **Model File** | `models/car_damage_segmentation_model.pt` |
| **Architecture** | YOLOv8 Instance Segmentation |
| **Training Dataset** | [Car Damage V5](https://roboflow.com/car-damage-kadad/car-damage-v5) |
| **Classes** | Dent, Scratch, Crack, Glass Shatter, Lamp Broken, Tire Flat, and more |
| **Task** | Instance segmentation and damage area cropping |

**Sample Output:**
```python
{
    "crop_001.jpg": {
        "damage_type": "dent",
        "confidence": 0.92,
        "source": "original_image.jpg",
        "index": 0
    }
}
```

---

### 2. Parts Classification Model (YOLOv11)

| Property | Value |
|----------|-------|
| **Model File** | `models/Parts_Name_Classification.pt` |
| **Architecture** | YOLOv11 Object Detection |
| **Training Dataset** | [Car Parts Dataset](https://roboflow.com/od-phi6w/car-parts-c1c2u/dataset/9) |
| **Classes** | 20+ car parts including bumper, fender, hood, door, headlight, etc. |
| **Task** | Identify which car part is damaged |

**Sample Output:**
```python
{
    "crop_001.jpg": [
        {
            "part_name": "front_bumper",
            "confidence": 0.89,
            "box": [x1, y1, x2, y2]
        }
    ]
}
```

---

### 3. Damage Severity Model (YOLOv8)

| Property | Value |
|----------|-------|
| **Model File** | `models/damage_severity_level_model.pt` |
| **Architecture** | YOLOv8 Classification |
| **Training Dataset** | [Car Damage Severity Dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset) |
| **Classes** | Minor, Moderate, Severe |
| **Task** | Classify the severity level of each damage instance |

**Sample Output:**
```python
{
    "crop_001.jpg": {
        "severity": "moderate",
        "confidence": 0.85,
        "all_scores": {
            "minor": 0.10,
            "moderate": 0.85,
            "severe": 0.05
        }
    }
}
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster inference)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Multimodal-Vehicle-Damage-Assessor.git
cd Multimodal-Vehicle-Damage-Assessor
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r src/requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
SERPER_API_KEY=your_serper_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

### Step 5: Download Models

Ensure the following model files are in the `models/` directory:
- `car_damage_segmentation_model.pt`
- `Parts_Name_Classification.pt`
- `damage_severity_level_model.pt`

---

## 💻 Usage

### Running the Streamlit App

```bash
cd src/App
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App

1. **Upload Images** - Select one or more vehicle damage images
2. **Enter Car Details** - Provide make, model, and year
3. **Analyze** - Click "Analyze Damage" to process
4. **Download Report** - Get a comprehensive PDF report

### Programmatic Usage

```python
from src.pipelines.inference_pipeline import run_inference

# Run full pipeline
pdf_path = run_inference(
    image_paths=["path/to/image1.jpg", "path/to/image2.jpg"],
    car_make="Toyota",
    car_model="Camry",
    car_year="2020"
)

print(f"Report saved to: {pdf_path}")
```

### Using Individual Modules

```python
# Damage Segmentation
from src.inference.damage_segmentation import segment_damage

cropped_images, labels = segment_damage(["image.jpg"])

# Parts Classification
from src.inference.Parts_name_classification import detect_car_parts_only

parts = detect_car_parts_only(cropped_images)

# Severity Detection
from src.inference.damage_severity import DamageSeverityDetector

detector = DamageSeverityDetector()
severity = detector.batch_detect(cropped_images)
```

---

## 📁 Project Structure

```
Multimodal-Vehicle-Damage-Assessor/
│
├── 📄 README.md                    # This file
├── 📄 .gitignore                   # Git ignore rules
├── 📄 .env                         # Environment variables (create this)
│
├── 📁 models/                      # Trained YOLO models
│   ├── car_damage_segmentation_model.pt
│   ├── Parts_Name_Classification.pt
│   └── damage_severity_level_model.pt
│
├── 📁 imgs/                        # Sample/test images
│
├── 📁 reports/                     # Generated PDF reports
│
└── 📁 src/                         # Source code
    ├── 📄 __init__.py
    ├── 📄 requirements.txt         # Python dependencies
    │
    ├── 📁 App/                     # Streamlit web application
    │   ├── 📄 __init__.py
    │   └── 📄 app.py               # Main Streamlit app
    │
    ├── 📁 inference/               # ML inference modules
    │   ├── 📄 __init__.py
    │   ├── 📄 damage_segmentation.py    # YOLOv8-seg damage detection
    │   ├── 📄 Parts_name_classification.py  # YOLOv11 parts classifier
    │   ├── 📄 damage_severity.py        # YOLOv8 severity classifier
    │   └── 📄 RAG.py                    # AI report generation
    │
    └── 📁 pipelines/               # Orchestration pipelines
        ├── 📄 __init__.py
        └── 📄 inference_pipeline.py # Main inference pipeline
```

---

## 📚 API Reference

### `segment_damage(image_paths: List[str]) -> Tuple[List[str], Dict]`

Detects and segments damage areas from input images.

**Parameters:**
- `image_paths`: List of paths to input images

**Returns:**
- `cropped_images`: List of paths to cropped damage images
- `labels`: Dictionary mapping crop filenames to damage metadata

---

### `detect_car_parts_only(image_paths: List[str]) -> Dict[str, List[Dict]]`

Classifies car parts in the given images.

**Parameters:**
- `image_paths`: List of paths to cropped damage images

**Returns:**
- Dictionary mapping image paths to detected parts with confidence scores

---

### `DamageSeverityDetector.batch_detect(image_paths: List[str]) -> Dict[str, Dict]`

Classifies severity level for each damage image.

**Parameters:**
- `image_paths`: List of paths to damage images

**Returns:**
- Dictionary mapping image paths to severity classification results

---

### `process_full_case(combined_results, car_details) -> str`

Generates a comprehensive damage assessment report.

**Parameters:**
- `combined_results`: Dictionary containing damage analysis results
- `car_details`: Dictionary with `make`, `model`, and `year`

**Returns:**
- Path to the generated PDF report

---

### `run_inference(image_paths, car_make, car_model, car_year) -> str`

Runs the complete inference pipeline.

**Parameters:**
- `image_paths`: List of input image paths
- `car_make`: Vehicle manufacturer
- `car_model`: Vehicle model name
- `car_year`: Vehicle year

**Returns:**
- Path to the generated PDF report

---

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | API key for Groq (Llama 3.3) | Yes |
| `SERPER_API_KEY` | API key for Serper (web search) | Yes |
| `GOOGLE_API_KEY` | API key for Google Gemini | Yes |

### Getting API Keys

1. **Groq API Key**: Sign up at [console.groq.com](https://console.groq.com)
2. **Serper API Key**: Sign up at [serper.dev](https://serper.dev)
3. **Google API Key**: Get from [Google AI Studio](https://aistudio.google.com)

---

## 📦 Dependencies

```
ultralytics>=8.0.0      # YOLO models
streamlit>=1.0.0        # Web interface
opencv-python>=4.5.0    # Image processing
supervision>=0.3.0      # Detection utilities
groq>=0.4.0             # LLM API
google-generativeai>=0.3.0  # Gemini API
requests>=2.28.0        # HTTP requests
python-dotenv>=1.0.0    # Environment variables
xhtml2pdf>=0.2.11       # PDF generation
numpy>=1.21.0           # Numerical operations
Pillow>=9.0.0           # Image handling
```

---

## 🔄 Pipeline Flow

```
Input Images
     │
     ▼
┌─────────────────────────────────────────┐
│         1. DAMAGE SEGMENTATION          │
│  • Load YOLOv8-seg model                │
│  • Detect damage instances              │
│  • Crop and save damaged areas          │
│  • Return: cropped images + labels      │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         2. PARTS CLASSIFICATION         │
│  • Load YOLOv11 model                   │
│  • Classify car parts in crops          │
│  • Select highest confidence match      │
│  • Return: part names per image         │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         3. SEVERITY ASSESSMENT          │
│  • Load YOLOv8 classifier               │
│  • Evaluate damage severity             │
│  • Classify: Minor/Moderate/Severe      │
│  • Return: severity per image           │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│         4. RAG REPORT GENERATION        │
│  • Groq: Decide repair strategy         │
│  • Serper: Search repair costs          │
│  • Gemini: Generate detailed report     │
│  • xhtml2pdf: Convert to PDF            │
│  • Return: PDF report path              │
└─────────────────────────────────────────┘
     │
     ▼
   PDF Report
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO models
- [Roboflow](https://roboflow.com/) for training datasets
- [Groq](https://groq.com/) for fast LLM inference
- [Google AI](https://ai.google/) for Gemini API
- [Streamlit](https://streamlit.io/) for the web framework

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

<p align="center">
  Made with ❤️ for the automotive industry
</p>
