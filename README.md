# SmartParking ANPR – AI-Powered Parking Management System

**SmartParking ANPR** is an intelligent parking management application that uses Computer Vision and Optical Character Recognition (OCR) to detect and recognize vehicle license plates from images.

The system combines YOLO, EasyOCR, OpenCV, Streamlit, and SQLite to support vehicle entry/exit management, parking history, and dashboard analytics.

## Live Demo

**Try the application:** [SmartParking ANPR – Live Demo](https://smartparking-k86i3gsnm8z8urxissadkt.streamlit.app/)

> The application runs on Streamlit Community Cloud. Initial loading and AI inference may take longer on free CPU resources.

## Key Features

### 1. Automatic License Plate Detection

- Upload vehicle images for analysis.
- Detect license plate regions using YOLO.
- Display bounding boxes and detection confidence.
- Retry detection with a lower confidence threshold when necessary.

### 2. License Plate Recognition

- Extract license plate text using EasyOCR.
- Apply OpenCV image preprocessing to improve readability.
- Normalize recognized characters.
- Display OCR confidence and alternative recognition results.
- Allow users to review and correct recognition errors.

### 3. Parking Entry and Exit Management

- Register vehicles entering the parking area.
- Record vehicle exit information.
- Track parking sessions and vehicle status.
- Maintain parking records using SQLite.

### 4. Dashboard and Statistics

- Monitor parking activity.
- View parking records and operational summaries.
- Visualize available statistics through the Streamlit dashboard.

## Technology Stack

| Component                     | Technology                |
| ----------------------------- | ------------------------- |
| Programming Language          | Python                    |
| User Interface                | Streamlit                 |
| Object Detection              | YOLO (Ultralytics)        |
| Optical Character Recognition | EasyOCR                   |
| Image Processing              | OpenCV                    |
| Deep Learning                 | PyTorch                   |
| Database                      | SQLite                    |
| Data Processing               | NumPy, Pandas             |
| Deployment                    | Streamlit Community Cloud |

## System Workflow

```text
Vehicle Image
     |
     v
Image Upload
     |
     v
YOLO License Plate Detection
     |
     v
License Plate Cropping
     |
     v
OpenCV Image Preprocessing
     |
     v
EasyOCR Text Recognition
     |
     v
License Plate Validation
     |
     v
User Review / Correction
     |
     v
Vehicle Entry / Exit
     |
     v
SQLite Database
     |
     v
Parking Dashboard
```

## Project Structure

```text
smartparking-anpr/
├── app.py                  # Streamlit application
├── detector.py             # YOLO license plate detection
├── ocr.py                  # OCR and image preprocessing
├── database.py             # Parking database operations
├── test_pipeline.py        # Recognition pipeline evaluation
├── models/
│   └── new_model/
│       └── best.pt         # YOLO model weights
├── requirements.txt        # Python dependencies
├── packages.txt            # Linux system dependencies
├── .gitignore
└── README.md
```

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

Open the local address displayed in the terminal, typically:

```text
http://localhost:8501
```

## Testing and Evaluation

The project includes a testing pipeline for evaluating license plate detection and OCR recognition using a collection of Vietnamese motorcycle license plate images.

The evaluation distinguishes between:

- Correctly recognized license plates.
- OCR recognition errors.
- Missed detections.

Results can be exported to a CSV report for further analysis.

Detection confidence and OCR confidence are not equivalent to overall model accuracy. Recognition performance depends on image quality, viewing angle, lighting, and license plate visibility.

## Limitations

- Recognition accuracy may decrease with blurry, low-resolution, or partially obscured license plates.
- Low-confidence detections require manual verification.
- CPU-based inference may be slower than GPU inference.
- SQLite storage on free cloud hosting is not guaranteed to persist across restarts or redeployments.
- The current application is a demonstration system and is not intended for unattended production parking operations.

## Future Improvements

- Improve Vietnamese license plate OCR accuracy.
- Optimize model inference speed.
- Add live camera and video processing.
- Migrate parking records to a persistent cloud database.
- Introduce user authentication and role-based access control.
- Improve dashboard analytics and reporting.

## Author

**Đinh Thành Vinh**

Backend Developer | AI/ML Enthusiast

**Project:** SmartParking ANPR – Intelligent License Plate Recognition and Parking Management System
