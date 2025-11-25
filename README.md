# AI Steganography Detection App

## Overview
This project provides an AI-enabled steganography detection tool that analyzes digital images to estimate whether hidden data (steganography) may be embedded within them. The application uses machine learning models trained on steganographic datasets and provides a user-friendly Streamlit interface for upload, evaluation, and reporting.

This tool was developed as part of an academic capstone project in Information Systems with a cybersecurity focus, exploring practical applications of machine learning for digital forensics.

---

## Features
- Upload single or batch images (JPG/PNG)
- AI-based steganography detection
- Model selection:
  - Basic CNN (Keras)
  - ResNet50 (Keras)
- Adjustable detection threshold
- Visual preview of uploaded images
- Results table with:
  - Filename
  - Predicted label (Clean/Stego)
  - Model score
  - Model engine
  - Expected label (if provided)
- Optional evaluation using labeled CSV
- Export results to CSV
- Automated metrics reporting:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix visualization
  - Performance tracking

---

## Folder Structure
ai-stego-app/

├── app.py                       # Main Streamlit application

├── requirements.txt

├── models/

│   ├── basic_cnn_model.keras

│   └── resnet50_model.keras

├── sample-images/               # Optional sample images

└── README.md

---

## Getting Started

### Prerequisites
- Python 3.8+
- pip or conda
- Streamlit
- TensorFlow / Keras
- scikit-learn
- (Optional) Streamlit Community Cloud account

---

## Installation

1. Clone the repository:

git clone https://github.com/schrodingers-garden/ai-stego-app.git
cd ai-stego-app

2. Install dependencies:

pip install -r requirements.txt

3. Run the application locally:

streamlit run app.py

---

## Usage
1. Select a model in the sidebar
2. Upload one or more images (JPG/PNG)
3. (Optional) Upload a labeled CSV with:
   filename, expected_label
4. Adjust the detection threshold
5. Click "Run detection"
6. Review predictions and metrics
7. Download results as CSV for reporting or audit
8. Select "reset session" to clear results.

---

## Streamlit Community Cloud Deployment
This application can be deployed for free using Streamlit Community Cloud:

1. Upload repository to GitHub
2. Connect Streamlit Cloud to GitHub
3. Deploy using:
   app.py as the entry point

Or visit: https://stego-detection-app-aumgajizq2ensgnco4ruuf.streamlit.app/

---

## Requirements
See `requirements.txt` for full dependency list.

---

## License
This project is for academic, non-commercial use only.

---

## Contact
Author: Kristy Seelnacht-Colombo
Email: kristyseelnacht-colombo2024@u.northwestern.edu

For issues or feature requests, please use the GitHub Issues tab.
