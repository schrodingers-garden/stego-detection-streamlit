# AI Steganography Detection App

## Overview
This project provides an AI-enabled steganography detection tool that analyzes digital images to estimate whether hidden data (steganography) may be embedded within them. The application uses a specialized **Hybrid Deep Learning Architecture** designed to detect steganographic noise residuals that traditional computer vision models often miss.

The tool provides a user-friendly Streamlit interface for upload, evaluation, and reporting. It was developed as part of an academic capstone project in Information Systems with a cybersecurity focus, exploring practical applications of machine learning for digital forensics.

---

## Technical Approach
Steganalysis differs from standard image classification because the signal (the hidden message) is indistinguishable from high-frequency noise. Standard CNNs (like ResNet) are designed to *ignore* noise to focus on content (e.g., "this is a cat").

To solve this, this project implements a **Spatial Rich Model (SRM) + EfficientNet** hybrid:
1.  **Preprocessing:** A custom, fixed-weight Convolutional layer initialized with 30 SRM filters to extract noise residuals and suppress image content.
2.  **Backbone:** An EfficientNet-B0 architecture trained on the noise maps rather than raw pixels.
3.  **Inference:** Uses **Test Time Augmentation (TTA)**, rotating images during analysis to verify hidden signals across different orientations.

---

## Features
- Upload single or batch images (JPG/PNG)
- Primary Detection Engine:
  - SRM + EfficientNet (PyTorch): High-accuracy model using residual noise extraction and TTA.
- Baseline/Demo Models:
  - Basic CNN (Keras) - *Included for UI demonstration*
  - ResNet50 (Keras) - *Included for UI demonstration*
- Adjustable detection threshold
- Visual preview of uploaded images
- Results table with:
  - Filename
  - Predicted label (Clean/Stego)
  - Model score (Probability)
  - Processing time
  - Engine used
- Optional evaluation using labeled CSV
- Automated metrics reporting:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix visualization

---

## Folder Structure
```text
ai-stego-app/
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── best_stego_model.pth         # PRIMARY MODEL (PyTorch/SRM)
├── models/
│   ├── basic_cnn_model.keras    # Demo model
│   └── resnet50_model.keras     # Demo model
├── sample-images/               # Optional sample images
└── README.md
```
---

## Getting Started

### Prerequisites
- Python 3.8+
- pip or conda
- Streamlit
- TensorFlow / Keras (for legacy demo models)
- PyTorch & Torchvision (for SRM Model)
- EfficientNet-PyTorch
- scikit-learn
- (Optional) Streamlit Community Cloud account

---

## Installation

1. Clone the repository:
```text
git clone https://github.com/schrodingers-garden/ai-stego-app.git
cd ai-stego-app
```
2. Install dependencies:
```text
pip install -r requirements.txt
```
3. Run the application locally:
```text
streamlit run app.py
```
---

## Usage
1. Select a model in the sidebar. Choose "SRM Model (Pytorch)" for best accuracy (recommended). Other models are available for comparison/demo purposes.
2. Upload one or more images (JPG/PNG)
3. (Optional) Evaluation: Upload a labeled CSV with filename, expected_label columns to calculate accuracy metrics.
4. Adjust the detection threshold.
5. Click "Run detection". The PyTorch model will utilize Test Time Augmentation (TTA), so processing may take slightly longer per image than standard inference.
6. Review predictions and metrics.
7. Download results as CSV for reporting or audit.
8. Select "reset session" to clear results.

---

## Streamlit Community Cloud Deployment
This application can be deployed for free using Streamlit Community Cloud:

1. Upload repository to GitHub
2. Connect Streamlit Cloud to GitHub
3. Deploy using:
   app.py as the entry point

Note: This application is deployed and also available for testing at: https://stego-detection-app-aumgajizq2ensgnco4ruuf.streamlit.app/

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
