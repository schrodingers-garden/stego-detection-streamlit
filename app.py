import time
import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# --- Keras Imports (For your old models) ---
from tensorflow.keras.models import load_model

# --- PyTorch Imports (For the new SRM model) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# =========================================================
# App Configuration
# =========================================================
st.set_page_config(
    page_title="AI Steganography Detection",
    page_icon="🕵️‍",
    layout="wide",
)

# =========================================================
# PYTORCH MODEL ARCHITECTURE (Required for loading)
# =========================================================

# 1. SRM Layer
class SRMConv2d(nn.Module):
    def __init__(self, stride=1, padding=0):
        super(SRMConv2d, self).__init__()
        self.in_channels = 3
        self.out_channels = 30
        self.kernel_size = (5, 5)
        self.padding = 2 if padding == 0 else padding
        self.stride = stride
        # Initialize weights (loaded later via state_dict)
        self.weight = nn.Parameter(torch.zeros(30, 3, 5, 5), requires_grad=False)
        self.threshold = 3.0 

    def forward(self, x):
        x = F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        x = torch.clamp(x, min=-self.threshold, max=self.threshold)
        return x

# 2. Main Network
class StegoNet(nn.Module):
    def __init__(self):
        super(StegoNet, self).__init__()
        self.srm = SRMConv2d()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Adapt first layer to accept 30 channels
        original_stem = self.backbone._conv_stem
        self.backbone._conv_stem = nn.Conv2d(
            in_channels=30, 
            out_channels=original_stem.out_channels, 
            kernel_size=original_stem.kernel_size, 
            stride=original_stem.stride, 
            padding=original_stem.padding, 
            bias=False
        )
        
        self.backbone._fc = nn.Linear(self.backbone._fc.in_features, 1)

    def forward(self, x):
        x = self.srm(x)
        x = self.backbone(x)
        return x

# =========================================================
# Global Styles (Restored from your original app)
# =========================================================
st.markdown(
    """
    <style>
        html, body, [class*="css"]  { font-family: "Segoe UI", sans-serif; }
        
        /* Buttons */
        .stButton>button, .stDownloadButton>button {
            color: #fff !important; background-color: #2f3333 !important;
            border: none; border-radius: 8px; box-shadow: 2px 2px 8px #55555555;
            font-weight: 600; font-size: 1.05rem; padding: 0.55rem 1.3rem;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background-color: #0f4cbd !important; box-shadow: 4px 4px 14px #44444488;
        }
        
        /* Sidebar Reset Button */
        div[data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #C0392B !important; color: #fff !important;
            border-radius: 8px; font-weight: 600; border: none; margin-top: 0.5rem;
        }

        /* Banner Styles (Restored binary background) */
        .app-banner {
            position: relative; margin-bottom: 1.5rem; padding: 1.4rem 1.8rem;
            border-radius: 14px; overflow: hidden;
            background: radial-gradient(circle at 0% 0%, #0f4cbd 0, #050816 45%, #02030a 100%);
            color: #f5f7fb; border: 1px solid #1f2937;
        }
        .app-banner::before {
            content: "01001001 00100000 01110011 01100101 01100101";
            position: absolute; top: -10px; left: -40px;
            font-family: "Consolas", monospace; font-size: 0.7rem; color: #3c82f6;
            opacity: 0.15; white-space: nowrap; transform: rotate(-15deg);
        }
        .app-banner::after {
            content: "01010011 01110100 01100101 01100111 01100001 01101110 01101111";
            position: absolute; bottom: -5px; right: -60px;
            font-family: "Consolas", monospace; font-size: 0.7rem; color: #22c55e;
            opacity: 0.15; white-space: nowrap; transform: rotate(12deg);
        }
        .app-banner-title { font-size: 1.65rem; font-weight: 700; margin-bottom: 0.25rem; }
        .app-banner-subtitle { font-size: 0.95rem; opacity: 0.9; }
        .app-banner-pill {
            display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.18rem 0.55rem;
            border-radius: 999px; background-color: #111827dd; border: 1px solid #374151;
            font-size: 0.75rem; margin-bottom: 0.35rem;
        }
        .app-banner-pill span.key { font-weight: 600; color: #93c5fd; }

        /* Results Card */
        .results-card {
            border: 2px solid #008080; border-radius: 10px; padding: 10px;
            background: #e8f7fa; margin-bottom: 15px;
        }
        .results-card table { width: 100%; }
        .results-card th, .results-card td { padding: 6px 8px; text-align: center; }
        .results-card thead { background-color: #d1f2f5; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Model configuration
# =========================================================
MODEL_CONFIG = {
    "SRM Model (Pytorch)": "models/best_stego_model.pth",  # New Model
    "Basic CNN (Keras)": "models/basic_cnn_model.keras",
    "ResNet50 (Keras)": "models/resnet50_model.keras",
}

MODEL_INPUT_SIZE = {
    "SRM Model (Pytorch)": (512, 512),
    "Basic CNN (Keras)": (256, 256),
    "ResNet50 (Keras)": (512, 512),
}

ALLOWED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# =========================================================
# Helper functions
# =========================================================

def predict_with_tta(model, image, device):
    """PyTorch: Rotates image 3 times and averages predictions."""
    model.eval()
    to_tensor = transforms.ToTensor()
    
    # Batch of 4 (Original + 3 Rotations)
    img_0 = to_tensor(image).unsqueeze(0)
    img_90 = to_tensor(image.rotate(90)).unsqueeze(0)
    img_180 = to_tensor(image.rotate(180)).unsqueeze(0)
    img_270 = to_tensor(image.rotate(270)).unsqueeze(0)
    
    batch = torch.cat([img_0, img_90, img_180, img_270], dim=0).to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.sigmoid(outputs)
    
    avg_prob = torch.mean(probs).item()
    return avg_prob

@st.cache_resource(show_spinner=False)
def load_stego_model(model_key: str):
    """Load model based on selection. Returns (model, device, engine_type)."""
    model_path = MODEL_CONFIG[model_key]

    # --- Case 1: PyTorch Model ---
    if "Pytorch" in model_key:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StegoNet()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device, "pytorch"

    # --- Case 2: Keras Models ---
    else:
        return load_model(model_path), None, "keras"

def preprocess_image_keras(image: Image.Image, model_key: str):
    """Old preprocessing for Keras models."""
    target_size = MODEL_INPUT_SIZE.get(model_key, (256, 256))
    im = image.convert("RGB")
    im = im.resize(target_size)
    arr = np.array(im) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def heuristic_score(image: Image.Image) -> float:
    """Fallback detector."""
    gray = image.convert("L").resize((64, 64))
    arr = np.array(gray, dtype="float32") / 255.0
    score = float(np.clip(arr.std(), 0.0, 1.0))
    return score

def compute_metrics(results):
    """Compute metrics if labels are present."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_true = [1 if r["expected_label"] == "Stego" else 0 for r in results]
    y_pred = [1 if r["predicted_label"] == "Stego" else 0 for r in results]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return acc, prec, rec, f1, cm

def render_confusion_matrix(cm):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Clean", "Stego"], yticklabels=["Clean", "Stego"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    st.pyplot(fig)

def validate_uploaded_images(files):
    if not files: return False, ["No files uploaded."]
    invalid = [f.name for f in files if not f.name.lower().endswith(ALLOWED_IMAGE_EXTENSIONS)]
    if invalid: return False, invalid
    return True, []

# =========================================================
# Sidebar Layout
# =========================================================
st.sidebar.header("Detection Controls")

model_choice = st.sidebar.selectbox(
    "Select model",
    list(MODEL_CONFIG.keys()),
    index=0,
    help="Choose between different trained models. 'SRM Model' is recommended.",
)

st.sidebar.caption(f"Model input size: {MODEL_INPUT_SIZE[model_choice]}")

threshold = st.sidebar.slider(
    "Suspicion threshold", 0.0, 1.0, 0.5, 0.01,
    help="Lower = More sensitive; Higher = More conservative."
)

st.sidebar.divider()

st.sidebar.info(
    """
    **How to use this tool**
    1. Select a **model** in the dropdown.
    2. Upload JPG/PNG images.
    3. (Optional) Upload a CSV (`filename, expected_label`) for evaluation.
    4. Click **Run detection**.
    """
)

if st.sidebar.button("Reset session", type="secondary"):
    for k in list(st.session_state.keys()): del st.session_state[k]
    st.rerun()

# =========================================================
# Main Layout
# =========================================================
st.markdown(
    """
    <div class="app-banner">
        <div class="app-banner-pill"><span class="key">Version:</span> <span>Beta 1.0</span></div>
        <div class="app-banner-title">AI Steganography Detection</div>
        <div class="app-banner-subtitle">Upload digital images to analyze hidden content.</div>
    </div>
    """, unsafe_allow_html=True
)

# Two-column layout for uploads
col_upload, col_csv = st.columns([2, 1])
with col_upload:
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, help="Drag and drop images here.")
with col_csv:
    csv_file = st.file_uploader("Optional CSV Labels", type=["csv"], help="CSV with 'filename' and 'expected_label' columns.")

# Parse CSV if provided
label_dict = {}
if csv_file:
    try:
        label_df = pd.read_csv(csv_file)
        if "filename" in label_df.columns and "expected_label" in label_df.columns:
            label_dict = dict(zip(label_df["filename"], label_df["expected_label"]))
        else:
            st.warning("CSV must have 'filename' and 'expected_label' columns.")
    except Exception as e:
        st.error(f"CSV Error: {e}")

# Preview Images
if uploaded_files:
    st.subheader("Selected image(s)")
    cols = st.columns(4)
    for idx, file in enumerate(uploaded_files):
        if idx < 4:
            cols[idx].image(file, caption=file.name, use_container_width=True)

run_detection = st.button("Run detection")

if "results" not in st.session_state: st.session_state["results"] = []

if run_detection:
    valid, invalid_files = validate_uploaded_images(uploaded_files)

    if not uploaded_files:
        st.error("No images uploaded.")
    elif not valid:
        st.error(f"Invalid files: {invalid_files}")
    else:
        results = []
        model_obj = None
        device = None
        engine_type = None
        model_loaded = False
        
        with st.spinner(f"Loading {model_choice}..."):
            try:
                model_obj, device, engine_type = load_stego_model(model_choice)
                model_loaded = True
            except Exception as e:
                st.error(f"Error loading model: {e}")
                model_loaded = False

        progress_bar = st.progress(0, text="Running detections...")
        total = len(uploaded_files)
        
        for i, file in enumerate(uploaded_files, start=1):
            image = Image.open(file).convert("RGB")
            start_time = time.perf_counter()

            if model_loaded:
                try:
                    if engine_type == "pytorch":
                        score = predict_with_tta(model_obj, image, device)
                        engine_label = "SRM (PyTorch)"
                    else: # Keras
                        input_arr = preprocess_image_keras(image, model_choice)
                        score = float(model_obj.predict(input_arr)[0][0])
                        engine_label = "Keras CNN"
                except Exception as e:
                    st.error(f"Prediction failed for {file.name}: {e}")
                    score = 0.0
                    engine_label = "Error"
            else:
                score = heuristic_score(image)
                engine_label = "Heuristic"

            pred_time = (time.perf_counter() - start_time) * 1000
            pred_label = "Stego" if score > threshold else "Clean"
            
            # Match with label
            raw_label = label_dict.get(file.name, "N/A")
            if raw_label != "N/A":
                expected_label = "Stego" if str(raw_label).lower() in ['1', 'stego'] else "Clean"
            else:
                expected_label = "N/A"

            results.append({
                "filename": file.name,
                "predicted_label": pred_label,
                "model_score": f"{score:.4f}",
                "expected_label": expected_label,
                "prediction_time_ms": f"{pred_time:.2f}",
                "engine": engine_label
            })
            progress_bar.progress(i / total)

        progress_bar.empty()
        st.session_state["results"] = results

# =========================================================
# Results Display
# =========================================================
if st.session_state["results"]:
    st.subheader("Results")
    df = pd.DataFrame(st.session_state["results"])
    
    st.markdown(f'<div class="results-card">{df.to_html(index=False)}</div>', unsafe_allow_html=True)
    
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "stego_results.csv", "text/csv")

    # Metrics Section
    if all(r["expected_label"] in ["Stego", "Clean"] for r in st.session_state["results"]):
        st.subheader("Batch Metrics")
        acc, prec, rec, f1, cm = compute_metrics(st.session_state["results"])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.2f}")
        c2.metric("Precision", f"{prec:.2f}")
        c3.metric("Recall", f"{rec:.2f}")
        c4.metric("F1 Score", f"{f1:.2f}")
        
        st.markdown("#### Confusion Matrix")
        render_confusion_matrix(cm)
    elif label_dict:
        st.info("Metrics not available: Some uploaded files did not match the provided CSV labels.")
