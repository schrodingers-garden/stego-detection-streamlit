import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

# =========================================================
# App Configuration
# =========================================================
st.set_page_config(
    page_title="AI Steganography Detection",
    page_icon="🕵️‍♀️",
    layout="wide",
)

# =========================================================
# Global Styles (Buttons, Sidebar, Banner, Table)
# =========================================================
st.markdown(
    """
    <style>
        /* Global font tweaks */
        html, body, [class*="css"]  {
            font-family: "Segoe UI", sans-serif;
        }

        /* Primary buttons & download buttons */
        .stButton>button, .stDownloadButton>button {
            color: #fff !important;
            background-color: #2f3333 !important;
            border: none;
            border-radius: 8px;
            box-shadow: 2px 2px 8px #55555555;
            font-weight: 600;
            font-size: 1.05rem;
            padding: 0.55rem 1.3rem;
        }
        .stButton>button:hover, .stDownloadButton>button:hover {
            background-color: #0f4cbd !important;
            color: #fff !important;
            box-shadow: 4px 4px 14px #44444488;
        }

        /* Sidebar reset button styling */
        div[data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #C0392B !important;
            color: #fff !important;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            border: none;
            box-shadow: 2px 2px 8px #82231b44;
            margin-top: 0.5rem;
        }
        div[data-testid="stSidebar"] button[kind="secondary"]:hover {
            background-color: #A93226 !important;
            color: #fff !important;
        }

        /* Sidebar info text color */
        .stAlert, .stAlert * {
            color: #111 !important;
        }

        /* Custom banner container */
        .app-banner {
            position: relative;
            margin-bottom: 1.5rem;
            padding: 1.4rem 1.8rem;
            border-radius: 14px;
            overflow: hidden;
            background: radial-gradient(circle at 0% 0%, #0f4cbd 0, #050816 45%, #02030a 100%);
            color: #f5f7fb;
            border: 1px solid #1f2937;
        }

        .app-banner::before {
            content: "01001001 00100000 01110011 01100101 01100101 00100000 01111001 01101111 01110101";
            position: absolute;
            top: -10px;
            left: -40px;
            font-family: "Consolas", monospace;
            font-size: 0.7rem;
            color: #3c82f6;
            opacity: 0.15;
            white-space: nowrap;
            transform: rotate(-15deg);
        }

        .app-banner::after {
            content: "01010011 01110100 01100101 01100111 01100001 01101110 01101111 01100111 01110010 01100001 01110000 01101000 01111001";
            position: absolute;
            bottom: -5px;
            right: -60px;
            font-family: "Consolas", monospace;
            font-size: 0.7rem;
            color: #22c55e;
            opacity: 0.15;
            white-space: nowrap;
            transform: rotate(12deg);
        }

        .app-banner-title {
            font-size: 1.65rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .app-banner-subtitle {
            font-size: 0.95rem;
            opacity: 0.9;
            max-width: 580px;
        }

        .app-banner-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            background-color: #111827dd;
            border: 1px solid #374151;
            font-size: 0.75rem;
            margin-bottom: 0.35rem;
        }

        .app-banner-pill span.key {
            font-weight: 600;
            color: #93c5fd;
        }

        /* Results table container */
        .results-card {
            border: 2px solid #008080;
            border-radius: 10px;
            padding: 10px;
            background: #e8f7fa;
            margin-bottom: 15px;
        }

        .results-card table {
            width: 100%;
        }

        .results-card thead {
            background-color: #d1f2f5;
        }

        .results-card th, .results-card td {
            padding: 6px 8px;
            font-size: 0.9rem;
        }

        .results-card th {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Model configuration (two models)
# =========================================================
# Update these paths to match your actual filenames in the models/ folder
MODEL_CONFIG = {
    "Basic CNN (Keras)": "models/basic_cnn_model.keras",
    "ResNet50 (Keras)": "models/resnet50_model.keras",
}

MODEL_INPUT_SIZE = {
    "Basic CNN (Keras)": (256, 256),
    "ResNet50 (Keras)": (512, 512),
}

# =========================================================
# Helper functions
# =========================================================

@st.cache_resource(show_spinner=True)
def load_stego_model(model_key: str):
    """Load and cache the selected Keras model based on model_key."""
    model_path = MODEL_CONFIG[model_key]
    return load_model(model_path)


def preprocess_image(image: Image.Image, model_key: str):
    """Resize based on model selection and normalize."""
    target_size = MODEL_INPUT_SIZE.get(model_key, (256, 256))

    im = image.convert("RGB")
    im = im.resize(target_size)
    arr = np.array(im) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def compute_metrics(results):
    """Compute batch metrics if ground-truth labels are available for all images."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    y_true = [1 if r["expected_label"] == "Stego" else 0 for r in results]
    y_pred = [1 if r["predicted_label"] == "Stego" else 0 for r in results]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return acc, prec, rec, f1, cm


def render_confusion_matrix(cm):
    """Render confusion matrix as a heatmap with seaborn/matplotlib."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Clean", "Stego"],
        yticklabels=["Clean", "Stego"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


# =========================================================
# Sidebar Layout
# =========================================================
st.sidebar.header("Detection Controls")

# Model selection dropdown
model_choice = st.sidebar.selectbox(
    "Select model",
    list(MODEL_CONFIG.keys()),
    index=0,
    help="Choose between different trained Keras models.",
)

st.sidebar.caption(f"Model input size: {MODEL_INPUT_SIZE[model_choice]}")

threshold = st.sidebar.slider(
    "Suspicion threshold",
    0.0,
    1.0,
    0.5,
    0.01,
    help=(
        "Lower = More sensitive (more images flagged as stego); "
        "Higher = More conservative (requires stronger model confidence)."
    ),
)

st.sidebar.divider()

st.sidebar.info(
    """
    **How to use this tool**

    1. Select a **model** in the dropdown.
    2. Upload one or more JPG/PNG images.
    3. (Optional) Upload a CSV (`filename, expected_label`) for evaluation.
    4. Adjust the **suspicion threshold** for sensitivity.
    5. Click **Run detection** to see predictions.
    6. Download results as CSV or review batch metrics.
    """
)

reset_action = st.sidebar.button("Reset session", type="secondary")

if reset_action:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    # ✅ Updated: use st.rerun() instead of deprecated st.experimental_rerun()
    st.rerun()

# =========================================================
# Main Banner
# =========================================================
st.markdown(
    """
    <div class="app-banner">
        <div class="app-banner-pill">
            <span class="key">Version:</span> <span>Beta 1.0</span>
        </div>
        <div class="app-banner-title">AI Steganography Detection</div>
        <div class="app-banner-subtitle">
            Upload digital images and let the selected model estimate the likelihood of hidden content.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# File Uploads
# =========================================================
col_upload, col_csv = st.columns([2, 1])

with col_upload:
    uploaded_files = st.file_uploader(
        "Upload one or more images for analysis",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="You can drag-and-drop multiple images here.",
    )

with col_csv:
    csv_file = st.file_uploader(
        "Optional labeled CSV for evaluation",
        type=["csv"],
        help="Expected structure: `filename, expected_label` where expected_label is 0/1 or Clean/Stego.",
    )

# Build filename -> expected_label mapping (if provided)
label_dict = {}
if csv_file is not None:
    try:
        label_df = pd.read_csv(csv_file)
        # Allow for flexibility: expected_label can be 0/1 or strings
        if "filename" in label_df.columns and "expected_label" in label_df.columns:
            label_dict = dict(zip(label_df["filename"], label_df["expected_label"]))
        else:
            st.warning(
                "CSV must contain at least two columns named `filename` and `expected_label`."
            )
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# =========================================================
# Preview Section
# =========================================================
if uploaded_files:
    st.subheader("Selected image(s)")
    # Display thumbnails in a responsive layout
    thumbs_per_row = 4
    rows = (len(uploaded_files) + thumbs_per_row - 1) // thumbs_per_row

    idx = 0
    for _ in range(rows):
        cols = st.columns(thumbs_per_row)
        for col in cols:
            if idx < len(uploaded_files):
                file = uploaded_files[idx]
                image = Image.open(file)
                # ✅ Updated: use use_container_width instead of deprecated use_column_width
                col.image(
                    image,
                    caption=file.name,
                    use_container_width=True,
                )
                idx += 1

# =========================================================
# Prediction
# =========================================================
run_detection = st.button("Run detection")

if "results" not in st.session_state:
    st.session_state["results"] = []

if run_detection and uploaded_files:
    results = []

    # First: load the selected model (lazy-load)
    with st.spinner(f"Loading {model_choice} model..."):
        model = load_stego_model(model_choice)

    # Then: run predictions with a progress bar
    total = len(uploaded_files)
    progress_text = "Running model predictions..."
    progress_bar = st.progress(0, text=progress_text)

    for i, file in enumerate(uploaded_files, start=1):
        image = Image.open(file)
        input_arr = preprocess_image(image, model_choice)
        score = float(model.predict(input_arr)[0][0])

        predicted_label = "Stego" if score > threshold else "Clean"

        # Normalize expected label if available
        raw_label = label_dict.get(file.name)
        if raw_label is None:
            expected_label = "N/A"
        else:
            # Accept 0/1 or strings like "Clean"/"Stego"
            if isinstance(raw_label, str):
                lbl = raw_label.strip().lower()
                if lbl in ["1", "stego", "steganography"]:
                    expected_label = "Stego"
                elif lbl in ["0", "clean", "cover"]:
                    expected_label = "Clean"
                else:
                    expected_label = "N/A"
            else:
                # Assume numeric
                expected_label = "Stego" if raw_label == 1 else "Clean"

        results.append(
            {
                "filename": file.name,
                "predicted_label": predicted_label,
                "model_score": f"{score:.4f}",
                "expected_label": expected_label,
            }
        )

        # Update progress bar
        progress_bar.progress(
            i / total,
            text=f"Analyzing image {i} of {total}...",
        )

    # Clear the progress bar when done
    progress_bar.empty()

    st.session_state["results"] = results

# =========================================================
# Results Display
# =========================================================
results = st.session_state.get("results", [])

if results:
    st.subheader("Results")

    results_df = pd.DataFrame(results)

    # Styled HTML table inside a card
    st.markdown(
        f"""
        <div class="results-card">
            {results_df.to_html(index=False)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Download results
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name="steg_detection_results.csv",
        mime="text/csv",
    )

    # =====================================================
    # Metrics Section
    # =====================================================
    st.subheader("Batch metrics (optional)")

    if all(r["expected_label"] in ["Stego", "Clean"] for r in results):
        acc, prec, rec, f1, cm = compute_metrics(results)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Accuracy", f"{acc:.2f}")
        col_m2.metric("Precision", f"{prec:.2f}")
        col_m3.metric("Recall", f"{rec:.2f}")
        col_m4.metric("F1-score", f"{f1:.2f}")

        st.markdown("#### Confusion matrix")
        render_confusion_matrix(cm)
    else:
        st.info(
            "Upload a labeled CSV with matching filenames and expected labels "
            "to view accuracy/precision/recall/F1 and a confusion matrix."
        )
else:
    st.caption(
        "Upload at least one image and click **Run detection** to see predictions."
    )
