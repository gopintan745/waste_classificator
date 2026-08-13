"""
Waste Material Classifier — Streamlit App

Production-ready app with:
- Confidence-aware classifications
- Material property recommendations
- User feedback for model improvement
- Disambiguation hints for visually similar materials
- Session analytics
"""

from env_setup import * 

import streamlit as st
from PIL import Image
from pathlib import Path

# Local imports
from classifier import WasteClassifier
from ui_components import (
    display_classification_result,
    show_feedback_widget,
    show_session_analytics,
    update_history,
)


# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="♻️ Waste Material Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================
# MODEL LOADING (CACHED)
# ============================================
@st.cache_resource(show_spinner="Loading model...")
def load_classifier(model_path: str, model_arch: str):
    """Load the classifier once and cache it."""
    return WasteClassifier(model_path, model_arch=model_arch)


# ============================================
# SIDEBAR — Model & Settings
# ============================================
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

MODEL_OPTIONS = {
    "EfficientNet-B0 (TrashNet)": (
        "experiments/transfer_trashnet_fianl/best_model.pth",
        "transfer",
    ),
    "EfficientNet-B0 (Merged)": (
        "experiments/transfer_merged_final/best_model.pth",
        "transfer",
    ),
    "Custom CNN (TrashNet)": (
        "experiments/custom_cnn_trashnet_final/best_model.pth",
        "custom",
    ),
    "Custom CNN (Merged)": (
        "experiments/custom_cnn_merged_final/best_model.pth",
        "custom",
    ),
}

selected_model_name = st.sidebar.selectbox(
    "Model",
    list(MODEL_OPTIONS.keys()),
    index=0,
    help="Different models trained on different datasets. EfficientNet-B0 generally performs best.",
)
model_path, model_arch = MODEL_OPTIONS[selected_model_name]

# Check if model exists
if not Path(model_path).exists():
    st.sidebar.error(f"Model not found: {model_path}")
    st.sidebar.info("Run the final training script first or update the path")
    st.stop()

# Confidence thresholds
st.sidebar.markdown("### 🎚️ Thresholds")
confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05,
    help="Below this confidence, the app will flag the prediction as uncertain.",
)
ambiguity_threshold = st.sidebar.slider(
    "Ambiguity threshold",
    min_value=0.0,
    max_value=0.5,
    value=0.20,
    step=0.05,
    help="If top-1 and top-2 confidences are within this gap, the classification is flagged as ambiguous.",
)

# Load model
classifier = load_classifier(model_path, model_arch)
classifier.confidence_threshold = confidence_threshold
classifier.ambiguity_threshold = ambiguity_threshold

# Model info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Info")
st.sidebar.info(f"""
**Selected:** {selected_model_name}

**Classes:** {len(classifier.class_names)} ({', '.join(classifier.class_names)})

**Known behavior:** Glass, metal, and plastic are visually similar.
The app will flag these for verification.
""")


# ============================================
# MAIN PAGE
# ============================================
st.title("♻️ Waste Material Classifier")
st.markdown(
    """
    Upload an image of a waste material, and the app will:
    1. **Identify** the material type
    2. **Show its properties** (recyclable, hazardous, etc.)
    3. **Provide handling steps** for proper disposal
    
    For best results:
    - Use a clear, well-lit photo
    - Center the material in the frame
    - Avoid cluttered backgrounds
    """
)

# ============================================
# INPUT: Upload or Camera
# ============================================
input_method = st.radio(
    "Choose input method:",
    ["📁 Upload image", "📸 Take photo"],
    horizontal=True,
)

img_source = None
if input_method == "📁 Upload image":
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        help="JPG, JPEG, or PNG. Max 200MB.",
    )
    if uploaded:
        img_source = uploaded

else:
    camera_input = st.camera_input("Take a photo")
    if camera_input:
        img_source = camera_input


# ============================================
# PROCESSING AND DISPLAY
# ============================================
if img_source is not None:
    try:
        img = Image.open(img_source).convert("RGB")
    except Exception as e:
        st.error(f"Could not load image: {e}")
        st.stop()

    # Display the original image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📷 Input Image")
        st.image(img, caption="Uploaded image", use_container_width=True)

    # Run inference
    with st.spinner("🔍 Analyzing material..."):
        try:
            result = classifier.predict(img)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # Update history
    update_history(result)

    # Display results
    with col2:
        st.markdown("### 🎯 Classification Result")
        display_classification_result(result)

    # Feedback widget
    show_feedback_widget(result, image_key=f"img_{len(st.session_state.get('history', []))}")

    # ============================================
    # ABOUT THIS PREDICTION
    # ============================================
    with st.expander("ℹ️ About this prediction"):
        st.markdown(f"""
        **Model:** {selected_model_name}
        
        **Top 3 predictions:**
        - {result['top_k'][0]['class'].title()}: {result['top_k'][0]['confidence']:.1%}
        - {result['top_k'][1]['class'].title()}: {result['top_k'][1]['confidence']:.1%}
        - {result['top_k'][2]['class'].title()}: {result['top_k'][2]['confidence']:.1%}
        
        **Confidence interpretation:**
        - 🟢 High (≥80%): Trust the prediction
        - 🟡 Medium (60-80%): Likely correct, verify if important
        - 🔴 Low (<60%): Don't trust — retake photo or verify manually
        
        **Class confusability** (based on training analysis):
        - Glass, metal, and plastic are visually similar
        - Cardboard and paper can be confused with each other
        - The `trash` class is intentionally broad
        """)

# ============================================================
# SESSION ANALYTICS (always at bottom)
# ============================================================
show_session_analytics()


# ============================================================
# FOOTER — Persistently visible
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 12px;'>
        Built with PyTorch + Streamlit · For educational and portfolio use
    </div>
    """,
    unsafe_allow_html=True,
)
