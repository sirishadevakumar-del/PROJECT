# ============================================================
# STREAMLIT DEMO APP - Adaptive AI Medical Scan Analysis
# Run locally AFTER downloading densenet_scan_model.pth from Colab
#
# Setup (run once in terminal):
#   pip install streamlit torch torchvision grad-cam pillow
#
# Run the app:
#   streamlit run app.py
# Then open the local URL it prints (usually http://localhost:8501)
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import csv
import os
from datetime import datetime

st.set_page_config(page_title="Adaptive AI Scan Analysis", layout="wide")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # must match the order from train_dataset.class_to_idx
MODEL_PATH = "densenet_scan_model.pth"
LOG_FILE = "decision_log.csv"

# ---- Load model once ----
@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()
target_layers = [model.features.denseblock4.denselayer16.conv2]
cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- UI ----
st.title("Adaptive AI-Driven Medical Scan Analysis")
st.caption("Upload a chest X-ray to get an AI finding with confidence and visual explanation.")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    # ---- Prediction + confidence ----
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item() * 100

    prediction = CLASS_NAMES[pred_idx]

    # Scan-quality flag: very simple check - flag low-confidence cases as "uncertain"
    is_uncertain = confidence < 70

    # ---- Grad-CAM heatmap ----
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    img_np = np.array(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # ---- Display ----
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Scan")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("AI Attention Heatmap (Grad-CAM)")
        st.image(visualization, use_container_width=True)

    st.divider()
    st.subheader("AI Finding")
    st.metric("Prediction", prediction)
    st.metric("Confidence", f"{confidence:.1f}%")

    if is_uncertain:
        st.warning("Low confidence - flagged as UNCERTAIN. Recommend manual review.")
    else:
        st.success("High confidence prediction.")

    # ---- Doctor feedback / decision logging ----
    st.divider()
    st.subheader("Doctor Review")
    decision = st.radio("Your decision:", ["Accept AI finding", "Correct the finding", "Reject the finding"])

    corrected_label = None
    if decision == "Correct the finding":
        corrected_label = st.selectbox("Correct label:", CLASS_NAMES)

    if st.button("Submit Decision"):
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "filename", "ai_prediction", "confidence", "doctor_decision", "corrected_label"])
            writer.writerow([
                datetime.now().isoformat(),
                uploaded_file.name,
                prediction,
                f"{confidence:.1f}",
                decision,
                corrected_label or "",
            ])
        st.success("Decision logged. Thank you.")

    st.caption("All decisions are saved to decision_log.csv for evaluation (sensitivity, specificity, correction rate).")
else:
    st.info("Upload a chest X-ray image to begin.")
