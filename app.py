import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import csv
import os
import json
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Adaptive AI Scan Analysis", layout="wide")

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
# 2-class pneumonia model:
#   CLASS_NAMES = ["NORMAL", "PNEUMONIA"]; MODEL_PATH = "densenet_scan_model.pth"
# 4-class multi-disease model (after train_multidisease_colab.py):
#   CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
#   MODEL_PATH = "densenet_multidisease_model.pth"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MODEL_PATH = "densenet_scan_model.pth"
NUM_CLASSES = len(CLASS_NAMES)

LOG_FILE = "decision_log.csv"
REFERENCE_DIR = "reference_cases"

HIGH_CONFIDENCE = 85
LOW_CONFIDENCE = 70


# ---- Load model once ----
@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
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


# ---------------------------------------------------------------
# SCAN-QUALITY CHECK (runs before the model touches the image)
# ---------------------------------------------------------------
def check_scan_quality(pil_image):
    gray = np.array(pil_image.convert("L"))
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    h, w = gray.shape
    issues = []
    if blur_score < 100:
        issues.append("Image appears blurry.")
    if brightness < 40:
        issues.append("Image appears too dark / underexposed.")
    if brightness > 220:
        issues.append("Image appears overexposed / washed out.")
    if h < 200 or w < 200:
        issues.append("Image resolution is very low.")
    return {"ok": len(issues) == 0, "issues": issues,
            "blur_score": round(blur_score, 1), "brightness": round(brightness, 1)}


# ---------------------------------------------------------------
# SIMILAR-CASE RETRIEVAL
# ---------------------------------------------------------------
@st.cache_resource
def load_reference_cases():
    npz_path = os.path.join(REFERENCE_DIR, "reference_embeddings.npz")
    if not os.path.isfile(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "labels": data["labels"],
        "filenames": data["filenames"],
    }

reference_data = load_reference_cases()

def get_embedding(pil_image):
    x = transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        feats = model.features(x)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
    return feats.numpy()[0]

def find_similar_cases(pil_image, top_k=3):
    if reference_data is None:
        return []
    query_emb = get_embedding(pil_image)
    ref_embs = reference_data["embeddings"]
    # cosine similarity
    sims = (ref_embs @ query_emb) / (
        np.linalg.norm(ref_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({
            "label": reference_data["labels"][i],
            "similarity": float(sims[i]) * 100,
            "path": os.path.join(REFERENCE_DIR, reference_data["filenames"][i]),
        })
    return results


# ---------------------------------------------------------------
# PERSISTENT LOGGING - Google Sheets with automatic CSV fallback
# ---------------------------------------------------------------
def get_gsheet():
    """Returns a gspread worksheet if secrets are configured, else None."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=scopes
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(st.secrets["sheet_id"]).sheet1
        return sheet
    except Exception:
        return None

def log_decision(row):
    """Try Google Sheets first; always also write to local CSV as backup."""
    sheet = get_gsheet()
    logged_to_sheets = False
    if sheet is not None:
        try:
            sheet.append_row(row)
            logged_to_sheets = True
        except Exception:
            pass

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "filename", "viewer_role", "ai_prediction",
                              "confidence", "detail_level", "doctor_decision", "corrected_label"])
        writer.writerow(row)

    return logged_to_sheets


# ---------------------------------------------------------------
# ROLE-BASED + CONFIDENCE-BASED ADAPTIVE EXPLANATION
# ---------------------------------------------------------------
ROLES = ["Radiologist", "Doctor / Physician", "Patient / Family"]

def get_detail_level(role, confidence):
    if role == "Radiologist":
        return "full"                     # radiologists always get full technical detail
    if role == "Doctor / Physician":
        if confidence >= HIGH_CONFIDENCE:
            return "summary"
        elif confidence >= LOW_CONFIDENCE:
            return "standard"
        else:
            return "full"
    # Patient / Family - always simplified, plain language
    return "patient"


# ---- UI ----
st.title("Adaptive AI-Driven Medical Scan Analysis")
st.caption("Upload a chest X-ray to get an AI finding, adapted to who is viewing it.")

role = st.selectbox("Who is viewing this result?", ROLES)
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # ---- Scan-quality check first ----
    quality = check_scan_quality(image)
    proceed = True
    if not quality["ok"]:
        st.warning("Scan-quality check flagged this image: " + " ".join(quality["issues"]))
        proceed = st.checkbox("Analyze anyway", value=False)
        st.caption(f"(blur score: {quality['blur_score']}, brightness: {quality['brightness']})")

    if proceed:
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item() * 100
        prediction = CLASS_NAMES[pred_idx]
        detail_level = get_detail_level(role, confidence)

        st.divider()
        st.subheader("AI Finding")

        # ---------------- PATIENT VIEW ----------------
        if detail_level == "patient":
            if confidence >= LOW_CONFIDENCE:
                st.success(f"Your scan was reviewed by an AI tool. The result suggests: **{prediction}**.")
            else:
                st.warning("The AI tool was not fully confident about this scan. "
                            "Your doctor will review it closely and explain the result to you.")
            st.caption("This is a support tool only. Please discuss the full result with your doctor.")
            with st.expander("See what the AI looked at (optional)"):
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
                img_np = np.array(image.resize((224, 224))) / 255.0
                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                st.image(visualization, use_container_width=True,
                         caption="Highlighted areas show what the AI focused on.")

        # ---------------- SUMMARY (confident doctor view) ----------------
        elif detail_level == "summary":
            col_a, col_b = st.columns(2)
            col_a.metric("Prediction", prediction)
            col_b.metric("Confidence", f"{confidence:.1f}%")
            st.success("High-confidence finding.")
            with st.expander("Show visual explanation"):
                grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
                img_np = np.array(image.resize((224, 224))) / 255.0
                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                st.image(visualization, caption="Grad-CAM heatmap", use_container_width=True)

        # ---------------- STANDARD (moderate confidence doctor view) ----------------
        elif detail_level == "standard":
            col_a, col_b = st.columns(2)
            col_a.metric("Prediction", prediction)
            col_b.metric("Confidence", f"{confidence:.1f}%")
            st.info("Moderate confidence - visual explanation shown below.")
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
            img_np = np.array(image.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            col1, col2 = st.columns(2)
            col1.image(image, caption="Original Scan", use_container_width=True)
            col2.image(visualization, caption="Grad-CAM Heatmap", use_container_width=True)

        # ---------------- FULL (radiologist, or low-confidence doctor view) ----------------
        else:
            col_a, col_b = st.columns(2)
            col_a.metric("Prediction", prediction)
            col_b.metric("Confidence", f"{confidence:.1f}%")
            if confidence < LOW_CONFIDENCE:
                st.warning("LOW CONFIDENCE - flagged as UNCERTAIN.")
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
            img_np = np.array(image.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            col1, col2 = st.columns(2)
            col1.image(image, caption="Original Scan", use_container_width=True)
            col2.image(visualization, caption="Grad-CAM Heatmap", use_container_width=True)

            st.write("**Full probability breakdown (all classes):**")
            prob_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability (%)": [round(p.item() * 100, 1) for p in probs]
            }).sort_values("Probability (%)", ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        # ---- Similar-case retrieval (shown for doctor/radiologist views, not patient) ----
        if detail_level in ("standard", "full", "summary") and reference_data is not None:
            similar = find_similar_cases(image, top_k=3)
            if similar:
                st.write("**Similar cases from the training set:**")
                cols = st.columns(len(similar))
                for c, case in zip(cols, similar):
                    with c:
                        st.image(case["path"], use_container_width=True)
                        st.caption(f"{case['label']} - {case['similarity']:.0f}% similar")

        # ---- Doctor feedback / decision logging ----
        st.divider()
        st.subheader("Review Decision")
        decision = st.radio("Decision:", ["Accept AI finding", "Correct the finding", "Reject the finding"])
        corrected_label = None
        if decision == "Correct the finding":
            corrected_label = st.selectbox("Correct label:", CLASS_NAMES)

        if st.button("Submit Decision"):
            row = [datetime.now().isoformat(), uploaded_file.name, role, prediction,
                   f"{confidence:.1f}", detail_level, decision, corrected_label or ""]
            logged_to_sheets = log_decision(row)
            if logged_to_sheets:
                st.success("Decision logged to Google Sheets (and local backup).")
            else:
                st.success("Decision logged locally.")
                st.caption("Google Sheets not configured yet - see setup notes - using local CSV for now.")
else:
    st.info("Upload a chest X-ray image to begin.")

# ---- Local decision log viewer + download (always available as backup) ----
st.divider()
st.subheader("Decision Log (local backup)")
if os.path.isfile(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
    st.dataframe(log_df, use_container_width=True, hide_index=True)
    with open(LOG_FILE, "rb") as f:
        st.download_button("Download decision log (CSV)", f, file_name="decision_log.csv")
else:
    st.caption("No decisions logged yet this session.")

