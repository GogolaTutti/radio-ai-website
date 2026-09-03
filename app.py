import base64
import io
import requests
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# Backend configuration
API_URL = st.secrets["API_URL"]

st.set_page_config(
    page_title="Radio AI — Clinical Assist",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Radio AI — Pediatric Fracture Detection")
st.markdown("Upload X-ray images to run classification with Grad-CAM heatmaps or object detection with IoU analysis.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Pipeline Configuration")

task_type = st.sidebar.radio(
    "Select Task",
    options=["classification", "segmentation"],
    format_func=lambda x: "Classification: Fracture or Not" if x == "classification" else "Segmentation: Fracture Localization"
)

if task_type == "classification":
#    model_choice = "cnn"
    model_choice = st.sidebar.selectbox(
            "Classification Model",
            options=["cnn", "vgg"],
            format_func=lambda x: "Custom CNN" if x == "cnn" else "VGG16 Transfer Learning"
        )
else:
    model_choice = "yolov8"

if task_type == "classification":
    target_mode = st.sidebar.selectbox(
        "Grad-CAM Target Mode",
        options=["fracture_only", "winning_class"],
        format_func=lambda x: "Fracture only" if x == "fracture_only" else "Winning class",
        help="Controls which probability class drives the gradient tape."
    )

# --- MAIN CONTENT AREA ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Data")
    uploaded_image = st.file_uploader(
        "Upload X-Ray Image (PNG, JPG, 16-bit PNG)",
        type=["png", "jpg", "jpeg"]
    )

    uploaded_gt = None
    if task_type == "segmentation":
        uploaded_gt = st.file_uploader(
            "Upload Ground Truth Bounding Boxes (.txt - YOLO format)",
            type=["txt"],
            help="Optional annotation file containing normalized 'class x_c y_c w h' lines."
        )

    if uploaded_image is not None:
        # Pre-process 16-bit / standard images safely to 8-bit RGB for preview & payload
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        # Handle 16-bit scaling
        if img is not None and img.dtype == np.uint16:
            img = (img / 256.0).astype(np.uint8)

        # Convert Grayscale to RGB for UI consistency
        if img is not None and len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img is not None and len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = None

        if img_rgb is not None:
            st.image(img_rgb, caption="Uploaded X-Ray", use_container_width=True)

            # Prepare image buffer for HTTP POST
            pil_img = Image.fromarray(img_rgb)
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format="PNG")
            img_bytes_payload = img_byte_arr.getvalue()

with col2:
    st.subheader("2. AI Diagnosis")

    if uploaded_image is not None and st.button("Run Model Inference", type="primary"):
        with st.spinner("Processing image through FastAPI backend..."):
            try:
                # Prepare Form Data
                files = {
                    "file": ("image.png", img_bytes_payload, "image/png")
                }

                if uploaded_gt is not None:
                    files["gt_file"] = (uploaded_gt.name, uploaded_gt.getvalue(), "text/plain")

                data = {
                    "task_type": task_type,
                    "model_choice": model_choice,
                    "target_mode": target_mode if task_type == "classification" else None
                }

                # Send Request to fast.py
                response = requests.post(f"{API_URL}/analyze", files=files, data=data)

                if response.status_code == 200:
                    res_json = response.json()

                    if task_type == "classification":
                        # Display Classification Results
                        prob = res_json.get("fracture_probability", 0.0)
                        label = res_json.get("prediction_label", "Unknown")

                        metric_col1, metric_col2 = st.columns(2)
                        metric_col1.metric("Prediction", label)
                        metric_col2.metric("Fracture Probability", f"{prob * 100:.1f}%")

                        # Render Base64 Grad-CAM Image
                        gradcam_b64 = res_json.get("gradcam_base64")
                        if gradcam_b64:
                            overlay_bytes = base64.b64decode(gradcam_b64)
                            overlay_pil = Image.open(io.BytesIO(overlay_bytes))
                            st.image(
                                overlay_pil,
                                caption=f"Grad-CAM Heatmap (Layer: {res_json.get('gradcam_layer')})",
                                use_container_width=True
                            )

                    elif task_type == "segmentation":
                        # Display Segmentation Results
                        det_count = res_json.get("detections_count", 0)
                        has_gt = res_json.get("has_ground_truth", False)
                        max_iou = res_json.get("max_iou", 0.0)

                        metric_col1, metric_col2 = st.columns(2)
                        metric_col1.metric("Detections Found", det_count)
                        if has_gt:
                            metric_col2.metric("Max IoU vs Ground Truth", f"{max_iou:.4f}")

                        # Render Base64 Segmented Image
                        segmented_b64 = res_json.get("segmented_image_base64")
                        if segmented_b64:
                            segmented_bytes = base64.b64decode(segmented_b64)
                            segmented_pil = Image.open(io.BytesIO(segmented_bytes))
                            st.image(
                                segmented_pil,
                                caption="YOLO Detections (Red = Predicted, Green = Ground Truth)",
                                use_container_width=True
                            )

                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI server. Make sure `fast.py` is running on port 8000.")
