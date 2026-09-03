import base64
import os
import requests
import streamlit as st
from PIL import Image
import io

# --- CONFIGURATION ---
API_URL = st.secrets.get(
    "API_URL",
    os.getenv("API_URL", "http://localhost:8000/analyze")
)

st.set_page_config(
    page_title="Radio AI - Medical Image Analysis",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Radio AI: X-Ray Analysis Dashboard")
st.markdown(
    "Upload a medical X-ray to run **Classification** (with Grad-CAM heatmaps) or **Segmentation** (YOLO localization)."
)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Model & Task Configuration")

task_type = st.sidebar.selectbox(
    "Select Task",
    options=["classification", "segmentation"],
    format_func=lambda x: x.capitalize(),
    help="Choose Classification for fracture risk + Grad-CAM, or Segmentation for YOLO object detection.",
)

# Dynamically filter model options based on selected task
if task_type == "classification":
    model_choice = st.sidebar.selectbox(
        "Select Model Architecture",
        options=["cnn", "vgg"],
        format_func=lambda x: x.upper(),
        help="Custom CNN (1-channel input) or VGG16 (3-channel input).",
    )
    target_mode = st.sidebar.radio(
        "Grad-CAM Target Focus",
        options=["fracture_only", "predicted_class"],
        format_func=lambda x: "Fracture Score Only" if x == "fracture_only" else "Predicted Class Direction",
        help="Choose how loss is backpropagated for heatmap generation.",
    )
else:
    model_choice = st.sidebar.selectbox(
        "Select Model Architecture",
        options=["yolo"],
        format_func=lambda x: x.upper(),
        help="Ultralytics YOLO for object localization.",
    )
    target_mode = "fracture_only"  # Default placeholder for API compliance

# --- FILE UPLOAD ---
st.subheader("1. Upload X-Ray Image")
uploaded_file = st.file_uploader(
    "Choose an image file (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Display Original Image Preview
    col_input, col_action = st.columns([1, 2])
    with col_input:
        st.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)

    with col_action:
        st.info(f"**Task:** {task_type.capitalize()} | **Model:** {model_choice.upper()}")
        run_analysis = st.button("🚀 Run Analysis", type="primary")

    if run_analysis:
        with st.spinner("Processing image via API..."):
            try:
                # Reset file pointer and build payload
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {
                    "task_type": task_type,
                    "model_choice": model_choice,
                    "target_mode": target_mode,
                }

                # Send Request to FastAPI Backend
                response = requests.post(API_URL, files=files, data=data, timeout=60)

                if response.status_code == 200:
                    result = response.json()
                    st.success("Analysis Complete!")
                    st.divider()

                    st.subheader("2. Diagnostic Results")

                    # --- RENDER CLASSIFICATION RESPONSE ---
                    if task_type == "classification":
                        res_col1, res_col2 = st.columns([1, 1])

                        with res_col1:
                            st.markdown("### Prediction Overview")
                            label = result.get("prediction_label", "Unknown")
                            prob = result.get("fracture_probability", 0.0)

                            if label == "Fracture":
                                st.error(f"**Diagnosis:** {label}")
                            else:
                                st.success(f"**Diagnosis:** {label}")

                            st.metric(
                                label="Fracture Probability",
                                value=f"{prob * 100:.2f}%",
                                delta=f"{'High Risk' if prob >= 0.5 else 'Low Risk'}",
                                delta_color="inverse" if prob >= 0.5 else "normal",
                            )

                            st.caption(f"**Grad-CAM Target Layer:** `{result.get('gradcam_layer', 'N/A')}`")

                        with res_col2:
                            st.markdown("### Grad-CAM Heatmap Visualization")
                            gradcam_b64 = result.get("gradcam_base64", "")

                            if gradcam_b64 and not gradcam_b64.startswith("ERROR"):
                                img_bytes = base64.b64decode(gradcam_b64)
                                heatmap_img = Image.open(io.BytesIO(img_bytes))
                                st.image(
                                    heatmap_img,
                                    caption="Grad-CAM Activation Overlay (JET Colormap)",
                                    use_container_width=True,
                                )
                            elif gradcam_b64.startswith("ERROR"):
                                st.warning(f"Grad-CAM could not be rendered: {gradcam_b64}")
                            else:
                                st.info("No heatmap returned.")

                    # --- RENDER SEGMENTATION RESPONSE ---
                    elif task_type == "segmentation":
                        res_col1, res_col2 = st.columns([1, 1])

                        with res_col1:
                            st.markdown("### Detections Summary")
                            count = result.get("detections_count", 0)
                            st.metric(label="Total Regions Detected", value=count)

                            boxes = result.get("detected_boxes", [])
                            if boxes:
                                st.markdown("**Bounding Box Coordinates & Confidence:**")
                                st.dataframe(boxes, use_container_width=True)
                            else:
                                st.info("No fracture regions or bounding boxes detected.")

                        with res_col2:
                            st.markdown("### YOLO Bounding Box Overlay")
                            seg_b64 = result.get("segmented_image_base64", "")

                            if seg_b64:
                                img_bytes = base64.b64decode(seg_b64)
                                segmented_img = Image.open(io.BytesIO(img_bytes))
                                st.image(
                                    segmented_img,
                                    caption="YOLO Localized Output",
                                    use_container_width=True,
                                )
                            else:
                                st.warning("No segmented image was returned.")

                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to FastAPI server at `{API_URL}`. Ensure `fast.py` is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
