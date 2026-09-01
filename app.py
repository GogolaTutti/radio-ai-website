import base64
from io import BytesIO
import requests
import streamlit as st
from PIL import Image

# --- CONFIG & STYLING ---
st.set_page_config(
    page_title="GRAZPEDWRI-DX Fracture Diagnostics",
    page_icon="🦴",
    layout="wide",
)

API_URL = "https://radioai-530270812256.europe-west1.run.app/"


# --- HELPER FUNCTIONS ---
def check_api_health(url: str) -> dict:
    """Verifies that the FastAPI backend is online and models are loaded."""
    try:
        response = requests.get(f"{url}/check", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return {"status": "offline", "loaded_models": []}
    return {"status": "offline", "loaded_models": []}


def base64_to_pil(base64_string: str) -> Image.Image:
    """Decodes a Base64 string back to a PIL Image for Streamlit rendering."""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_bytes))


# --- SIDEBAR: STATUS & SETTINGS ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Health Check Status
    health = check_api_health(API_URL)
    if health.get("status") == "online":
        st.success(f"🟢 API Status: Online")
        st.caption(f"Loaded Models: {', '.join(health.get('loaded_models', []))}")
    else:
        st.error("🔴 API Status: Offline")
        st.warning(f"Ensure FastAPI is running at `{API_URL}`")

    st.divider()
    st.markdown("### About")
    st.info(
        "This diagnostic tool assists in detecting pediatric wrist fractures using "
        "the **GRAZPEDWRI-DX** dataset via binary classification (Grad-CAM) "
        "and YOLO object detection."
    )


# --- MAIN INTERFACE HEADER ---
st.title("🦴 GRAZPEDWRI-DX Fracture Diagnostics")
st.markdown("Upload a pediatric wrist X-ray to perform AI-assisted classification or segmentation.")

# File Uploader
uploaded_file = st.file_uploader(
    "Choose an X-ray image...",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Display Original Image Preview
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📷 Original X-Ray")
        original_img = Image.open(uploaded_file)
        st.image(original_img, use_container_width=True, caption=uploaded_file.name)

    with col_output:
        # Create tabs for the two API workflows
        tab_class, tab_seg = st.tabs(["🔍 Classification (CNN/VGG)", "🎯 Bounding Box Segmentation (YOLO)"])

        # =====================================================================
        # TAB 1: CLASSIFICATION & GRAD-CAM
        # =====================================================================
        with tab_class:
            st.markdown("### Fracture Detection & Heatmap")

            with st.form("classification_form"):
                model_choice = st.selectbox(
                    "Select Classification Backbone",
                    options=["cnn", "vgg"],
                    format_func=lambda x: x.upper(),
                )
                target_mode = st.radio(
                    "Grad-CAM Target Focus",
                    options=["fracture_only", "winning_class"],
                    help="'fracture_only' highlights fracture evidence; 'winning_class' highlights features supporting the top prediction.",
                    horizontal=True,
                )
                submit_class = st.form_submit_button("Run Classification", type="primary")

            if submit_class:
                if health.get("status") != "online":
                    st.error("Cannot proceed: API backend is offline.")
                else:
                    with st.spinner("Analyzing image and computing Grad-CAM..."):
                        # Prepare payload for FastAPI Form data endpoint
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {
                            "task_type": "classification",
                            "model_choice": model_choice,
                            "target_mode": target_mode,
                        }

                        try:
                            res = requests.post(f"{API_URL}/analyze", files=files, data=data)

                            if res.status_code == 200:
                                result = res.json()

                                # Display Diagnostic Metrics
                                is_fracture = result.get("is_fracture")
                                prob = result.get("fracture_probability", 0.0)

                                if is_fracture:
                                    st.error(f"🚨 **Prediction:** {result['prediction_label']}")
                                else:
                                    st.success(f"✅ **Prediction:** {result['prediction_label']}")

                                st.progress(prob, text=f"Fracture Probability: {prob * 100:.2f}%")
                                st.caption(f"Target Layer Used: `{result.get('gradcam_layer')}`")

                                # Render Grad-CAM Heatmap Image
                                gradcam_b64 = result.get("gradcam_base64")
                                if gradcam_b64:
                                    gradcam_img = base64_to_pil(gradcam_b64)
                                    st.image(
                                        gradcam_img,
                                        caption=f"Grad-CAM Overlay ({model_choice.upper()})",
                                        use_container_width=True
                                    )
                                else:
                                    st.warning("Grad-CAM visualization could not be generated.")
                            else:
                                st.error(f"Error {res.status_code}: {res.json().get('detail')}")

                        except Exception as e:
                            st.error(f"Failed to connect to backend: {e}")

        # =====================================================================
        # TAB 2: SEGMENTATION (YOLO)
        # =====================================================================
        with tab_seg:
            st.markdown("### Bounding Box Localization")

            with st.form("segmentation_form"):
                st.caption("Runs object detection using YOLOv8/YOLOv11 to localize fractures.")
                submit_seg = st.form_submit_button("Run Segmentation", type="primary")

            if submit_seg:
                if health.get("status") != "online":
                    st.error("Cannot proceed: API backend is offline.")
                else:
                    with st.spinner("Running YOLO detection model..."):
                        uploaded_file.seek(0)
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {
                            "task_type": "segmentation",
                            "model_choice": "yolo",
                        }

                        try:
                            res = requests.post(f"{API_URL}/analyze", files=files, data=data)

                            if res.status_code == 200:
                                result = res.json()
                                count = result.get("detections_count", 0)

                                if count > 0:
                                    st.warning(f"⚠️ Detected **{count}** lesion/fracture area(s).")
                                else:
                                    st.success("✅ No lesions or fractures detected by YOLO.")

                                # Display Segmented Image
                                seg_b64 = result.get("segmented_image_base64")
                                if seg_b64:
                                    seg_img = base64_to_pil(seg_b64)
                                    st.image(
                                        seg_img,
                                        caption="Annotated YOLO Detection Output",
                                        use_container_width=True
                                    )

                                # Show Bounding Box Coordinates in expander
                                boxes = result.get("detected_boxes", [])
                                if boxes:
                                    with st.expander("View Bounding Box Coordinates"):
                                        st.json(boxes)

                            else:
                                st.error(f"Error {res.status_code}: {res.json().get('detail')}")

                        except Exception as e:
                            st.error(f"Failed to connect to backend: {e}")
