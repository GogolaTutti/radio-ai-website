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

API_URL = st.secrets.toml["API_URL"]

# GRAZPEDWRI-DX Class Label Map
CLASS_NAMES = {
    0: "boneanomaly",
    1: "bonelesion",
    2: "foreignbody",
    3: "fracture",
    4: "metal",
    5: "periostealreaction",
    6: "pronatorsign",
    7: "softtissue",
    8: "text",
}


# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=30)
def check_api_health(url: str) -> dict:
    """Verifies that the FastAPI backend is online and models are loaded (cached for 30s)."""
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
        st.success("🟢 API Status: Online")
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
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📷 Original X-Ray")
        original_img = Image.open(uploaded_file)
        st.image(original_img, use_container_width=True, caption=uploaded_file.name)

    with col_output:
        tab_class, tab_seg = st.tabs([
            "🔍 Classification: Fracture or Not (CNN/VGG)",
            "🎯 Fracture Localization Box (YOLO)"
        ])

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

                                is_fracture = result.get("is_fracture")
                                prob = result.get("fracture_probability", 0.0)

                                if is_fracture:
                                    st.error(f"🚨 **Prediction:** {result.get('prediction_label', 'FRACTURE')}")
                                else:
                                    st.success(f"✅ **Prediction:** {result.get('prediction_label', 'NO FRACTURE')}")

                                st.progress(prob, text=f"Fracture Probability: {prob * 100:.2f}%")
                                st.caption(f"Target Layer Used: `{result.get('gradcam_layer')}`")

                                gradcam_b64 = result.get("gradcam_base64")
                                if gradcam_b64:
                                    gradcam_img = base64_to_pil(gradcam_b64)
                                    st.image(
                                        gradcam_img,
                                        caption=f"Grad-CAM Overlay ({model_choice.upper()})",
                                        use_container_width=True
                                    )

                                    # Download Option
                                    buf = BytesIO()
                                    gradcam_img.save(buf, format="PNG")
                                    st.download_button(
                                        label="💾 Download Grad-CAM Image",
                                        data=buf.getvalue(),
                                        file_name=f"gradcam_{uploaded_file.name}",
                                        mime="image/png",
                                    )
                                else:
                                    st.warning("Grad-CAM visualization could not be generated.")
                            else:
                                st.error(f"Error {res.status_code}: {res.json().get('detail')}")

                        except Exception as e:
                            st.error(f"Failed to connect to backend: {e}")

        # =====================================================================
        # TAB 2: SEGMENTATION & GROUND TRUTH COMPARISON (YOLO)
        # =====================================================================
        with tab_seg:
            st.markdown("### Bounding Box Localization & Evaluation")

            # Optional Ground Truth File Uploader
            uploaded_gt_file = st.file_uploader(
                "Optional: Upload Ground Truth Annotation (.txt)",
                type=["txt"],
                help="Upload the corresponding normalized YOLO .txt file to compute IoU metrics."
            )

            with st.form("segmentation_form"):
                conf_threshold = st.slider(
                    "Detection Confidence Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.25,
                    step=0.05,
                    help="Minimum confidence score required to display a bounding box.",
                )
                submit_seg = st.form_submit_button("Run Detection & Evaluate", type="primary")

            if submit_seg:
                if health.get("status") != "online":
                    st.error("Cannot proceed: API backend is offline.")
                else:
                    with st.spinner("Running YOLO model and computing IoU..."):
                        uploaded_file.seek(0)

                        # Prepare multi-part file dictionary
                        files = {
                            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                        }
                        if uploaded_gt_file is not None:
                            uploaded_gt_file.seek(0)
                            files["gt_file"] = (
                                uploaded_gt_file.name,
                                uploaded_gt_file.getvalue(),
                                "text/plain",
                            )

                        data = {
                            "task_type": "segmentation",
                            "model_choice": "yolo",
                            "conf_threshold": conf_threshold,
                        }

                        try:
                            res = requests.post(f"{API_URL}/analyze", files=files, data=data)

                            if res.status_code == 200:
                                result = res.json()

                                gt_boxes = result.get("ground_truth_boxes", [])
                                pred_boxes = result.get("detected_boxes", [])
                                mean_iou = result.get("mean_iou", 0.0)
                                count = len(pred_boxes)

                                # Display Metric Highlights
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Ground Truth Boxes", len(gt_boxes))
                                m2.metric("YOLO Detections", count)
                                m3.metric(
                                    "Mean IoU Score",
                                    f"{mean_iou:.2%}" if uploaded_gt_file else "N/A"
                                )

                                st.divider()

                                if count > 0:
                                    st.warning(f"⚠️ Detected **{count}** area(s).")
                                else:
                                    st.success("✅ No lesions or fractures detected by YOLO.")

                                seg_b64 = result.get("segmented_image_base64")
                                if seg_b64:
                                    seg_img = base64_to_pil(seg_b64)
                                    st.image(
                                        seg_img,
                                        caption="Annotated YOLO Detection Output",
                                        use_container_width=True
                                    )

                                    # Download Option
                                    buf = BytesIO()
                                    seg_img.save(buf, format="PNG")
                                    st.download_button(
                                        label="💾 Download Detection Image",
                                        data=buf.getvalue(),
                                        file_name=f"yolo_{uploaded_file.name}",
                                        mime="image/png",
                                    )

                                # Display Detailed Box Table
                                if pred_boxes:
                                    with st.expander("View Detection Details & IoU Scores"):
                                        formatted_boxes = []
                                        for p in pred_boxes:
                                            cls_id = p.get("class_id")
                                            formatted_boxes.append({
                                                "Class": CLASS_NAMES.get(cls_id, str(cls_id)),
                                                "Confidence": f"{p.get('confidence', 0.0):.2%}",
                                                "Matched IoU": f"{p.get('matched_iou', 0.0):.2%}" if uploaded_gt_file else "N/A",
                                                "Box [xmin, ymin, xmax, ymax]": p.get("box"),
                                            })
                                        st.dataframe(formatted_boxes, use_container_width=True)

                            else:
                                st.error(f"Error {res.status_code}: {res.json().get('detail')}")

                        except Exception as e:
                            st.error(f"Failed to connect to backend: {e}")
