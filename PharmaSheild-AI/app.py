import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2

from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_PATH = "model/pharmashield_efficientnet_b0.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = ["Fake", "Real"]

@st.cache_resource
def load_model():

    model = models.efficientnet_b0(weights=None)

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        num_features,
        2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def calculate_risk_score(probabilities):

    fake_probability = probabilities[0, 0].item()

    # Current baseline score:
    # higher fake probability = higher suspicion
    score = fake_probability * 100

    return score


def get_risk_level(score):

    if score < 30:
        return "LOW"

    elif score < 65:
        return "MEDIUM"

    else:
        return "HIGH"

def generate_gradcam(model, input_tensor, predicted_class, rgb_image):

    target_layer = model.features[-1]

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    targets = [
        ClassifierOutputTarget(predicted_class)
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]

    visualization = show_cam_on_image(
        rgb_image.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    return visualization

st.set_page_config(
    page_title="PharmaShield AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ PharmaShield AI")

st.subheader(
    "AI-Powered Counterfeit Medicine Packaging Screening"
)

st.write(
    "Upload a medicine package image to obtain an "
    "AI-assisted screening result and visual explanation."
)

st.info(
    "PharmaShield is a screening tool, not a regulatory "
    "certification system."
)

uploaded_file = st.file_uploader(
    "Upload medicine packaging image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Package",
        width=450
    )

    # Convert image
    input_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    # Image for Grad-CAM
    rgb_image = np.array(
        image.resize((224, 224))
    ) / 255.0

    with st.spinner("Analyzing packaging..."):

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_class = outputs.argmax(
                dim=1
            ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    risk_score = calculate_risk_score(
        probabilities
    )

    risk_level = get_risk_level(
        risk_score
    )

    prediction = CLASS_NAMES[
        predicted_class
    ]

    st.divider()

    st.header("Analysis Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prediction",
            prediction
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Risk Score",
            f"{risk_score:.1f}/100"
        )


    if risk_level == "HIGH":

        st.error(
            f"🔴 HIGH SUSPICION — {risk_score:.1f}/100"
        )

    elif risk_level == "MEDIUM":

        st.warning(
            f"🟠 MEDIUM SUSPICION — {risk_score:.1f}/100"
        )

    else:

        st.success(
            f"🟢 LOW SUSPICION — {risk_score:.1f}/100"
        )

    st.divider()

    st.header("💡 Explainable AI")

    with st.spinner("Generating AI explanation..."):

        heatmap = generate_gradcam(
            model,
            input_tensor,
            predicted_class,
            rgb_image
        )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Original Package",
            width=450
        )

    with col2:

        st.image(
            heatmap,
            caption="AI Attention / Grad-CAM",
            width=450
        )


    st.write(
        "Highlighted regions show areas that contributed "
        "strongly to the model's prediction."
    )

    st.divider()

    st.warning(
        "This result is an AI-assisted screening assessment. "
        "It should not be treated as definitive proof of "
        "authenticity or counterfeiting. Verify suspicious "
        "products through authorized channels."
    )