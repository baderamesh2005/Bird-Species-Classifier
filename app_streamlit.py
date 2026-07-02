# app_streamlit.py

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
import json
from pathlib import Path

# ================= SETTINGS =================
BG_FILE = "bg.jpg"

BG_URL = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1950&q=80"

MODEL_PATH = "model.pth"
LABELS_PATH = "labels.json"

# ============================================

st.set_page_config(
    page_title="Bird Species Classifier",
    layout="centered"
)

# ================= BACKGROUND CSS =================

def build_css():

    if Path(BG_FILE).exists():
        bg_style = f"""
        background-image: url('{BG_FILE}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        """
    else:
        bg_style = f"""
        background-image: url('{BG_URL}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        """

    return f"""
    <style>

    .stApp {{
        {bg_style}
    }}

    .main-card {{
        background: rgba(255,255,255,0.82);
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        max-width: 950px;
        margin: auto;
        margin-top: 25px;
        backdrop-filter: blur(10px);
    }}

    .title {{
        text-align:center;
        font-size:48px;
        font-weight:800;
        color:#05204A;
        margin-bottom:10px;
    }}

    .subtitle {{
        text-align:center;
        font-size:18px;
        color:#1f3b5b;
        margin-bottom:25px;
    }}

    .mode-btn {{
        width:100%;
        padding:14px;
        border-radius:14px;
        text-align:center;
        font-weight:700;
        font-size:18px;
    }}

    .result {{
        font-size:28px;
        font-weight:800;
        color:#05204A;
        margin-top:20px;
        text-align:center;
    }}

    .confidence {{
        display:inline-block;
        margin-top:12px;
        padding:10px 18px;
        border-radius:999px;
        background:linear-gradient(90deg,#06b6d4,#2563eb);
        color:white;
        font-size:20px;
        font-weight:700;
    }}

    .invalid {{
        font-size:28px;
        font-weight:800;
        color:red;
        text-align:center;
        margin-top:20px;
    }}

    </style>
    """

st.markdown(build_css(), unsafe_allow_html=True)

# ================= LOAD LABELS =================

@st.cache_resource
def load_labels(path=LABELS_PATH):

    with open(path, "r") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict):
        return {int(k): v for k, v in loaded.items()}

    elif isinstance(loaded, list):
        return {i: name for i, name in enumerate(loaded)}

    else:
        raise RuntimeError("labels.json format invalid")


# ================= BUILD MODEL =================

def build_model(num_classes, device):

    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.to(device)

    return model


# ================= LOAD MODEL =================

def robust_load_state(model, path, device):

    raw = torch.load(path, map_location=device)

    if isinstance(raw, dict):

        candidates = [raw]

        for key in ("state_dict", "model_state", "model", "params"):
            if key in raw and isinstance(raw[key], dict):
                candidates.append(raw[key])

        for cand in candidates:

            try:
                model.load_state_dict(cand)
                return model

            except:
                try:
                    new = {
                        k.replace("module.", ""): v
                        for k, v in cand.items()
                    }

                    model.load_state_dict(new)
                    return model

                except:
                    pass

    raise RuntimeError("Unable to load model weights")


@st.cache_resource
def load_model_and_labels():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    labels = load_labels()

    model = build_model(len(labels), device)

    model = robust_load_state(
        model,
        MODEL_PATH,
        device
    )

    model.eval()

    return model, labels, device


# ================= IMAGE TRANSFORM =================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= GENERAL OBJECT CHECKER =================

weights = ResNet18_Weights.DEFAULT

imagenet_model = resnet18(weights=weights)

imagenet_model.eval()

imagenet_labels = weights.meta["categories"]

imagenet_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

BIRD_KEYWORDS = [

"bird",
"sparrow",
"eagle",
"duck",
"cock",
"hen",
"goose",
"owl",
"parrot",
"macaw",
"falcon",
"crow",
"peacock",
"penguin",
"albatross",
"jay",
"bulbul",
"finch",
"robin",
"canary",
"woodpecker",
"kingfisher",
"warbler",
"oriole",
"hawk",
"vulture",
"toucan",
"flamingo",
"crane",
"stork"

]

# ================= PREDICTION FUNCTION =================

def predict_pil_image(pil_img, model, labels, device):

    # STEP 1 → Detect object

    img_check = imagenet_transform(pil_img).unsqueeze(0)

    with torch.no_grad():

        output = imagenet_model(img_check)

        probs = torch.softmax(output, dim=1)

        conf_check, pred_check = torch.max(probs, 1)

    object_name = imagenet_labels[pred_check.item()].lower()

    bird_found = False

    for word in BIRD_KEYWORDS:

        if word in object_name:

            bird_found = True

            break


    # Reject if not bird
    if (not bird_found) or (conf_check.item() < 0.45):

        return "INVALID", 0


    # STEP 2 → Bird species prediction

    img = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():

        out = model(img)

        probs = torch.softmax(out, dim=1)

        conf, pred = torch.max(probs, dim=1)

    confidence = float(conf.item() * 100)

    if confidence < 55:

        return "INVALID", confidence

    label = labels.get(
        int(pred.item()),
        "UNKNOWN"
    )

    return label, confidence

# ================= LOAD MODEL =================

with st.spinner("Loading AI Model..."):

    model, labels, device = load_model_and_labels()


# ================= UI =================

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown(
    '<div class="title">🦜 Bird Species Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload bird image or capture real-time image</div>',
    unsafe_allow_html=True
)

# ================= MODE =================

if "mode" not in st.session_state:
    st.session_state.mode = "upload"

col1, col2 = st.columns(2)

with col1:

    if st.button("📸 Real-time Capture"):

        st.session_state.mode = "camera"

with col2:

    if st.button("📁 Upload Image"):

        st.session_state.mode = "upload"

st.write("")

# ================= CAMERA MODE =================

if st.session_state.mode == "camera":

    camera_file = st.camera_input("Take a picture")

    if camera_file:

        pil_img = Image.open(camera_file).convert("RGB")

        st.image(pil_img, use_container_width=True)

        if st.button("Predict Camera Image"):

            label, conf = predict_pil_image(
                pil_img,
                model,
                labels,
                device
            )

            if label == "INVALID":

                st.markdown(
                    '<div class="invalid">❌ Invalid Input (Not a Bird)</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f'<div class="result">Predicted: {label}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div style="text-align:center;"><span class="confidence">{conf:.2f}%</span></div>',
                    unsafe_allow_html=True
                )

# ================= UPLOAD MODE =================

else:

    uploaded = st.file_uploader(
        "Upload Bird Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        pil_img = Image.open(uploaded).convert("RGB")

        st.image(pil_img, use_container_width=True)

        if st.button("Predict this upload"):

            label, conf = predict_pil_image(
                pil_img,
                model,
                labels,
                device
            )

            if label == "INVALID":

                st.markdown(
                    '<div class="invalid">❌ Invalid Input (Not a Bird)</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f'<div class="result">Predicted: {label}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div style="text-align:center;"><span class="confidence">{conf:.2f}%</span></div>',
                    unsafe_allow_html=True
                )

st.markdown("</div>", unsafe_allow_html=True)
# streamlit run app_streamlit.py
