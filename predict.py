import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import argparse
from collections import OrderedDict

# ---- load labels (support dict or list) ----
with open("labels.json", "r") as f:
    loaded_labels = json.load(f)

# Normalize labels into a mapping: idx_to_class[int] -> class_name (string)
if isinstance(loaded_labels, dict):
    # labels.json might be {"0":"CROW","1":"SPARROW", ...}
    try:
        idx_to_class = {int(k): v for k, v in loaded_labels.items()}
    except Exception:
        # If keys are already ints (unlikely), just convert
        idx_to_class = {int(k): v for k, v in loaded_labels.items()}
elif isinstance(loaded_labels, list):
    # labels.json might be ["CROW","SPARROW", ...]
    idx_to_class = {i: name for i, name in enumerate(loaded_labels)}
else:
    raise RuntimeError("Unexpected labels.json format. It must be a list or dict.")

num_classes = len(idx_to_class)

# ---- SAME MODEL AS TRAINING ----
def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ---- Image transforms ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(img_path):
    # Build model
    model = build_model(num_classes)

    # Load weights (support both state_dict and full model object)
    loaded = torch.load("model.pth", map_location="cpu")
    if isinstance(loaded, (dict, OrderedDict)):
        model.load_state_dict(loaded)
    else:
        # saved object was entire model
        model = loaded

    model.eval()

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    predicted_class = idx_to_class[int(pred.item())]
    confidence = conf.item() * 100

    print("\n=========================")
    print(f"Predicted Bird Species: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("=========================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to image")
    args = parser.parse_args()

    predict(args.img)

 