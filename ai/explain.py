"""Grad-CAM explanation module for ResNet50 crack detection."""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.pth"
IMG_SIZE = 224
CLASSES = ["negative", "positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRACK_THRESHOLD = float(os.getenv("CRACK_THRESHOLD", "0.5"))

_model = None
_target_layer = None
_activations = None
_gradients = None


def _hook_activations(module, input, output):
    global _activations
    _activations = output.detach()


def _hook_gradients(module, grad_input, grad_output):
    global _gradients
    _gradients = grad_output[0].detach()


def _load_model():
    global _model, _target_layer
    if _model is not None:
        return _model
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    _target_layer = model.layer4[-1]
    _target_layer.register_forward_hook(_hook_activations)
    _target_layer.register_full_backward_hook(_hook_gradients)
    _model = model
    return model


_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_image(image_path):
    """Load as RGB with EXIF orientation applied, so phone photos are upright.
    The same image is used for model input and heatmap overlay, so the two
    always line up."""
    with Image.open(image_path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _preprocess(img):
    return _TRANSFORM(img).unsqueeze(0).to(DEVICE)


def _generate_gradcam(input_tensor, class_idx):
    global _activations, _gradients
    model = _load_model()
    model.zero_grad()
    output = model(input_tensor)
    target = output[0, class_idx]
    target.backward()
    weights = _gradients.mean(dim=[2, 3], keepdim=True)
    cam = (weights * _activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return cam


def predict_with_heatmap(image_path, output_heatmap_path=None):
    """Returns: (label, confidence, heatmap_overlay_path)"""
    model = _load_model()
    img = _load_image(image_path)
    input_tensor = _preprocess(img)

    with torch.no_grad():
        logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)
    p_positive = probs[0, 1].item()

    if p_positive >= CRACK_THRESHOLD:
        label = "positive"
        confidence = p_positive
    else:
        label = "negative"
        confidence = 1.0 - p_positive

    pred_idx = CLASSES.index(label)

    cam = _generate_gradcam(input_tensor, pred_idx)

    orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)

    if output_heatmap_path is None:
        p = Path(image_path)
        output_heatmap_path = str(p.parent / f"{p.stem}_heatmap.jpg")
    cv2.imwrite(str(output_heatmap_path), overlay)

    return label, confidence, str(output_heatmap_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m ai.explain <image_path>")
        sys.exit(1)
    img_path = sys.argv[1]
    lbl, conf, hm = predict_with_heatmap(img_path)
    print(f"Label: {lbl}, Confidence: {conf:.4f}, Heatmap: {hm}")
