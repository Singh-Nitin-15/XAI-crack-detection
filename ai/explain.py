"""
Grad-CAM explanation module.
Loads trained ResNet18 and produces prediction + heatmap overlay.
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.pth"
IMG_SIZE = 224
CLASSES = ["negative", "positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    _target_layer = model.layer4[-1]
    _target_layer.register_forward_hook(_hook_activations)
    _target_layer.register_full_backward_hook(_hook_gradients)
    _model = model
    return model


def _preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)


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
    """
    Returns: (label, confidence, heatmap_overlay_path)
    """
    model = _load_model()
    input_tensor = _preprocess(image_path)

    with torch.no_grad():
        logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)
    confidence, pred_idx = probs.max(1)
    confidence = confidence.item()
    pred_idx = pred_idx.item()
    label = CLASSES[pred_idx]

    input_tensor_grad = _preprocess(image_path)
    input_tensor_grad.requires_grad_(False)
    cam = _generate_gradcam(input_tensor_grad, pred_idx)

    orig = cv2.imread(str(image_path))
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
        print("Usage: python explain.py <image_path>")
        sys.exit(1)
    img_path = sys.argv[1]
    lbl, conf, hm = predict_with_heatmap(img_path)
    print(f"Label: {lbl}, Confidence: {conf:.4f}, Heatmap: {hm}")
