import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

def load_model():
    # Example: resnet18 with 2 classes
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

def predict_image(model, img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ])

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        confidence = torch.softmax(output, dim=1)[0][predicted].item()

    label = "glaucoma" if predicted.item() == 0 else "normal"
    return label, confidence


def generate_gradcam(model, img_path, predicted_label, save_path):
    """
    Minimal Grad-CAM:
    1. Hook final conv layer
    2. Backprop predicted class
    3. Overlay heatmap on original image
    4. Save to `save_path`
    """
    import torch.nn.functional as F

    # Decide the target class index based on predicted_label
    class_idx = 0 if predicted_label == "glaucoma" else 1

    # We need the input tensor again
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ])
    image_pil = Image.open(img_path).convert("RGB")
    input_tensor = transform(image_pil).unsqueeze(0)
    input_tensor.requires_grad = True  # so we can do backward

    # Hooks to capture activations & gradients
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks on the last conv layer
    final_conv = model.layer4[-1].conv2
    fwd = final_conv.register_forward_hook(forward_hook)
    bwd = final_conv.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    # Target class score
    score = output[0, class_idx]
    
    # Backward pass
    model.zero_grad()
    score.backward()

    # Detach from the graph
    act = activations[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]

    # Remove hooks
    fwd.remove()
    bwd.remove()

    # Average gradients spatially
    alpha = np.mean(grad, axis=(1, 2))

    # Weight the channels by alpha
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(alpha):
        cam += w * act[i]

    # ReLU
    cam = np.maximum(cam, 0)

    # Normalize to [0,1]
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()

    # Resize to original image size
    cam = cv2.resize(cam, (224, 224))

    # Convert cam to 3-channel heatmap
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on original image
    orig = np.array(image_pil.resize((224, 224)))
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    # Save Grad-CAM
    cv2.imwrite(save_path, overlay)
