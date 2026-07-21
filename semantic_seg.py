import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import cv2
import numpy as np
from torchvision import transforms

def load_seg_model(model_path):
    # Determine the available hardware device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if model_path == "deeplabv3_base":
        # Load the original pretrained DeepLabV3 (Clean baseline)
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    else:
        # Load a finetuned student DeepLabV3 model
        model = deeplabv3_resnet50(weights=None, num_classes=21, aux_loss=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    model.to(device)
    model.eval()
    return model

def predict_seg_mask(model, img_rgb, conf=None):
    # Ensure operations run on the same device as the model
    device = next(model.parameters()).device
    
    # Standard PyTorch transformations for DeepLabV3
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Add batch dimension and push to device
    input_tensor = preprocess(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        
    # Extract the predicted classes (0 is background)
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    # Create a visual colored mask for the GUI
    mask_colored = cv2.applyColorMap(output_predictions * 10, cv2.COLORMAP_JET)
    
    # Create the overlay (blending the original image and the mask)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, mask_colored, 0.4, 0)
    
    # Return everything in standard RGB for Tkinter display
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return output_predictions, overlay_rgb


def iou_mask(gt_mask, pred_mask):
    """
    Calculates Intersection over Union (IoU) ratio [0.0 - 1.0] between two class masks.
    Returns None if both masks are None.
    """
    if gt_mask is None or pred_mask is None:
        return None
    valid_pixels = (gt_mask > 0) | (pred_mask > 0)
    union = valid_pixels.sum()
    if union == 0:
        return 1.0
    intersection = ((gt_mask == pred_mask) & (gt_mask > 0)).sum()
    return float(intersection / union)