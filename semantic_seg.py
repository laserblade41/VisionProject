"""
Semantic segmentation module using SegFormer models.
Provides functions to load models, predict segmentation masks, and evaluate IoU.
"""

import torch
import numpy as np
import cv2
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image

# Select device: prefer GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Semantic Segmentation using device: {device}")


def load_seg_model(model_name='nvidia/segformer-b1-ade20k-512-512'):
    """
    Load a SegFormer segmentation model from Hugging Face.
    
    Args:
        model_name (str): Model identifier from Hugging Face Hub
            Common models:
            - nvidia/segformer-b0-ade20k-512-512 (smallest, fastest)
            - nvidia/segformer-b1-ade20k-512-512 (recommended)
            - nvidia/segformer-b2-ade20k-512-512
            - nvidia/segformer-b3-ade20k-512-512
            - nvidia/segformer-b4-ade20k-512-512
            - nvidia/segformer-b5-ade20k-512-512 (largest, most accurate)
    
    Returns:
        tuple: (model, image_processor) loaded segmentation model and processor
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, processor


def predict_seg_mask(model_tuple, img_rgb, conf=0.25, device_override=None):
    """
    Run segmentation prediction on an image and extract the mask.
    
    Args:
        model_tuple (tuple): (model, processor) from load_seg_model()
        img_rgb (np.ndarray): Input image in RGB format (uint8, shape H×W×3)
        conf (float): Confidence threshold (currently unused for SegFormer, kept for API compatibility)
        device_override (str): Optional device override (defaults to global device)
    
    Returns:
        tuple: (mask, overlay_img) where:
            - mask (np.ndarray): Binary segmentation mask (H×W, values 0-255)
            - overlay_img (np.ndarray): RGB image with mask overlaid, or None if extraction failed
    """
    model, processor = model_tuple
    dev = device_override if device_override else device
    
    try:
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(img_rgb)
        
        # Preprocess
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get logits and upsample to original size
        logits = outputs.logits  # [B, num_classes, H, W]
        
        # Upsample to match input image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=img_rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        
        # Get predicted class for each pixel
        predicted = upsampled_logits.argmax(dim=1)  # [B, H, W]
        
        # Convert to numpy and create binary mask (background=0, anything else=255)
        pred_np = predicted.cpu().numpy()[0]  # [H, W]
        
        # Create binary mask: foreground (non-background classes) = 255
        combined_mask = ((pred_np > 0).astype(np.uint8)) * 255
        
        # Create overlay visualization
        overlay = create_mask_overlay(img_rgb, combined_mask)
        
        return combined_mask, overlay
    
    except Exception as e:
        print(f"Error during segmentation prediction: {e}")
        return None, None


def create_mask_overlay(img_rgb, mask, color=(0, 255, 0), alpha=0.5):
    """
    Create a visualization of mask overlaid on the image.
    
    Args:
        img_rgb (np.ndarray): Input image in RGB format
        mask (np.ndarray): Binary segmentation mask (H×W)
        color (tuple): RGB color for the mask overlay
        alpha (float): Transparency of mask overlay (0-1)
    
    Returns:
        np.ndarray: Overlaid image in RGB format
    """
    if img_rgb is None or mask is None:
        return None
    
    overlay = img_rgb.copy().astype(np.float32)
    
    # Create colored mask
    mask_colored = np.zeros_like(img_rgb, dtype=np.float32)
    mask_pixels = mask > 0
    mask_colored[mask_pixels] = color
    
    # Blend mask with image
    overlay[mask_pixels] = overlay[mask_pixels] * (1 - alpha) + mask_colored[mask_pixels] * alpha
    
    return np.clip(overlay, 0, 255).astype(np.uint8)


def iou_mask(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1 (np.ndarray): First binary mask (H×W, values 0-255 or 0-1)
        mask2 (np.ndarray): Second binary mask (H×W, values 0-255 or 0-1)
    
    Returns:
        float: IoU value in range [0, 1], or None if masks are invalid
    """
    if mask1 is None or mask2 is None:
        return None
    
    try:
        # Normalize to binary (0 or 1)
        m1 = (mask1 > 127).astype(np.uint8)
        m2 = (mask2 > 127).astype(np.uint8)
        
        # Compute intersection and union
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        
        if union == 0:
            return 0.0
        
        iou = float(intersection) / float(union)
        return iou
    
    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return None


def dice_coefficient(mask1, mask2):
    """
    Calculate Dice coefficient between two binary masks.
    Useful alternative to IoU.
    
    Args:
        mask1 (np.ndarray): First binary mask
        mask2 (np.ndarray): Second binary mask
    
    Returns:
        float: Dice coefficient in range [0, 1]
    """
    if mask1 is None or mask2 is None:
        return None
    
    try:
        m1 = (mask1 > 127).astype(np.uint8)
        m2 = (mask2 > 127).astype(np.uint8)
        
        intersection = np.logical_and(m1, m2).sum()
        dice = 2.0 * intersection / (m1.sum() + m2.sum())
        
        return float(dice)
    
    except Exception as e:
        print(f"Error calculating Dice coefficient: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python semantic_seg.py <image_path> [--model MODEL_NAME]")
        sys.exit(1)
    
    img_path = sys.argv[1]
    model_name = 'nvidia/segformer-b1-ade20k-512-512'
    
    # Parse optional model name
    if '--model' in sys.argv:
        idx = sys.argv.index('--model')
        if idx + 1 < len(sys.argv):
            model_name = sys.argv[idx + 1]
    
    # Load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: Could not load image from {img_path}")
        sys.exit(1)
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Load model and predict
    print(f"Loading model: {model_name}")
    model_tuple = load_seg_model(model_name)
    
    print(f"Running segmentation on {img_path}")
    mask, overlay = predict_seg_mask(model_tuple, img_rgb, conf=0.25)
    
    if mask is not None:
        print(f"Segmentation mask shape: {mask.shape}")
        print(f"Mask coverage: {100.0 * (mask > 0).sum() / mask.size:.2f}%")
        
        # Save results
        if overlay is not None:
            out_path = img_path.rsplit('.', 1)[0] + '_seg_overlay.png'
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"Saved overlay to {out_path}")
        
        mask_path = img_path.rsplit('.', 1)[0] + '_seg_mask.png'
        cv2.imwrite(mask_path, mask)
        print(f"Saved mask to {mask_path}")
    else:
        print("Failed to extract segmentation mask")
