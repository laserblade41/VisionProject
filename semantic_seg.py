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


def load_seg_model(model_name='nvidia/segformer-b1-finetuned-ade-512-512'):
    """
    Load a SegFormer segmentation model from Hugging Face.
    
    Args:
        model_name (str): Model identifier from Hugging Face Hub
            Common models:
            - nvidia/segformer-b0-finetuned-ade-512-512 (smallest, fastest)
            - nvidia/segformer-b1-finetuned-ade-512-512 (recommended)
            - nvidia/segformer-b2-finetuned-ade-512-512
            - nvidia/segformer-b3-finetuned-ade-512-512
            - nvidia/segformer-b4-finetuned-ade-512-512
            - nvidia/segformer-b5-finetuned-ade-512-512 (largest, most accurate)
    
    Returns:
        tuple: (model, image_processor) loaded segmentation model and processor
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, processor


BACKGROUND_CLASSES = {
    0,   # wall
    1,   # building
    2,   # sky
    3,   # floor
    4,   # tree
    5,   # ceiling
    6,   # road
    7,   # bed 
    8,   # windowpane
    9,   # grass
    11,  # sidewalk
    13,  # earth
    14,  # door
    16,  # mountain
    17,  # plant
    18,  # curtain
    21,  # water
    25,  # house
    26,  # sea
    28,  # rug
    29,  # field
    32,  # fence
    34,  # rock
    46,  # sand
    52,  # path
    53,  # stairs
    54,  # runway
    57,  # pillow
    59,  # stairway
    60,  # river
    61,  # bridge
    63,  # blind
    66,  # flower
    68,  # hill
    70,  # countertop
    72,  # palm
    91,  # dirt track
    94,  # pole
    95,  # land
    96,  # bannister
    101, # poster
    102, # stage
    105, # canopy
    109, # swimming pool
    113, # waterfall
    114, # tent
    121, # step
    122, # tank
    127, # lake
    131, # blanket
    140, # pier
    145, # glass
}


def create_color_overlay(img_rgb, pred_np, closed_mask, background_classes, alpha=0.4):
    """
    Create a visualization of class-specific masks overlaid on the image,
    excluding background classes and grouping components to ensure consistent coloring.
    """
    if img_rgb is None or pred_np is None or closed_mask is None:
        return None
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask)
    
    # Generate a deterministic palette of distinct colors for 150 classes
    rng = np.random.default_rng(42)
    palette = rng.integers(50, 255, size=(150, 3), dtype=np.uint8)
    
    overlay = img_rgb.copy().astype(np.float32)
    mask_colored = np.zeros_like(img_rgb, dtype=np.float32)
    
    any_foreground = False
    
    # Process each component (0 is background label)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 100:  # Ignore small noise components
            continue
            
        comp_mask = labels == i
        comp_classes = pred_np[comp_mask]
        
        # Find dominant foreground class in the component
        unique, counts = np.unique(comp_classes, return_counts=True)
        fg_unique = [u for u in unique if u not in background_classes]
        
        if fg_unique:
            fg_counts = [counts[np.where(unique == u)[0][0]] for u in fg_unique]
            dominant_cls = fg_unique[np.argmax(fg_counts)]
        else:
            dominant_cls = unique[np.argmax(counts)]
            
        # Color the entire component with the dominant class's color
        mask_colored[comp_mask] = palette[dominant_cls % 150]
        any_foreground = True
        
    if not any_foreground:
        return img_rgb.copy()
        
    # Blend mask with image
    blend_mask = closed_mask > 0
    overlay[blend_mask] = overlay[blend_mask] * (1 - alpha) + mask_colored[blend_mask] * alpha
    
    return np.clip(overlay, 0, 255).astype(np.uint8)


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
        
        # Convert to numpy
        pred_np = predicted.cpu().numpy()[0]  # [H, W]
        
        # Create binary mask: foreground (non-background classes) = 255
        is_foreground = ~np.isin(pred_np, list(BACKGROUND_CLASSES))
        fg_mask = (is_foreground.astype(np.uint8)) * 255
        
        # Apply morphological closing to bridge gaps/stripes in foreground objects
        H, W = pred_np.shape
        kernel_size = max(5, int(min(H, W) * 0.02))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create overlay visualization
        overlay = create_color_overlay(img_rgb, pred_np, closed_mask, BACKGROUND_CLASSES)
        
        return closed_mask, overlay
    
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
    model_name = 'nvidia/segformer-b1-finetuned-ade-512-512'
    
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
