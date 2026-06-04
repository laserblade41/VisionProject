import cv2
import numpy as np


def apply_clahe(img_rgb):
    # Convert RGB to LAB color space for better CLAHE application
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # Create CLAHE object with typical parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE only to the L channel (lightness)
    l_channel = img_lab[:, :, 0]
    l_clahe = clahe.apply(l_channel)
    
    # Merge the CLAHE-enhanced L channel back with A and B channels
    img_lab[:, :, 0] = l_clahe
    
    # Convert back to RGB
    img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return img_clahe


def apply_median_filter(img_rgb, ksize=3):
    # Median filtering is effective for salt-and-pepper noise because it removes
    # isolated impulse pixels while preserving edges better than a mean filter.
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError("ksize must be an odd integer >= 3")

    return cv2.medianBlur(img_rgb, ksize)


def apply_restoration_filter(img_rgb, blur_ksize=5, sharpen_amount=1.5):
    # A lightweight restoration filter: unsharp masking restores edge detail,
    # then CLAHE improves local contrast for haze and fog.
    if blur_ksize % 2 == 0 or blur_ksize < 3:
        raise ValueError("blur_ksize must be an odd integer >= 3")
    if sharpen_amount < 0:
        raise ValueError("sharpen_amount must be >= 0")

    blurred_img = cv2.GaussianBlur(img_rgb, (blur_ksize, blur_ksize), 0)
    sharpened_img = cv2.addWeighted(img_rgb, 1.0 + sharpen_amount, blurred_img, -sharpen_amount, 0)
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)
    return apply_clahe(sharpened_img)