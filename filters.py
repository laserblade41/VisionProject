import cv2


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