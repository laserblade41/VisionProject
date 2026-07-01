import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
from filters import apply_clahe, apply_median_filter, apply_restoration_filter

class SmartRestorationModel(nn.Module):
    def __init__(self):
        super(SmartRestorationModel, self).__init__()
        # Initialize the backbone classifier
        self.classifier = models.resnet18(pretrained=True)
        
        # five output classes:
        # 0: clean
        # 1: low light
        # 2: salt-and-pepper noise
        # 3: motion blur
        # 4: fog
        
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, 5)
        
    def _restore_brightness(self, img):
        # as per our report, to solve the low light problem, we use clahe
        return apply_clahe(img)
    
    def _restore_salt_pepper_noise(self, img):
        # Use median filtering to remove salt-and-pepper noise
        return apply_median_filter(img)
    
    def _restore_motion_blur(self, img):
        # Use a restoration filter that combines unsharp masking and CLAHE
        return apply_restoration_filter(img)
    
    def _restore_fog(self, img):
        # Use a restoration filter that combines unsharp masking and CLAHE
        return apply_restoration_filter(img)


    def forward(self, x_tensor, original_rgb_numpy):
        """
        x_tensor: Normalized torch tensor for classification (Shape: [1, 3, 224, 224])
        original_rgb_numpy: The raw original RGB image array to apply filters on
        """
        # Predict the distortion type
        logits = self.classifier(x_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
        
        # Dynamically route to the right filter processing layer
        # case 1: low light -> apply CLAHE
        if predicted_class == 1:
            corrected_img = self._restore_brightness(original_rgb_numpy)
        # case 2: salt-and-pepper noise -> apply median filtering
        elif predicted_class == 2:
            corrected_img = self._restore_salt_pepper_noise(original_rgb_numpy)
        # case 3: motion blur -> apply restoration filter
        elif predicted_class == 3:
            corrected_img = self._restore_motion_blur(original_rgb_numpy)
        # case 4: fog -> apply restoration filter
        elif predicted_class == 4:
            corrected_img = self._restore_fog(original_rgb_numpy)
        else:
            corrected_img = original_rgb_numpy  # Class 0: Image is already clean
            
        return corrected_img, predicted_class