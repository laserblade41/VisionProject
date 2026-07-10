import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2

# Import your custom distortion functions
from distorsions import reduce_brightness, apply_salt_and_pepper_noise, apply_motion_blur, apply_fog 

# 1. Define your distortions dictionary
distortions = {
    "LowLight": reduce_brightness,
    "SaltAndPepper": apply_salt_and_pepper_noise,
    "MotionBlur": apply_motion_blur,
    "Fog": apply_fog
}

# =====================================================================
# 2. Smart Restoration Model Definition (Torch Module)
# =====================================================================
class SmartRestorationModel(nn.Module):
    def __init__(self):
        super(SmartRestorationModel, self).__init__()
        # Load a lightweight, pre-trained feature extractor
        self.classifier = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 5 output classes: Clean (0), Low Light (1), Salt-and-Pepper (2), Motion Blur (3), Fog (4)
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_ftrs, 5)
        
    def _restore_lowlight(self, img_np):
        # Gamma correction + CLAHE local contrast enhancement
        gamma = 0.35
        lut = (np.arange(256) / 255.0) ** gamma * 255
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        img_gamma = cv2.LUT(img_np, lut)
        
        lab = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        
    def _restore_salt_and_pepper(self, img_np):
        # Median blur is highly effective at wiping out severe impulse pixels
        return cv2.medianBlur(img_np, 5)
        
    def _restore_motion_blur(self, img_np):
        # Unsharp masking filter to re-sharpen high-frequency edge gradients
        blurred = cv2.GaussianBlur(img_np, (9, 9), 0)
        sharpened = cv2.addWeighted(img_np, 1.5, blurred, -0.5, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
        
    def _restore_fog(self, img_np):
        # Local CLAHE pass across luminance channel to pull out washed out contrast
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    def forward(self, x_tensor, original_rgb_numpy):
        """
        x_tensor: Normalized classification tensor [Batch, 3, 224, 224]
        original_rgb_numpy: Raw uint8 RGB array to apply classical filters on
        """
        # 1. Predict the distortion type using the trainable ResNet head
        logits = self.classifier(x_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
        
        # 2. Route to the optimized restoration channel based on prediction
        if predicted_class == 1:
            corrected_img = self._restore_lowlight(original_rgb_numpy)
        elif predicted_class == 2:
            corrected_img = self._restore_salt_and_pepper(original_rgb_numpy)
        elif predicted_class == 3:
            corrected_img = self._restore_motion_blur(original_rgb_numpy)
        elif predicted_class == 4:
            corrected_img = self._restore_fog(original_rgb_numpy)
        else:
            corrected_img = original_rgb_numpy  # Class 0: Image is clean
            
        return corrected_img, predicted_class


# =====================================================================
# 3. Dataset Generation Utilities
# =====================================================================
class DistortionDataset(Dataset):
    def __init__(self, hf_dataset_split="train"):
        print(f"Initializing {hf_dataset_split} dataset...")
        
        # FIX: Switch to a pre-converted Parquet repository variant
        self.base_ds = load_dataset("vikhyatk/scene_parse_150", split=f"{hf_dataset_split}[:2000]")
        self.num_base_images = len(self.base_ds)
        
        print(f"Successfully cached {self.num_base_images} base images!")
        
        # Standard preprocessing transform for ResNet-18 classification
        self.tensor_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # Each base image yields 5 variations: Clean (0), Low Light (1), Salt-and-Pepper (2), Motion Blur (3), Fog (4)
        return self.num_base_images * 5

    def __getitem__(self, idx):
        base_idx = idx // 5
        distortion_type = idx % 5
        
        raw_pil_img = self.base_ds[base_idx]["image"].convert("RGB")
        raw_np_img = np.array(raw_pil_img)
        
        if distortion_type == 0:
            processed_np = raw_np_img
            label = 0  # Clean
        elif distortion_type == 1:
            processed_np = reduce_brightness(raw_np_img, factor=0.25)
            label = 1  # Low Light
        elif distortion_type == 2:
            processed_np = apply_salt_and_pepper_noise(raw_np_img, amount=0.05)
            label = 2  # Salt-and-Pepper Noise
        elif distortion_type == 3:
            processed_np = apply_motion_blur(raw_np_img, kernel_size=9)
            label = 3  # Motion Blur
        elif distortion_type == 4:
            processed_np = apply_fog(raw_np_img, intensity=0.5)
            label = 4  # Fog
            
        processed_pil = Image.fromarray(processed_np)
        tensor_img = self.tensor_transform(processed_pil)
        
        # We return the raw_np_img too so our model forward pass can apply the filters
        return tensor_img, label, raw_np_img
    
def collate_fn(batch):
    transposed = list(zip(*batch))
    tensor_imgs = torch.stack(transposed[0], 0)
    labels = torch.tensor(transposed[1])
    raw_np_imgs = list(transposed[2])
    return tensor_imgs, labels, raw_np_imgs

def get_dataloaders(batch_size=16, split="train"):
    train_dataset = DistortionDataset(hf_dataset_split=split)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        collate_fn=collate_fn
    )
    return train_loader

if __name__ == "__main__":
    loader = get_dataloaders(batch_size=1)
    images, labels, raw_arrays = next(iter(loader))
    print("Batch Check -> Image Tensor Shape:", images.shape, "| Labels Target Shape:", labels.shape)
    
    # Quick sanity validation of the integrated module structure
    model = SmartRestorationModel()
    single_img_numpy = raw_arrays[0]
    
    corrected, predicted_cls = model(images, single_img_numpy)
    print(f"Sanity Pass Verification Successful! Model predicted class: {predicted_cls}")