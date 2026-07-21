import os
import random
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torch.optim import AdamW

# Import your custom distortions (Removed brightness and fog)
from distorsions import (
    apply_salt_and_pepper_noise,
    apply_motion_blur,
    apply_gaussian_noise
)


class UnsupervisedRobustDataset(Dataset):
    """
    Dataset that loads only images, creating a clean version and a
    specifically distorted version for Teacher-Student training using DeepLabV3.
    """

    def __init__(self, image_dir, preprocess_transform, noise_type):
        self.image_dir = image_dir
        self.preprocess = preprocess_transform
        self.noise_type = noise_type
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

        # Load all valid image files from the single folder
        self.images = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in valid_exts]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img_bgr = cv2.imread(img_path)

        # Handle gracefully if an image fails to load
        if img_bgr is None:
            return self.__getitem__((idx + 1) % len(self.images))

        clean_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        distorted_rgb = clean_rgb.copy()

        # Apply specific environmental distortion based on the expert type
        if self.noise_type == "sp":
            # Random salt and pepper severity
            distorted_rgb = apply_salt_and_pepper_noise(distorted_rgb, amount=random.uniform(0.1, 0.4))

        elif self.noise_type == "blur":
            # Random kernel size for motion blur
            k = random.choice([5, 11, 19])
            distorted_rgb = apply_motion_blur(distorted_rgb, kernel_size=k)

        elif self.noise_type == "gaussian":
            # Random standard deviation for Gaussian noise to improve robustness
            std = random.choice([25, 50, 75])
            distorted_rgb = apply_gaussian_noise(distorted_rgb, std=std)

        clean_pil = Image.fromarray(clean_rgb)
        distorted_pil = Image.fromarray(distorted_rgb)

        # Process both images using torchvision's specific DeepLabV3 transforms
        clean_tensor = self.preprocess(clean_pil)
        distorted_tensor = self.preprocess(distorted_pil)

        return {
            "clean_pixel_values": clean_tensor,
            "distorted_pixel_values": distorted_tensor
        }


def train_expert_deeplab(noise_type):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    IMAGE_DIR = os.path.join(current_dir, "coco_5k_images")

    # Define the save directory for the specific expert
    save_dir = os.path.join(current_dir, f"finetuned_robust_student_deeplab_{noise_type}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "robust_deeplab_student.pth")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[{noise_type.upper()} EXPERT] Training using Teacher-Student Distillation on {device}...")

    # Load DeepLabV3 default weights
    weights = DeepLabV3_ResNet50_Weights.DEFAULT

    # Define a custom preprocess pipeline that forces a strict 520x520 size
    preprocess = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. Load Teacher Model (Frozen, evaluates clean images to create pseudo-labels)
    teacher_model = deeplabv3_resnet50(weights=weights)
    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 2. Load Student Model (Active, learns from distorted images)
    if os.path.exists(save_path):
        print(f"Loading existing student model weights from {save_path} to continue training...")
        student_model = deeplabv3_resnet50(weights=None, num_classes=21, aux_loss=True)
        student_model.load_state_dict(torch.load(save_path, map_location=device))
        learning_rate = 0.00001
    else:
        print("No existing model found. Starting fresh from base DeepLabV3 weights...")
        student_model = deeplabv3_resnet50(weights=weights)
        learning_rate = 0.00005

    student_model.to(device)
    student_model.train()

    # Setup Dataset and DataLoader with the specific noise type
    dataset = UnsupervisedRobustDataset(IMAGE_DIR, preprocess, noise_type)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    optimizer = AdamW(student_model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"  Epoch {epoch + 1}/{num_epochs}")
        print("  " + "-" * 15)

        for idx, batch in enumerate(dataloader):
            clean_pixels = batch["clean_pixel_values"].to(device)
            distorted_pixels = batch["distorted_pixel_values"].to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(clean_pixels)['out']
                pseudo_labels = teacher_outputs.argmax(dim=1)

            student_outputs = student_model(distorted_pixels)['out']
            loss = loss_fn(student_outputs, pseudo_labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if idx % 10 == 0:
                print(f"    Batch {idx} | Distillation Loss: {loss.item():.4f}")

    # Save the robust student model weights
    torch.save(student_model.state_dict(), save_path)
    print(f"Success! {noise_type.upper()} Robust model saved to {save_path}\n")


if __name__ == "__main__":
    # Train the three required experts seamlessly
    experts = ["sp", "blur", "gaussian"]
    for expert in experts:
        train_expert_deeplab(expert)