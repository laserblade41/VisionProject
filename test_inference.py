import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Import the model architecture from your dataset preparation file
from dataset_preparation import SmartRestorationModel

# Define the text labels corresponding to the output classes
CLASS_LABELS = {
    0: "Clean (No Distortion)",
    1: "Low Light",
    2: "Salt-and-Pepper Noise",
    3: "Motion Blur",
    4: "Fog"
}

def run_inference(image_path, weights_path="distortion_classifier.pth"):
    # 1. Setup processing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    # 2. Load the image from file
    raw_bgr = cv2.imread(image_path)
    if raw_bgr is None:
        print(f"Error: Could not read image at '{image_path}'. Please check the path.")
        return
    
    # Convert BGR to RGB for PyTorch and Matplotlib
    original_rgb_np = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    
    # 3. Preprocess the image for the ResNet classification head
    # Must match the exact transforms used in DistortionDataset
    tensor_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pil_img = Image.fromarray(original_rgb_np)
    # Add a batch dimension [1, 3, 224, 224] and send to device
    input_tensor = tensor_transform(pil_img).unsqueeze(0).to(device)

    # 4. Initialize model and load trained state weights
    model = SmartRestorationModel()
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Successfully loaded trained weights from '{weights_path}'")
    except FileNotFoundError:
        print(f"Error: Weights file '{weights_path}' not found. Did you run the training script first?")
        return
        
    model.to(device)
    model.eval()  # Set to evaluation mode

    # 5. Execute Forward Pass
    with torch.no_grad():
        # Pass the classification tensor and the original raw numpy array
        corrected_img, predicted_class_idx = model(input_tensor, original_rgb_np)
        
    predicted_label = CLASS_LABELS.get(predicted_class_idx, "Unknown")
    print(f"\nPrediction Result -> Detected Artifact: Layer {predicted_class_idx} ({predicted_label})")

    # 6. Visualize the Before and After results side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left subplot: Distorted Input
    axes[0].imshow(original_rgb_np)
    axes[0].set_title(f"Input Image\n(Predicted: {predicted_label})", fontsize=12, color='red' if predicted_class_idx > 0 else 'green')
    axes[0].axis("off")
    
    # Right subplot: Restored Output
    axes[1].imshow(corrected_img)
    axes[1].set_title("Restored Output Image\n(After Targeted Filter Processing)", fontsize=12, color='blue')
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure a local image path argument was given via the console
    if len(sys.argv) < 2:
        print("Usage: python test_inference.py <path_to_your_test_image.jpg>")
        sys.exit(1)
        
    test_image = sys.argv[1]
    run_inference(test_image)