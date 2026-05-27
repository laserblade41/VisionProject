import sys
import cv2
import numpy as np
import random
from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image

def run_orb(img_rgb, nfeatures=800):
    # 1. Convert image to grayscale (ORB operates on single-channel images)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. Initialize the ORB detector with a set limit on features
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    # 3. Detect the keypoints
    keypoints = orb.detect(gray, None)
    
    # 4. Draw the keypoints on top of the original image for visualization
    output_img = cv2.drawKeypoints(img_rgb, keypoints, None, flags=0)
    
    return output_img, keypoints

def overlay_mask(img_pil, mask_pil, alpha=0.45):
    # Convert PIL Image to a float32 numpy array
    img = np.array(img_pil.convert("RGB")).astype(np.float32)
    
    # Convert the mask to an int32 numpy array
    m = np.array(mask_pil).astype(np.int32)
    
    # Generate a random color palette for the 150 possible classes in ADE20K
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, size=(256, 3), dtype=np.uint8)
    
    # Apply the colors to the mask based on the class integer
    color = palette[(m % 256).astype(np.int32)]
    
    # Blend the original image and the colored mask using the alpha value
    out = (img * (1 - alpha) + color.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(out)

# main loop:
def main():
    # 1. Load the tiny version of the ADE20K dataset from Hugging Face
    print("Loading dataset...")
    ds = load_dataset("nateraw/ade20k-tiny", split="train")
    N = len(ds)

    # 2. Select 4 random images to test your pipeline
    random.seed(7)
    idxs = random.sample(range(N), 4)
    samples = [ds[i] for i in idxs]

    # 3. Separate the raw images and their ground truth labels (masks)
    images = [s["image"] for s in samples]
    masks = [s["label"] for s in samples]
    
    # Plot the 4 samples side-by-side
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, m in zip(axes, images, masks):
        ax.imshow(overlay_mask(img, m))
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

def test_individual_image(img_path):
    print(f"Processing {img_path}...")
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Could not load image at {img_path}. Check the path.")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # use overlay_mask to visualize the image with a dummy mask (for testing purposes)
    dummy_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    dummy_mask[img_rgb.shape[0]//4:3*img_rgb.shape[0]//4, img_rgb.shape[1]//4:3*img_rgb.shape[1]//4] = 1  # a simple square mask
    img_with_mask = overlay_mask(Image.fromarray(img_rgb), Image.fromarray(dummy_mask))
    
    # Run ORB on the masked image
    output_img, keypoints = run_orb(np.array(img_with_mask))
    
    # show the OUTPUT image with the detected keypoints (this was the bug!)
    plt.figure(figsize=(10, 10))
    plt.imshow(output_img)
    plt.title(f"ORB keypoints: {len(keypoints)}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # If a system argument is provided, test the individual image
    if len(sys.argv) > 1:
        test_individual_image(sys.argv[1])
    # Otherwise, run the default dataset test
    else:
        main()