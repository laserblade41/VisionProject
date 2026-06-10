from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

# Select a heavier YOLOv8 model and prefer GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLO("yolov8x.pt")
model.to(device)

def yolo_overlay(img_rgb, conf=0.25):
    # Run prediction on the image (use GPU if available)
    r = model.predict(img_rgb, conf=conf, verbose=False, device=device)[0]
    # Plot the bounding boxes and labels onto the image
    out = r.plot()
    return out, r

def test_individual_image():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        return
        
    img_path = sys.argv[1]
    
    # Read the image and convert it to RGB
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    # RUN YOLO INSTEAD OF ORB
    output_img, results = yolo_overlay(img_rgb)
    
    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(output_img)
    plt.title(f"YOLO detections: {len(results.boxes)}")
    plt.axis("off")
    plt.show()