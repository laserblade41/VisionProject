import cv2
import numpy as np
import random
from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib

matplotlib.use('TkAgg')


def run_orb(img_rgb, nfeatures=800):
    # Convert image to grayscale for ORB processing
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # Detect the keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Draw the keypoints on top of the RGB image
    output_img = cv2.drawKeypoints(img_rgb, keypoints, None, flags=0)

    return output_img, keypoints, descriptors


def reduce_brightness(img_np, factor=0.25):
    # Reduce brightness by scaling pixel values
    # We use np.clip to prevent overflow/underflow outside the [0, 255] range
    dark_img = np.clip(img_np * factor, 0, 255).astype(np.uint8)
    return dark_img


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


def main():
    # Load the tiny version of the ADE20K dataset
    ds = load_dataset("nateraw/ade20k-tiny", split="train")
    N = len(ds)

    # Select 4 random images
    random.seed(42)
    idxs = random.sample(range(N), 4)
    samples = [ds[i] for i in idxs]

    # Convert PIL Images to numpy arrays once at the beginning to save processing time
    images_np = [np.array(s["image"].convert("RGB")) for s in samples]

    # Create a 3x4 grid for plotting
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Make room at the bottom of the window for the slider and button
    plt.subplots_adjust(bottom=0.2)

    # Pre-calculate ORB for original images (top row) since they will never change
    original_data = []  # Store (keypoints, descriptors) for each original image
    for col, img_np in enumerate(images_np):
        orb_orig, kp_orig, desc_orig = run_orb(img_np)
        original_data.append((kp_orig, desc_orig))
        ax_orig = axes[0, col]
        ax_orig.imshow(orb_orig)
        ax_orig.axis("off")
        ax_orig.set_title(f"Original\nKeypoints: {len(kp_orig)}")

    # Define the update function that will be called when the button is clicked
    def apply_changes(event):
        # Fetch the current value from the slider
        factor = factor_slider.val

        for col, img_np in enumerate(images_np):
            # Generate the low-light version based on the new factor
            dark_img_np = reduce_brightness(img_np, factor=factor)

            # Run ORB on the newly darkened image
            orb_dark, kp_dark, desc_dark = run_orb(dark_img_np)

            # Clear the previous image on the axis and plot the updated one
            ax_dark = axes[1, col]
            ax_dark.clear()
            ax_dark.imshow(orb_dark)
            ax_dark.axis("off")
            ax_dark.set_title(f"Dark (Factor: {factor:.2f})\nKeypoints: {len(kp_dark)}")
            
            # Apply CLAHE correction to the darkened image
            dark_clahe_img = apply_clahe(dark_img_np)
            
            # Run ORB on the CLAHE-corrected image
            orb_clahe, kp_clahe, desc_clahe = run_orb(dark_clahe_img.astype(np.uint8))
            
            # Plot the CLAHE-corrected image
            ax_clahe = axes[2, col]
            ax_clahe.clear()
            ax_clahe.imshow(orb_clahe)
            ax_clahe.axis("off")
            ax_clahe.set_title(f"Dark + CLAHE\nKeypoints: {len(kp_clahe)}")

        # Redraw the matplotlib canvas to display the updates
        fig.canvas.draw_idle()

    # --- UI Elements Setup ---

    # Define axes for the slider and button [left, bottom, width, height]
    ax_slider = plt.axes([0.25, 0.05, 0.4, 0.03])
    ax_button = plt.axes([0.7, 0.05, 0.1, 0.03])

    # Create the interactive slider
    factor_slider = Slider(
        ax=ax_slider,
        label='Darkness Factor ',
        valmin=0.0,
        valmax=1.0,
        valinit=0.25,
        valstep=0.01
    )

    # Create the Apply button
    apply_btn = Button(ax_button, 'Apply', hovercolor='0.9')

    # Link the button click event to our apply_changes function
    apply_btn.on_clicked(apply_changes)

    # Trigger an initial draw for the bottom row using a dummy event
    class DummyEvent:
        pass

    apply_changes(DummyEvent())

    # Start the GUI event loop
    plt.show()


if __name__ == "__main__":
    main()