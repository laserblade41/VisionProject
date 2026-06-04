import random

import cv2
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from matplotlib.widgets import RadioButtons, Slider

from distorsions import (
    apply_fog,
    apply_haze,
    apply_motion_blur,
    apply_salt_and_pepper_noise,
    reduce_brightness,
)
from filters import apply_clahe, apply_median_filter, apply_restoration_filter


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


def apply_distortion(img_np, distortion_name, strength, kernel_size):
    if distortion_name == "None":
        return img_np
    if distortion_name == "Brightness":
        return reduce_brightness(img_np, factor=strength)
    if distortion_name == "Salt & Pepper":
        return apply_salt_and_pepper_noise(img_np, amount=strength)
    if distortion_name == "Motion Blur":
        return apply_motion_blur(img_np, kernel_size=kernel_size, angle=0)
    if distortion_name == "Fog":
        return apply_fog(img_np, intensity=strength, blur_ksize=kernel_size)
    return img_np


def apply_filter(img_np, filter_name, kernel_size):
    if filter_name == "None":
        return img_np
    if filter_name == "Median":
        return apply_median_filter(img_np, ksize=kernel_size)
    if filter_name == "CLAHE":
        return apply_clahe(img_np)
    if filter_name == "Restoration":
        return apply_restoration_filter(img_np, blur_ksize=kernel_size)
    return img_np


def main():
    # Load the tiny version of the ADE20K dataset
    ds = load_dataset("nateraw/ade20k-tiny", split="train")
    image_count = len(ds)

    # Select 4 random images
    random.seed(42)
    idxs = random.sample(range(image_count), 4)
    samples = [ds[i] for i in idxs]

    # Convert PIL Images to numpy arrays once at the beginning to save processing time
    images_np = [np.array(sample["image"].convert("RGB")) for sample in samples]

    # Create a 2x4 grid for plotting: original images and selected output
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    plt.subplots_adjust(left=0.28, bottom=0.22, right=0.98, top=0.9, wspace=0.05, hspace=0.15)
    fig.suptitle("General Image Processing UI", fontsize=16)

    # Precompute the original row because it never changes
    for col, img_np in enumerate(images_np):
        original_img, original_kp, _ = run_orb(img_np)
        ax_original = axes[0, col]
        ax_original.imshow(original_img)
        ax_original.axis("off")
        ax_original.set_title(f"Original\nKeypoints: {len(original_kp)}")

    ax_distortion = plt.axes([0.03, 0.46, 0.20, 0.40], facecolor="#f6f1e8")
    ax_filter = plt.axes([0.03, 0.18, 0.20, 0.18], facecolor="#f6f1e8")
    ax_strength = plt.axes([0.30, 0.13, 0.62, 0.03])
    ax_kernel = plt.axes([0.30, 0.08, 0.62, 0.03])

    distortion_selector = RadioButtons(
        ax_distortion,
        ("None", "Brightness", "Salt & Pepper", "Motion Blur", "Fog"),
        active=1,
    )
    filter_selector = RadioButtons(
        ax_filter,
        ("None", "Median", "CLAHE", "Restoration"),
        active=2,
    )

    strength_slider = Slider(
        ax=ax_strength,
        label="Strength",
        valmin=0.0,
        valmax=1.0,
        valinit=0.35,
        valstep=0.01,
    )
    kernel_slider = Slider(
        ax=ax_kernel,
        label="Kernel Size",
        valmin=3,
        valmax=31,
        valinit=9,
        valstep=2,
    )

    def render_preview(event=None):
        distortion_name = distortion_selector.value_selected
        filter_name = filter_selector.value_selected
        strength = float(strength_slider.val)
        kernel_size = int(kernel_slider.val)

        for col, img_np in enumerate(images_np):
            distorted_img = apply_distortion(img_np, distortion_name, strength, kernel_size)
            filtered_img = apply_filter(distorted_img, filter_name, kernel_size)

            orb_img, keypoints, _ = run_orb(filtered_img)

            ax_result = axes[1, col]
            ax_result.clear()
            ax_result.imshow(orb_img)
            ax_result.axis("off")
            ax_result.set_title(
                f"{distortion_name} + {filter_name}\nKeypoints: {len(keypoints)}"
            )

        fig.canvas.draw_idle()

    distortion_selector.on_clicked(render_preview)
    filter_selector.on_clicked(render_preview)
    strength_slider.on_changed(render_preview)
    kernel_slider.on_changed(render_preview)

    render_preview()
    plt.show()


if __name__ == "__main__":
    main()