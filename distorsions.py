import cv2
import numpy as np


def reduce_brightness(img_np, factor=0.25):
    # Reduce brightness by scaling pixel values
    # We use np.clip to prevent overflow/underflow outside the [0, 255] range
    dark_img = np.clip(img_np * factor, 0, 255).astype(np.uint8)
    return dark_img


def apply_salt_and_pepper_noise(img_np, amount=0.05, salt_vs_pepper=0.5):
    # Add random white and black pixels to simulate salt-and-pepper noise
    noisy_img = np.array(img_np, copy=True)
    total_pixels = noisy_img.shape[0] * noisy_img.shape[1]
    num_salt = int(total_pixels * amount * salt_vs_pepper)
    num_pepper = int(total_pixels * amount * (1.0 - salt_vs_pepper))

    if noisy_img.ndim == 2:
        salt_coords = (
            np.random.randint(0, noisy_img.shape[0], num_salt),
            np.random.randint(0, noisy_img.shape[1], num_salt),
        )
        pepper_coords = (
            np.random.randint(0, noisy_img.shape[0], num_pepper),
            np.random.randint(0, noisy_img.shape[1], num_pepper),
        )
        noisy_img[salt_coords] = 255
        noisy_img[pepper_coords] = 0
    else:
        salt_coords = (
            np.random.randint(0, noisy_img.shape[0], num_salt),
            np.random.randint(0, noisy_img.shape[1], num_salt),
        )
        pepper_coords = (
            np.random.randint(0, noisy_img.shape[0], num_pepper),
            np.random.randint(0, noisy_img.shape[1], num_pepper),
        )
        noisy_img[salt_coords[0], salt_coords[1], :] = 255
        noisy_img[pepper_coords[0], pepper_coords[1], :] = 0

    return noisy_img.astype(np.uint8)


def apply_motion_blur(img_np, kernel_size=9, angle=0):
    # Motion blur is modeled with a line kernel that can be rotated to any angle.
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0

    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))

    kernel_sum = kernel.sum()
    if kernel_sum == 0:
        kernel[kernel_size // 2, :] = 1.0
        kernel_sum = kernel.sum()

    kernel /= kernel_sum
    blurred_img = cv2.filter2D(img_np, -1, kernel)
    return blurred_img.astype(np.uint8)


def apply_haze(img_np, intensity=0.35):
    # Haze reduces contrast and washes the image toward white.
    if intensity < 0 or intensity > 1:
        raise ValueError("intensity must be between 0 and 1")

    white_overlay = np.full_like(img_np, 255)
    hazy_img = cv2.addWeighted(img_np, 1.0 - intensity, white_overlay, intensity, 0)
    return hazy_img.astype(np.uint8)


def apply_fog(img_np, intensity=0.5, blur_ksize=15):
    # Fog is a denser atmospheric distortion, so we blur first and then wash out.
    if intensity < 0 or intensity > 1:
        raise ValueError("intensity must be between 0 and 1")
    if blur_ksize % 2 == 0 or blur_ksize < 3:
        raise ValueError("blur_ksize must be an odd integer >= 3")

    blurred_img = cv2.GaussianBlur(img_np, (blur_ksize, blur_ksize), 0)
    white_overlay = np.full_like(img_np, 255)
    foggy_img = cv2.addWeighted(blurred_img, 1.0 - intensity, white_overlay, intensity, 0)
    return foggy_img.astype(np.uint8)

