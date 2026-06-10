import random
import numpy as np
import cv2


def add_gaussian_noise(img, sigma=10):
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noised = img.astype(np.float32) + gauss
    noised = np.clip(noised, 0, 255).astype(np.uint8)
    return noised


def blur_image(img, k=5):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def adjust_brightness_contrast(img, alpha=1.0, beta=0):
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def rotate_image(img, angle=10):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_random_distortion(img, strength=1.0, seed=None):
    """Apply a random combination of distortions to the RGB uint8 image.
    strength: 0..1 controlling magnitude
    seed: optional int to make distortions reproducible
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_out = img.copy()
    ops = random.choices(['noise', 'blur', 'brightness', 'rotate'], k=random.randint(1,3))
    for op in ops:
        if op == 'noise':
            sigma = max(1.0, int(30 * strength))
            img_out = add_gaussian_noise(img_out, sigma=sigma)
        elif op == 'blur':
            k = int(1 + 8 * strength)
            img_out = blur_image(img_out, k)
        elif op == 'brightness':
            alpha = 1.0 + (random.uniform(-0.6, 0.6) * strength)
            beta = int(30 * (random.uniform(-1, 1) * strength))
            img_out = adjust_brightness_contrast(img_out, alpha=alpha, beta=beta)
        elif op == 'rotate':
            ang = random.uniform(-25, 25) * strength
            img_out = rotate_image(img_out, ang)
    return img_out


# Provide a helper to enumerate available distortions (useful for tests)
def available_distortions():
    return ['noise', 'blur', 'brightness', 'rotate']
