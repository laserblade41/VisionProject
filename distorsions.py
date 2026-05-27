import numpy as np


def reduce_brightness(img_np, factor=0.25):
    # Reduce brightness by scaling pixel values
    # We use np.clip to prevent overflow/underflow outside the [0, 255] range
    dark_img = np.clip(img_np * factor, 0, 255).astype(np.uint8)
    return dark_img