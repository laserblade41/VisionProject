import cv2

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