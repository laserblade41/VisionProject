import os
import urllib.request
import zipfile
import shutil


def download_coco_5k_images():
    # Define the target directory for the images
    base_dir = "coco_5k_images"
    os.makedirs(base_dir, exist_ok=True)

    # URL for the COCO 2017 validation images (exactly 5,000 images, ~1GB)
    img_url = "http://images.cocodataset.org/zips/val2017.zip"
    img_zip = "val2017.zip"

    print("Downloading COCO 2017 Validation Images (~1GB)...")
    # Download the zip file
    urllib.request.urlretrieve(img_url, img_zip)

    print("Extracting images...")
    # Extract the images to a temporary folder first
    with zipfile.ZipFile(img_zip, 'r') as zip_ref:
        zip_ref.extractall("temp_extract")

    print("Moving images to the final directory...")
    # Move all the .jpg files from the extracted folder to our base_dir
    source_folder = os.path.join("temp_extract", "val2017")
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg"):
            shutil.move(os.path.join(source_folder, filename), os.path.join(base_dir, filename))

    print("Cleaning up temporary files...")
    # Remove the zip file and the temporary extraction folder
    os.remove(img_zip)
    shutil.rmtree("temp_extract")

    print(f"Success! 5,000 images are ready in the '{base_dir}' folder.")


if __name__ == "__main__":
    download_coco_5k_images()
