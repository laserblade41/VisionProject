import os
import cv2
import glob
import shutil
import random
from ultralytics import YOLO
from ultralytics.utils.downloads import download
from distorsions import apply_motion_blur, apply_salt_and_pepper_noise, reduce_brightness


def create_coco_yaml(dataset_dir):
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(dataset_dir)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        for i, name in enumerate(classes):
            f.write(f"  {i}: {name}\n")
    return yaml_path


def build_specialized_datasets():
    base_dir = os.path.abspath("datasets")

    # Define directories for the three specialized experts
    experts = {
        "sp": os.path.join(base_dir, "coco1000_sp"),
        "blur": os.path.join(base_dir, "coco1000_blur"),
        "gauss": os.path.join(base_dir, "coco1000_gauss")
    }

    # Verify if datasets already exist to skip creation
    if all(os.path.exists(d) for d in experts.values()):
        print("Specialized datasets already exist. Skipping creation.")
        return {k: os.path.join(v, "data.yaml") for k, v in experts.items()}

    print("Building base clean dataset first...")
    download('http://images.cocodataset.org/zips/val2017.zip', dir=base_dir)
    download('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip', dir=base_dir)

    raw_imgs = glob.glob(os.path.join(base_dir, "val2017/*.jpg"))
    random.shuffle(raw_imgs)

    # Filter 1200 valid images (1000 train, 200 val)
    valid_images = []
    for img in raw_imgs:
        bn = os.path.basename(img).replace('.jpg', '')
        lbl = os.path.join(base_dir, "coco/labels/val2017", f"{bn}.txt")
        if os.path.exists(lbl):
            valid_images.append((img, lbl))
        if len(valid_images) >= 1200:
            break

    train_data = valid_images[:1000]
    val_data = valid_images[1000:1200]

    yaml_paths = {}

    for noise_type, dataset_dir in experts.items():
        print(f"Generating dataset for: {noise_type}")

        # Create Train and Val folders
        for split in ["train", "val"]:
            os.makedirs(os.path.join(dataset_dir, f"images/{split}"), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, f"labels/{split}"), exist_ok=True)

        # Apply multi-level augmentation to Train data
        for img_path, lbl_path in train_data:
            img = cv2.imread(img_path)
            if img is None:
                continue
            bn = os.path.basename(img_path).replace('.jpg', '')

            if noise_type == "sp":
                # 3 levels of Salt and Pepper noise
                for sp in [0.2, 0.3, 0.4]:
                    aug_img = apply_salt_and_pepper_noise(img, sp)
                    cv2.imwrite(os.path.join(dataset_dir, "images/train", f"{bn}_sp{int(sp * 100)}.jpg"), aug_img)
                    shutil.copy(lbl_path, os.path.join(dataset_dir, "labels/train", f"{bn}_sp{int(sp * 100)}.txt"))

            elif noise_type == "blur":
                # 3 levels of Motion Blur
                for k in [5, 11, 19]:
                    aug_img = apply_motion_blur(img, k)
                    cv2.imwrite(os.path.join(dataset_dir, "images/train", f"{bn}_blur{k}.jpg"), aug_img)
                    shutil.copy(lbl_path, os.path.join(dataset_dir, "labels/train", f"{bn}_blur{k}.txt"))

            elif noise_type == "bright":
                # 3 levels of Brightness Reduction
                for bf in [0.6, 0.4, 0.2]:
                    aug_img = reduce_brightness(img, factor=bf)
                    cv2.imwrite(os.path.join(dataset_dir, "images/train", f"{bn}_bright{int(bf * 100)}.jpg"), aug_img)
                    shutil.copy(lbl_path, os.path.join(dataset_dir, "labels/train", f"{bn}_bright{int(bf * 100)}.txt"))

        # Copy Val data (Clean images to objectively test generalization)
        for img_path, lbl_path in val_data:
            bn = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(dataset_dir, f"images/val/{bn}"))
            shutil.copy(lbl_path, os.path.join(dataset_dir, f"labels/val/{bn.replace('.jpg', '.txt')}"))

        yaml_paths[noise_type] = create_coco_yaml(dataset_dir)

    return yaml_paths


if __name__ == "__main__":
    yaml_files = build_specialized_datasets()
    print("\nStarting specialized trainings...")

    for noise_type, yaml_file in yaml_files.items():
        print(f"\n--- Training Expert Model for {noise_type.upper()} ---")

        # Path to this specific expert's best weights
        best_weights_path = f"runs/detect/train_{noise_type}/weights/best.pt"

        if os.path.exists(best_weights_path):
            print(f"Found previous best weights for {noise_type}. Starting extended training (fine-tuning)...")
            model = YOLO(best_weights_path)
        else:
            print(f"Starting a fresh training session for {noise_type}...")
            model = YOLO("yolov8x.pt")

        # Train and allow overriding in the same dedicated folder
        model.train(
            data=yaml_file,
            epochs=50,
            imgsz=640,
            device="cuda",
            cache=False,
            workers=4,
            project="",
            name=f"train_{noise_type}",
            exist_ok=True
        )