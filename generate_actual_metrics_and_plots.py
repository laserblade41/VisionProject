import os
import glob
import json
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Headless backend for plot generation

import torch
from ORB import run_orb
from YOLO import yolo_overlay, model as yolo_model
from distorsions import (
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_motion_blur,
)
from filters import (
    apply_bilateral_filter,
    apply_median_filter,
    apply_restoration_filter,
)
from semantic_seg import load_seg_model, predict_seg_mask, iou_mask

# Helper function to compute Signal-to-Noise Ratio (SNR in dB)
def compute_snr(clean_rgb, dist_rgb):
    clean = clean_rgb.astype(np.float64)
    dist = dist_rgb.astype(np.float64)
    noise = clean - dist
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power <= 1e-10:
        return 50.0  # Cap clean/identical at 50 dB
    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return float(snr_db)

# Helper function to extract bounding boxes and classes from YOLO result
def extract_yolo_boxes_and_classes(result, conf_thresh=0.25):
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return [], [], []
    boxes = result.boxes
    confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
    clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else []
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else []
    
    valid_boxes = []
    valid_clss = []
    valid_confs = []
    for i in range(len(confs)):
        if confs[i] >= conf_thresh:
            valid_boxes.append(xyxy[i])
            valid_clss.append(int(clss[i]))
            valid_confs.append(float(confs[i]))
    return valid_boxes, valid_clss, valid_confs

# Calculate IoU between two bounding boxes xyxy
def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)

# Target COCO class mapping for readable evaluation
COCO_CLASSES = {
    0: 'person', 2: 'car', 5: 'bus', 7: 'truck', 15: 'cat', 16: 'dog',
    24: 'backpack', 26: 'handbag', 39: 'bottle', 41: 'cup', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 66: 'keyboard',
    67: 'cell phone', 72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock'
}

# Helper function to compute true ORB descriptor match ratio using Lowe's ratio test
def compute_orb_match_ratio(clean_kps, clean_des, dist_kps, dist_des, ratio_thresh=0.75):
    if clean_kps is None or len(clean_kps) == 0:
        return None  # Skip image if clean image detects no keypoints
    if dist_kps is None or len(dist_kps) == 0 or clean_des is None or dist_des is None or len(clean_des) < 2 or len(dist_des) < 2:
        return 0.0
    try:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn_matches = bf.knnMatch(clean_des, dist_des, k=2)
        good_matches = []
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
        return float(min(1.0, len(good_matches) / len(clean_kps)))
    except Exception:
        return 0.0

def main():
    print("Starting comprehensive benchmark measurements across all 3 vision tasks...")
    os.makedirs("assets", exist_ok=True)

    NUM_IMAGES = 250

    # Load images for benchmark (sample 15 representative images from dataset)
    image_paths = sorted(glob.glob("test_dataset_250/images/*.jpg"))
    clean_image_paths = [p for p in image_paths if not any(x in p for x in ['_blur', '_bright', '_sp'])][:NUM_IMAGES]
    
    if len(clean_image_paths) == 0:
        print("Fallback: searching datasets/val2017...")
        clean_image_paths = sorted(glob.glob("datasets/val2017/*.jpg"))[:NUM_IMAGES]

    if len(clean_image_paths) == 0:
        print("Fallback: searching test_dataset_250/images...")
        clean_image_paths = sorted(glob.glob("test_dataset_250/images/*.jpg"))[:NUM_IMAGES]

    if len(clean_image_paths) == 0:
        print("Fallback: downloading sample COCO benchmark images...")
        os.makedirs("datasets/benchmark_clean", exist_ok=True)
        sample_urls = [
            "http://images.cocodataset.org/val2017/000000000139.jpg",
            "http://images.cocodataset.org/val2017/000000000285.jpg",
            "http://images.cocodataset.org/val2017/000000000632.jpg",
            "http://images.cocodataset.org/val2017/000000000724.jpg",
            "http://images.cocodataset.org/val2017/000000000776.jpg",
            "http://images.cocodataset.org/val2017/000000000785.jpg",
            "http://images.cocodataset.org/val2017/000000000802.jpg",
            "http://images.cocodataset.org/val2017/000000000872.jpg",
            "http://images.cocodataset.org/val2017/000000000885.jpg",
            "http://images.cocodataset.org/val2017/000000001000.jpg",
            "http://images.cocodataset.org/val2017/000000001268.jpg",
            "http://images.cocodataset.org/val2017/000000001296.jpg",
            "http://images.cocodataset.org/val2017/000000001353.jpg",
            "http://images.cocodataset.org/val2017/000000001425.jpg",
            "http://images.cocodataset.org/val2017/000000001490.jpg"
        ]
        import urllib.request
        for url in sample_urls:
            fn = os.path.join("datasets/benchmark_clean", os.path.basename(url))
            if not os.path.exists(fn):
                try:
                    urllib.request.urlretrieve(url, fn)
                except Exception as e:
                    print(f"Could not download {url}: {e}")
        clean_image_paths = sorted(glob.glob("datasets/benchmark_clean/*.jpg"))[:NUM_IMAGES]

    print(f"Loaded {len(clean_image_paths)} clean benchmark images.")

    # Load Segmentation Model
    print("Loading DeepLabV3 segmentation model...")
    seg_model_tuple = load_seg_model('deeplabv3_base')

    # Define distortion sweep parameters
    gaussian_stds = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0]
    sp_noise_amounts = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    blur_kernel_sizes = [1, 5, 9, 15, 21, 29]

    # --- 1. BENCHMARK TASK 1: ORB KEYPOINT DETECTION ---
    print("\n--- Measuring Task 1: ORB Keypoint Detection ---")
    orb_results = {'Gaussian Noise': [], 'S&P Noise': [], 'Motion Blur': []}

    # Gaussian Noise sweep
    for std_val in gaussian_stds:
        snr_list, match_ratio_list, kps_list = [], [], []
        for p in clean_image_paths:
            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            clean_out, clean_kps, clean_des = run_orb(img_rgb, nfeatures=800)
            
            dist_rgb = apply_gaussian_noise(img_rgb, std=std_val) if std_val > 0 else img_rgb.copy()
            snr = compute_snr(img_rgb, dist_rgb)
            dist_out, dist_kps, dist_des = run_orb(dist_rgb, nfeatures=800)
            
            match_ratio = compute_orb_match_ratio(clean_kps, clean_des, dist_kps, dist_des)
            if match_ratio is not None:
                snr_list.append(snr)
                match_ratio_list.append(match_ratio)
                kps_list.append(len(dist_kps))
        
        avg_snr = float(np.mean(snr_list)) if snr_list else 50.0
        avg_ratio = float(np.mean(match_ratio_list)) if match_ratio_list else 0.0
        avg_kps = float(np.mean(kps_list)) if kps_list else 0.0
        orb_results['Gaussian Noise'].append({'intensity': std_val, 'snr': avg_snr, 'match_ratio': avg_ratio, 'kps': avg_kps})

    # Noise sweep
    for sp in sp_noise_amounts:
        snr_list, match_ratio_list, kps_list = [], [], []
        for p in clean_image_paths:
            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            clean_out, clean_kps, clean_des = run_orb(img_rgb, nfeatures=800)
            
            dist_rgb = apply_salt_and_pepper_noise(img_rgb, amount=sp) if sp > 0 else img_rgb.copy()
            snr = compute_snr(img_rgb, dist_rgb)
            dist_out, dist_kps, dist_des = run_orb(dist_rgb, nfeatures=800)
            
            match_ratio = compute_orb_match_ratio(clean_kps, clean_des, dist_kps, dist_des)
            if match_ratio is not None:
                snr_list.append(snr)
                match_ratio_list.append(match_ratio)
                kps_list.append(len(dist_kps))
        
        avg_snr = float(np.mean(snr_list)) if snr_list else 50.0
        avg_ratio = float(np.mean(match_ratio_list)) if match_ratio_list else 0.0
        avg_kps = float(np.mean(kps_list)) if kps_list else 0.0
        orb_results['S&P Noise'].append({'intensity': sp, 'snr': avg_snr, 'match_ratio': avg_ratio, 'kps': avg_kps})

    # Motion Blur sweep
    for k in blur_kernel_sizes:
        snr_list, match_ratio_list, kps_list = [], [], []
        for p in clean_image_paths:
            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            clean_out, clean_kps, clean_des = run_orb(img_rgb, nfeatures=800)
            
            dist_rgb = apply_motion_blur(img_rgb, kernel_size=k) if k > 1 else img_rgb.copy()
            snr = compute_snr(img_rgb, dist_rgb)
            dist_out, dist_kps, dist_des = run_orb(dist_rgb, nfeatures=800)
            
            match_ratio = compute_orb_match_ratio(clean_kps, clean_des, dist_kps, dist_des)
            if match_ratio is not None:
                snr_list.append(snr)
                match_ratio_list.append(match_ratio)
                kps_list.append(len(dist_kps))
        
        avg_snr = float(np.mean(snr_list)) if snr_list else 50.0
        avg_ratio = float(np.mean(match_ratio_list)) if match_ratio_list else 0.0
        avg_kps = float(np.mean(kps_list)) if kps_list else 0.0
        orb_results['Motion Blur'].append({'intensity': k, 'snr': avg_snr, 'match_ratio': avg_ratio, 'kps': avg_kps})


    # --- 2. BENCHMARK TASK 2: YOLO OBJECT DETECTION (PER-CLASS & PER-SNR) ---
    print("\n--- Measuring Task 2: YOLO Object Detection (Per-Class & Per-SNR) ---")
    
    clean_baselines = []
    class_counts = {}
    for p in clean_image_paths:
        img_bgr = cv2.imread(p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        _, r_clean = yolo_overlay(img_rgb, conf=0.25)
        boxes, clss, confs = extract_yolo_boxes_and_classes(r_clean, conf_thresh=0.25)
        clean_baselines.append({'path': p, 'boxes': boxes, 'clss': clss, 'confs': confs})
        for c in clss:
            class_counts[int(c)] = class_counts.get(int(c), 0) + 1

    top_classes = sorted(class_counts.keys(), key=lambda c: class_counts[c], reverse=True)[:6]
    top_class_names = {int(c): COCO_CLASSES.get(int(c), f"class_{c}") for c in top_classes}
    print(f"Top evaluated classes: {top_class_names}")

    yolo_snr_results = {'Gaussian Noise': [], 'S&P Noise': [], 'Motion Blur': []}
    yolo_per_class_results = {int(c): {'name': top_class_names[int(c)], 'Gaussian Noise': [], 'S&P Noise': [], 'Motion Blur': []} for c in top_classes}

    # Evaluate YOLO Gaussian Noise sweep
    for std_val in gaussian_stds:
        snr_list = []
        overall_recalls = []
        class_recalls = {int(c): [] for c in top_classes}

        for idx, p in enumerate(clean_image_paths):
            base = clean_baselines[idx]
            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            dist_rgb = apply_gaussian_noise(img_rgb, std=std_val) if std_val > 0 else img_rgb.copy()
            snr = compute_snr(img_rgb, dist_rgb)
            snr_list.append(snr)

            _, r_dist = yolo_overlay(dist_rgb, conf=0.25)
            d_boxes, d_clss, d_confs = extract_yolo_boxes_and_classes(r_dist, conf_thresh=0.25)

            base_boxes = base['boxes']
            base_clss = base['clss']

            if len(base_boxes) == 0:
                continue

            matched = 0
            used_d_indices = set()
            for b_box, b_cls in zip(base_boxes, base_clss):
                b_cls = int(b_cls)
                found = False
                for d_idx, (d_box, d_cls) in enumerate(zip(d_boxes, d_clss)):
                    if d_idx in used_d_indices:
                        continue
                    d_cls = int(d_cls)
                    if b_cls == d_cls and bbox_iou(b_box, d_box) >= 0.3:
                        found = True
                        used_d_indices.add(d_idx)
                        break
                if found:
                    matched += 1
                if b_cls in top_classes:
                    class_recalls[b_cls].append(1.0 if found else 0.0)

            overall_recalls.append(matched / len(base_boxes))

        avg_snr = float(np.mean(snr_list))
        avg_rec = float(np.mean(overall_recalls)) if overall_recalls else 0.0
        yolo_snr_results['Gaussian Noise'].append({'intensity': std_val, 'snr': avg_snr, 'recall': avg_rec})

        for c in top_classes:
            c = int(c)
            c_rec = float(np.mean(class_recalls[c])) if class_recalls[c] else 0.0
            yolo_per_class_results[c]['Gaussian Noise'].append({'intensity': std_val, 'snr': avg_snr, 'recall': c_rec})

    # Evaluate YOLO S&P Noise sweep
    for sp in sp_noise_amounts:
        snr_list = []
        overall_recalls = []
        class_recalls = {int(c): [] for c in top_classes}

        for idx, p in enumerate(clean_image_paths):
            base = clean_baselines[idx]
            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            dist_rgb = apply_salt_and_pepper_noise(img_rgb, amount=sp) if sp > 0 else img_rgb.copy()
            snr = compute_snr(img_rgb, dist_rgb)
            snr_list.append(snr)

            _, r_dist = yolo_overlay(dist_rgb, conf=0.25)
            d_boxes, d_clss, d_confs = extract_yolo_boxes_and_classes(r_dist, conf_thresh=0.25)

            base_boxes = base['boxes']
            base_clss = base['clss']
            if len(base_boxes) == 0: continue

            matched = 0
            used_d_indices = set()
            for b_box, b_cls in zip(base_boxes, base_clss):
                b_cls = int(b_cls)
                found = False
                for d_idx, (d_box, d_cls) in enumerate(zip(d_boxes, d_clss)):
                    if d_idx in used_d_indices:
                        continue
                    d_cls = int(d_cls)
                    if b_cls == d_cls and bbox_iou(b_box, d_box) >= 0.3:
                        found = True
                        used_d_indices.add(d_idx)
                        break
                if found: matched += 1
                if b_cls in top_classes:
                    class_recalls[b_cls].append(1.0 if found else 0.0)

            overall_recalls.append(matched / len(base_boxes))

        avg_snr = float(np.mean(snr_list))
        avg_rec = float(np.mean(overall_recalls)) if overall_recalls else 0.0
        yolo_snr_results['S&P Noise'].append({'intensity': sp, 'snr': avg_snr, 'recall': avg_rec})

        for c in top_classes:
            c = int(c)
            c_rec = float(np.mean(class_recalls[c])) if class_recalls[c] else 0.0
            yolo_per_class_results[c]['S&P Noise'].append({'intensity': sp, 'snr': avg_snr, 'recall': c_rec})

    # Evaluate YOLO Motion Blur sweep
    for k in blur_kernel_sizes:
        snr_list = []
        overall_recalls = []
        class_recalls = {int(c): [] for c in top_classes}

        for idx, p in enumerate(clean_image_paths):
            base = clean_baselines[idx]
            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            dist_rgb = apply_motion_blur(img_rgb, kernel_size=k) if k > 1 else img_rgb.copy()
            snr = compute_snr(img_rgb, dist_rgb)
            snr_list.append(snr)

            _, r_dist = yolo_overlay(dist_rgb, conf=0.25)
            d_boxes, d_clss, d_confs = extract_yolo_boxes_and_classes(r_dist, conf_thresh=0.25)

            base_boxes = base['boxes']
            base_clss = base['clss']
            if len(base_boxes) == 0: continue

            matched = 0
            used_d_indices = set()
            for b_box, b_cls in zip(base_boxes, base_clss):
                b_cls = int(b_cls)
                found = False
                for d_idx, (d_box, d_cls) in enumerate(zip(d_boxes, d_clss)):
                    if d_idx in used_d_indices:
                        continue
                    d_cls = int(d_cls)
                    if b_cls == d_cls and bbox_iou(b_box, d_box) >= 0.3:
                        found = True
                        used_d_indices.add(d_idx)
                        break
                if found: matched += 1
                if b_cls in top_classes:
                    class_recalls[b_cls].append(1.0 if found else 0.0)

            overall_recalls.append(matched / len(base_boxes))

        avg_snr = float(np.mean(snr_list))
        avg_rec = float(np.mean(overall_recalls)) if overall_recalls else 0.0
        yolo_snr_results['Motion Blur'].append({'intensity': k, 'snr': avg_snr, 'recall': avg_rec})

        for c in top_classes:
            c = int(c)
            c_rec = float(np.mean(class_recalls[c])) if class_recalls[c] else 0.0
            yolo_per_class_results[c]['Motion Blur'].append({'intensity': k, 'snr': avg_snr, 'recall': c_rec})


    # --- 3. BENCHMARK TASK 3: SEMANTIC SEGMENTATION (PER-CLASS & PER-SNR) ---
    print("\n--- Measuring Task 3: Semantic Segmentation (Per-Class & Per-SNR) ---")
    
    seg_baselines = []
    for p in clean_image_paths:
        img_bgr = cv2.imread(p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        clean_mask, _ = predict_seg_mask(seg_model_tuple, img_rgb)
        seg_baselines.append({'path': p, 'mask': clean_mask})

    seg_snr_results = {'Gaussian Noise': [], 'S&P Noise': [], 'Motion Blur': []}

    # Gaussian Noise sweep
    for std_val in gaussian_stds:
        snr_list, iou_list = [], []
        for idx, p in enumerate(clean_image_paths):
            base_mask = seg_baselines[idx]['mask']
            if base_mask is None: continue
            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            dist_rgb = apply_gaussian_noise(img_rgb, std=std_val) if std_val > 0 else img_rgb.copy()
            snr = compute_snr(img_rgb, dist_rgb)
            
            dist_mask, _ = predict_seg_mask(seg_model_tuple, dist_rgb)
            iou = iou_mask(base_mask, dist_mask)
            if iou is not None:
                snr_list.append(snr)
                iou_list.append(iou)

        avg_snr = float(np.mean(snr_list))
        avg_iou = float(np.mean(iou_list))
        seg_snr_results['Gaussian Noise'].append({'intensity': std_val, 'snr': avg_snr, 'iou': avg_iou})

    # Noise sweep
    for sp in sp_noise_amounts:
        snr_list, iou_list = [], []
        for idx, p in enumerate(clean_image_paths):
            base_mask = seg_baselines[idx]['mask']
            if base_mask is None: continue
            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            dist_rgb = apply_salt_and_pepper_noise(img_rgb, amount=sp) if sp > 0 else img_rgb.copy()
            snr = compute_snr(img_rgb, dist_rgb)
            
            dist_mask, _ = predict_seg_mask(seg_model_tuple, dist_rgb)
            iou = iou_mask(base_mask, dist_mask)
            if iou is not None:
                snr_list.append(snr)
                iou_list.append(iou)

        avg_snr = float(np.mean(snr_list))
        avg_iou = float(np.mean(iou_list))
        seg_snr_results['S&P Noise'].append({'intensity': sp, 'snr': avg_snr, 'iou': avg_iou})

    # Motion Blur sweep
    for k in blur_kernel_sizes:
        snr_list, iou_list = [], []
        for idx, p in enumerate(clean_image_paths):
            base_mask = seg_baselines[idx]['mask']
            if base_mask is None: continue
            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            dist_rgb = apply_motion_blur(img_rgb, kernel_size=k) if k > 1 else img_rgb.copy()
            snr = compute_snr(img_rgb, dist_rgb)
            
            dist_mask, _ = predict_seg_mask(seg_model_tuple, dist_rgb)
            iou = iou_mask(base_mask, dist_mask)
            if iou is not None:
                snr_list.append(snr)
                iou_list.append(iou)

        avg_snr = float(np.mean(snr_list))
        avg_iou = float(np.mean(iou_list))
        seg_snr_results['Motion Blur'].append({'intensity': k, 'snr': avg_snr, 'iou': avg_iou})


    # --- 4. BENCHMARK MITIGATION EVALUATION FOR PLOT 5 & 6 (YOLO & SEGMENTATION MITIGATION) ---
    print("\n--- Measuring actual performance for Plot 5 (YOLO Object Detection Mitigation) ---")
    
    mitigation_categories = ['Gaussian Noise', 'S&P Noise', 'Motion Blur']
    
    mitigation_configs = [
        {
            'name': 'Gaussian Noise',
            'distort': lambda img: apply_gaussian_noise(img, std=50.0),
            'restore': lambda img: apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75),
            'yolo_weights': ['weights/train_gaussian/best.pt', 'runs/detect/train_gaussian/weights/best.pt'],
            'seg_weights': ['weights/train_gaussian/best.pth', 'finetuned_robust_student_deeplab_gaussian/robust_deeplab_student.pth']
        },
        {
            'name': 'S&P Noise',
            'distort': lambda img: apply_salt_and_pepper_noise(img, amount=0.35),
            'restore': lambda img: apply_median_filter(img, ksize=5),
            'yolo_weights': ['weights/train_sp/best.pt', 'runs/detect/train_sp/weights/best.pt'],
            'seg_weights': ['weights/train_sp/best.pth', 'finetuned_robust_student_deeplab_sp/robust_deeplab_student.pth']
        },
        {
            'name': 'Motion Blur',
            'distort': lambda img: apply_motion_blur(img, kernel_size=15),
            'restore': lambda img: apply_restoration_filter(img),
            'yolo_weights': ['weights/train_blur/best.pt', 'runs/detect/train_blur/weights/best.pt'],
            'seg_weights': ['weights/train_blur/best.pth', 'finetuned_robust_student_deeplab_blur/robust_deeplab_student.pth']
        }
    ]

    # 4A. YOLO Object Detection Mitigation
    distorted_scores = []
    restored_scores = []
    finetuned_scores = []

    for cfg in mitigation_configs:
        cat_name = cfg['name']
        ft_weights_list = cfg.get('yolo_weights', [])
        ft_model = None
        if isinstance(ft_weights_list, str):
            ft_weights_list = [ft_weights_list]
        for w_path in ft_weights_list:
            if os.path.exists(w_path):
                try:
                    from ultralytics import YOLO
                    ft_model = YOLO(w_path)
                    print(f"Loaded fine-tuned YOLO model for '{cat_name}' from {w_path}")
                    break
                except Exception as e:
                    print(f"Could not load YOLO model from {w_path}: {e}")

        dist_recalls, rest_recalls, ft_recalls = [], [], []

        for idx, p in enumerate(clean_image_paths):
            base = clean_baselines[idx]
            base_boxes = base['boxes']
            base_clss = base['clss']
            if len(base_boxes) == 0:
                continue

            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            dist_rgb = cfg['distort'](img_rgb)
            rest_rgb = cfg['restore'](dist_rgb)

            # 1. Base YOLO on Distorted
            _, r_dist = yolo_overlay(dist_rgb, conf=0.25)
            d_boxes, d_clss, _ = extract_yolo_boxes_and_classes(r_dist, conf_thresh=0.25)
            matched_dist = 0
            used_d_indices = set()
            for b_box, b_cls in zip(base_boxes, base_clss):
                b_cls = int(b_cls)
                for d_idx, (d_box, d_cls) in enumerate(zip(d_boxes, d_clss)):
                    if d_idx in used_d_indices:
                        continue
                    if b_cls == int(d_cls) and bbox_iou(b_box, d_box) >= 0.3:
                        matched_dist += 1
                        used_d_indices.add(d_idx)
                        break
            dist_recalls.append(matched_dist / len(base_boxes))

            # 2. Base YOLO on Restored
            _, r_rest = yolo_overlay(rest_rgb, conf=0.25)
            res_boxes, res_clss, _ = extract_yolo_boxes_and_classes(r_rest, conf_thresh=0.25)
            matched_rest = 0
            used_r_indices = set()
            for b_box, b_cls in zip(base_boxes, base_clss):
                b_cls = int(b_cls)
                for r_idx, (r_box, r_cls) in enumerate(zip(res_boxes, res_clss)):
                    if r_idx in used_r_indices:
                        continue
                    if b_cls == int(r_cls) and bbox_iou(b_box, r_box) >= 0.3:
                        matched_rest += 1
                        used_r_indices.add(r_idx)
                        break
            rest_recalls.append(matched_rest / len(base_boxes))

            # 3. Fine-Tuned YOLO Model on Distorted
            if ft_model is not None:
                ft_res = ft_model.predict(dist_rgb, conf=0.25, verbose=False)[0]
                ft_boxes, ft_clss, _ = extract_yolo_boxes_and_classes(ft_res, conf_thresh=0.25)
                matched_ft = 0
                used_ft_indices = set()
                for b_box, b_cls in zip(base_boxes, base_clss):
                    b_cls = int(b_cls)
                    for f_idx, (f_box, f_cls) in enumerate(zip(ft_boxes, ft_clss)):
                        if f_idx in used_ft_indices:
                            continue
                        if b_cls == int(f_cls) and bbox_iou(b_box, f_box) >= 0.3:
                            matched_ft += 1
                            used_ft_indices.add(f_idx)
                            break
                ft_recalls.append(matched_ft / len(base_boxes))

        avg_dist = float(np.mean(dist_recalls)) if dist_recalls else 0.0
        avg_rest = float(np.mean(rest_recalls)) if rest_recalls else 0.0
        distorted_scores.append(round(avg_dist, 3))
        restored_scores.append(round(avg_rest, 3))
        if ft_model is not None:
            avg_ft = float(np.mean(ft_recalls)) if ft_recalls else 0.0
            finetuned_scores.append(round(avg_ft, 3))
            print(f"  [YOLO] {cat_name:15s} | Distorted Score: {avg_dist:.4f} | Filtered Score: {avg_rest:.4f} | Fine-Tuned Score: {avg_ft:.4f}")
        else:
            print(f"  [YOLO] {cat_name:15s} | Distorted Score: {avg_dist:.4f} | Filtered Score: {avg_rest:.4f} | (No Fine-Tuned YOLO Weights Found)")

    mitigation_results = {
        'categories': mitigation_categories,
        'distorted_scores': distorted_scores,
        'restored_scores': restored_scores,
        'finetuned_scores': finetuned_scores
    }

    # 4B. Semantic Segmentation Mitigation
    print("\n--- Measuring actual performance for Plot 6 (Semantic Segmentation Mitigation) ---")
    seg_distorted_scores = []
    seg_restored_scores = []
    seg_finetuned_scores = []

    for cfg in mitigation_configs:
        cat_name = cfg['name']
        ft_weights_list = cfg.get('seg_weights', [])
        ft_seg_model = None
        if isinstance(ft_weights_list, str):
            ft_weights_list = [ft_weights_list]
        for w_path in ft_weights_list:
            if os.path.exists(w_path):
                try:
                    ft_seg_model = load_seg_model(w_path)
                    print(f"Loaded fine-tuned DeepLabV3 student model for '{cat_name}' from {w_path}")
                    break
                except Exception as e:
                    print(f"Could not load DeepLabV3 student model from {w_path}: {e}")

        seg_dist_ious, seg_rest_ious, seg_ft_ious = [], [], []

        for idx, p in enumerate(clean_image_paths):
            base_mask = seg_baselines[idx]['mask']
            if base_mask is None:
                continue

            img_bgr = cv2.imread(p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            dist_rgb = cfg['distort'](img_rgb)
            rest_rgb = cfg['restore'](dist_rgb)

            # 1. Base DeepLabV3 on Distorted
            dist_mask, _ = predict_seg_mask(seg_model_tuple, dist_rgb)
            iou_dist = iou_mask(base_mask, dist_mask)
            if iou_dist is not None:
                seg_dist_ious.append(iou_dist)

            # 2. Base DeepLabV3 on Restored
            rest_mask, _ = predict_seg_mask(seg_model_tuple, rest_rgb)
            iou_rest = iou_mask(base_mask, rest_mask)
            if iou_rest is not None:
                seg_rest_ious.append(iou_rest)

            # 3. Fine-Tuned DeepLabV3 Student Model on Distorted
            if ft_seg_model is not None:
                ft_mask, _ = predict_seg_mask(ft_seg_model, dist_rgb)
                iou_ft = iou_mask(base_mask, ft_mask)
                if iou_ft is not None:
                    seg_ft_ious.append(iou_ft)

        avg_seg_dist = float(np.mean(seg_dist_ious)) if seg_dist_ious else 0.0
        avg_seg_rest = float(np.mean(seg_rest_ious)) if seg_rest_ious else 0.0
        seg_distorted_scores.append(round(avg_seg_dist, 3))
        seg_restored_scores.append(round(avg_seg_rest, 3))
        if ft_seg_model is not None:
            avg_seg_ft = float(np.mean(seg_ft_ious)) if seg_ft_ious else 0.0
            seg_finetuned_scores.append(round(avg_seg_ft, 3))
            print(f"  [Segmentation] {cat_name:15s} | Distorted mIoU: {avg_seg_dist:.4f} | Filtered mIoU: {avg_seg_rest:.4f} | Fine-Tuned mIoU: {avg_seg_ft:.4f}")
        else:
            print(f"  [Segmentation] {cat_name:15s} | Distorted mIoU: {avg_seg_dist:.4f} | Filtered mIoU: {avg_seg_rest:.4f} | (No Fine-Tuned Model Weights Found)")

    seg_mitigation_results = {
        'categories': mitigation_categories,
        'distorted_scores': seg_distorted_scores,
        'restored_scores': seg_restored_scores,
        'finetuned_scores': seg_finetuned_scores
    }

    # Convert keys in yolo_per_class_results to strings for JSON serializability
    json_yolo_per_class = {str(c): data for c, data in yolo_per_class_results.items()}

    measured_data = {
        'orb_results': orb_results,
        'yolo_snr_results': yolo_snr_results,
        'yolo_per_class_results': json_yolo_per_class,
        'seg_snr_results': seg_snr_results,
        'mitigation_results': mitigation_results,
        'seg_mitigation_results': seg_mitigation_results
    }
    with open("assets/measured_metrics.json", "w") as f:
        json.dump(measured_data, f, indent=2)
    print("Saved measured metrics to assets/measured_metrics.json")


    # --- 5. GENERATE HIGH-QUALITY PLOTS ---
    print("\n--- Generating High-Quality Benchmark Plots ---")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot 1: ORB Keypoint Match Ratio vs SNR
    plt.figure(figsize=(8, 5), dpi=300)
    for dist_name, data in orb_results.items():
        sorted_data = sorted(data, key=lambda x: x['snr'], reverse=True)
        snrs = [d['snr'] for d in sorted_data]
        ratios = [d['match_ratio'] for d in sorted_data]
        plt.plot(snrs, ratios, 'o-', linewidth=2.5, label=f"ORB under {dist_name}")
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label="Clean Baseline (1.0)")
    plt.gca().invert_xaxis()
    plt.title("Task 1: ORB Keypoint Match Ratio vs SNR (dB)", fontsize=13, fontweight='bold')
    plt.xlabel("Signal-to-Noise Ratio (SNR dB) \u2192 Decreasing Quality", fontsize=11)
    plt.ylabel("Good Matches / Clean Baseline Keypoints", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("assets/task1_orb_degradation_snr.png")
    plt.close()

    # Plot 2: YOLO Detection Recall per Class vs SNR (Gaussian Noise)
    plt.figure(figsize=(9, 5.5), dpi=300)
    for c in top_classes:
        c = int(c)
        c_name = top_class_names[c]
        data = yolo_per_class_results[c]['Gaussian Noise']
        sorted_data = sorted(data, key=lambda x: x['snr'], reverse=True)
        snrs = [d['snr'] for d in sorted_data]
        recalls = [d['recall'] for d in sorted_data]
        plt.plot(snrs, recalls, 's-', linewidth=2.0, label=f"Class: {c_name}")
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.7, label="Clean Baseline (1.0)")
    plt.gca().invert_xaxis()
    plt.title("Task 2: YOLO Object Detection Recall Per-Class vs SNR (Gaussian Noise Sweep)", fontsize=13, fontweight='bold')
    plt.xlabel("Signal-to-Noise Ratio (SNR dB) \u2192 Higher Noise", fontsize=11)
    plt.ylabel("Class Detection Recall", fontsize=11)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig("assets/task2_yolo_per_class_snr.png")
    plt.close()

    # Plot 3: Task 3 Segmentation IoU vs SNR
    plt.figure(figsize=(8, 5), dpi=300)
    for dist_name, data in seg_snr_results.items():
        sorted_data = sorted(data, key=lambda x: x['snr'], reverse=True)
        snrs = [d['snr'] for d in sorted_data]
        ious = [d['iou'] for d in sorted_data]
        plt.plot(snrs, ious, '^--', linewidth=2.5, label=f"DeepLabV3 under {dist_name}")
    plt.gca().invert_xaxis()
    plt.title("Task 3: Semantic Segmentation IoU vs SNR (dB)", fontsize=13, fontweight='bold')
    plt.xlabel("Signal-to-Noise Ratio (SNR dB) \u2192 Higher Distortion", fontsize=11)
    plt.ylabel("Intersection over Union (IoU vs Baseline)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("assets/task3_seg_per_class_snr.png")
    plt.close()

    # Plot 4: Overall 3-Tasks Robustness Curve Comparison
    plt.figure(figsize=(9, 5.5), dpi=300)
    sorted_orb = sorted(orb_results['Gaussian Noise'], key=lambda x: x['snr'], reverse=True)
    snrs_orb = [d['snr'] for d in sorted_orb]
    val_orb = [d['match_ratio'] for d in sorted_orb]
    
    sorted_yolo = sorted(yolo_snr_results['Gaussian Noise'], key=lambda x: x['snr'], reverse=True)
    snrs_yolo = [d['snr'] for d in sorted_yolo]
    val_yolo = [d['recall'] for d in sorted_yolo]

    sorted_seg = sorted(seg_snr_results['Gaussian Noise'], key=lambda x: x['snr'], reverse=True)
    snrs_seg = [d['snr'] for d in sorted_seg]
    val_seg = [d['iou'] for d in sorted_seg]

    plt.plot(snrs_orb, val_orb, 'o-', color='#e74c3c', linewidth=2.5, label="Task 1: ORB Feature Detection (Match Ratio)")
    plt.plot(snrs_yolo, val_yolo, 's-', color='#2ecc71', linewidth=2.5, label="Task 2: YOLO Object Detection (Recall)")
    plt.plot(snrs_seg, val_seg, '^-', color='#3498db', linewidth=2.5, label="Task 3: DeepLabV3 Segmentation (IoU)")
    plt.gca().invert_xaxis()
    plt.title("Cross-Task Robustness Comparison Under Gaussian Noise Degradation", fontsize=13, fontweight='bold')
    plt.xlabel("Signal-to-Noise Ratio (SNR dB) \u2192 Higher Gaussian Noise", fontsize=11)
    plt.ylabel("Normalized Performance Score [0.0 - 1.0]", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("assets/overall_3tasks_snr_comparison.png")
    plt.close()

    # Plot 5: Bar Chart Comparing Clean vs Distorted vs Restored (Pre-Filter) vs Fine-Tuned (YOLO Performance)
    plt.figure(figsize=(10, 5), dpi=300)
    x = np.arange(len(mitigation_categories))

    if len(finetuned_scores) == len(mitigation_categories):
        width = 0.25
        plt.bar(x - width, distorted_scores, width, label='Distorted Input', color='#e74c3c')
        plt.bar(x, restored_scores, width, label='Pre-Filtered Restored', color='#f39c12')
        plt.bar(x + width, finetuned_scores, width, label='Fine-Tuned YOLO Model', color='#27ae60')
    else:
        width = 0.35
        plt.bar(x - width / 2, distorted_scores, width, label='Distorted Input', color='#e74c3c')
        plt.bar(x + width / 2, restored_scores, width, label='Pre-Filtered Restored', color='#f39c12')

    plt.axhline(1.0, color='gray', linestyle='--', label='Clean Baseline (1.00)')
    plt.ylabel('YOLO Detection Recall Score', fontsize=11)
    plt.title('YOLO Object Detection Mitigation Evaluation', fontsize=13, fontweight='bold')
    plt.xticks(x, mitigation_categories, fontsize=11)
    plt.ylim(0, 1.15)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("assets/yolo_restoration_vs_finetuned_bar.png")
    plt.close()

    # Plot 6: Bar Chart Comparing Clean vs Distorted vs Restored (Pre-Filter) vs Fine-Tuned (Semantic Segmentation Performance)
    plt.figure(figsize=(10, 5), dpi=300)

    if len(seg_finetuned_scores) == len(mitigation_categories):
        width = 0.25
        plt.bar(x - width, seg_distorted_scores, width, label='Distorted Input (Base Model)', color='#e74c3c')
        plt.bar(x, seg_restored_scores, width, label='Pre-Filtered Restored (Base Model)', color='#f39c12')
        plt.bar(x + width, seg_finetuned_scores, width, label='Fine-Tuned Student Model', color='#27ae60')
    else:
        width = 0.35
        plt.bar(x - width / 2, seg_distorted_scores, width, label='Distorted Input (Base Model)', color='#e74c3c')
        plt.bar(x + width / 2, seg_restored_scores, width, label='Pre-Filtered Restored (Base Model)', color='#f39c12')

    plt.axhline(1.0, color='gray', linestyle='--', label='Clean Baseline (1.00 mIoU)')
    plt.ylabel('Semantic Segmentation mIoU Score', fontsize=11)
    plt.title('Semantic Segmentation Mitigation Evaluation', fontsize=13, fontweight='bold')
    plt.xticks(x, mitigation_categories, fontsize=11)
    plt.ylim(0, 1.15)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("assets/seg_restoration_vs_finetuned_bar.png")
    plt.close()

    print("Successfully generated all benchmark plots in assets/")


    # --- 6. GENERATE SIDE-BY-SIDE VISUAL COMPARISON GRIDS ---
    print("\n--- Generating Side-by-Side Visual Inspection Grids ---")
    sample_img_path = clean_image_paths[0]
    img_bgr = cv2.imread(sample_img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # A. Task 1 ORB Visual Grid
    clean_orb_viz, _, _ = run_orb(img_rgb, nfeatures=800)
    gauss_rgb = apply_gaussian_noise(img_rgb, std=35)
    gauss_orb_viz, _, _ = run_orb(gauss_rgb, nfeatures=800)
    restored_gauss = apply_bilateral_filter(gauss_rgb, d=9, sigma_color=75, sigma_space=75)
    restored_orb_viz, _, _ = run_orb(restored_gauss, nfeatures=800)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    axes[0].imshow(clean_orb_viz)
    axes[0].set_title("Clean Baseline (ORB Keypoints)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(gauss_orb_viz)
    axes[1].set_title("Distorted: Gaussian Noise (\u03c3=35)", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(restored_orb_viz)
    axes[2].set_title("Restored: Bilateral Filtered", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig("assets/vis_orb_clean_distorted_restored.png")
    plt.close()

    # B. Task 2 YOLO Visual Grid
    clean_yolo_viz, _ = yolo_overlay(img_rgb, conf=0.25)
    noisy_rgb = apply_salt_and_pepper_noise(img_rgb, amount=0.25)
    noisy_yolo_viz, _ = yolo_overlay(noisy_rgb, conf=0.25)
    restored_noise = apply_median_filter(noisy_rgb, ksize=5)
    restored_yolo_viz, _ = yolo_overlay(restored_noise, conf=0.25)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    axes[0].imshow(clean_yolo_viz)
    axes[0].set_title("Clean Baseline (YOLO Detections)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(noisy_yolo_viz)
    axes[1].set_title("Distorted: S&P Noise (p=0.25)", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(restored_yolo_viz)
    axes[2].set_title("Restored: Median Filtered", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig("assets/vis_yolo_clean_distorted_restored.png")
    plt.close()

    # C. Task 3 Segmentation Visual Grid
    _, clean_seg_viz = predict_seg_mask(seg_model_tuple, img_rgb)
    blur_rgb = apply_motion_blur(img_rgb, kernel_size=19)
    _, blur_seg_viz = predict_seg_mask(seg_model_tuple, blur_rgb)
    restored_blur = apply_bilateral_filter(blur_rgb, d=9, sigma_color=75, sigma_space=75)
    _, restored_seg_viz = predict_seg_mask(seg_model_tuple, restored_blur)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    axes[0].imshow(clean_seg_viz if clean_seg_viz is not None else img_rgb)
    axes[0].set_title("Clean Baseline (DeepLabV3 Mask)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(blur_seg_viz if blur_seg_viz is not None else blur_rgb)
    axes[1].set_title("Distorted: Motion Blur (k=19)", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(restored_seg_viz if restored_seg_viz is not None else restored_blur)
    axes[2].set_title("Restored: Bilateral Filtered", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig("assets/vis_seg_clean_distorted_restored.png")
    plt.close()

    # D. Multi-panel Per-Class Object Bounding Box Comparison Grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    sample_imgs = clean_image_paths[:4]
    for idx, p in enumerate(sample_imgs):
        r_ax = axes[idx // 2, idx % 2]
        im_bgr = cv2.imread(p)
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        im_gauss = apply_gaussian_noise(im_rgb, std=35)
        im_viz, _ = yolo_overlay(im_gauss, conf=0.20)
        r_ax.imshow(im_viz)
        r_ax.set_title(f"Sample {idx+1} under Gaussian Noise (\u03c3=35)", fontsize=10, fontweight='bold')
        r_ax.axis('off')
    plt.tight_layout()
    plt.savefig("assets/vis_per_class_bbox_grid.png")
    plt.close()

    print("Successfully generated all visual comparison image grids in assets/")

if __name__ == '__main__':
    main()
