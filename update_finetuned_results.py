import os
import glob
import json
import argparse
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from YOLO import yolo_overlay
from distorsions import (
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_motion_blur,
)
from semantic_seg import load_seg_model, predict_seg_mask, iou_mask
from generate_actual_metrics_and_plots import extract_yolo_boxes_and_classes, bbox_iou


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update fine-tuned model evaluation scores and regenerate mitigation plots in assets."
    )
    parser.add_argument("--json-path", type=str, default="assets/measured_metrics.json", help="Path to measured_metrics.json")
    parser.add_argument("--num-images", type=int, default=15, help="Number of benchmark clean images")
    
    # Custom weight paths for YOLO models (.pt)
    parser.add_argument("--yolo-gaussian", type=str, default=None, help="Path to fine-tuned YOLO model (.pt) for Gaussian noise")
    parser.add_argument("--yolo-sp", type=str, default=None, help="Path to fine-tuned YOLO model (.pt) for S&P noise")
    parser.add_argument("--yolo-blur", type=str, default=None, help="Path to fine-tuned YOLO model (.pt) for Motion blur")

    # Custom weight paths for DeepLabV3 segmentation models (.pth)
    parser.add_argument("--seg-gaussian", type=str, default=None, help="Path to fine-tuned DeepLabV3 model (.pth) for Gaussian noise")
    parser.add_argument("--seg-sp", type=str, default=None, help="Path to fine-tuned DeepLabV3 model (.pth) for S&P noise")
    parser.add_argument("--seg-blur", type=str, default=None, help="Path to fine-tuned DeepLabV3 model (.pth) for Motion blur")

    return parser.parse_args()


def load_clean_benchmark_images(num_images=15):
    image_paths = sorted(glob.glob("test_dataset_250/images/*.jpg"))
    clean_image_paths = [p for p in image_paths if not any(x in p for x in ['_blur', '_bright', '_sp'])][:num_images]
    
    if len(clean_image_paths) == 0:
        clean_image_paths = sorted(glob.glob("datasets/val2017/*.jpg"))[:num_images]

    if len(clean_image_paths) == 0:
        clean_image_paths = sorted(glob.glob("test_dataset_250/images/*.jpg"))[:num_images]

    if len(clean_image_paths) == 0:
        clean_image_paths = sorted(glob.glob("datasets/benchmark_clean/*.jpg"))[:num_images]

    return clean_image_paths


def resolve_weight_path(user_arg, default_candidates):
    if user_arg:
        if os.path.exists(user_arg):
            return user_arg
        else:
            print(f"Warning: Specified weight file '{user_arg}' not found.")
            return None
    for cand in default_candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


def main():
    args = parse_args()
    
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"Cannot find '{args.json_path}'. Run generate_actual_metrics_and_plots.py first.")

    with open(args.json_path, "r") as f:
        measured_data = json.load(f)

    print(f"Loaded existing metrics from {args.json_path}")
    clean_image_paths = load_clean_benchmark_images(args.num_images)
    print(f"Using {len(clean_image_paths)} benchmark clean images.")

    # 1. Pre-compute clean baseline annotations
    print("Computing clean baselines for YOLO and DeepLabV3...")
    seg_model_base = load_seg_model('deeplabv3_base')

    yolo_clean_baselines = []
    seg_clean_baselines = []

    for p in clean_image_paths:
        img_bgr = cv2.imread(p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # YOLO clean baseline
        _, r_clean = yolo_overlay(img_rgb, conf=0.25)
        boxes, clss, confs = extract_yolo_boxes_and_classes(r_clean, conf_thresh=0.25)
        yolo_clean_baselines.append({'boxes': boxes, 'clss': clss})

        # DeepLabV3 clean baseline
        clean_mask, _ = predict_seg_mask(seg_model_base, img_rgb)
        seg_clean_baselines.append({'mask': clean_mask})

    categories_config = [
        {
            'name': 'Gaussian Noise',
            'distort': lambda img: apply_gaussian_noise(img, std=75.0),
            'yolo_candidates': [args.yolo_gaussian, 'weights/train_gaussian/best.pt'],
            'seg_candidates': [args.seg_gaussian, 'weights/train_gaussian/best.pth']
        },
        {
            'name': 'S&P Noise',
            'distort': lambda img: apply_salt_and_pepper_noise(img, amount=0.25),
            'yolo_candidates': [args.yolo_sp, 'weights/train_sp/best.pt'],
            'seg_candidates': [args.seg_sp, 'weights/train_sp/best.pth']
        },
        {
            'name': 'Motion Blur',
            'distort': lambda img: apply_motion_blur(img, kernel_size=15),
            'yolo_candidates': [args.yolo_blur, 'weights/train_blur/best.pt'],
            'seg_candidates': [args.seg_blur, 'weights/train_blur/best.pth']
        }
    ]

    yolo_ft_scores = []
    seg_ft_scores = []

    print("\n--- Re-evaluating Fine-Tuned Weights ---")

    for cfg in categories_config:
        cat_name = cfg['name']
        distort_fn = cfg['distort']

        # A. YOLO Fine-Tuned Evaluation
        yolo_path = resolve_weight_path(cfg['yolo_candidates'][0], [c for c in cfg['yolo_candidates'][1:] if c])
        if yolo_path:
            try:
                from ultralytics import YOLO
                yolo_ft_model = YOLO(yolo_path)
                print(f"Evaluating YOLO [{cat_name}] from '{yolo_path}'...")
                yolo_recalls = []
                for idx, p in enumerate(clean_image_paths):
                    base = yolo_clean_baselines[idx]
                    base_boxes, base_clss = base['boxes'], base['clss']
                    if len(base_boxes) == 0:
                        continue

                    img_bgr = cv2.imread(p)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    dist_rgb = distort_fn(img_rgb)

                    ft_res = yolo_ft_model.predict(dist_rgb, conf=0.25, verbose=False)[0]
                    ft_boxes, ft_clss, _ = extract_yolo_boxes_and_classes(ft_res, conf_thresh=0.25)
                    matched_ft = 0
                    used_ft = set()
                    for b_box, b_cls in zip(base_boxes, base_clss):
                        b_cls = int(b_cls)
                        for f_idx, (f_box, f_cls) in enumerate(zip(ft_boxes, ft_clss)):
                            if f_idx in used_ft:
                                continue
                            if b_cls == int(f_cls) and bbox_iou(b_box, f_box) >= 0.3:
                                matched_ft += 1
                                used_ft.add(f_idx)
                                break
                    yolo_recalls.append(matched_ft / len(base_boxes))

                avg_yolo = float(np.mean(yolo_recalls)) if yolo_recalls else 0.0
                yolo_ft_scores.append(round(avg_yolo, 3))
            except Exception as e:
                print(f"Error evaluating YOLO model ({yolo_path}): {e}")
                yolo_ft_scores.append(0.0)
        else:
            print(f"No valid YOLO fine-tuned weights for [{cat_name}]")
            yolo_ft_scores.append(0.0)

        # B. Semantic Segmentation Fine-Tuned Evaluation
        seg_path = resolve_weight_path(cfg['seg_candidates'][0], [c for c in cfg['seg_candidates'][1:] if c])
        if seg_path:
            try:
                seg_ft_model = load_seg_model(seg_path)
                print(f"Evaluating DeepLabV3 [{cat_name}] from '{seg_path}'...")
                seg_ious = []
                for idx, p in enumerate(clean_image_paths):
                    base_mask = seg_clean_baselines[idx]['mask']
                    if base_mask is None:
                        continue

                    img_bgr = cv2.imread(p)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    dist_rgb = distort_fn(img_rgb)

                    ft_mask, _ = predict_seg_mask(seg_ft_model, dist_rgb)
                    iou_val = iou_mask(base_mask, ft_mask)
                    if iou_val is not None:
                        seg_ious.append(iou_val)

                avg_seg = float(np.mean(seg_ious)) if seg_ious else 0.0
                seg_ft_scores.append(round(avg_seg, 3))
            except Exception as e:
                print(f"Error evaluating DeepLabV3 model ({seg_path}): {e}")
                seg_ft_scores.append(0.0)
        else:
            print(f"No valid DeepLabV3 fine-tuned weights for [{cat_name}]")
            seg_ft_scores.append(0.0)

    # 2. Update JSON
    measured_data['mitigation_results']['finetuned_scores'] = yolo_ft_scores
    measured_data['seg_mitigation_results']['finetuned_scores'] = seg_ft_scores

    with open(args.json_path, "w") as f:
        json.dump(measured_data, f, indent=2)

    print(f"\nSuccessfully updated fine-tuned scores in {args.json_path}!")
    print(f"  YOLO Fine-Tuned Scores: {yolo_ft_scores}")
    print(f"  Segmentation Fine-Tuned Scores: {seg_ft_scores}")

    # 3. Regenerate Plots 5 & 6
    mitigation_categories = measured_data['mitigation_results']['categories']
    yolo_dist = measured_data['mitigation_results']['distorted_scores']
    yolo_rest = measured_data['mitigation_results']['restored_scores']

    seg_dist = measured_data['seg_mitigation_results']['distorted_scores']
    seg_rest = measured_data['seg_mitigation_results']['restored_scores']

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    x = np.arange(len(mitigation_categories))

    # Plot 5: YOLO Bar Chart
    plt.figure(figsize=(10, 5), dpi=300)
    width = 0.25
    plt.bar(x - width, yolo_dist, width, label='Distorted Input', color='#e74c3c')
    plt.bar(x, yolo_rest, width, label='Pre-Filtered Restored', color='#f39c12')
    plt.bar(x + width, yolo_ft_scores, width, label='Fine-Tuned YOLO Model', color='#27ae60')

    plt.axhline(1.0, color='gray', linestyle='--', label='Clean Baseline (1.00)')
    plt.ylabel('YOLO Detection Recall Score', fontsize=11)
    plt.title('YOLO Object Detection Mitigation Evaluation', fontsize=13, fontweight='bold')
    plt.xticks(x, mitigation_categories, fontsize=11)
    plt.ylim(0, 1.15)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("assets/yolo_restoration_vs_finetuned_bar.png")
    plt.close()
    print("Updated asset plot: assets/yolo_restoration_vs_finetuned_bar.png")

    # Plot 6: Semantic Segmentation Bar Chart
    plt.figure(figsize=(10, 5), dpi=300)
    plt.bar(x - width, seg_dist, width, label='Distorted Input (Base Model)', color='#e74c3c')
    plt.bar(x, seg_rest, width, label='Pre-Filtered Restored (Base Model)', color='#f39c12')
    plt.bar(x + width, seg_ft_scores, width, label='Fine-Tuned Student Model', color='#27ae60')

    plt.axhline(1.0, color='gray', linestyle='--', label='Clean Baseline (1.00 mIoU)')
    plt.ylabel('Semantic Segmentation mIoU Score', fontsize=11)
    plt.title('Semantic Segmentation Mitigation Evaluation', fontsize=13, fontweight='bold')
    plt.xticks(x, mitigation_categories, fontsize=11)
    plt.ylim(0, 1.15)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("assets/seg_restoration_vs_finetuned_bar.png")
    plt.close()
    print("Updated asset plot: assets/seg_restoration_vs_finetuned_bar.png")


if __name__ == "__main__":
    main()
