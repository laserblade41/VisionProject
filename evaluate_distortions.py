import argparse
import time
import json
import random
import numpy as np
import cv2
from YOLO import yolo_overlay
from distortions import apply_random_distortion
from filters import apply_restoration_filter, apply_median_filter, apply_clahe

# Utilities to extract detections from ultralytics result object

def extract_detections(r, conf_thresh=0.25):
    detections = []
    try:
        boxes = r.boxes
        # attempt common attribute access
        confs = None
        clss = None
        xyxy = None
        if hasattr(boxes, 'conf'):
            confs = boxes.conf.cpu().numpy()
        if hasattr(boxes, 'cls'):
            clss = boxes.cls.cpu().numpy()
        if hasattr(boxes, 'xyxy'):
            xyxy = boxes.xyxy.cpu().numpy()
        # fallback when attributes missing
        if confs is None or clss is None:
            # try iterating boxes
            for b in boxes:
                try:
                    c = float(b.conf)
                    cl = int(b.cls)
                    coords = tuple(map(float, b.xyxy))
                    if c >= conf_thresh:
                        detections.append((cl, c, coords))
                except Exception:
                    continue
        else:
            for i in range(len(confs)):
                c = float(confs[i])
                if c < conf_thresh:
                    continue
                cl = int(clss[i])
                coords = tuple(map(float, xyxy[i])) if xyxy is not None else None
                detections.append((cl, c, coords))
    except Exception:
        pass
    return detections

# Simple comparison: baseline classes must be present in trial classes

def is_correct_detection(baseline_dets, trial_dets):
    base_classes = set([d[0] for d in baseline_dets])
    trial_classes = set([d[0] for d in trial_dets])
    return base_classes.issubset(trial_classes)

# Distortions are provided from distortions.py; filters available in filters.py
# (imported at top of file)
# apply_random_distortion(img, strength, seed=None) -> RGB uint8 image


def run_evaluation(image_path, trials=50, conf=0.25, strength=0.6, require_exact_match=False, save_json=None, use_filter=False, filter_type='restoration', seed=None):
    # Load image (BGR -> RGB)
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print(f"Running baseline (no distortion) on {image_path}")
    _, r_base = yolo_overlay(img_rgb, conf=conf)
    baseline_dets = extract_detections(r_base, conf_thresh=conf)
    print(f"Baseline detections (classes): {set([d[0] for d in baseline_dets])}")

    correct = 0
    trial_records = []
    start = time.time()
    for i in range(trials):
        # use per-trial seed when provided to make trials reproducible
        trial_seed = None if seed is None else int(seed) + i
        img_dist = apply_random_distortion(img_rgb, strength=strength, seed=trial_seed)

        # Optionally apply a restoration/filter from filters.py
        if use_filter:
            if filter_type == 'restoration':
                img_dist = apply_restoration_filter(img_dist)
            elif filter_type == 'median':
                img_dist = apply_median_filter(img_dist, ksize=3)
            elif filter_type == 'clahe':
                img_dist = apply_clahe(img_dist)

        _, r_trial = yolo_overlay(img_dist, conf=conf)
        trial_dets = extract_detections(r_trial, conf_thresh=conf)
        ok = is_correct_detection(baseline_dets, trial_dets)
        if require_exact_match:
            ok = ok and (set([d[0] for d in baseline_dets]) == set([d[0] for d in trial_dets]))
        if ok:
            correct += 1
        trial_records.append({
            'trial': i+1,
            'ok': bool(ok),
            'trial_classes': list(set([int(d[0]) for d in trial_dets])),
            'trial_count': len(trial_dets)
        })
        if (i+1) % 10 == 0:
            print(f"Completed {i+1}/{trials} trials...")

    elapsed = time.time() - start
    pct = 100.0 * correct / trials if trials > 0 else 0.0
    summary = {
        'image': image_path,
        'trials': trials,
        'correct': correct,
        'percent': pct,
        'conf_threshold': conf,
        'strength': strength,
        'time_seconds': elapsed,
        'baseline_classes': list(set([int(d[0]) for d in baseline_dets]))
    }
    print("\nEvaluation summary:")
    print(json.dumps(summary, indent=2))
    if save_json:
        out = {'summary': summary, 'trials': trial_records}
        with open(save_json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"Saved detailed results to {save_json}")
    return summary, trial_records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate YOLO robustness to image distortions')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--trials', type=int, default=50, help='Number of distorted trials')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--strength', type=float, default=0.6, help='Max distortion strength in [0,1]')
    parser.add_argument('--exact', action='store_true', help='Require exact class set match (not just subset)')
    parser.add_argument('--out', help='Save detailed JSON results to file')
    parser.add_argument('--use-filter', action='store_true', help='Apply restoration/filter from filters.py to distorted images before detection')
    parser.add_argument('--filter-type', choices=['restoration','median','clahe'], default='restoration', help='Which filter to apply when --use-filter is set')
    parser.add_argument('--seed', type=int, help='Optional base seed for reproducible distortions')
    args = parser.parse_args()

    run_evaluation(args.image, trials=args.trials, conf=args.conf, strength=args.strength, require_exact_match=args.exact, save_json=args.out, use_filter=args.use_filter, filter_type=args.filter_type, seed=args.seed)
