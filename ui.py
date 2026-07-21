import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from pathlib import Path
from ORB import run_orb
from YOLO import yolo_overlay, model as yolo_model, device as yolo_device

# Make sure these functions exist in your distorsions.py and filters.py files!
from distorsions import (
    apply_salt_and_pepper_noise,
    apply_motion_blur,
    apply_gaussian_noise
)

from filters import (
    apply_median_filter,
    apply_restoration_filter,
    apply_bilateral_filter
)


class ImageProcessingUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Technique Comparison")
        self.root.geometry("1600x900")

        self.original_image = None
        self.distorted_image = None
        self.processed_image = None
        self.current_results = None

        # Dataset variables
        self.dataset_folder = None
        self.dataset_images = []
        self.current_image_index = 0
        self.current_image_path = None

        # Segmentation model cache
        self.seg_model_tuple = None
        self.seg_model_name = None

        # Map display names to PyTorch model keywords or paths
        self.seg_model_map = {
            'DeepLabV3 Pretrained (Clean Base)': 'deeplabv3_base',
            'DeepLabV3 FineTuned (Salt & Pepper)': 'finetuned_robust_student_deeplab_sp/robust_deeplab_student.pth',
            'DeepLabV3 FineTuned (Motion Blur)': 'finetuned_robust_student_deeplab_blur/robust_deeplab_student.pth',
            'DeepLabV3 FineTuned (Gaussian)': 'finetuned_robust_student_deeplab_gaussian/robust_deeplab_student.pth'
        }

        # YOLO comparison cache
        self.last_yolo_results = None
        self.last_yolo_input = None

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: Controls
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Image selection
        ttk.Label(left_frame, text="Image Selection", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(0, 5)
        )
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5)
        )
        ttk.Button(button_frame, text="Load Dataset", command=self.load_dataset).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        # Dataset navigation
        self.dataset_frame = ttk.Frame(left_frame)
        self.dataset_frame.pack(fill=tk.X, pady=(0, 15))

        self.dataset_label = ttk.Label(self.dataset_frame, text="No dataset loaded", font=("Arial", 9))
        self.dataset_label.pack(anchor=tk.W)

        nav_frame = ttk.Frame(self.dataset_frame)
        nav_frame.pack(fill=tk.X, pady=(5, 0))
        # ttk.Button(nav_frame, text="< Prev", command=self.prev_image, width=10).pack(side=tk.LEFT, padx=(0, 5))
        # ttk.Button(nav_frame, text="Next >", command=self.next_image, width=10).pack(side=tk.LEFT)

        # Technique selection
        ttk.Label(left_frame, text="Processing Technique", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(10, 5)
        )
        self.technique_var = tk.StringVar(value="ORB")
        techniques = ["ORB", "YOLO", "Segmentation"]
        for tech in techniques:
            ttk.Radiobutton(
                left_frame, text=tech, variable=self.technique_var, value=tech
            ).pack(anchor=tk.W)

        # Distortion selection
        ttk.Label(left_frame, text="Distortions", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(10, 5)
        )
        self.distortion_vars = {}
        self.distortion_params = {}

        # Add Gaussian Noise to the list with default 25, min 1, max 100, step 1
        distortions = [
            ("Salt & Pepper", "salt_pepper", 0.05, 0.0, 0.75),
            ("Motion Blur", "motion_blur", 9, 3, 31, 2),
            ("Gaussian Noise", "gaussian_noise", 25, 1, 100, 1)
        ]

        for name, key, default, min_val, max_val, *step in distortions:
            frame = ttk.Frame(left_frame)
            frame.pack(anchor=tk.W, pady=2)
            self.distortion_vars[key] = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                frame, text=name, variable=self.distortion_vars[key]
            ).pack(side=tk.LEFT)

            step_val = step[0] if step else 0.01
            self.distortion_params[key] = {
                "var": tk.DoubleVar(value=default),
                "min": min_val,
                "max": max_val,
                "step": step_val,
            }
            ttk.Scale(
                frame,
                from_=min_val,
                to=max_val,
                variable=self.distortion_params[key]["var"],
                orient=tk.HORIZONTAL,
                length=150,
            ).pack(side=tk.LEFT, padx=(5, 0))

        # Filter selection
        ttk.Label(left_frame, text="Restoration Filters", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(10, 5)
        )
        self.filter_vars = {}

        # Add Bilateral Filter to the options
        filters = ["Median Filter", "Restoration Filter", "Bilateral Filter"]
        for filt in filters:
            self.filter_vars[filt] = tk.BooleanVar(value=False)
            ttk.Checkbutton(left_frame, text=filt, variable=self.filter_vars[filt]).pack(
                anchor=tk.W
            )

        # ORB parameters
        ttk.Label(left_frame, text="ORB Parameters", font=("Arial", 11, "bold")).pack(
            anchor=tk.W, pady=(10, 5)
        )
        ttk.Label(left_frame, text="Number of Features:").pack(anchor=tk.W)
        self.orb_nfeatures = tk.IntVar(value=800)
        ttk.Scale(
            left_frame,
            from_=100,
            to=2000,
            variable=self.orb_nfeatures,
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, pady=(0, 10))
        ttk.Label(left_frame, textvariable=self.orb_nfeatures).pack(anchor=tk.W, pady=(0, 10))

        # YOLO parameters
        ttk.Label(left_frame, text="YOLO Parameters", font=("Arial", 11, "bold")).pack(
            anchor=tk.W, pady=(10, 5)
        )
        ttk.Label(left_frame, text="Select YOLO Model:").pack(anchor=tk.W)

        self.yolo_model_type = tk.StringVar(value="Base")
        ttk.Radiobutton(left_frame, text="Base Pretrained (Clean)", variable=self.yolo_model_type, value="Base").pack(
            anchor=tk.W)
        ttk.Radiobutton(left_frame, text="Fine-Tuned (Salt & Pepper)", variable=self.yolo_model_type,
                        value="FineTuned_SP").pack(
            anchor=tk.W)
        ttk.Radiobutton(left_frame, text="Fine-Tuned (Motion Blur)", variable=self.yolo_model_type,
                        value="FineTuned_Blur").pack(
            anchor=tk.W)
        ttk.Radiobutton(left_frame, text="Fine-Tuned (Gaussian)", variable=self.yolo_model_type,
                        value="FineTuned_Gaussian").pack(
            anchor=tk.W)

        ttk.Label(left_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        self.yolo_conf = tk.DoubleVar(value=0.25)
        ttk.Scale(
            left_frame,
            from_=0.0,
            to=1.0,
            variable=self.yolo_conf,
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, pady=(0, 10))
        ttk.Label(left_frame, textvariable=self.yolo_conf).pack(anchor=tk.W, pady=(0, 10))
        # self.yolo_compare_btn = ttk.Button(
        #     left_frame, text="Enhance & Compare YOLO Detections", command=self.open_yolo_comparison, state=tk.DISABLED
        # )
        # self.yolo_compare_btn.pack(fill=tk.X, pady=(0, 10))

        # Segmentation parameters
        ttk.Label(left_frame, text="Segmentation Parameters", font=("Arial", 11, "bold")).pack(
            anchor=tk.W, pady=(10, 5)
        )

        seg_models_display = list(self.seg_model_map.keys())

        self.seg_model_var = tk.StringVar(value=seg_models_display[0])
        seg_combobox = ttk.Combobox(left_frame, values=seg_models_display, textvariable=self.seg_model_var,
                                    state="readonly")
        seg_combobox.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(left_frame, text="Preload DeepLabV3 Model", command=self.preload_seg_model).pack(fill=tk.X,
                                                                                                    pady=(0, 5))

        self.seg_status_label = ttk.Label(left_frame, text="Segmentation model: not loaded", font=("Arial", 9))
        self.seg_status_label.pack(anchor=tk.W, pady=(0, 10))

        # Process button
        ttk.Button(left_frame, text="Process", command=self.process_image).pack(
            fill=tk.X, pady=(15, 5)
        )

        # Save Distorted Image button
        self.btn_save_distorted = ttk.Button(
            left_frame, text="Save Distorted Image", command=self.save_distorted_image, state=tk.DISABLED
        )
        self.btn_save_distorted.pack(fill=tk.X, pady=(0, 15))

        # Reset button
        ttk.Button(left_frame, text="Reset All", command=self.reset_all).pack(
            fill=tk.X, pady=(0, 20)
        )

        # Right panel: Images and results
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        images_container = tk.Frame(canvas_frame)
        images_container.pack(fill=tk.BOTH, expand=True, pady=10)

        # 1. Original
        frame_original = tk.Frame(images_container)
        frame_original.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(frame_original, text="Original", font=("Arial", 11, "bold")).pack(side=tk.TOP, pady=5)
        self.canvas_original = tk.Canvas(frame_original, bg="gray", height=350)
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        # 2. Distorted
        frame_distorted = tk.Frame(images_container)
        frame_distorted.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(frame_distorted, text="Distorted (if applied)", font=("Arial", 11, "bold")).pack(side=tk.TOP, pady=5)
        self.canvas_distorted = tk.Canvas(frame_distorted, bg="gray", height=350)
        self.canvas_distorted.pack(fill=tk.BOTH, expand=True)

        # 3. Processed
        frame_processed = tk.Frame(images_container)
        frame_processed.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        ttk.Label(frame_processed, text="Processed", font=("Arial", 11, "bold")).pack(side=tk.TOP, pady=5)
        self.canvas_processed = tk.Canvas(frame_processed, bg="gray", height=350)
        self.canvas_processed.pack(fill=tk.BOTH, expand=True)

        # Results text
        ttk.Label(right_frame, text="Results", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(10, 0))
        self.results_text = tk.Text(right_frame, height=10, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def save_distorted_image(self):
        if not hasattr(self, 'distorted_image') or self.distorted_image is None:
            messagebox.showwarning("Warning", "No distorted image available to save!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")],
            title="Save Distorted Image"
        )

        if file_path:
            try:
                img_bgr = cv2.cvtColor(self.distorted_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, img_bgr)
                messagebox.showinfo("Success", f"Distorted image saved successfully to:\n{file_path}")
            except Exception as e:
                err_msg = str(e)
                messagebox.showerror("Error", f"Failed to save image:\n{err_msg}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            self._load_image_from_path(file_path)
            self.dataset_folder = None
            self.dataset_images = []
            self.dataset_label.config(text="No dataset loaded")

    def load_dataset(self):
        folder_path = filedialog.askdirectory(title="Select folder containing images")
        if folder_path:
            self.dataset_folder = folder_path
            self.dataset_images = []

            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
            for ext in extensions:
                self.dataset_images.extend(Path(folder_path).glob(f"*{ext}"))

            self.dataset_images.sort()

            if self.dataset_images:
                self.current_image_index = 0
                self.dataset_label.config(
                    text=f"Dataset: {len(self.dataset_images)} images | Image 1/{len(self.dataset_images)}"
                )
                self._load_image_from_path(str(self.dataset_images[0]))
            else:
                messagebox.showwarning("Warning", "No image files found in the selected folder")
                self.dataset_folder = None

    def prev_image(self):
        if not self.dataset_images:
            messagebox.showwarning("Warning", "No dataset loaded")
            return

        self.current_image_index = (self.current_image_index - 1) % len(self.dataset_images)
        self._load_image_from_path(str(self.dataset_images[self.current_image_index]))
        self.dataset_label.config(
            text=f"Dataset: {len(self.dataset_images)} images | Image {self.current_image_index + 1}/{len(self.dataset_images)}"
        )

    def next_image(self):
        if not self.dataset_images:
            messagebox.showwarning("Warning", "No dataset loaded")
            return

        self.current_image_index = (self.current_image_index + 1) % len(self.dataset_images)
        self._load_image_from_path(str(self.dataset_images[self.current_image_index]))
        self.dataset_label.config(
            text=f"Dataset: {len(self.dataset_images)} images | Image {self.current_image_index + 1}/{len(self.dataset_images)}"
        )

    def _load_image_from_path(self, file_path):
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            messagebox.showerror("Error", f"Failed to load image: {file_path}")
            return

        self.original_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.current_image_path = file_path
        self.distorted_image = None
        self.processed_image = None
        self.display_image(self.original_image, self.canvas_original)
        self.canvas_distorted.delete("all")
        self.canvas_processed.delete("all")
        self.results_text.delete("1.0", tk.END)

        if hasattr(self, 'btn_save_distorted'):
            self.btn_save_distorted.config(state=tk.DISABLED)

    def display_image(self, img_rgb, canvas):
        h, w = img_rgb.shape[:2]
        max_width, max_height = 400, 350
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        photo = ImageTk.PhotoImage(img_pil)

        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        canvas.image = photo

    def apply_distortions(self, img):
        result = img.copy()
        applied = []

        if self.distortion_vars["salt_pepper"].get():
            amount = self.distortion_params["salt_pepper"]["var"].get()
            result = apply_salt_and_pepper_noise(result, amount)
            applied.append(f"Salt & Pepper: {amount:.3f}")

        if self.distortion_vars["motion_blur"].get():
            kernel_size = int(self.distortion_params["motion_blur"]["var"].get())
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            result = apply_motion_blur(result, kernel_size)
            applied.append(f"Motion Blur: kernel={kernel_size}")

        # Applied new Gaussian Noise logic
        if self.distortion_vars["gaussian_noise"].get():
            std = self.distortion_params["gaussian_noise"]["var"].get()
            result = apply_gaussian_noise(result, std=std)
            applied.append(f"Gaussian Noise: std={std:.1f}")

        return result, applied

    def apply_filters(self, img):
        result = img.copy()
        applied = []

        if self.filter_vars["Median Filter"].get():
            result = apply_median_filter(result, ksize=5)
            applied.append("Median Filter")

        if self.filter_vars["Restoration Filter"].get():
            result = apply_restoration_filter(result)
            applied.append("Restoration Filter")

        # Applied new Bilateral Filter logic
        if self.filter_vars["Bilateral Filter"].get():
            result = apply_bilateral_filter(result)
            applied.append("Bilateral Filter")

        return result, applied

    def process_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Processing... please wait")
        self.root.update()

        thread = threading.Thread(target=self._process_worker)
        thread.start()

    def _process_worker(self):
        try:
            img = self.original_image.copy()
            distortion_info, distortion_names = self.apply_distortions(img)
            self.distorted_image = distortion_info

            if distortion_names:
                self.display_image(distortion_info, self.canvas_distorted)

            filtered_img, filter_names = self.apply_filters(distortion_info)

            technique = self.technique_var.get()

            if technique == "ORB":
                processed_img, keypoints, descriptors = run_orb(
                    filtered_img, nfeatures=self.orb_nfeatures.get()
                )
                results_text = f"ORB Detection Results:\n"
                results_text += f"  Keypoints detected: {len(keypoints)}\n"
                results_text += f"  Descriptors shape: {descriptors.shape if descriptors is not None else 'None'}\n"

            elif technique == "YOLO":
                selected_model = self.yolo_model_type.get()

                if selected_model == "Base":
                    processed_img, yolo_results = yolo_overlay(
                        filtered_img, conf=self.yolo_conf.get()
                    )
                else:
                    import os
                    from ultralytics import YOLO

                    model_paths = {
                        "FineTuned_SP": "weights/train_sp/best.pt",
                        "FineTuned_Blur": "weights/train_blur/best.pt",
                        "FineTuned_Bright": "weights/train_bright/best.pt",
                        "FineTuned_Gaussian": "weights/train_gaussian/best.pt"
                    }

                    weights_path = model_paths.get(selected_model)

                    if not os.path.exists(weights_path):
                        raise Exception(f"Fine-tuned weights not found at: {weights_path}")

                    ft_model = YOLO(weights_path)
                    yolo_results = ft_model.predict(filtered_img, conf=self.yolo_conf.get(), verbose=False)[0]
                    processed_img = yolo_results.plot()

                self.last_yolo_results = yolo_results
                self.last_yolo_input = filtered_img.copy()

                results_text = f"YOLO Detection Results ({selected_model} Model):\n"
                results_text += f"  Objects detected: {len(yolo_results.boxes)}\n"

                if len(yolo_results.boxes) > 0:
                    for i, box in enumerate(yolo_results.boxes):
                        cls_name = yolo_results.names[int(box.cls)]
                        conf = float(box.conf)
                        results_text += f"  {i + 1}. {cls_name}: {conf:.2%}\n"

            elif technique == "Segmentation":
                try:
                    from semantic_seg import load_seg_model, predict_seg_mask
                except Exception as e:
                    processed_img = filtered_img
                    results_text = f"Segmentation import error: {e}\n"
                else:
                    display_name = getattr(self, 'seg_model_var',
                                           tk.StringVar(value=list(self.seg_model_map.keys())[0])).get()
                    actual_model_path = self.seg_model_map.get(display_name, display_name)

                    try:
                        # Fallback to original image if no distortion was applied
                        dist_img_to_use = self.distorted_image if self.distorted_image is not None else self.original_image

                        # 1. Generate Ground Truth using Base Model on CLEAN original image
                        base_model = load_seg_model("deeplabv3_base")
                        clean_mask, _ = predict_seg_mask(base_model, self.original_image)

                        # 2. Test 1: Base Model -> Distorted Image
                        distorted_mask_base, _ = predict_seg_mask(base_model, dist_img_to_use)

                        # 3. Test 2: Base Model -> Filtered/Restored Image
                        filtered_mask_base, _ = predict_seg_mask(base_model, filtered_img)

                        # 4. Load the requested Expert model
                        if getattr(self, 'seg_model_tuple', None) is None or getattr(self, 'seg_model_name',
                                                                                     None) != actual_model_path:
                            self.seg_model_tuple = load_seg_model(actual_model_path)
                            self.seg_model_name = actual_model_path
                            self.root.after(0,
                                            lambda: self.seg_status_label.config(text=f"Model loaded: {display_name}"))

                        # 5. Test 3: Expert Model -> Distorted Image
                        distorted_mask_expert, overlay_expert = predict_seg_mask(self.seg_model_tuple, dist_img_to_use)

                        # Display the Expert overlay in the UI
                        if overlay_expert is not None:
                            processed_img = overlay_expert
                        elif distorted_mask_expert is not None:
                            processed_img = cv2.cvtColor(distorted_mask_expert, cv2.COLOR_GRAY2RGB)
                        else:
                            processed_img = dist_img_to_use

                        # --- Helper function for IoU ---
                        def calc_iou(mask_a, mask_b):
                            intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
                            union = np.logical_or(mask_a > 0, mask_b > 0).sum()
                            return (intersection / union) * 100 if union > 0 else 0.0

                        # Calculate metrics
                        iou_base_dist = calc_iou(clean_mask, distorted_mask_base)
                        iou_base_filt = calc_iou(clean_mask, filtered_mask_base)
                        iou_expert_dist = calc_iou(clean_mask, distorted_mask_expert)

                        results_text = "Semantic Segmentation IoU Comparison:\n"
                        results_text += "-" * 45 + "\n"
                        results_text += f"1. Base Model -> Distorted Image:\n   IoU = {iou_base_dist:.2f}%\n\n"
                        results_text += f"2. Base Model -> Filtered Image (Classical Restoration):\n   IoU = {iou_base_filt:.2f}%\n\n"
                        results_text += f"3. {display_name} -> Distorted Image:\n   IoU = {iou_expert_dist:.2f}%\n"
                        results_text += "-" * 45 + "\n"

                    except Exception as e:
                        processed_img = filtered_img
                        results_text = f"Segmentation error: {e}\n"

            self.processed_image = processed_img

            full_results = results_text
            if distortion_names:
                full_results += f"\nDistortions applied:\n  " + ", ".join(distortion_names) + "\n"
            if filter_names:
                full_results += f"\nFilters applied:\n  " + ", ".join(filter_names) + "\n"

            self.root.after(0, self._update_ui, processed_img, full_results)

        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda msg=err_msg: messagebox.showerror("Error", msg))

    def reset_all(self):
        self.distorted_image = None
        self.processed_image = None
        self.last_yolo_results = None
        self.last_yolo_input = None

        self.canvas_distorted.delete("all")
        self.canvas_processed.delete("all")
        self.results_text.delete("1.0", tk.END)

        for _, var in self.distortion_vars.items():
            var.set(False)

        for _, var in self.filter_vars.items():
            var.set(False)

        if hasattr(self, 'yolo_compare_btn'):
            self.yolo_compare_btn.config(state=tk.DISABLED)
        if hasattr(self, 'btn_save_distorted'):
            self.btn_save_distorted.config(state=tk.DISABLED)

        if getattr(self, 'seg_status_label', None):
            self.seg_status_label.config(text=(
                f"Model loaded: {self.seg_model_var.get()}" if self.seg_model_name else "Segmentation model: not loaded"))

        messagebox.showinfo("Reset", "All processing results and settings have been reset")

    def preload_seg_model(self):
        display_name = getattr(self, 'seg_model_var', tk.StringVar(value=list(self.seg_model_map.keys())[0])).get()
        actual_model_path = self.seg_model_map.get(display_name, display_name)

        thread = threading.Thread(target=self._preload_worker, args=(actual_model_path, display_name))
        thread.start()

    def _preload_worker(self, actual_model_path, display_name):
        try:
            from semantic_seg import load_seg_model
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda msg=err_msg: messagebox.showerror("SegModel Import Error", msg))
            return
        try:
            self.root.after(0, lambda: self.results_text.delete("1.0", tk.END))
            self.root.after(0,
                            lambda: self.results_text.insert(tk.END, f"Loading segmentation model: {display_name}..."))

            model_tuple = load_seg_model(actual_model_path)
            self.seg_model_tuple = model_tuple
            self.seg_model_name = actual_model_path

            self.root.after(0, lambda: self.seg_status_label.config(text=f"Model loaded: {display_name}"))
            self.root.after(0, lambda: self.results_text.delete("1.0", tk.END))
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda msg=err_msg: messagebox.showerror("SegModel Load Error", msg))

    def _update_ui(self, processed_img, results_text):
        self.display_image(processed_img, self.canvas_processed)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, results_text)

        if self.technique_var.get() == "YOLO" and getattr(self, "last_yolo_results", None) is not None:
            self.yolo_compare_btn.config(state=tk.NORMAL)
        else:
            self.yolo_compare_btn.config(state=tk.DISABLED)

        if getattr(self, "distorted_image", None) is not None:
            self.btn_save_distorted.config(state=tk.NORMAL)

    def open_yolo_comparison(self):
        if getattr(self, "last_yolo_results", None) is None or len(self.last_yolo_results.boxes) == 0:
            messagebox.showwarning("Warning",
                                   "No YOLO detections are available. Please select YOLO and click Process first.")
            return
        YOLOComparisonDialog(self, self.original_image, self.distorted_image, self.last_yolo_results,
                             self.last_yolo_input)


class YOLOComparisonDialog:
    def __init__(self, parent, original_img, distorted_img, yolo_results, yolo_input):
        self.parent = parent
        self.original_img = original_img
        self.distorted_img = distorted_img if distorted_img is not None else original_img
        self.yolo_results = yolo_results
        self.yolo_input = yolo_input

        self.win = tk.Toplevel(parent.root)
        self.win.title("YOLO Object Detection: Image Enhancement & Comparison Tool")
        self.win.geometry("1200x850")
        self.win.transient(parent.root)
        self.win.grab_set()

        self.detected_boxes = []
        if yolo_results and len(yolo_results.boxes) > 0:
            for i, box in enumerate(yolo_results.boxes):
                cls_idx = int(box.cls[0].cpu().numpy())
                cls_name = yolo_results.names[cls_idx]
                conf = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy()
                self.detected_boxes.append({
                    'index': i,
                    'class_idx': cls_idx,
                    'class_name': cls_name,
                    'conf': conf,
                    'xyxy': xyxy
                })

        self.setup_ui()

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.win)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Crop-Level Enhancement & Comparison")

        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Full-Image Enhancement & Comparison")

        self.setup_tab1()
        self.setup_tab2()

    def setup_tab1(self):
        ctrl_frame = ttk.Frame(self.tab1)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(ctrl_frame, text="Crop Evaluation Controls", font=("Arial", 11, "bold")).pack(anchor=tk.W,
                                                                                                pady=(0, 10))

        ttk.Label(ctrl_frame, text="Select Detected Object:").pack(anchor=tk.W, pady=(5, 2))
        self.object_combo = ttk.Combobox(ctrl_frame, state="readonly", width=30)
        self.object_combo.pack(fill=tk.X, pady=(0, 10))

        obj_values = []
        for box in self.detected_boxes:
            obj_values.append(f"Object {box['index'] + 1}: {box['class_name']} ({box['conf']:.1%})")
        self.object_combo['values'] = obj_values
        if obj_values:
            self.object_combo.current(0)

        ttk.Label(ctrl_frame, text="Crop Enhancement Method:").pack(anchor=tk.W, pady=(5, 2))
        self.enhance_combo = ttk.Combobox(ctrl_frame, state="readonly", width=30)
        self.enhance_combo.pack(fill=tk.X, pady=(0, 10))
        self.enhance_combo['values'] = [
            "None (Original Crop)",
            "Median Filter (Noise Removal)",
            "Restoration (Motion Blur)",
            "Sharpening (Sharpen)",
            "Histogram Equalization",
            "Gamma Correction (Gamma=1.5)",
            "Gamma Correction (Gamma=0.6)",
            "Bilateral Denoise",
            "Combined (CLAHE + Sharpen)",
            "Median + CLAHE + Sharpen"
        ]
        self.enhance_combo.current(0)

        self.object_combo.bind("<<ComboboxSelected>>", lambda e: self.update_crop_evaluation())
        self.enhance_combo.bind("<<ComboboxSelected>>", lambda e: self.update_crop_evaluation())

        self.status_label = ttk.Label(ctrl_frame, text="Ready", font=("Arial", 9, "italic"))
        self.status_label.pack(anchor=tk.W, pady=(20, 2))

        display_frame = ttk.Frame(self.tab1)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        images_frame = ttk.Frame(display_frame)
        images_frame.pack(fill=tk.BOTH, expand=True)

        c1_frame = ttk.LabelFrame(images_frame, text="Original Crop (Clean)")
        c1_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.lbl_orig_crop = ttk.Label(c1_frame, borderwidth=1, relief="solid")
        self.lbl_orig_crop.pack(padx=10, pady=10)

        c2_frame = ttk.LabelFrame(images_frame, text="Distorted/Input Crop")
        c2_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.lbl_dist_crop = ttk.Label(c2_frame, borderwidth=1, relief="solid")
        self.lbl_dist_crop.pack(padx=10, pady=10)

        c3_frame = ttk.LabelFrame(images_frame, text="Enhanced Crop")
        c3_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.lbl_enh_crop = ttk.Label(c3_frame, borderwidth=1, relief="solid")
        self.lbl_enh_crop.pack(padx=10, pady=10)

        images_frame.grid_columnconfigure(0, weight=1)
        images_frame.grid_columnconfigure(1, weight=1)
        images_frame.grid_columnconfigure(2, weight=1)

        table_frame = ttk.LabelFrame(display_frame, text="Quantitative & Class Detection Comparison")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.tree = ttk.Treeview(table_frame, columns=("Metric", "Original (Clean)", "Distorted/Input", "Enhanced"),
                                 show="headings", height=6)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tree.heading("Metric", text="Metric / Attribute")
        self.tree.heading("Original (Clean)", text="Original (Clean)")
        self.tree.heading("Distorted/Input", text="Distorted / Input")
        self.tree.heading("Enhanced", text="Enhanced")

        self.tree.column("Metric", width=200, anchor=tk.W)
        self.tree.column("Original (Clean)", width=150, anchor=tk.CENTER)
        self.tree.column("Distorted/Input", width=150, anchor=tk.CENTER)
        self.tree.column("Enhanced", width=150, anchor=tk.CENTER)

        if self.detected_boxes:
            self.update_crop_evaluation()

    def setup_tab2(self):
        ctrl_frame = ttk.Frame(self.tab2)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(ctrl_frame, text="Distorted Image Head-to-Head Evaluation", font=("Arial", 11, "bold")).pack(
            anchor=tk.W, pady=(0, 10))

        ttk.Label(ctrl_frame, text="Select Enhancement Filter:").pack(anchor=tk.W, pady=(5, 2))
        self.tab2_enhance_combo = ttk.Combobox(ctrl_frame, state="readonly", width=30)
        self.tab2_enhance_combo.pack(fill=tk.X, pady=(0, 10))

        self.tab2_enhance_combo['values'] = [
            "None (Original/Distorted)",
            "Median Filter (Noise Removal)",
            "Restoration (Motion Blur)",
            "Sharpening (Sharpen)",
            "Histogram Equalization",
            "Gamma Correction (Gamma=1.5)",
            "Gamma Correction (Gamma=0.6)",
            "Bilateral Denoise",
            "Combined (CLAHE + Sharpen)",
            "Median + CLAHE + Sharpen"
        ]
        self.tab2_enhance_combo.current(1)

        self.tab2_run_btn = ttk.Button(ctrl_frame, text="Evaluate Strategies", command=self.run_full_image_yolo)
        self.tab2_run_btn.pack(fill=tk.X, pady=10)

        self.tab2_status_label = ttk.Label(ctrl_frame, text="Ready", font=("Arial", 9, "italic"))
        self.tab2_status_label.pack(anchor=tk.W, pady=(5, 10))

        ttk.Label(ctrl_frame, text="Evaluation Summary:").pack(anchor=tk.W, pady=(10, 2))
        self.tab2_results_text = tk.Text(ctrl_frame, height=20, width=32, wrap=tk.WORD)
        self.tab2_results_text.pack(fill=tk.BOTH, expand=True)

        display_frame = ttk.Frame(self.tab2)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        img_frame = ttk.Frame(display_frame)
        img_frame.pack(fill=tk.BOTH, expand=True)

        f0_frame = ttk.LabelFrame(img_frame, text="1. Base YOLO on Distorted Image")
        f0_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.tab2_lbl_orig = ttk.Label(f0_frame, borderwidth=1, relief="solid")
        self.tab2_lbl_orig.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        f1_frame = ttk.LabelFrame(img_frame, text="2. Base YOLO on Enhanced Image")
        f1_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.tab2_lbl_dist = ttk.Label(f1_frame, borderwidth=1, relief="solid")
        self.tab2_lbl_dist.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        selected_model_str = self.parent.yolo_model_type.get().replace('_', ' ')
        f2_frame = ttk.LabelFrame(img_frame, text=f"3. {selected_model_str} on Distorted")
        f2_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        self.tab2_lbl_enh = ttk.Label(f2_frame, borderwidth=1, relief="solid")
        self.tab2_lbl_enh.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        img_frame.grid_columnconfigure(0, weight=1)
        img_frame.grid_columnconfigure(1, weight=1)
        img_frame.grid_columnconfigure(2, weight=1)
        img_frame.grid_rowconfigure(0, weight=1)

    def display_crop(self, img_np, label_widget):
        if img_np is None or img_np.size == 0:
            label_widget.config(image='', text="No Image")
            return
        h, w = img_np.shape[:2]
        max_size = 220
        scale = min(max_size / w, max_size / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img_resized = cv2.resize(img_np, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        photo = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=photo, text="")
        label_widget.image = photo

    def display_full_img(self, img_np, label_widget):
        if img_np is None or img_np.size == 0:
            label_widget.config(image='', text="No Image")
            return
        h, w = img_np.shape[:2]
        max_width, max_height = 290, 200
        scale = min(max_width / w, max_height / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img_resized = cv2.resize(img_np, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        photo = ImageTk.PhotoImage(img_pil)
        label_widget.config(image=photo, text="")
        label_widget.image = photo

    def get_crop(self, img, box):
        x1, y1, x2, y2 = map(int, box['xyxy'])
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        return img[y1:y2, x1:x2]

    def apply_enhancement(self, crop, method):
        if method.startswith("None"):
            return crop.copy()
        elif method.startswith("Restoration"):
            from filters import apply_restoration_filter
            return apply_restoration_filter(crop)
        elif method.startswith("Sharpening"):
            blurred = cv2.GaussianBlur(crop, (5, 5), 0)
            sharpened = cv2.addWeighted(crop, 2.5, blurred, -1.5, 0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        elif method.startswith("Histogram Equalization"):
            img_yuv = cv2.cvtColor(crop, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        elif "Gamma=1.5" in method:
            invGamma = 1.0 / 1.5
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(crop, table)
        elif "Gamma=0.6" in method:
            invGamma = 1.0 / 0.6
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(crop, table)
        elif method.startswith("Median Filter"):
            return cv2.medianBlur(crop, 5)
        elif method.startswith("Bilateral Denoise"):
            return cv2.bilateralFilter(crop, 9, 75, 75)
        elif method.startswith("Combined"):
            img_lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
            clahe_crop = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            blurred = cv2.GaussianBlur(clahe_crop, (5, 5), 0)
            sharpened = cv2.addWeighted(clahe_crop, 2.5, blurred, -1.5, 0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        elif "Median + CLAHE + Sharpen" in method:
            med = cv2.medianBlur(crop, 5)
            img_lab = cv2.cvtColor(med, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
            clahe_crop = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            blurred = cv2.GaussianBlur(clahe_crop, (5, 5), 0)
            sharpened = cv2.addWeighted(clahe_crop, 2.5, blurred, -1.5, 0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        return crop.copy()

    def calc_sharpness(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return 0.0

    def calc_contrast(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return np.std(gray)
        except Exception:
            return 0.0

    def calc_brightness(self, img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return np.mean(gray)
        except Exception:
            return 0.0

    def find_best_match(self, yolo_results, target_class_idx):
        if len(yolo_results.boxes) == 0:
            return "None", 0.0
        target_boxes = [b for b in yolo_results.boxes if int(b.cls[0].cpu().numpy()) == target_class_idx]
        if target_boxes:
            best_box = max(target_boxes, key=lambda b: float(b.conf[0].cpu().numpy()))
            cls_idx = int(best_box.cls[0].cpu().numpy())
            cls_name = yolo_results.names[cls_idx]
            conf = float(best_box.conf[0].cpu().numpy())
            return cls_name, conf
        else:
            best_box = max(yolo_results.boxes, key=lambda b: float(b.conf[0].cpu().numpy()))
            cls_idx = int(best_box.cls[0].cpu().numpy())
            cls_name = yolo_results.names[cls_idx]
            conf = float(best_box.conf[0].cpu().numpy())
            return f"{cls_name} (diff)", conf

    def update_crop_evaluation(self):
        box_idx = self.object_combo.current()
        method = self.enhance_combo.get()
        if box_idx < 0 or box_idx >= len(self.detected_boxes):
            return
        self.status_label.config(text="Evaluating crop with YOLO...")
        box = self.detected_boxes[box_idx]
        thread = threading.Thread(target=self._eval_worker, args=(box, method))
        thread.start()

    def _eval_worker(self, box, method):
        try:
            orig_crop = self.get_crop(self.original_img, box)
            dist_crop = self.get_crop(self.distorted_img, box)
            enh_crop = self.apply_enhancement(dist_crop, method)

            metrics = {
                'orig': {
                    'sharpness': self.calc_sharpness(orig_crop),
                    'contrast': self.calc_contrast(orig_crop),
                    'brightness': self.calc_brightness(orig_crop),
                    'class': box['class_name'],
                    'conf': box['conf']
                },
                'dist': {
                    'sharpness': self.calc_sharpness(dist_crop),
                    'contrast': self.calc_contrast(dist_crop),
                    'brightness': self.calc_brightness(dist_crop),
                },
                'enh': {
                    'sharpness': self.calc_sharpness(enh_crop),
                    'contrast': self.calc_contrast(enh_crop),
                    'brightness': self.calc_brightness(enh_crop),
                }
            }

            target_class_idx = box['class_idx']

            dist_results = yolo_model.predict(dist_crop, conf=0.01, verbose=False, device=yolo_device)[0]
            dist_class, dist_conf = self.find_best_match(dist_results, target_class_idx)
            metrics['dist']['class'] = dist_class
            metrics['dist']['conf'] = dist_conf

            enh_results = yolo_model.predict(enh_crop, conf=0.01, verbose=False, device=yolo_device)[0]
            enh_class, enh_conf = self.find_best_match(enh_results, target_class_idx)
            metrics['enh']['class'] = enh_class
            metrics['enh']['conf'] = enh_conf

            self.win.after(0, self._update_crop_ui, orig_crop, dist_crop, enh_crop, metrics)
        except Exception as e:
            err_msg = str(e)
            self.win.after(0, lambda msg=err_msg: messagebox.showerror("Evaluation Error",
                                                                       f"Error evaluating crop: {msg}"))

    def _update_crop_ui(self, orig_crop, dist_crop, enh_crop, metrics):
        self.display_crop(orig_crop, self.lbl_orig_crop)
        self.display_crop(dist_crop, self.lbl_dist_crop)
        self.display_crop(enh_crop, self.lbl_enh_crop)

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.tree.insert("", "end", values=(
            "YOLO Class Detected",
            metrics['orig']['class'],
            metrics['dist']['class'],
            metrics['enh']['class']
        ))
        self.tree.insert("", "end", values=(
            "YOLO Confidence",
            f"{metrics['orig']['conf']:.2%}",
            f"{metrics['dist']['conf']:.2%}",
            f"{metrics['enh']['conf']:.2%}"
        ))
        self.tree.insert("", "end", values=(
            "Sharpness (Laplacian Var)",
            f"{metrics['orig']['sharpness']:.1f}",
            f"{metrics['dist']['sharpness']:.1f}",
            f"{metrics['enh']['sharpness']:.1f}"
        ))
        self.tree.insert("", "end", values=(
            "Contrast (Intensity SD)",
            f"{metrics['orig']['contrast']:.1f}",
            f"{metrics['dist']['contrast']:.1f}",
            f"{metrics['enh']['contrast']:.1f}"
        ))
        self.tree.insert("", "end", values=(
            "Average Brightness (Mean)",
            f"{metrics['orig']['brightness']:.1f}",
            f"{metrics['dist']['brightness']:.1f}",
            f"{metrics['enh']['brightness']:.1f}"
        ))
        self.status_label.config(text="Ready")

    def run_full_image_yolo(self):
        method = self.tab2_enhance_combo.get()
        self.tab2_status_label.config(text="Evaluating strategies... this may take a moment.")
        self.tab2_run_btn.config(state=tk.DISABLED)
        thread = threading.Thread(target=self._full_image_worker, args=(method,))
        thread.start()

    def _full_image_worker(self, method):
        try:
            enhanced_full = self.apply_enhancement(self.distorted_img, method)

            from YOLO import yolo_overlay
            import os
            from ultralytics import YOLO

            dist_overlay_base, dist_res_base = yolo_overlay(self.distorted_img, conf=self.parent.yolo_conf.get())
            enh_overlay_base, enh_res_base = yolo_overlay(enhanced_full, conf=self.parent.yolo_conf.get())

            selected_model = self.parent.yolo_model_type.get()
            if selected_model == "Base":
                ft_model = YOLO("yolov8n.pt")
                expert_name = "BASE"
            else:
                model_paths = {
                    "FineTuned_SP": "runs/detect/train_sp/weights/best.pt",
                    "FineTuned_Blur": "runs/detect/train_blur/weights/best.pt",
                    "FineTuned_Bright": "runs/detect/train_bright/weights/best.pt",
                    "FineTuned_Gaussian": "runs/detect/train_gaussian/weights/best.pt"
                }
                weights_path = model_paths.get(selected_model)
                expert_name = selected_model.upper()

                if not os.path.exists(weights_path):
                    raise Exception(f"Weights for {selected_model} not found at {weights_path}.")

                ft_model = YOLO(weights_path)

            ft_res = ft_model.predict(self.distorted_img, conf=self.parent.yolo_conf.get(), verbose=False)[0]
            ft_overlay = ft_res.plot()

            stats_text = "Detection Statistics:\n\n"

            stats_text += f"1. BASE MODEL -> DISTORTED IMAGE:\n"
            stats_text += f"  Total objects: {len(dist_res_base.boxes)}\n"
            for i, box in enumerate(dist_res_base.boxes):
                c_idx = int(box.cls[0].cpu().numpy())
                c_name = dist_res_base.names[c_idx]
                c_conf = float(box.conf[0].cpu().numpy())
                stats_text += f"  - {c_name}: {c_conf:.1%}\n"

            stats_text += f"\n2. BASE MODEL -> ENHANCED IMAGE ({method}):\n"
            stats_text += f"  Total objects: {len(enh_res_base.boxes)}\n"
            for i, box in enumerate(enh_res_base.boxes):
                c_idx = int(box.cls[0].cpu().numpy())
                c_name = enh_res_base.names[c_idx]
                c_conf = float(box.conf[0].cpu().numpy())
                stats_text += f"  - {c_name}: {c_conf:.1%}\n"

            stats_text += f"\n3. {expert_name} MODEL -> DISTORTED IMAGE:\n"
            stats_text += f"  Total objects: {len(ft_res.boxes)}\n"
            for i, box in enumerate(ft_res.boxes):
                c_idx = int(box.cls[0].cpu().numpy())
                c_name = ft_res.names[c_idx]
                c_conf = float(box.conf[0].cpu().numpy())
                stats_text += f"  - {c_name}: {c_conf:.1%}\n"

            stats_text += f"\nCOMPARATIVE SUMMARY (vs Base Distorted):\n"

            diff_enh = len(enh_res_base.boxes) - len(dist_res_base.boxes)
            if diff_enh > 0:
                stats_text += f"  [Enhancement]: ✔ Found {diff_enh} MORE object(s).\n"
            elif diff_enh < 0:
                stats_text += f"  [Enhancement]: ⚠ Found {abs(diff_enh)} FEWER object(s).\n"
            else:
                stats_text += f"  [Enhancement]: • Same object count.\n"

            diff_ft = len(ft_res.boxes) - len(dist_res_base.boxes)
            if diff_ft > 0:
                stats_text += f"  [Fine-Tuned]:  ✔ Found {diff_ft} MORE object(s).\n"
            elif diff_ft < 0:
                stats_text += f"  [Fine-Tuned]:  ⚠ Found {abs(diff_ft)} FEWER object(s).\n"
            else:
                stats_text += f"  [Fine-Tuned]:  • Same object count.\n"

            self.win.after(0, self._update_full_image_ui, dist_overlay_base, enh_overlay_base, ft_overlay, stats_text)

        except Exception as e:
            err_msg = str(e)
            self.win.after(0, lambda msg=err_msg: messagebox.showerror("Full Image Error",
                                                                       f"Error processing full image: {msg}"))
            self.win.after(0, lambda: self.tab2_run_btn.config(state=tk.NORMAL))

    def _update_full_image_ui(self, dist_overlay_base, enh_overlay_base, ft_overlay, stats_text):
        self.display_full_img(dist_overlay_base, self.tab2_lbl_orig)
        self.display_full_img(enh_overlay_base, self.tab2_lbl_dist)
        self.display_full_img(ft_overlay, self.tab2_lbl_enh)

        self.tab2_results_text.delete("1.0", tk.END)
        self.tab2_results_text.insert(tk.END, stats_text)

        self.tab2_status_label.config(text="Evaluation Complete")
        self.tab2_run_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingUI(root)
    root.mainloop()