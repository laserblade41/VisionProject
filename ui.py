import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from pathlib import Path
from ORB import run_orb
from YOLO import yolo_overlay
from distorsions import (
    reduce_brightness,
    apply_salt_and_pepper_noise,
    apply_motion_blur,
    apply_fog,
)
from filters import (
    apply_clahe,
    apply_median_filter,
    apply_restoration_filter,
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

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: Controls
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))

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
        ttk.Button(nav_frame, text="< Prev", command=self.prev_image, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Next >", command=self.next_image, width=10).pack(side=tk.LEFT)

        # Technique selection
        ttk.Label(left_frame, text="Processing Technique", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(10, 5)
        )
        self.technique_var = tk.StringVar(value="ORB")
        techniques = ["ORB", "YOLO"]
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

        distortions = [
            ("Reduce Brightness", "brightness", 0.25, 0.1, 0.9),
            ("Salt & Pepper", "salt_pepper", 0.05, 0.0, 0.2),
            ("Motion Blur", "motion_blur", 9, 3, 31, 2),
            ("Fog", "fog", 0.5, 0.0, 1.0),
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
        filters = ["CLAHE", "Median Filter", "Restoration Filter"]
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

        # Process button
        ttk.Button(left_frame, text="Process", command=self.process_image).pack(
            fill=tk.X, pady=20
        )
        
        # Reset button
        ttk.Button(left_frame, text="Reset All", command=self.reset_all).pack(
            fill=tk.X, pady=(0, 20)
        )

        # Right panel: Images and results
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas for images
        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(canvas_frame, text="Original", font=("Arial", 10, "bold")).pack()
        self.canvas_original = tk.Canvas(canvas_frame, bg="gray", height=300)
        self.canvas_original.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(canvas_frame, text="Distorted (if applied)", font=("Arial", 10, "bold")).pack()
        self.canvas_distorted = tk.Canvas(canvas_frame, bg="gray", height=300)
        self.canvas_distorted.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(canvas_frame, text="Processed", font=("Arial", 10, "bold")).pack()
        self.canvas_processed = tk.Canvas(canvas_frame, bg="gray", height=300)
        self.canvas_processed.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Results text
        ttk.Label(canvas_frame, text="Results", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.results_text = tk.Text(canvas_frame, height=8, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True)

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
            
            # Supported image extensions
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

    def display_image(self, img_rgb, canvas):
        h, w = img_rgb.shape[:2]
        max_width, max_height = 500, 300
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

        if self.distortion_vars["brightness"].get():
            factor = self.distortion_params["brightness"]["var"].get()
            result = reduce_brightness(result, factor)
            applied.append(f"Brightness: {factor:.2f}")

        if self.distortion_vars["salt_pepper"].get():
            amount = self.distortion_params["salt_pepper"]["var"].get()
            result = apply_salt_and_pepper_noise(result, amount)
            applied.append(f"Salt & Pepper: {amount:.3f}")

        if self.distortion_vars["motion_blur"].get():
            kernel_size = int(self.distortion_params["motion_blur"]["var"].get())
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            result = apply_motion_blur(result, kernel_size)
            applied.append(f"Motion Blur: kernel={kernel_size}")

        if self.distortion_vars["fog"].get():
            intensity = self.distortion_params["fog"]["var"].get()
            result = apply_fog(result, intensity)
            applied.append(f"Fog: {intensity:.2f}")

        return result, applied

    def apply_filters(self, img):
        result = img.copy()
        applied = []

        if self.filter_vars["CLAHE"].get():
            result = apply_clahe(result)
            applied.append("CLAHE")

        if self.filter_vars["Median Filter"].get():
            result = apply_median_filter(result, ksize=5)
            applied.append("Median Filter")

        if self.filter_vars["Restoration Filter"].get():
            result = apply_restoration_filter(result)
            applied.append("Restoration Filter")

        return result, applied

    def process_image(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        # Show processing message
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Processing... please wait")
        self.root.update()

        # Run in thread to prevent UI freezing
        thread = threading.Thread(target=self._process_worker)
        thread.start()

    def _process_worker(self):
        try:
            # Apply distortions
            img = self.original_image.copy()
            distortion_info, distortion_names = self.apply_distortions(img)
            self.distorted_image = distortion_info

            if distortion_names:
                self.display_image(distortion_info, self.canvas_distorted)

            # Apply filters
            filtered_img, filter_names = self.apply_filters(distortion_info)

            # Apply technique
            technique = self.technique_var.get()

            if technique == "ORB":
                processed_img, keypoints, descriptors = run_orb(
                    filtered_img, nfeatures=self.orb_nfeatures.get()
                )
                results_text = f"ORB Detection Results:\n"
                results_text += f"  Keypoints detected: {len(keypoints)}\n"
                results_text += f"  Descriptors shape: {descriptors.shape if descriptors is not None else 'None'}\n"

            elif technique == "YOLO":
                processed_img, yolo_results = yolo_overlay(
                    filtered_img, conf=self.yolo_conf.get()
                )
                results_text = f"YOLO Detection Results:\n"
                results_text += f"  Objects detected: {len(yolo_results.boxes)}\n"
                if len(yolo_results.boxes) > 0:
                    for i, box in enumerate(yolo_results.boxes):
                        cls_name = yolo_results.names[int(box.cls)]
                        conf = float(box.conf)
                        results_text += f"  {i + 1}. {cls_name}: {conf:.2%}\n"

            self.processed_image = processed_img

            # Build full results text
            full_results = results_text
            if distortion_names:
                full_results += f"\nDistortions applied:\n  " + ", ".join(distortion_names) + "\n"
            if filter_names:
                full_results += f"\nFilters applied:\n  " + ", ".join(filter_names) + "\n"

            # Update UI in main thread
            self.root.after(0, self._update_ui, processed_img, full_results)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
    
    def reset_all(self):
        """Reset all processing results and clear distortion settings"""
        # Clear images
        self.distorted_image = None
        self.processed_image = None
        
        # Clear canvases
        self.canvas_distorted.delete("all")
        self.canvas_processed.delete("all")
        self.results_text.delete("1.0", tk.END)
        
        # Reset distortion checkboxes and sliders
        for key, var in self.distortion_vars.items():
            var.set(False)
        
        # Reset filter checkboxes
        for key, var in self.filter_vars.items():
            var.set(False)
        
        messagebox.showinfo("Reset", "All processing results and settings have been reset")

    def _update_ui(self, processed_img, results_text):
        self.display_image(processed_img, self.canvas_processed)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, results_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingUI(root)
    root.mainloop()
