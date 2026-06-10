# Image Processing UI Guide

## Overview

This is a unified GUI application for comparing image processing techniques (ORB and YOLO) with various distortions and filters. The application allows you to:

- Load and process images
- Select between ORB feature detection or YOLO object detection
- Apply multiple distortions to test robustness
- Apply restoration filters
- View before/after comparisons
- See detailed metrics for each technique

## Features

### Processing Techniques

**ORB (Oriented FAST and Rotated BRIEF)**
- Detects and computes keypoint features
- Useful for feature matching and tracking
- Parameter: Number of features (100-2000, default 800)
- Output: Keypoint count and descriptor information

**YOLO (You Only Look Once)**
- Real-time object detection
- Detects multiple object classes
- Parameter: Confidence threshold (0.0-1.0, default 0.25)
- Output: List of detected objects with confidence scores

### Distortions

**Reduce Brightness**
- Multiplies pixel values by a factor
- Range: 0.1-0.9 (default 0.25)
- Use to test low-light robustness

**Salt & Pepper Noise**
- Random white and black pixels
- Range: 0.0-0.2 (default 0.05)
- Tests noise robustness

**Motion Blur**
- Simulates motion with directional blur
- Kernel size: 3-31 (default 9)
- Tests blur robustness

**Haze**
- Reduces contrast with white overlay
- Intensity: 0.0-1.0 (default 0.35)
- Tests hazy/dusty conditions

**Fog**
- Dense atmospheric distortion (blur + white overlay)
- Intensity: 0.0-1.0 (default 0.5)
- Tests foggy conditions

### Restoration Filters

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- Improves local contrast
- Effective for haze/fog restoration

**Median Filter**
- Removes impulse noise
- Effective for salt & pepper noise

**Restoration Filter**
- Combined unsharp mask + CLAHE
- Comprehensive restoration approach

## Usage

### Running the Application

```bash
python ui.py
```

### Basic Workflow

#### Single Image Processing
1. **Load Image**: Click "Load Image" button to select an image file
2. **Choose Technique**: Select either ORB or YOLO
3. **Apply Distortions**: Check desired distortions and adjust sliders
4. **Apply Filters** (optional): Check restoration filters
5. **Adjust Parameters**: 
   - For ORB: Set number of features
   - For YOLO: Set confidence threshold
6. **Process**: Click "Process" button
7. **View Results**: See before/distorted/processed images and metrics

#### Batch Processing with Dataset
1. **Load Dataset**: Click "Load Dataset" button and select a folder containing images
2. **Navigate**: Use "< Prev" and "Next >" buttons to browse through images
3. **Configure Settings**: Set up technique, distortions, and filters
4. **Process**: Click "Process" for each image (results update in place)
5. **Compare**: Easily switch between images to compare results

#### Reset Settings
- Click **"Reset All"** to clear all distortion selections and processing results
- This allows you to quickly try different parameter combinations

### Tips

- Start with default settings to understand baseline performance
- Apply one distortion at a time to understand individual effects
- Use restoration filters after applying distortions to see recovery
- For ORB: Increase features for more detail, decrease for speed
- For YOLO: Lower confidence threshold to detect more objects (with more false positives)
- **Dataset Mode**: Organize test images in a folder and use "Load Dataset" to batch test
- **Reset Button**: Use "Reset All" between experiments to clear settings quickly

## Output

The results panel shows:

**For ORB:**
- Number of keypoints detected
- Descriptor shape (number of descriptors × 32 bytes)

**For YOLO:**
- Total number of objects detected
- List of detected objects with their confidence scores (e.g., "person: 95%")

**Distortions Applied:**
- List of all distortions with their parameters

**Filters Applied:**
- List of all restoration filters applied

## Dataset Mode

The application supports batch processing of multiple images:

### Dataset Structure
Simply create a folder with your test images:
```
my_dataset/
├── image1.jpg
├── image2.png
├── image3.bmp
└── ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp` (case-insensitive)

### Navigation
- **Load Dataset**: Select the folder containing your images
- **Prev/Next Buttons**: Navigate through images in the dataset
- Current progress shown as "Image X/Y"
- Images are automatically sorted alphabetically

## Implementation Details

### Architecture

- **ui.py**: Main Tkinter GUI application
- **ORB.py**: ORB feature detection implementation
- **YOLO.py**: YOLO object detection integration
- **distorsions.py**: Image distortion functions
- **filters.py**: Image restoration/filter functions

### Processing Pipeline

```
Original Image → Apply Distortions → Apply Filters → Apply Technique → Display Results
```

### Threading

Image processing runs in a background thread to prevent UI freezing during computation.

## System Requirements

- Python 3.8+
- OpenCV
- Pillow
- Tkinter (usually included with Python)
- Ultralytics YOLOv8
- NumPy, Matplotlib, SciPy (from requirements.txt)

## Performance Notes

- YOLO requires GPU for real-time performance (CPU mode is slower)
- ORB is fast on both CPU and GPU
- Larger images take longer to process
- Multiple distortions compound processing time
