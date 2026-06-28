# HPatches Descriptor Evaluation Pipeline

A comprehensive benchmark and evaluation suite for local image descriptors using the **HPatches (Homography Patches)** dataset. 

This pipeline evaluates classical hand-crafted descriptors (SIFT, ORB, BRISK, LATCH, and optionally SURF) alongside a deep-learning-based descriptor (LIFT) across three standard tasks: **Patch Verification**, **Image Matching**, and **Patch Retrieval**.

---

## Table of Contents
1. [Evaluation Tasks & Metrics](#evaluation-tasks--metrics)
2. [Supported Descriptors](#supported-descriptors)
3. [Dataset Modes](#dataset-modes)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [Results & Visualizations](#results--visualizations)
7. [References](#references)

---

## Evaluation Tasks & Metrics

The pipeline evaluates descriptors across three classic computer vision problems:

1. **Patch Verification** (FPR95)
   - *Goal*: Determine if two image patches depict the same 3D point under varying viewpoints or illumination.
   - *Metric*: **False Positive Rate at 95% Recall (FPR95)**. Lower is better.
   
2. **Image Matching** (MMA)
   - *Goal*: Match interest points between a reference image and target images related by a known homography.
   - *Metric*: **Mean Matching Accuracy (MMA)** at pixel thresholds from 1px to 5px. Higher is better. Matches are found via Lowe's ratio-test.
   
3. **Patch Retrieval** (mAP)
   - *Goal*: Retrieve and rank patches from a large gallery that correspond to a query patch.
   - *Metric*: **mean Average Precision (mAP)**. Higher is better.

---

## Supported Descriptors

The pipeline evaluates the following descriptors, utilizing OpenCV's native implementations and PyTorch for neural network features:

| Descriptor | Type | Keypoint Detector | Descriptor Extractor | Distance Norm | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SIFT** | Gradient-based float | SIFT | SIFT | L2 Norm | Classical baseline |
| **ORB** | Binary | ORB | ORB | Hamming Norm | Fast, efficient |
| **BRISK** | Binary | BRISK | BRISK | Hamming Norm | High keypoint density |
| **LATCH** | Binary | FAST | LATCH | Hamming Norm | Learned 3-patch arrangements |
| **LIFT** | CNN-based float | SIFT | PyTorch CNN | L2 Norm | ECCV 2016 learned descriptor |
| **SURF** | Gradient-based float | SURF | SURF | L2 Norm | Optional (patent-restricted) |

---

## Dataset Modes

The pipeline supports two dataset configurations and an on-the-fly demonstration mode:

### Mode A: Full-Image Sequences (`--image_mode`)
- **Structure**: Each sequence directory in `hpatches-sequences-release/` contains:
  - `1.ppm` (Reference image)
  - `2.ppm` ... `6.ppm` (Target images under viewpoint or illumination changes)
  - `H_1_2` ... `H_1_6` (Ground-truth 3x3 homography matrices)
- **Capabilities**: Runs all three evaluation tasks (Verification, Matching, Retrieval).

### Mode B: Pre-extracted Patches (`--patch_mode`)
- **Structure**: Each sequence directory in `hpatches-sequences-release/` contains:
  - `ref.png` (Reference patch stack: $N \times 65 \times 65$ grayscale stack)
  - `e1.png` ... `e5.png` (Easy positive patches)
  - `h1.png` ... `h5.png` (Hard positive patches)
  - `t1.png` ... `t5.png` (Tough positive patches)
- **Capabilities**: Runs **Patch Verification** and **Retrieval**. Image Matching (MMA) is skipped because full images and homographies are not present.

### Mode C: Demo Mode (`--demo`)
- **Description**: Generates synthetic image sequences on the fly (no actual dataset download required) to test the installation and pipeline structure.

---

## Installation & Setup

### Prerequisites
Make sure Python 3.8+ is installed on your system.

### Install Dependencies
Install the required packages using `pip`:
```bash
pip install opencv-contrib-python numpy matplotlib tqdm
```

*(Optional)* To run the **LIFT** descriptor, install PyTorch and torchvision:
```bash
pip install torch torchvision
```

---

## Usage Guide

Run the pipeline using the command-line interface provided by `hpatches_descriptor_eval.py`.

### 1. Running on Demo Data (Fast Verification)
No dataset files are required to run in demo mode:
```bash
python hpatches_descriptor_eval.py --demo
```

### 2. Running on the Real Dataset
Point the script to your dataset directory:
```bash
python hpatches_descriptor_eval.py --hpatches_dir ./hpatches-sequences-release
```

### 3. Forcing Patch Mode or Image Mode
By default, the script auto-detects the mode (prefers patch mode if `ref.png` is found). You can override this behavior:
```bash
# Force patch mode
python hpatches_descriptor_eval.py --hpatches_dir ./hpatches-sequences-release --patch_mode

# Force image mode
python hpatches_descriptor_eval.py --hpatches_dir ./hpatches-sequences-release --image_mode
```

### 4. Specifying LIFT CNN Weights
If you have a trained checkpoint for LIFT (`.pth` weights format file), supply it via `--lift_weights`:
```bash
python hpatches_descriptor_eval.py --demo --lift_weights ./lift_desc.pth
```
*Note: If weights are omitted, LIFT will run with random initialization (useful for testing the pipeline, but metrics will not be meaningful).*

### Command Line Arguments Reference
- `--hpatches_dir`: Path to the HPatches sequences folder.
- `--demo`: Flag to run with synthetic demo data.
- `--out_dir`: Directory where evaluation reports, charts, and CSV results are saved (default: `./hpatches_results`).
- `--max_seqs`: Maximum number of sequences to process (default: `30` to keep execution times reasonable).
- `--n_patches`: Number of patches per sequence to use in the verification task (default: `150`).
- `--lift_weights`: Path to pre-trained LIFT model parameters.

---

## Results & Visualizations

All pipeline outputs are saved in the directory specified by `--out_dir` (default: `./hpatches_results/`).

### Output Directory Structure
```
hpatches_results/
├── results.csv
├── benchmark_results.png
├── mma_curves.png
└── matches_<Descriptor>_<Sequence>.png
```

- **`results.csv`**: A CSV file compiling all performance metrics (FPR95, mAP, MMA, execution speed, and keypoint counts) for each descriptor.
- **`benchmark_results.png`**: A summary dashboard bar plot displaying descriptor performance across tasks.
- **`mma_curves.png`**: Curve plots showing Mean Matching Accuracy (MMA) as a function of the pixel threshold.
- **`matches_*.png`**: Visual overlay matches on sequence pairs showing true positive (green lines) and false positive (red lines) correspondences.

---

## References
1. Yi, K. M., Trulls, E., Lepetit, V., & Fua, P. (2016). **LIFT: Learned Invariant Feature Transform**. *European Conference on Computer Vision (ECCV)*.
2. Balntas, V., Lenc, K., Vedaldi, A., & Mikolajczyk, K. (2017). **HPatches: A benchmark and evaluation of local descriptors**. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
