Here's the updated and fully aligned `README.md` with spots left for your sample code runs:

---

# 🩻 XRay-Clarity-Biometrics

A pipeline for processing X-Ray images of consumer electronics using segmentation, alignment, and deep feature comparison. Designed for Clarity project use cases in biometric forensics.

---

## 📦 Features

* Dual-reference image alignment (front and back)
* Semantic segmentation using YOLOv8
* Rotation correction using DINO / ResNet
* Patch-based anomaly detection
* Batch processing from structured CSVs
* Visual debugging and performance evaluation

---

## 🔧 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/TheGhoul27/XRay-Clarity-Biometrics.git
cd XRay-Clarity-Biometrics
```

---

### 2. Create a Python Environment

```bash
conda create -n xray_clarity python=3.9 -y
conda activate xray_clarity
```

---

### 3. Install Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> 📌 For GPU users, install the appropriate PyTorch version from [https://pytorch.org](https://pytorch.org)

---

## 📁 Project Structure

```
XRay-Clarity-Biometrics/
├── data/                        # Input images, segmentation masks, and CSVs
│   ├── images/                  # X-Ray images
│   ├── seg/                     # Ground truth or generated masks
│   └── Clarity-test-images...   # Master CSV input file
│
├── matcher/                    # Patch-level feature match & benchmarking
│   ├── benchmark.py
│   └── matches.py
│
├── pipeline/                   # Main pipeline modules
│   ├── pipeline.py
│   ├── align_utils.py
│   └── seg_utils.py
│
├── pipeline_eval/             # Evaluation + ablation scripts
│   ├── anomaly_eval.py
│   ├── anomaly_eval_ablation_study.py
│   └── orb_ransac_ablation.py
│
├── pipeline_single_file/      # Minimal scripts for standalone runs
│   ├── segment_obb_align.py
│   ├── segment_obb_omniglue_align.py
│   ├── rotation_align.py
│   └── rotation_align_general.py
│
├── yolo-matcher/              # YOLO training, inference, visualization
│   ├── training.py
│   ├── visualize.py
│   ├── save_masks.py
│   └── yolov8n-seg.pt         # Pretrained YOLOv8n segmentation model
│
├── aligning_all.py            # Main end-to-end batch processing script
├── test_pipeline.py           # Sample runner for sanity check
├── requirements.txt
└── README.md                  # This file
```

---

## 🚀 How to Run

> Make sure `data/Clarity-test-images-data-Sheet.csv` is formatted properly with `Item`, `Side`, and `Path`.

### 🧪 # Sample usage heading

```bash
python #TODO: Add real usage here
```

---

## 📌 Notes

* Supported formats: `.bmp`, `.png`, `.jpg`
* Ensure pretrained weights (e.g., `yolov8n-seg.pt`) are present
* Outputs are saved under `vis/` or `output/` depending on the script
* The project expects masks in YOLOv8 segmentation format

---

## 🛠️ Troubleshooting

* **PermissionError**: Ensure all image files are accessible and not locked by the OS
* **CUDA Errors**: Ensure CUDA, drivers, and PyTorch versions are compatible
* **Empty Outputs**: Double-check segmentation masks and CSV path consistency

---