# AdaptSeg
<div align="center">

# 🏜️ AdaptSeg

### Off-Road Semantic Scene Segmentation with a Frozen DINOv2 Backbone

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DL%20Framework-EE4C2C?logo=pytorch&logoColor=white)
![Backbone](https://img.shields.io/badge/Backbone-DINOv2%20ViT--S%2F14-6f42c1)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-success)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20GPU-20BEFF?logo=kaggle&logoColor=white)

*Pixel-level understanding of unstructured outdoor terrain, built as an individual project.*

</div>

---

## 📌 Overview

**AdaptSeg** is a deep learning project that performs **semantic segmentation** on off-road scenes. Instead of assigning one label to a whole image, the model predicts a class for **every pixel**, separating terrain, vegetation, obstacles and other elements of an outdoor environment.

This kind of scene understanding is a building block for:

- 🚙 Autonomous off-road navigation
- 🤖 Robotics and perception systems
- 🗺️ Traversability and path analysis

AdaptSeg pairs a **frozen Vision Transformer (DINOv2 ViT-S/14)** with a **lightweight CNN segmentation head**. The transformer provides strong global features, and only the small head is trained, which keeps training efficient.

---

## 🧠 How It Works

```
Input Image ──▶ DINOv2 (frozen) ──▶ Feature Tokens ──▶ CNN Segmentation Head ──▶ Output Mask
```

1. The image is resized and normalized.
2. The frozen DINOv2 backbone extracts patch-level feature tokens.
3. The CNN head converts the tokens into spatial feature maps.
4. The output is upsampled to the input resolution to produce a per-pixel class prediction.

### Architecture

| Component | Details |
|---|---|
| **Backbone** | DINOv2 ViT-S/14, transformer-based, captures global image relationships, **frozen** during training |
| **Head** | Lightweight CNN that turns feature tokens into spatial maps and pixel-wise class logits |

---

## ⚙️ Training Setup

| Parameter | Value |
|---|---|
| Input size | 448 × 896 |
| Batch size | 16 |
| Learning rate | 3e-4 |
| Epochs | 15+ |
| Optimizer | AdamW |
| Classes | 10 |
| Normalization | ImageNet mean / std |
| Augmentation | Random horizontal flip, color jitter |

### Loss Function

```
Total Loss = 0.5 × CrossEntropy + 0.5 × Dice Loss
```

- **Cross-Entropy** drives correct per-pixel classification.
- **Dice Loss** improves region overlap and helps with class imbalance.

### Mask Conversion

Raw mask pixel values are remapped to contiguous class indices before training:

```python
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
```

The best checkpoint is selected automatically based on validation performance.

---

## 📊 Evaluation

- **Pixel-level IoU** as a basic overlap measure.
- **Class-wise mean IoU (mIoU)** as the primary, more reliable metric.
- **Test-Time Augmentation (TTA):** predictions on the original and horizontally flipped image are averaged. The evaluation script also supports multi-scale inference.

### Results

| Split | IoU |
|---|---|
| Training | ~0.7+ |
| Validation / Test | ~0.35 – 0.6 (varies with generalization) |

> The gap between training and validation/test IoU shows room to improve generalization. See [Future Work](#-future-work).

---

## 🚧 Challenges & Solutions

| Challenge | Solution |
|---|---|
| High memory use from large input size | Frozen backbone, feature extraction instead of full fine-tuning |
| Slow training with a transformer | Train only the lightweight head, GPU training |
| Class imbalance | Combined Cross-Entropy + Dice loss |
| Limited generalization | Data augmentation and flip TTA |

---

## 📁 Dataset Structure

```
Offroad_Segmentation_Training_Dataset/
├── train/
│   ├── Color_Images/
│   └── Segmentation/
└── val/
    ├── Color_Images/
    └── Segmentation/
```

Each image in `Color_Images/` has a matching mask in `Segmentation/`.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd <repo-folder>
```

### 2. Install dependencies (Python 3.8+)

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision timm opencv-python numpy tqdm pillow matplotlib
```

### 3. Set up the dataset

Place the dataset in the structure shown above and update the path in the script:

```python
DATA_DIR = "/kaggle/input/datasets/roheeeeet/rohit-xeno"
```

### 4. Train

```bash
python train.py
```

Or run all training cells in a Kaggle notebook. The best model is saved to `/kaggle/working/best_model.pth`.

### 5. Evaluate

```bash
python test.py
```

Computes mean IoU using flip TTA and multi-scale inference.

### 6. Visualize

```bash
python visualize.py
```

Results are saved to `/kaggle/working/visualizations/`. Each output image shows:

```
[ Original | Ground Truth | Prediction | Overlay ]
```

To download from Kaggle:

```bash
cd /kaggle/working
zip -r results.zip visualizations
```

---

## 🔍 Inference

```python
import torch

model.load_state_dict(torch.load("best_model.pth"))
model.eval()
```

Inference runs feature extraction, segmentation prediction and mask generation.

---

## 💡 Tips

- Enable a **GPU** for much faster training.
- If DataLoader errors occur on Kaggle, set `num_workers=0`.
- Use the **same image resolution** for training and testing.

---

## 🔮 Future Work

- Fine-tune the transformer backbone
- Stronger segmentation head (e.g., SegFormer-style decoder)
- Class-balanced loss
- Multi-scale training and richer augmentation
- Faster inference toward real-time use

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `torchvision` · `timm` · `OpenCV` · `NumPy` · `Pillow` · `Matplotlib` · `tqdm`

---

## 👤 Author

**Kunal Thakur**
B.Tech Computer Science & Engineering, Delhi Technical Campus, Greater Noida

*AdaptSeg is an individual project, designed, implemented and evaluated by me.*

---

## 📄 License

Add your license here (e.g., MIT).
