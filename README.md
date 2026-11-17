# AdaptSeg: Off-Road Semantic Scene Segmentation

## About the Project

AdaptSeg is a deep learning-based project focused on understanding complex off-road environments through semantic segmentation. Unlike basic image classification, this project assigns a label to every pixel in an image, allowing the model to distinguish between terrain, vegetation, obstacles, and other elements present in outdoor scenes.

The goal of this project is to build an intelligent system capable of interpreting real-world environments, which can be applied in areas such as autonomous navigation, robotics, and off-road path analysis.

What makes this project powerful is the use of a Vision Transformer (DINOv2) combined with a custom segmentation head, enabling both global understanding and efficient prediction.

---

## How It Works

The model follows a simple but effective pipeline:

```
Input Image → DINOv2 Feature Extraction → Segmentation Head → Output Mask
```

Instead of directly learning from raw pixels, the model first extracts meaningful features using a transformer-based backbone and then converts those features into a segmentation map.

---

## Model Architecture

The system consists of two main components:

**1. DINOv2 Backbone (ViT-S/14)**
- A transformer-based architecture
- Captures global relationships in images
- Outputs feature tokens instead of convolution maps
- Used in frozen mode to improve efficiency

**2. Segmentation Head**
- A lightweight CNN-based module
- Converts transformer features into spatial feature maps
- Produces pixel-wise class predictions

Each image has a corresponding segmentation mask.

---

## Data Processing

Before feeding images into the model:

- Images are resized to **448 × 896**
- Normalized using ImageNet statistics
- Augmented using:
  - Random Horizontal Flip
  - Color Jitter

### Mask Conversion

Custom pixel values are mapped into class indices:

```python
value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
```

This ensures consistency during training.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Batch Size | 16 |
| Learning Rate | 3e-4 |
| Epochs | 15+ |
| Image Size | 448 × 896 |
| Number of Classes | 10 |

### Loss Function

A combination of two loss functions is used:

```
Total Loss = 0.5 × CrossEntropy + 0.5 × Dice Loss
```

- Cross Entropy ensures correct classification
- Dice Loss improves segmentation overlap

### Optimization Strategy

To improve performance, the following techniques were used:

- AdamW optimizer for better generalization
- Frozen backbone to reduce computation
- Efficient feature extraction using DINOv2
- GPU-based training for speed

---

## Training Process

During training:

1. Images are passed through the DINOv2 backbone
2. Feature tokens are extracted
3. The segmentation head processes the features
4. Output is resized to match original dimensions
5. Loss is computed and backpropagated
6. The best model is saved based on validation performance

---

## Evaluation & Metrics

### IoU (Intersection over Union)

IoU measures how well the predicted segmentation overlaps with the ground truth.

Two approaches were used:

- Basic IoU (pixel accuracy)
- Class-wise mean IoU (more reliable metric)

### Test Time Augmentation (TTA)

To improve prediction accuracy:

- The model runs on the original image
- The model runs on the horizontally flipped image
- The final output is the average of both predictions

This helps reduce prediction errors.

### Metrics

| Split | IoU |
|---|---|
| Training | ~0.7+ |
| Validation / Test | ~0.35 – 0.6 (depends on generalization) |

---

## Results

- Model performance improved progressively during training
- The best model is selected automatically
- Final evaluation is based on the mean IoU score

---

## Model Usage

To load the trained model:

```python
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
```

Inference includes:

- Feature extraction
- Segmentation prediction
- Mask generation

---

## Dependencies

Install the required libraries:

```bash
pip install torch torchvision timm opencv-python numpy tqdm pillow
```

---

## Challenges Faced

- High memory usage due to large image size
- Long training time with a transformer-based architecture
- Class imbalance in the dataset

## Solutions Implemented

- Used feature extraction instead of full fine-tuning
- Combined Dice and Cross-Entropy loss
- Applied data augmentation and TTA
- Optimized the training pipeline

---

## Steps to Run the Project

Follow the steps below to train, evaluate, and visualize the semantic segmentation model.

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd <repo-folder>
```

### 2. Install Dependencies

Make sure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision timm opencv-python tqdm numpy pillow matplotlib
```

### 3. Setup Dataset

Download the dataset and place it inside:

```
/kaggle/input/datasets/roheeeeet/rohit-xeno/
```

Ensure the folder structure is:

```
Offroad_Segmentation_Training_Dataset/
│
├── train/
│   ├── Color_Images/
│   └── Segmentation/
│
└── val/
    ├── Color_Images/
    └── Segmentation/
```

### 4. Configure Paths

Update the dataset path in the script:

```python
DATA_DIR = "/kaggle/input/datasets/roheeeeet/rohit-xeno"
```

### 5. Train the Model

Run the training script:

```bash
python train.py
```

Or, in a Kaggle Notebook, run all training cells.

- The model trains for the specified number of epochs
- The best model is saved at: `/kaggle/working/best_model.pth`

### 6. Evaluate the Model

After training, run evaluation:

```bash
python test.py
```

Computes Mean IoU using:

- Flip TTA
- Multi-scale inference

### 7. Generate Visualizations

Run the visualization script:

```bash
python visualize.py
```

Outputs are saved in `/kaggle/working/visualizations/`. Each output contains:

```
[ Original | Ground Truth | Prediction | Overlay ]
```

### 8. Download Results (Kaggle)

To download the results:

```bash
cd /kaggle/working
zip -r results.zip visualizations
```

Then download `results.zip` from the Kaggle output panel.

---

## Notes

- Ensure a GPU is enabled for faster training
- Set `num_workers=0` in the DataLoader if facing issues on Kaggle
- Use the same image resolution during training and testing

---

## Future Improvements

- Improve the IoU score further
- Fine-tune the transformer backbone
- Optimize inference speed
- Extend to real-time applications
- Multi-scale training
- Better segmentation head (e.g., SegFormer)
- Class-balanced loss
- Data augmentation improvements
