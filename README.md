Markdown
# Breast Cancer Tissue Segmentation using U-Net  
**Bachelor's Thesis Project – 2020**

Semantic segmentation of cancerous regions in histopathological images using classic U-Net architecture.

## Abstract
> Machine learning models based on neural networks have emerged as one of the most significant methods in medical science for extracting useful information and identifying tissue lesions or segmenting cancerous regions.  
> In this project, medical image analysis and cancerous tissue segmentation were carried out using the **U-Net** architecture. Techniques such as data augmentation and dropout layers were applied to enhance model efficiency and prevent overfitting.  
> Implementation was done in Python using Keras/TensorFlow on Google Colab.

**Keywords**: Deep Learning · Convolutional Neural Networks · U-Net · Medical Image Segmentation · Breast Cancer


## Key Features
- ROI-based preprocessing: automatic cropping of the region of interest using ground-truth masks
- Image resizing to 256×256 with anti-aliasing
- Classic U-Net with ELU activation and He-normal initialization
- Dropout (0.1–0.2) and data augmentation for regularization
- Binary semantic segmentation (cancerous vs non-cancerous tissue)
- Full pipeline in Jupyter Notebook (Colab-ready)


## Dataset
Public collection of breast histopathology images with pixel-level binary masks (.TIF format)  
→ Automatically cropped & resized to 256×256×3 during preprocessing


## Model Architecture (Vanilla U-Net)
Input (256×256×3)
│
├─ Conv2D(16) → ELU → Conv2D(16) → Dropout(0.1) → MaxPool2D
├─ Conv2D(32) → ELU → Conv2D(32) → Dropout(0.1) → MaxPool2D
├─ Conv2D(64) → ELU → Conv2D(64) → Dropout(0.2) → MaxPool2D
├─ Bottleneck Conv2D(128)
↑ (Up-sampling + Concat + Conv blocks symmetrically)
└─ Final 1×1 Conv → Sigmoid → Output (256×256×1)


## Results & Visualization (included in notebook)
- Training/validation loss & accuracy curves
- Before/after preprocessing examples (full image → cropped ROI → resized)
- Overlay of ground-truth vs predicted masks
- Sample predictions on test set


## Repository Contents
├── notebooks/
│   └── Breast_Cancer_U_Net_Segmentation.ipynb    # Complete Colab notebook
├── data/                                         # Sample images & masks (full dataset on request)
├── results/                                       # Plots, predictions, preprocessing examples
├── Bachelor's_thesis_abstract_and_code.pdf        # Thesis abstract + full code
└── README.md


## Tech Stack
- Python 3.6–3.8
- TensorFlow 2.x + Keras
- scikit-image, OpenCV, NumPy, Matplotlib
- Google Colab (GPU)

## How to Run
1. Open the notebook in Google Colab
2. Upload `samples.rar` or change paths to your dataset
3. Run all cells → model trains and displays results

## Links
- GitHub → https://github.com/armin-ds/cnn-breast-cancer-segmentation
- Kaggle → X
- LinkedIn → X
- Harvard CS50P & Math for ML certificates (2025)


Made with enthusiasm for MSc Data Science applications – Italy 2026
