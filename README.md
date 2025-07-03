# CardiacNet: Detection of Cardiovascular Diseases in ECG Images

CardiacNet is a deep learning-based system for detecting cardiovascular diseases using ECG images. It utilizes Convolutional Neural Networks (CNNs) trained on public ECG datasets to accurately identify cardiac abnormalities. The model achieves a classification accuracy of **98.73%**, demonstrating its effectiveness in clinical decision support systems.

---

## 🚀 Project Overview

- **Model**: Custom Convolutional Neural Network (CNN) – *CardiacNet*
- **Accuracy**: 98.73% on test set
- **Input**: ECG images (JPEG/PNG format)
- **Output**: Detected cardiovascular condition (multi-class classification)
- **Tech Stack**: Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib

---

## 📁 Dataset Sources

This project uses publicly available ECG datasets:

1. **[ECG Image Dataset - Kaggle](https://www.kaggle.com/datasets/jayaprakashpondy/ecgimages)**
   - Contains categorized ECG images of various cardiac conditions.
   - Format: `.jpg` images organized by folder name per disease.
   - Size: ~5,000 images.

2. **[PTB-XL ECG Dataset - PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)**
   - Large-scale ECG dataset with clinical labels.
   - 12-lead ECG signals stored as `.npy` and `.csv`, convertible to images.
   - Used in conjunction with preprocessing and conversion scripts.

---

## 🧠 Model Architecture: CardiacNet

The CNN architecture consists of the following:

- 3 Convolutional Blocks (Conv2D + ReLU + MaxPooling)
- Dropout layers for regularization
- Fully Connected Dense Layers
- Output Layer with Softmax for multi-class classification

Trained with:
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Epochs**: 50
- **Hardware**: NVIDIA Tesla V100 GPU

---

## 🛠️ Preprocessing Pipeline

To improve signal clarity and model performance:
- **Resizing** images to 224x224 pixels
- **Gaussian Filtering** to reduce image noise
- **Edge Detection (Sobel/Canny)** to enhance waveform boundaries
- **Normalization** of pixel intensity
- **One-Hot Encoding** of labels for multi-class output

---

## 📊 Results

- **Training Accuracy**: 99.25%
- **Validation Accuracy**: 98.73%
- **Confusion Matrix**: High true positives across all cardiac classes
- **ROC-AUC Score**: 0.98

---

## 📂 Folder Structure

```bash
CardiacNet/
├── data/
│   ├── ecgimages/              # Kaggle images
│   └── ptbxl/                  # Converted PhysioNet ECGs
├── models/
│   └── cardiacnet_model.h5     # Trained model
├── notebooks/
│   └── training_pipeline.ipynb
├── preprocessing/
│   └── image_preprocessing.py
├── utils/
│   └── data_loader.py
├── evaluate.py
├── predict.py
└── README.md


🧪 How to Run
Clone this repository:
git clone https://github.com/rameshavinash94/Cardiovascular-Detection-using-ECG-images.git
cd Cardiovascular-Detection-using-ECG-images

Install requirements:
pip install -r requirements.txt

Train the model:
python training.py

Predict on a sample image:
python predict.py --image test_ecg.jpg

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.


