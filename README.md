# 🌙 Lunar-Mars-Crater Detection

<div align="center">

![Crater Detection Demo](crater_detection_demo.gif)

**Advanced crater detection on Lunar and Martian surfaces using YOLOv8**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/criamadei/Lunar-Mars-Crater/blob/main/Lunar_Mars_Crater.ipynb)
[![License:  MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blueviolet)](https://github.com/ultralytics/ultralytics)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔭 Overview

This project leverages **YOLOv8** (You Only Look Once, version 8) for real-time crater detection on Martian and Lunar surface images. The model is trained to identify and localize craters with high precision, which has applications in: 

- 🛰️ Autonomous spacecraft landing site selection
- 🗺️ Planetary surface mapping and geological analysis
- 🚀 Mission planning for lunar and Mars exploration
- 🔬 Scientific research in planetary geology

The notebook provides a complete end-to-end pipeline from data preparation to model deployment and visualization.

---

## ✨ Features

- **🎯 High Accuracy Detection**: Fine-tuned YOLOv8s model with advanced augmentation techniques
- **⚡ GPU Acceleration**: Optimized for NVIDIA T4 GPU training on Google Colab
- **📊 Comprehensive Metrics**:  Detailed training analytics including mAP, precision, recall, and confusion matrices
- **🎬 Visual Results**:  Automated GIF generation showcasing detection capabilities
- **🔄 Reproducible**: Seeded random states ensure consistent results
- **📦 Export Ready**: Download trained model weights for deployment
- **🛠️ Production Ready**: Includes data augmentation, dropout, learning rate scheduling, and early stopping

---

## 🎥 Demo

The model successfully detects craters across diverse terrain conditions:

![Crater Detection Demo](crater_detection_demo.gif)

> **Note**: Replace `crater_detection_demo.gif` with your own GIF file by adding it to the repository or linking to an external URL.

---

## 📦 Dataset

This project uses the **[Martian Lunar Crater Detection Dataset](https://www.kaggle.com/datasets/lincolnzh/martianlunar-crater-detection-dataset)** from Kaggle. 

### Dataset Characteristics:
- **Source**: High-resolution orbital imagery from Mars and Moon missions
- **Format**: YOLO format annotations (`.txt` files with bounding boxes)
- **Structure**:
  ```
  dataset_mars_moon/
  └── craters/
      ├── train/
      │   ├── images/
      │   └── labels/
      ├── valid/
      │   ├── images/
      │   └── labels/
      └── test/
          ├── images/
          └── labels/
  ```

---

## 🚀 Installation

### Option 1: Google Colab (Recommended)

Click the badge below to open directly in Colab with GPU support:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/criamadei/Lunar-Mars-Crater/blob/main/Lunar_Mars_Crater.ipynb)

**Setup in Colab:**
1. Set runtime to GPU:  `Runtime > Change runtime type > T4 GPU`
2. Add Kaggle credentials to Colab Secrets: 
   - Click the 🔑 key icon in the left sidebar
   - Add `KAGGLE_USERNAME` with your Kaggle username
   - Add `KAGGLE_KEY` with your Kaggle API key

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/criamadei/Lunar-Mars-Crater.git
cd Lunar-Mars-Crater

# Install dependencies
pip install ultralytics pandas numpy matplotlib seaborn scikit-learn imageio opencv-python

# Set Kaggle credentials
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

---

## 💻 Usage

### Quick Start

1. **Run the notebook**:  Open `Lunar_Mars_Crater.ipynb` in Jupyter or Colab
2. **Execute all cells**: The notebook will automatically: 
   - Download and prepare the dataset
   - Train the YOLOv8 model
   - Generate visualizations and metrics
   - Create a detection demo GIF

### Training Configuration

Key hyperparameters (adjust in cell 8):

```python
results = model.train(
    data='craters.yaml',
    epochs=100,           # Training epochs
    patience=10,          # Early stopping patience
    imgsz=1240,          # Image size
    batch=16,            # Batch size
    dropout=0.3,         # Dropout rate
    freeze=10,           # Freeze first 10 layers
    optimizer='AdamW',   # Optimizer
    lr0=0.001,          # Initial learning rate
    cos_lr=True         # Cosine LR scheduler
)
```

### Inference on New Images

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('lunar_crater_project/yolo_run/weights/best.pt')

# Run inference
results = model.predict(
    source='path/to/your/image.jpg',
    conf=0.25,  # Confidence threshold
    save=True   # Save annotated image
)
```

---

## 📈 Model Performance

### Training Metrics

The model achieves strong performance after 100 epochs:

| Metric | Value |
|--------|-------|
| **mAP50** | Check results after training |
| **mAP50-95** | Check results after training |
| **Precision** | See results. png |
| **Recall** | See results.png |

### Training Curves

![Training Results](lunar_crater_project/yolo_run/results.png)

### Confusion Matrix

![Confusion Matrix](lunar_crater_project/yolo_run/confusion_matrix. png)

---

## 📁 Project Structure

```
Lunar-Mars-Crater/
├── Lunar_Mars_Crater.ipynb    # Main notebook with complete pipeline
├── README.md                   # Project documentation
├── craters.yaml               # YOLO dataset configuration (auto-generated)
├── crater_detection_demo.gif  # Demo GIF (generated after training)
└── lunar_crater_project/      # Training outputs (auto-generated)
    └── yolo_run/
        ├── weights/
        │   ├── best.pt        # Best model weights
        │   └── last.pt        # Last epoch weights
        ├── results.png        # Training curves
        └── confusion_matrix. png
```

---

## 🎯 Results

### Key Achievements

- ✅ Successfully detects craters of varying sizes and terrain conditions
- ✅ Robust to lighting variations and surface textures
- ✅ Real-time inference capability on GPU
- ✅ Minimal false positives through optimized confidence thresholds

### Sample Detections

The model performs inference on test images, drawing bounding boxes around detected craters with confidence scores. 

---

## 🤝 Contributing

Contributions are welcome!  Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions

- 🔧 Hyperparameter optimization
- 📊 Additional evaluation metrics
- 🎨 Enhanced visualization tools
- 📝 Multi-language documentation
- 🔬 Transfer learning to other planetary bodies

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** for the YOLOv8 framework
- **[Lincoln Zhang](https://www.kaggle.com/lincolnzh)** for the Martian Lunar Crater Detection Dataset
- **NASA/ESA** for the original planetary imagery
- **Google Colab** for providing free GPU resources

---

## 📧 Contact

**Project Maintainer**:  [@criamadei](https://github.com/criamadei)

For questions or suggestions, please [open an issue](https://github.com/criamadei/Lunar-Mars-Crater/issues) or reach out via GitHub. 

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! **

Made with ❤️ for space exploration and computer vision

</div>
