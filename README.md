# 🩺 Chest X-ray Pneumonia Detection — MLflow Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/DenseNet121-grey?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Conda-44A833?style=for-the-badge&logo=anaconda&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accuracy-96--97%25-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  <b>End-to-end MLOps pipeline for pneumonia detection from chest X-rays — with full experiment tracking, model versioning, and reproducible runs.</b>
</p>

---

## 📌 Overview

**Chest X-ray Pneumonia Detection** is a production-structured **deep learning + MLOps project** that detects pneumonia from chest X-ray images using a **DenseNet121** backbone. The entire training lifecycle — hyperparameters, metrics, and model artifacts — is tracked using **MLflow Projects**, ensuring reproducibility and clean experiment management.

Achieved **~96–97% validation accuracy** on the Chest X-ray dataset (Normal vs. Pneumonia binary classification).

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🏥 **Medical Image Classification** | Binary classification: Normal vs. Pneumonia |
| 🧠 **DenseNet121 Backbone** | Pretrained CNN with dense skip connections |
| 🧪 **MLflow Experiment Tracking** | Logs hyperparameters, metrics, and model artifacts per run |
| 📦 **Reproducible Environment** | Conda-based setup via `MLproject` + `conda.yaml` |
| 🖥️ **MLflow UI** | Visual dashboard for comparing runs and metrics |
| 🔁 **One-Command Training** | `mlflow run .` handles setup and training end-to-end |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **Task** | Binary Classification (Normal vs. Pneumonia) |
| **Backbone** | DenseNet121 |
| **Validation Accuracy** | ~96–97% |
| **Loss Function** | Cross-Entropy Loss |
| **Optimizer** | Adam |
| **Input Size** | 224 × 224 × 3 |

---

## 🧠 How It Works

```
Chest X-ray Images (train / val / test)
            │
            ▼
  Data Preprocessing & Augmentation
  (Resize 224×224, Normalize)
            │
            ▼
   DenseNet121 Feature Extraction
            │
            ▼
     Binary Classification Head
     (Normal vs. Pneumonia)
            │
            ▼
  MLflow Run Logging
  ┌─────────────────────────┐
  │ Parameters: epochs,     │
  │ batch_size, lr          │
  │ Metrics: loss, accuracy │
  │ Artifacts: model.pt     │
  └─────────────────────────┘
            │
            ▼
     MLflow UI Dashboard
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch, Torchvision |
| **Model Architecture** | DenseNet121 |
| **Experiment Tracking** | MLflow |
| **Environment Management** | Conda |
| **Image Processing** | OpenCV / PIL, NumPy |
| **Future Deployment** | FastAPI, Docker |

---

## 📂 Project Structure

```
chest_xray_mlflow/
│
├── train.py            # Model training + MLflow logging
├── MLproject           # MLflow Project configuration
├── conda.yaml          # Reproducible Conda environment
├── requirements.txt    # Python dependencies
├── README.md
│
└── data/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## 📋 MLflow Tracking

The following are automatically logged per training run:

### Parameters
| Parameter | Description |
|---|---|
| `epochs` | Number of training epochs |
| `batch_size` | Batch size used during training |
| `learning_rate` | Adam optimizer learning rate |

### Metrics
| Metric | Description |
|---|---|
| `train_loss` | Training loss per epoch |
| `val_accuracy` | Validation accuracy per epoch |

### Artifacts
- Trained PyTorch model (`.pt`)
- Model metadata and configuration

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/nirusanathara/chest-xray-pneumonia.git
cd chest_xray_mlflow
```

### 2️⃣ Create & Activate Conda Environment
```bash
conda env create -f conda.yaml
conda activate chest-xray-env
```

### 3️⃣ Verify Setup
```bash
python -c "import torch, mlflow; print('Environment Ready ✅')"
```

### 4️⃣ Prepare Dataset
Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle and place it under `data/`:
```
data/
├── train/
├── val/
└── test/
```

---

## 🚀 Run Training

### Using MLflow Project (Recommended)
```bash
mlflow run . --env-manager=local
```

This will automatically:
- Start an MLflow experiment run
- Log all hyperparameters
- Log training and validation metrics
- Save the trained model as an MLflow artifact

### Launch MLflow UI
```bash
mlflow ui
```
Open in browser: `http://127.0.0.1:5000`

In the UI you can:
- Compare multiple experiment runs side-by-side
- Visualize training/validation metrics over time
- Download or register trained model artifacts

---

## 🧪 Model Architecture

```
Input: 224 × 224 × 3 (RGB Chest X-ray)
        │
        ▼
DenseNet121 (pretrained on ImageNet)
  - 121 layers with dense connections
  - Each layer receives feature maps from all previous layers
        │
        ▼
Global Average Pooling
        │
        ▼
Fully Connected Layer → 2 classes
        │
        ▼
Output: Normal (0) | Pneumonia (1)
```

---

## 🔮 Roadmap

- [ ] MLflow Model Registry (Staging → Production workflow)
- [ ] FastAPI inference service (`/predict` endpoint)
- [ ] Dockerized deployment
- [ ] CI/CD pipeline integration
- [ ] GPU training support
- [ ] Dataset versioning (DVC)
- [ ] Multi-class extension (Bacterial vs. Viral Pneumonia)

---

## 📚 Dataset Reference

**Chest X-Ray Images (Pneumonia)**  
Source: [Kaggle — Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
Classes: `NORMAL` · `PNEUMONIA`

---

## 👨‍💻 Author

**Niru Sanathara**  
AI/ML Engineer | Applied Scientist  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sanatharaniru/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/Niru-112)

---



<p align="center"><i>🏥 Applying deep learning and MLOps to make medical diagnostics more accessible.</i></p>
