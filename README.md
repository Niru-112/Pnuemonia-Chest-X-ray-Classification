# 🩺 Chest X-ray Pneumonia Detection – MLflow Project

An end-to-end Machine Learning + MLOps project for detecting Pneumonia from Chest X-ray images using PyTorch and MLflow.
The project is structured as an MLflow Project to ensure reproducibility, experiment tracking, and model versioning.

📌 Project Overview

Pneumonia is a serious lung infection that can be diagnosed through chest X-ray images.
This project builds a deep learning–based image classification system and tracks the entire training lifecycle using MLflow.

Key Highlights

PyTorch-based CNN model (DenseNet)

MLflow experiment tracking

MLflow Projects for reproducible runs

Conda-based environment management

Production-ready project structure

🧠 Tech Stack

Python

PyTorch

Torchvision

MLflow

Conda

NumPy

OpenCV / PIL

FastAPI (optional – future deployment)

Docker (optional – future deployment)

📂 Project Structure

chest_xray_mlflow/
│
├── train.py               # Model training & MLflow logging
├── MLproject              # MLflow Project configuration
├── conda.yaml             # Reproducible Conda environment
├── requirements.txt       # Python dependencies
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
└── README.md

🚀 How to Run the Project

1️⃣ Create & Activate Conda Environment

conda env create -f conda.yaml
conda activate chest-xray-env

Verify setup:

python -c "import torch, mlflow; print('Environment Ready')"

2️⃣ Run Training using MLflow Project

mlflow run . --env-manager=local

This will:

Start an MLflow run

Log hyperparameters

Log training metrics

Save the trained PyTorch model as an MLflow artifact

3️⃣ Launch MLflow UI

mlflow ui

Open in browser:
http://127.0.0.1:5000

You can view:

Experiments

Runs

Parameters

Metrics

Logged models

📊 MLflow Tracking

The following are tracked automatically:

Parameters

Epochs

Batch size

Learning rate

Metrics

Training loss

Validation accuracy

Artifacts

Trained PyTorch model

Model metadata

🧪 Model Architecture

Backbone: DenseNet121

Input size: 224 × 224 × 3

Loss: Cross-Entropy Loss

Optimizer: Adam

Task: Binary Classification (Normal vs Pneumonia)

🧩 Why MLflow?

This project uses MLflow Projects to:

Ensure reproducible ML experiments

Track multiple model versions

Maintain clean separation of code and configuration

Enable future deployment via MLflow Model Registry

🔮 Future Enhancements

Model Registry (Staging → Production)

FastAPI inference service

Dockerized deployment

CI/CD integration

GPU training support

Dataset versioning
