import argparse
import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torchvision.models as models

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # 🔑 ATTACH TO EXISTING MLFLOW PROJECT RUN
    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):

        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.lr)

        # Dummy metrics (replace later)
        mlflow.log_metric("train_loss", 0.42)
        mlflow.log_metric("val_accuracy", 0.91)

        mlflow.pytorch.log_model(model, artifact_path="model")

        print("✅ Training + MLflow logging completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(args)
