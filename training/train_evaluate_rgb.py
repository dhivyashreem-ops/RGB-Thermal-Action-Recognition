import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

FEATURE_DIR = r"data\processed\features"

RGB_TRAIN = os.path.join(
    FEATURE_DIR,
    "rgb_train_features.npy"
)

RGB_TRAIN_LABELS = os.path.join(
    FEATURE_DIR,
    "rgb_train_labels.npy"
)

RGB_TEST = os.path.join(
    FEATURE_DIR,
    "rgb_test_features.npy"
)

RGB_TEST_LABELS = os.path.join(
    FEATURE_DIR,
    "rgb_test_labels.npy"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

NUM_CLASSES = 27
FEATURE_DIM = 512

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class RGBClassifier(nn.Module):

    def __init__(
        self,
        input_dim=512,
        num_classes=27
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        return self.network(x)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

print("=" * 70)
print("RGB BASELINE — TRAIN + TEST EVALUATION")
print("=" * 70)

print("\nDevice:", DEVICE)


X_train = np.load(RGB_TRAIN)
y_train = np.load(RGB_TRAIN_LABELS)

X_test = np.load(RGB_TEST)
y_test = np.load(RGB_TEST_LABELS)


print("\nTraining features:", X_train.shape)
print("Training labels:", y_train.shape)

print("\nTesting features:", X_test.shape)
print("Testing labels:", y_test.shape)


# ---------------------------------------------------------
# Tensor conversion
# ---------------------------------------------------------

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = RGBClassifier(
    input_dim=FEATURE_DIM,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    loss_value = (
        running_loss / total
    )

    accuracy = (
        correct / total
    ) * 100

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {loss_value:.4f} | "
        f"Train Accuracy: {accuracy:.2f}%"
    )


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

model.eval()

all_predictions = []

with torch.no_grad():

    for start in range(
        0,
        len(X_test),
        BATCH_SIZE
    ):

        batch = X_test[
            start:start + BATCH_SIZE
        ].to(DEVICE)

        outputs = model(
            batch
        )

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


y_pred = np.array(
    all_predictions
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 70)
print("RGB TEST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy      : {accuracy:.4f}"
)

print(
    f"Macro Precision: {precision:.4f}"
)

print(
    f"Macro Recall   : {recall:.4f}"
)

print(
    f"Macro F1       : {macro_f1:.4f}"
)

print(
    f"Weighted F1    : {weighted_f1:.4f}"
)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

os.makedirs(
    r"outputs\results",
    exist_ok=True
)

np.save(
    r"outputs\results\rgb_confusion_matrix.npy",
    cm
)


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

torch.save(
    model.state_dict(),
    r"outputs\models\rgb_classifier.pth"
)


print("\nConfusion matrix saved:")
print(
    r"outputs\results\rgb_confusion_matrix.npy"
)

print("\nRGB baseline evaluation completed.")