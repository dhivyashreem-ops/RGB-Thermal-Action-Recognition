import os

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


# =========================================================
# CONFIGURATION
# =========================================================

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

THERMAL_TRAIN = os.path.join(
    FEATURE_DIR,
    "thermal_train_features.npy"
)

THERMAL_TRAIN_LABELS = os.path.join(
    FEATURE_DIR,
    "thermal_train_labels.npy"
)

THERMAL_TEST = os.path.join(
    FEATURE_DIR,
    "thermal_test_features.npy"
)

THERMAL_TEST_LABELS = os.path.join(
    FEATURE_DIR,
    "thermal_test_labels.npy"
)


BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

RGB_DIM = 512
THERMAL_DIM = 512

FUSION_DIM = 1024
NUM_CLASSES = 27


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# FUSION CLASSIFIER
# =========================================================

class FusionClassifier(nn.Module):

    def __init__(
        self,
        input_dim=1024,
        num_classes=27
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        return self.network(x)


# =========================================================
# START
# =========================================================

print("=" * 70)
print("RGB + THERMAL FUSION BASELINE")
print("=" * 70)

print("\nDevice:", DEVICE)


# =========================================================
# LOAD DATA
# =========================================================

rgb_train = np.load(
    RGB_TRAIN
)

rgb_train_labels = np.load(
    RGB_TRAIN_LABELS
)

rgb_test = np.load(
    RGB_TEST
)

rgb_test_labels = np.load(
    RGB_TEST_LABELS
)


thermal_train = np.load(
    THERMAL_TRAIN
)

thermal_train_labels = np.load(
    THERMAL_TRAIN_LABELS
)

thermal_test = np.load(
    THERMAL_TEST
)

thermal_test_labels = np.load(
    THERMAL_TEST_LABELS
)


print("\nRGB train:", rgb_train.shape)
print("Thermal train:", thermal_train.shape)

print("\nRGB test:", rgb_test.shape)
print("Thermal test:", thermal_test.shape)


# =========================================================
# VERIFY ALIGNMENT
# =========================================================

if not np.array_equal(
    rgb_train_labels,
    thermal_train_labels
):

    raise ValueError(
        "RGB and Thermal training labels do not match."
    )


if not np.array_equal(
    rgb_test_labels,
    thermal_test_labels
):

    raise ValueError(
        "RGB and Thermal testing labels do not match."
    )


# =========================================================
# FUSION
# =========================================================

X_train = np.concatenate(
    [
        rgb_train,
        thermal_train
    ],
    axis=1
)

X_test = np.concatenate(
    [
        rgb_test,
        thermal_test
    ],
    axis=1
)


y_train = rgb_train_labels
y_test = rgb_test_labels


print("\nFused training features:")
print(X_train.shape)

print("\nFused testing features:")
print(X_test.shape)


# =========================================================
# TENSORS
# =========================================================

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


train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =========================================================
# MODEL
# =========================================================

model = FusionClassifier(
    input_dim=FUSION_DIM,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# TRAINING
# =========================================================

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for features, labels in train_loader:

        features = features.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            features
        )

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


    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    ) * 100


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Train Accuracy: "
        f"{epoch_accuracy:.2f}%"
    )


# =========================================================
# TEST
# =========================================================

print("\n" + "=" * 70)
print("FUSION TEST EVALUATION")
print("=" * 70)


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


# =========================================================
# METRICS
# =========================================================

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


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 70)
print("RGB + THERMAL FUSION TEST RESULTS")
print("=" * 70)

print(
    f"\nAccuracy       : {accuracy:.4f}"
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


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


os.makedirs(
    r"outputs\results",
    exist_ok=True
)

np.save(
    r"outputs\results\fusion_confusion_matrix.npy",
    cm
)


# =========================================================
# SAVE MODEL
# =========================================================

os.makedirs(
    r"outputs\models",
    exist_ok=True
)

torch.save(
    model.state_dict(),
    r"outputs\models\fusion_baseline.pth"
)


print("\nConfusion matrix saved:")
print(
    r"outputs\results\fusion_confusion_matrix.npy"
)

print("\nModel saved:")
print(
    r"outputs\models\fusion_baseline.pth"
)


print("\n" + "=" * 70)
print("FUSION BASELINE COMPLETED")
print("=" * 70)