import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from models.fusion_baseline import FusionBaseline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RGB_FEATURES = (
    r"data\processed\features"
    r"\rgb_train_100_features.npy"
)

RGB_LABELS = (
    r"data\processed\features"
    r"\rgb_train_100_labels.npy"
)

THERMAL_FEATURES = (
    r"data\processed\features"
    r"\thermal_train_100_features.npy"
)

THERMAL_LABELS = (
    r"data\processed\features"
    r"\thermal_train_100_labels.npy"
)


BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

NUM_CLASSES = 27


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Load features
# ---------------------------------------------------------

print("=" * 70)
print("RGB + THERMAL FUSION TRAINING")
print("=" * 70)

print("\nDevice:", DEVICE)


rgb_features = np.load(
    RGB_FEATURES
)

rgb_labels = np.load(
    RGB_LABELS
)

thermal_features = np.load(
    THERMAL_FEATURES
)

thermal_labels = np.load(
    THERMAL_LABELS
)


print("\nRGB features:")
print(rgb_features.shape)

print("\nThermal features:")
print(thermal_features.shape)


# ---------------------------------------------------------
# Verify labels
# ---------------------------------------------------------

if not np.array_equal(
    rgb_labels,
    thermal_labels
):

    raise ValueError(
        "RGB and Thermal labels do not match!"
    )


labels = rgb_labels


# ---------------------------------------------------------
# Convert to tensors
# ---------------------------------------------------------

rgb_tensor = torch.tensor(
    rgb_features,
    dtype=torch.float32
)

thermal_tensor = torch.tensor(
    thermal_features,
    dtype=torch.float32
)

label_tensor = torch.tensor(
    labels,
    dtype=torch.long
)


dataset = TensorDataset(
    rgb_tensor,
    thermal_tensor,
    label_tensor
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = FusionBaseline(
    rgb_dim=512,
    thermal_dim=512,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ---------------------------------------------------------
# Loss + optimizer
# ---------------------------------------------------------

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

    for rgb, thermal, labels_batch in loader:

        rgb = rgb.to(DEVICE)
        thermal = thermal.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            rgb,
            thermal
        )

        loss = criterion(
            outputs,
            labels_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * labels_batch.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels_batch
        ).sum().item()

        total += labels_batch.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    ) * 100

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

os.makedirs(
    r"outputs\models",
    exist_ok=True
)

model_path = (
    r"outputs\models"
    r"\fusion_baseline.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


print("\n" + "=" * 70)
print("FUSION TRAINING COMPLETED")
print("=" * 70)

print("\nModel saved:")
print(model_path)