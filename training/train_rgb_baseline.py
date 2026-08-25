import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from datasets.darkact_dataset import DarkActDataset
from models.rgb_baseline import RGBBaseline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MANIFEST = r"data\splits\darkact_multimodal_manifest.csv"
DATASET_ROOT = r"data\processed\darkact"

BATCH_SIZE = 2
NUM_FRAMES = 8
IMAGE_SIZE = 224
NUM_CLASSES = 27
EPOCHS = 2
MAX_TRAIN_SAMPLES = 500
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

print("=" * 70)
print("RGB BASELINE TRAINING")
print("=" * 70)

print("\nDevice:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

train_dataset = DarkActDataset(
    manifest_path=MANIFEST,
    dataset_root=DATASET_ROOT,
    split="train",
    num_frames=NUM_FRAMES,
    image_size=IMAGE_SIZE,
    max_samples=MAX_TRAIN_SAMPLES
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = RGBBaseline(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ---------------------------------------------------------
# Loss and optimizer
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

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    for batch_index, batch in enumerate(
        train_loader,
        start=1
    ):

        rgb = batch["rgb"].to(
            DEVICE
        )

        labels = batch["label"].to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(rgb)

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

        predictions = (
            outputs.argmax(dim=1)
        )

        correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += labels.size(0)

        if batch_index % 50 == 0:

            print(
                f"Batch {batch_index} | "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    ) * 100

    print(
        f"\nEpoch {epoch + 1} Result"
    )

    print(
        f"Loss: {epoch_loss:.4f}"
    )

    print(
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
    r"\rgb_baseline.pth"
)

torch.save(
    model.state_dict(),
    model_path
)

print("\n" + "=" * 70)
print("RGB BASELINE TRAINING COMPLETED")
print("=" * 70)

print("\nModel saved to:")
print(model_path)