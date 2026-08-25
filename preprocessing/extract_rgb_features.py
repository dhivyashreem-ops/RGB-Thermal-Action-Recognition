import os
import sys

import torch
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from torch.utils.data import DataLoader
from datasets.darkact_dataset import DarkActDataset
from models.feature_extractor import ResNetFeatureExtractor


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MANIFEST = r"data\splits\darkact_multimodal_manifest.csv"
DATASET_ROOT = r"data\processed\darkact"

OUTPUT_DIR = r"data\processed\features"

NUM_FRAMES = 8
IMAGE_SIZE = 224

BATCH_SIZE = 2

MAX_SAMPLES = 100

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

print("=" * 70)
print("RGB FEATURE EXTRACTION")
print("=" * 70)

print("\nDevice:", DEVICE)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

dataset = DarkActDataset(
    manifest_path=MANIFEST,
    dataset_root=DATASET_ROOT,
    split="train",
    num_frames=NUM_FRAMES,
    image_size=IMAGE_SIZE,
    max_samples=MAX_SAMPLES
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ---------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------

model = ResNetFeatureExtractor()

model = model.to(DEVICE)

model.eval()


# ---------------------------------------------------------
# Extraction
# ---------------------------------------------------------

all_features = []
all_labels = []


with torch.no_grad():

    for batch_index, batch in enumerate(
        loader,
        start=1
    ):

        rgb = batch["rgb"].to(
            DEVICE
        )

        labels = batch["label"]

        batch_size, time_steps, channels, height, width = rgb.shape

        # Combine batch and temporal dimensions
        rgb = rgb.reshape(
            batch_size * time_steps,
            channels,
            height,
            width
        )

        frame_features = model(
            rgb
        )

        # Restore temporal dimension
        frame_features = frame_features.reshape(
            batch_size,
            time_steps,
            -1
        )

        # Temporal average
        video_features = frame_features.mean(
            dim=1
        )

        all_features.append(
            video_features.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )

        if batch_index % 10 == 0:

            print(
                f"Processed "
                f"{min(batch_index * BATCH_SIZE, len(dataset))}"
                f"/{len(dataset)} samples"
            )


# ---------------------------------------------------------
# Combine
# ---------------------------------------------------------

features = np.concatenate(
    all_features,
    axis=0
)

labels = np.concatenate(
    all_labels,
    axis=0
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

feature_file = os.path.join(
    OUTPUT_DIR,
    f"rgb_{'train'}_{MAX_SAMPLES}_features.npy"
)

label_file = os.path.join(
    OUTPUT_DIR,
    f"rgb_{'train'}_{MAX_SAMPLES}_labels.npy"
)

np.save(
    feature_file,
    features
)

np.save(
    label_file,
    labels
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RGB FEATURE EXTRACTION COMPLETED")
print("=" * 70)

print("\nFeatures shape:")
print(features.shape)

print("\nLabels shape:")
print(labels.shape)

print("\nFeature file:")
print(feature_file)

print("\nLabel file:")
print(label_file)