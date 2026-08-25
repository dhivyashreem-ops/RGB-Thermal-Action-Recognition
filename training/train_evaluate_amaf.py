import os
import sys
import importlib.util

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
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# =========================================================
# LOAD AMAF-NET DIRECTLY
# =========================================================

AMAF_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "amaf_net.py"
)

if not os.path.isfile(AMAF_PATH):

    raise FileNotFoundError(
        f"AMAF model not found:\n{AMAF_PATH}"
    )


spec = importlib.util.spec_from_file_location(
    "amaf_net",
    AMAF_PATH
)

amaf_module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    amaf_module
)

AMAFNet = amaf_module.AMAFNet


# =========================================================
# FEATURE DIRECTORY
# =========================================================

FEATURE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "features"
)


# =========================================================
# FEATURE FILES
# =========================================================

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


# =========================================================
# CONFIGURATION
# =========================================================

RGB_DIM = 512
THERMAL_DIM = 512

HIDDEN_DIM = 256
NUM_CLASSES = 27

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# FILE CHECK
# =========================================================

required_files = [
    RGB_TRAIN,
    RGB_TRAIN_LABELS,
    RGB_TEST,
    RGB_TEST_LABELS,
    THERMAL_TRAIN,
    THERMAL_TRAIN_LABELS,
    THERMAL_TEST,
    THERMAL_TEST_LABELS
]


for file_path in required_files:

    if not os.path.isfile(file_path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )


# =========================================================
# START
# =========================================================

print("=" * 70)
print("AMAF-NET — ADAPTIVE MODALITY ATTENTION FUSION")
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nProject root:")
print(PROJECT_ROOT)


# =========================================================
# LOAD RGB FEATURES
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


# =========================================================
# LOAD THERMAL FEATURES
# =========================================================

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


# =========================================================
# DISPLAY DATA SHAPES
# =========================================================

print("\nRGB train:")
print(rgb_train.shape)

print("RGB train labels:")
print(rgb_train_labels.shape)

print("\nThermal train:")
print(thermal_train.shape)

print("Thermal train labels:")
print(thermal_train_labels.shape)

print("\nRGB test:")
print(rgb_test.shape)

print("RGB test labels:")
print(rgb_test_labels.shape)

print("\nThermal test:")
print(thermal_test.shape)

print("Thermal test labels:")
print(thermal_test_labels.shape)


# =========================================================
# DATA VALIDATION
# =========================================================

if rgb_train.shape[0] != len(rgb_train_labels):

    raise ValueError(
        "RGB training feature/label count mismatch."
    )


if thermal_train.shape[0] != len(thermal_train_labels):

    raise ValueError(
        "Thermal training feature/label count mismatch."
    )


if rgb_test.shape[0] != len(rgb_test_labels):

    raise ValueError(
        "RGB testing feature/label count mismatch."
    )


if thermal_test.shape[0] != len(thermal_test_labels):

    raise ValueError(
        "Thermal testing feature/label count mismatch."
    )


if rgb_train.shape[1] != RGB_DIM:

    raise ValueError(
        f"RGB feature dimension should be "
        f"{RGB_DIM}, but found "
        f"{rgb_train.shape[1]}"
    )


if thermal_train.shape[1] != THERMAL_DIM:

    raise ValueError(
        f"Thermal feature dimension should be "
        f"{THERMAL_DIM}, but found "
        f"{thermal_train.shape[1]}"
    )


# =========================================================
# VERIFY RGB / THERMAL ALIGNMENT
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


print("\nRGB/Thermal label alignment: VERIFIED")


# =========================================================
# CONVERT TO TENSORS
# =========================================================

rgb_train_tensor = torch.tensor(
    rgb_train,
    dtype=torch.float32
)

thermal_train_tensor = torch.tensor(
    thermal_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    rgb_train_labels,
    dtype=torch.long
)


rgb_test_tensor = torch.tensor(
    rgb_test,
    dtype=torch.float32
)

thermal_test_tensor = torch.tensor(
    thermal_test,
    dtype=torch.float32
)


# =========================================================
# DATASET
# =========================================================

train_dataset = TensorDataset(
    rgb_train_tensor,
    thermal_train_tensor,
    y_train_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# =========================================================
# MODEL
# =========================================================

model = AMAFNet(
    rgb_dim=RGB_DIM,
    thermal_dim=THERMAL_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
)


model = model.to(
    DEVICE
)


print("\nModel:")
print(model)


# =========================================================
# LOSS AND OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# TRAINING
# =========================================================

print("\n" + "=" * 70)
print("AMAF-NET TRAINING")
print("=" * 70)


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for rgb, thermal, labels in train_loader:

        rgb = rgb.to(
            DEVICE
        )

        thermal = thermal.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = model(
            rgb,
            thermal
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
# TEST EVALUATION
# =========================================================

print("\n" + "=" * 70)
print("AMAF-NET TEST EVALUATION")
print("=" * 70)


model.eval()


all_predictions = []

all_attention = []


with torch.no_grad():

    for start in range(
        0,
        len(rgb_test_tensor),
        BATCH_SIZE
    ):

        rgb_batch = rgb_test_tensor[
            start:start + BATCH_SIZE
        ].to(DEVICE)


        thermal_batch = thermal_test_tensor[
            start:start + BATCH_SIZE
        ].to(DEVICE)


        outputs, attention = model(
            rgb_batch,
            thermal_batch,
            return_attention=True
        )


        predictions = outputs.argmax(
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_attention.append(
            attention.cpu().numpy()
        )


# =========================================================
# CONVERT RESULTS
# =========================================================

y_pred = np.array(
    all_predictions
)


attention_values = np.concatenate(
    all_attention,
    axis=0
)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    rgb_test_labels,
    y_pred
)


precision = precision_score(
    rgb_test_labels,
    y_pred,
    average="macro",
    zero_division=0
)


recall = recall_score(
    rgb_test_labels,
    y_pred,
    average="macro",
    zero_division=0
)


macro_f1 = f1_score(
    rgb_test_labels,
    y_pred,
    average="macro",
    zero_division=0
)


weighted_f1 = f1_score(
    rgb_test_labels,
    y_pred,
    average="weighted",
    zero_division=0
)


# =========================================================
# MAIN RESULTS
# =========================================================

print("\n" + "=" * 70)
print("AMAF-NET TEST RESULTS")
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


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")


print(
    classification_report(
        rgb_test_labels,
        y_pred,
        zero_division=0
    )
)


# =========================================================
# ATTENTION ANALYSIS
# =========================================================

mean_rgb_weight = (
    attention_values[:, 0].mean()
)


mean_thermal_weight = (
    attention_values[:, 1].mean()
)


print("\n" + "=" * 70)
print("MODALITY ATTENTION ANALYSIS")
print("=" * 70)


print(
    f"\nMean RGB attention     : "
    f"{mean_rgb_weight:.4f}"
)


print(
    f"Mean Thermal attention : "
    f"{mean_thermal_weight:.4f}"
)


print(
    f"\nAttention sum          : "
    f"{mean_rgb_weight + mean_thermal_weight:.4f}"
)


# =========================================================
# ATTENTION DISTRIBUTION
# =========================================================

rgb_dominant = (
    attention_values[:, 0]
    >
    attention_values[:, 1]
).sum()


thermal_dominant = (
    attention_values[:, 1]
    >
    attention_values[:, 0]
).sum()


equal_attention = (
    attention_values[:, 0]
    ==
    attention_values[:, 1]
).sum()


total_samples = len(
    attention_values
)


print("\nRGB-dominant samples:")
print(
    f"{rgb_dominant} "
    f"({rgb_dominant / total_samples * 100:.2f}%)"
)


print("\nThermal-dominant samples:")
print(
    f"{thermal_dominant} "
    f"({thermal_dominant / total_samples * 100:.2f}%)"
)


print("\nEqual-attention samples:")
print(
    f"{equal_attention} "
    f"({equal_attention / total_samples * 100:.2f}%)"
)


# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "results"
)


MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "models"
)


os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    rgb_test_labels,
    y_pred
)


confusion_file = os.path.join(
    RESULT_DIR,
    "amaf_confusion_matrix.npy"
)


np.save(
    confusion_file,
    cm
)


# =========================================================
# ATTENTION WEIGHTS
# =========================================================

attention_file = os.path.join(
    RESULT_DIR,
    "amaf_attention_weights.npy"
)


np.save(
    attention_file,
    attention_values
)


# =========================================================
# PREDICTIONS
# =========================================================

prediction_file = os.path.join(
    RESULT_DIR,
    "amaf_predictions.npy"
)


np.save(
    prediction_file,
    y_pred
)


# =========================================================
# SAVE MODEL
# =========================================================

model_file = os.path.join(
    MODEL_DIR,
    "amaf_net.pth"
)


torch.save(
    model.state_dict(),
    model_file
)


# =========================================================
# FINAL FILE INFORMATION
# =========================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)


print("\nModel:")
print(model_file)


print("\nConfusion matrix:")
print(confusion_file)


print("\nAttention weights:")
print(attention_file)


print("\nPredictions:")
print(prediction_file)


print("\n" + "=" * 70)
print("AMAF-NET EVALUATION COMPLETED")
print("=" * 70)