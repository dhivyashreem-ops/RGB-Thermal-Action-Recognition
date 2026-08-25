import random
import argparse
import os
import sys
import importlib.util

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

# =========================================================
# COMMAND-LINE SEED
# =========================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible training"
)

args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=" * 70)
print("REPRODUCIBLE EXPERIMENT")
print("=" * 70)
print(f"Random seed: {SEED}")

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
    sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# LOAD CMAF MODEL
# =========================================================

CMAF_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cmaf_net.py"
)

if not os.path.isfile(CMAF_PATH):
    raise FileNotFoundError(
        f"CMAF model not found:\n{CMAF_PATH}"
    )


spec = importlib.util.spec_from_file_location(
    "cmaf_net",
    CMAF_PATH
)

cmaf_module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    cmaf_module
)

CMAFNet = cmaf_module.CMAFNet


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

NUM_TOKENS = 16
NUM_HEADS = 4

NUM_CLASSES = 27

BATCH_SIZE = 64

EPOCHS = 50

LEARNING_RATE = 0.0005

VALIDATION_SIZE = 0.15

PATIENCE = 7

RANDOM_STATE = 42


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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


BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cmaf_v3_balanced_best.pth"
)


# =========================================================
# START
# =========================================================

print("=" * 70)
print("CMAF-NET V3 — CLASS-BALANCED TRAINING")
print("=" * 70)

print("\nDevice:")
print(DEVICE)

print("\nProject root:")
print(PROJECT_ROOT)


# =========================================================
# CHECK FILES
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
# LOAD RGB
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
# LOAD THERMAL
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
# DATA SHAPES
# =========================================================

print("\nRGB train:")
print(rgb_train.shape)

print("Thermal train:")
print(thermal_train.shape)

print("\nRGB test:")
print(rgb_test.shape)

print("Thermal test:")
print(thermal_test.shape)


# =========================================================
# BASIC VALIDATION
# =========================================================

if len(rgb_train) != len(thermal_train):

    raise ValueError(
        "RGB and Thermal training sample counts differ."
    )


if len(rgb_test) != len(thermal_test):

    raise ValueError(
        "RGB and Thermal test sample counts differ."
    )


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
        "RGB and Thermal test labels do not match."
    )


print(
    "\nRGB/Thermal label alignment: VERIFIED"
)


# =========================================================
# CHECK NUMBER OF CLASSES
# =========================================================

unique_classes = np.unique(
    rgb_train_labels
)

print("\nClasses found:")
print(unique_classes)

print(
    "\nNumber of classes:",
    len(unique_classes)
)


if len(unique_classes) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"found {len(unique_classes)}."
    )


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

class_counts = np.bincount(
    rgb_train_labels,
    minlength=NUM_CLASSES
)

for class_id, count in enumerate(
    class_counts
):

    print(
        f"Class {class_id:02d}: {count}"
    )


# =========================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# =========================================================

indices = np.arange(
    len(rgb_train)
)


train_indices, val_indices = train_test_split(

    indices,

    test_size=VALIDATION_SIZE,

    random_state=RANDOM_STATE,

    stratify=rgb_train_labels
)


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "\nTraining samples:",
    len(train_indices)
)

print(
    "Validation samples:",
    len(val_indices)
)

print(
    "Test samples:",
    len(rgb_test)
)


# =========================================================
# SPLIT DATA
# =========================================================

rgb_train_split = rgb_train[
    train_indices
]

thermal_train_split = thermal_train[
    train_indices
]

y_train_split = rgb_train_labels[
    train_indices
]


rgb_val_split = rgb_train[
    val_indices
]

thermal_val_split = thermal_train[
    val_indices
]

y_val_split = rgb_train_labels[
    val_indices
]


# =========================================================
# CONVERT TO TENSORS
# =========================================================

rgb_train_tensor = torch.tensor(
    rgb_train_split,
    dtype=torch.float32
)

thermal_train_tensor = torch.tensor(
    thermal_train_split,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_split,
    dtype=torch.long
)


rgb_val_tensor = torch.tensor(
    rgb_val_split,
    dtype=torch.float32
)

thermal_val_tensor = torch.tensor(
    thermal_val_split,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val_split,
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
# DATASETS
# =========================================================

train_dataset = TensorDataset(
    rgb_train_tensor,
    thermal_train_tensor,
    y_train_tensor
)


val_dataset = TensorDataset(
    rgb_val_tensor,
    thermal_val_tensor,
    y_val_tensor
)


generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=" * 70)
print("REPRODUCIBLE EXPERIMENT")
print("=" * 70)

print(
    f"Random seed: {SEED}"
)
# =========================================================
# CLASS WEIGHTS
# =========================================================

train_class_counts = np.bincount(
    y_train_split,
    minlength=NUM_CLASSES
)


# Inverse-frequency weighting

class_weights = (
    len(y_train_split)
    /
    (
        NUM_CLASSES
        *
        train_class_counts
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)


print("\n" + "=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

for class_id, weight in enumerate(
    class_weights
):

    print(
        f"Class {class_id:02d}: "
        f"{weight.item():.4f}"
    )


class_weights = class_weights.to(
    DEVICE
)


# =========================================================
# MODEL
# =========================================================

model = CMAFNet(

    rgb_dim=RGB_DIM,

    thermal_dim=THERMAL_DIM,

    hidden_dim=HIDDEN_DIM,

    num_tokens=NUM_TOKENS,

    num_heads=NUM_HEADS,

    num_classes=NUM_CLASSES
)


model = model.to(
    DEVICE
)


print("\nModel created successfully.")


# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4
)


# =========================================================
# SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=3
)


# =========================================================
# VALIDATION FUNCTION
# =========================================================

def evaluate_validation(
    model,
    loader
):

    model.eval()

    all_predictions = []

    all_labels = []

    total_loss = 0.0

    total_samples = 0


    with torch.no_grad():

        for (
            rgb,
            thermal,
            labels
        ) in loader:

            rgb = rgb.to(
                DEVICE
            )

            thermal = thermal.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                rgb,
                thermal
            )


            loss = criterion(
                outputs,
                labels
            )


            total_loss += (
                loss.item()
                *
                labels.size(0)
            )

            total_samples += (
                labels.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


    all_predictions = np.array(
        all_predictions
    )

    all_labels = np.array(
        all_labels
    )


    avg_loss = (
        total_loss
        /
        total_samples
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    return (
        avg_loss,
        accuracy,
        macro_f1
    )


# =========================================================
# TRAINING
# =========================================================

print("\n" + "=" * 70)
print("CMAF-NET V3 BALANCED TRAINING")
print("=" * 70)


best_val_f1 = -1.0

epochs_without_improvement = 0


for epoch in range(
    EPOCHS
):

    model.train()


    running_loss = 0.0

    correct = 0

    total = 0


    for (
        rgb,
        thermal,
        labels
    ) in train_loader:

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


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        running_loss += (
            loss.item()
            *
            labels.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += (
            labels.size(0)
        )


    train_loss = (
        running_loss
        /
        total
    )


    train_accuracy = (
        correct
        /
        total
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    (
        val_loss,
        val_accuracy,
        val_macro_f1
    ) = evaluate_validation(
        model,
        val_loader
    )


    scheduler.step(
        val_macro_f1
    )


    current_lr = optimizer.param_groups[0][
        "lr"
    ]


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}% | "
        f"Val Macro-F1: {val_macro_f1:.4f} | "
        f"LR: {current_lr:.6f}"
    )


    # =====================================================
    # BEST MODEL
    # =====================================================

    if val_macro_f1 > best_val_f1:

        best_val_f1 = val_macro_f1

        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )


        print(
            f"  --> BEST MODEL SAVED "
            f"(Val Macro-F1: "
            f"{best_val_f1:.4f})"
        )


    else:

        epochs_without_improvement += 1


    # =====================================================
    # EARLY STOPPING
    # =====================================================

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        print(
            f"No improvement for "
            f"{PATIENCE} epochs."
        )

        break


# =========================================================
# LOAD BEST MODEL
# =========================================================

print("\n" + "=" * 70)
print("LOADING BEST CMAF MODEL")
print("=" * 70)

print(
    "\nBest validation Macro-F1:",
    f"{best_val_f1:.4f}"
)

print(
    "\nLoading:",
    BEST_MODEL_PATH
)


model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )
)


model.eval()


# =========================================================
# TEST EVALUATION
# =========================================================

print("\n" + "=" * 70)
print("CMAF-NET TEST EVALUATION")
print("=" * 70)


all_predictions = []

all_rgb_attention = []

all_thermal_attention = []


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


        (
            outputs,
            rgb_attention,
            thermal_attention
        ) = model(
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


        all_rgb_attention.append(
            rgb_attention.cpu().numpy()
        )


        all_thermal_attention.append(
            thermal_attention.cpu().numpy()
        )


# =========================================================
# RESULTS
# =========================================================

y_pred = np.array(
    all_predictions
)


rgb_attention = np.concatenate(
    all_rgb_attention,
    axis=0
)


thermal_attention = np.concatenate(
    all_thermal_attention,
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
# TEST RESULTS
# =========================================================

print("\n" + "=" * 70)
print("CMAF-NET BALANCED TEST RESULTS")
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

print("\n" + "=" * 70)
print("CROSS-MODAL ATTENTION ANALYSIS")
print("=" * 70)


print(
    "\nRGB -> Thermal attention shape:"
)

print(
    rgb_attention.shape
)


print(
    "\nThermal -> RGB attention shape:"
)

print(
    thermal_attention.shape
)


print(
    "\nMean RGB -> Thermal attention:"
)

print(
    rgb_attention.mean()
)


print(
    "\nMean Thermal -> RGB attention:"
)

print(
    thermal_attention.mean()
)


# =========================================================
# SAVE RESULTS
# =========================================================

np.save(
    os.path.join(
        RESULT_DIR,
        "cmaf_balanced_predictions.npy"
    ),
    y_pred
)


np.save(
    os.path.join(
        RESULT_DIR,
        "cmaf_balanced_confusion_matrix.npy"
    ),
    confusion_matrix(
        rgb_test_labels,
        y_pred
    )
)


np.save(
    os.path.join(
        RESULT_DIR,
        "cmaf_balanced_rgb_attention.npy"
    ),
    rgb_attention
)


np.save(
    os.path.join(
        RESULT_DIR,
        "cmaf_balanced_thermal_attention.npy"
    ),
    thermal_attention
)


# =========================================================
# SAVE FINAL MODEL
# =========================================================

final_model_path = os.path.join(
    MODEL_DIR,
    "cmaf_v3_balanced_final.pth"
)


torch.save(
    model.state_dict(),
    final_model_path
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics_path = os.path.join(
    RESULT_DIR,
    "cmaf_v3_balanced_metrics.txt"
)


with open(
    metrics_path,
    "w"
) as f:

    f.write(
        "CMAF-NET V3 BALANCED RESULTS\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        f"Best Validation Macro-F1: "
        f"{best_val_f1:.4f}\n"
    )

    f.write(
        f"Test Accuracy: "
        f"{accuracy:.4f}\n"
    )

    f.write(
        f"Macro Precision: "
        f"{precision:.4f}\n"
    )

    f.write(
        f"Macro Recall: "
        f"{recall:.4f}\n"
    )

    f.write(
        f"Macro F1: "
        f"{macro_f1:.4f}\n"
    )

    f.write(
        f"Weighted F1: "
        f"{weighted_f1:.4f}\n"
    )


# =========================================================
# COMPLETED
# =========================================================

print("\n" + "=" * 70)
print("CMAF-NET BALANCED EVALUATION COMPLETED")
print("=" * 70)

print(
    "\nBest model:"
)

print(
    BEST_MODEL_PATH
)

print(
    "\nFinal model:"
)

print(
    final_model_path
)

print(
    "\nResults:"
)

print(
    RESULT_DIR
)

print("\n" + "=" * 70)