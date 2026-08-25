import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report
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


RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "results"
)

PLOT_DIR = os.path.join(
    RESULT_DIR,
    "plots"
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# =========================================================
# FILES
# =========================================================

PREDICTIONS_FILE = os.path.join(
    RESULT_DIR,
    "cmaf_balanced_predictions.npy"
)

CONFUSION_FILE = os.path.join(
    RESULT_DIR,
    "cmaf_balanced_confusion_matrix.npy"
)

RGB_ATTENTION_FILE = os.path.join(
    RESULT_DIR,
    "cmaf_balanced_rgb_attention.npy"
)

THERMAL_ATTENTION_FILE = os.path.join(
    RESULT_DIR,
    "cmaf_balanced_thermal_attention.npy"
)

RGB_LABEL_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "features",
    "rgb_test_labels.npy"
)


# =========================================================
# CHECK FILES
# =========================================================

required_files = [
    PREDICTIONS_FILE,
    CONFUSION_FILE,
    RGB_ATTENTION_FILE,
    THERMAL_ATTENTION_FILE,
    RGB_LABEL_FILE
]


for file_path in required_files:

    if not os.path.isfile(file_path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )


print("=" * 70)
print("FINAL CMAF-NET RESULTS ANALYSIS")
print("=" * 70)


# =========================================================
# LOAD DATA
# =========================================================

y_pred = np.load(
    PREDICTIONS_FILE
)

y_true = np.load(
    RGB_LABEL_FILE
)

conf_matrix = np.load(
    CONFUSION_FILE
)

rgb_attention = np.load(
    RGB_ATTENTION_FILE
)

thermal_attention = np.load(
    THERMAL_ATTENTION_FILE
)


print("\nPrediction shape:")
print(y_pred.shape)

print("\nGround-truth shape:")
print(y_true.shape)

print("\nConfusion matrix shape:")
print(conf_matrix.shape)

print("\nRGB attention shape:")
print(rgb_attention.shape)

print("\nThermal attention shape:")
print(thermal_attention.shape)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    y_true,
    y_pred,
    output_dict=True,
    zero_division=0
)


# =========================================================
# PER-CLASS F1
# =========================================================

class_ids = list(
    range(27)
)

f1_scores = []

precisions = []

recalls = []

supports = []


for class_id in class_ids:

    class_result = report[
        str(class_id)
    ]

    precisions.append(
        class_result["precision"]
    )

    recalls.append(
        class_result["recall"]
    )

    f1_scores.append(
        class_result["f1-score"]
    )

    supports.append(
        class_result["support"]
    )


# =========================================================
# SAVE PER-CLASS TABLE
# =========================================================

per_class_file = os.path.join(
    RESULT_DIR,
    "cmaf_per_class_results.csv"
)


with open(
    per_class_file,
    "w"
) as f:

    f.write(
        "class,precision,recall,f1,support\n"
    )

    for i in class_ids:

        f.write(
            f"{i},"
            f"{precisions[i]:.6f},"
            f"{recalls[i]:.6f},"
            f"{f1_scores[i]:.6f},"
            f"{supports[i]}\n"
        )


# =========================================================
# 1. CONFUSION MATRIX
# =========================================================

plt.figure(
    figsize=(14, 12)
)

plt.imshow(
    conf_matrix,
    interpolation="nearest"
)

plt.title(
    "CMAF-Net Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.colorbar()

plt.xticks(
    class_ids
)

plt.yticks(
    class_ids
)

plt.tight_layout()


confusion_path = os.path.join(
    PLOT_DIR,
    "cmaf_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 2. PER-CLASS F1
# =========================================================

plt.figure(
    figsize=(14, 6)
)

plt.bar(
    class_ids,
    f1_scores
)

plt.xlabel(
    "Action Class"
)

plt.ylabel(
    "F1 Score"
)

plt.title(
    "CMAF-Net Per-Class F1 Score"
)

plt.xticks(
    class_ids
)

plt.ylim(
    0,
    1
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


f1_path = os.path.join(
    PLOT_DIR,
    "cmaf_per_class_f1.png"
)

plt.savefig(
    f1_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 3. PRECISION / RECALL / F1
# =========================================================

x = np.arange(
    len(class_ids)
)

width = 0.25


plt.figure(
    figsize=(16, 7)
)

plt.bar(
    x - width,
    precisions,
    width,
    label="Precision"
)

plt.bar(
    x,
    recalls,
    width,
    label="Recall"
)

plt.bar(
    x + width,
    f1_scores,
    width,
    label="F1"
)

plt.xlabel(
    "Action Class"
)

plt.ylabel(
    "Score"
)

plt.title(
    "CMAF-Net Per-Class Precision, Recall and F1"
)

plt.xticks(
    x,
    class_ids
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


prf_path = os.path.join(
    PLOT_DIR,
    "cmaf_per_class_metrics.png"
)

plt.savefig(
    prf_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 4. RGB -> THERMAL ATTENTION
# =========================================================

rgb_mean_attention = (
    rgb_attention.mean(
        axis=(0, 1)
    )
)


plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    rgb_mean_attention,
    interpolation="nearest"
)

plt.title(
    "Mean RGB → Thermal Cross-Modal Attention"
)

plt.xlabel(
    "Thermal Tokens"
)

plt.ylabel(
    "RGB Tokens"
)

plt.colorbar()

plt.tight_layout()


rgb_attention_path = os.path.join(
    PLOT_DIR,
    "rgb_to_thermal_attention.png"
)

plt.savefig(
    rgb_attention_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 5. THERMAL -> RGB ATTENTION
# =========================================================

thermal_mean_attention = (
    thermal_attention.mean(
        axis=(0, 1)
    )
)


plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    thermal_mean_attention,
    interpolation="nearest"
)

plt.title(
    "Mean Thermal → RGB Cross-Modal Attention"
)

plt.xlabel(
    "RGB Tokens"
)

plt.ylabel(
    "Thermal Tokens"
)

plt.colorbar()

plt.tight_layout()


thermal_attention_path = os.path.join(
    PLOT_DIR,
    "thermal_to_rgb_attention.png"
)

plt.savefig(
    thermal_attention_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 6. ATTENTION ENTROPY
# =========================================================

def attention_entropy(
    attention
):

    eps = 1e-12

    entropy = -np.sum(
        attention
        *
        np.log(
            attention + eps
        ),
        axis=-1
    )

    return entropy


rgb_entropy = attention_entropy(
    rgb_attention
)

thermal_entropy = attention_entropy(
    thermal_attention
)


print("\n" + "=" * 70)
print("ATTENTION ENTROPY")
print("=" * 70)

print(
    "\nRGB -> Thermal mean entropy:"
)

print(
    rgb_entropy.mean()
)

print(
    "\nThermal -> RGB mean entropy:"
)

print(
    thermal_entropy.mean()
)


# =========================================================
# 7. ATTENTION CONCENTRATION
# =========================================================

rgb_max_attention = np.max(
    rgb_attention,
    axis=-1
)

thermal_max_attention = np.max(
    thermal_attention,
    axis=-1
)


print("\n" + "=" * 70)
print("ATTENTION CONCENTRATION")
print("=" * 70)

print(
    "\nRGB -> Thermal mean maximum attention:"
)

print(
    rgb_max_attention.mean()
)

print(
    "\nThermal -> RGB mean maximum attention:"
)

print(
    thermal_max_attention.mean()
)


# =========================================================
# 8. SUMMARY FILE
# =========================================================

summary_file = os.path.join(
    RESULT_DIR,
    "cmaf_final_analysis_summary.txt"
)


with open(
    summary_file,
    "w"
) as f:

    f.write(
        "CMAF-NET FINAL ANALYSIS\n"
    )

    f.write(
        "=======================\n\n"
    )

    f.write(
        f"Mean RGB->Thermal attention: "
        f"{rgb_attention.mean():.6f}\n"
    )

    f.write(
        f"Mean Thermal->RGB attention: "
        f"{thermal_attention.mean():.6f}\n"
    )

    f.write(
        f"RGB->Thermal entropy: "
        f"{rgb_entropy.mean():.6f}\n"
    )

    f.write(
        f"Thermal->RGB entropy: "
        f"{thermal_entropy.mean():.6f}\n"
    )

    f.write(
        f"RGB->Thermal max attention: "
        f"{rgb_max_attention.mean():.6f}\n"
    )

    f.write(
        f"Thermal->RGB max attention: "
        f"{thermal_max_attention.mean():.6f}\n"
    )


# =========================================================
# COMPLETED
# =========================================================

print("\n" + "=" * 70)
print("FINAL CMAF ANALYSIS COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(
    "\nConfusion matrix:"
)

print(
    confusion_path
)

print(
    "\nPer-class F1:"
)

print(
    f1_path
)

print(
    "\nPer-class metrics:"
)

print(
    prf_path
)

print(
    "\nRGB -> Thermal attention:"
)

print(
    rgb_attention_path
)

print(
    "\nThermal -> RGB attention:"
)

print(
    thermal_attention_path
)

print(
    "\nSummary:"
)

print(
    summary_file
)

print("\n" + "=" * 70)