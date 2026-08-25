import zipfile
import csv
import os
from collections import Counter

ZIP_PATH = r"data\raw\darkact_datasets.zip"

TRAIN_ANNOTATION = "ARIM_v1/annotations/_multimodal_train_annotations.txt"
TEST_ANNOTATION = "ARIM_v1/annotations/_multimodal_test_annotations.txt"

OUTPUT_DIR = r"data\splits"


def read_annotations(zip_file, annotation_file, split):
    records = []

    with zip_file.open(annotation_file) as f:
        lines = f.read().decode(
            "utf-8",
            errors="ignore"
        ).splitlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 3:
            print(
                f"WARNING: Unexpected format in "
                f"{annotation_file}, line {line_number}"
            )
            print(line)
            continue

        rgb_path = parts[0]
        thermal_path = parts[1]
        label = int(parts[2])

        records.append({
            "split": split,
            "rgb_path": rgb_path,
            "thermal_path": thermal_path,
            "label": label
        })

    return records


print("=" * 70)
print("DARKACT MULTIMODAL MANIFEST CREATION")
print("=" * 70)

if not os.path.exists(ZIP_PATH):
    print("ERROR: Dataset ZIP not found:")
    print(ZIP_PATH)
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    zip_files = set(z.namelist())

    print("\nReading training annotations...")

    train_records = read_annotations(
        z,
        TRAIN_ANNOTATION,
        "train"
    )

    print("Training samples:", len(train_records))

    print("\nReading testing annotations...")

    test_records = read_annotations(
        z,
        TEST_ANNOTATION,
        "test"
    )

    print("Testing samples:", len(test_records))

    records = train_records + test_records

    print("\nTotal multimodal samples:", len(records))

    # ---------------------------------------------------------
    # Verify RGB and Thermal files
    # ---------------------------------------------------------

    missing_rgb = []
    missing_thermal = []

    valid_records = []

    for record in records:

        rgb_exists = (
            "ARIM_v1/videos/" + record["rgb_path"]
            in zip_files
        )

        thermal_exists = (
            "ARIM_v1/videos/" + record["thermal_path"]
            in zip_files
        )

        if not rgb_exists:
            missing_rgb.append(record["rgb_path"])

        if not thermal_exists:
            missing_thermal.append(record["thermal_path"])

        if rgb_exists and thermal_exists:
            valid_records.append(record)

    print("\n" + "-" * 70)
    print("FILE VERIFICATION")
    print("-" * 70)

    print("Valid paired samples:", len(valid_records))
    print("Missing RGB files:", len(missing_rgb))
    print("Missing Thermal files:", len(missing_thermal))

    # ---------------------------------------------------------
    # Class statistics
    # ---------------------------------------------------------

    train_classes = Counter(
        r["label"]
        for r in train_records
    )

    test_classes = Counter(
        r["label"]
        for r in test_records
    )

    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION")
    print("-" * 70)

    print("\nTraining classes:")
    for label, count in sorted(train_classes.items()):
        print(f"Class {label:2d}: {count}")

    print("\nTesting classes:")
    for label, count in sorted(test_classes.items()):
        print(f"Class {label:2d}: {count}")

    # ---------------------------------------------------------
    # Save manifest
    # ---------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        "darkact_multimodal_manifest.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "rgb_path",
                "thermal_path",
                "label"
            ]
        )

        writer.writeheader()
        writer.writerows(valid_records)

    print("\n" + "=" * 70)
    print("MANIFEST CREATED")
    print("=" * 70)

    print("Output:")
    print(output_file)

    print("\nTotal valid paired samples:", len(valid_records))

    if missing_rgb or missing_thermal:
        print("\nWARNING:")
        print("Some files referenced by annotations were not found.")

    print("\nCompleted.")