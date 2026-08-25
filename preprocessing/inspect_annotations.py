import zipfile
import os

ZIP_PATH = r"data\raw\darkact_datasets.zip"

print("=" * 70)
print("DARKACT ANNOTATION INSPECTION")
print("=" * 70)

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    annotation_files = [
        name for name in z.namelist()
        if "annotations/" in name.lower()
        and name.lower().endswith(".txt")
    ]

    print("\nAnnotation files found:")
    print("-" * 70)

    for file in annotation_files:
        print(file)

    print("\n" + "=" * 70)
    print("SAMPLE ANNOTATION CONTENT")
    print("=" * 70)

    for file in annotation_files:

        print("\n" + "-" * 70)
        print(file)
        print("-" * 70)

        try:
            with z.open(file) as f:

                lines = f.read().decode(
                    "utf-8",
                    errors="ignore"
                ).splitlines()

                for line in lines[:10]:
                    print(line)

        except Exception as e:
            print("Could not read file:", e)

print("\n" + "=" * 70)
print("Annotation inspection completed.")
print("=" * 70)