import zipfile
import csv
import os

ZIP_PATH = r"data\raw\darkact_datasets.zip"
MANIFEST_PATH = r"data\splits\darkact_multimodal_manifest.csv"
OUTPUT_ROOT = r"data\processed\darkact"


print("=" * 70)
print("DARKACT MULTIMODAL DATASET EXTRACTION")
print("=" * 70)


if not os.path.exists(ZIP_PATH):
    print("ERROR: ZIP file not found.")
    exit()

if not os.path.exists(MANIFEST_PATH):
    print("ERROR: Manifest file not found.")
    exit()


# ---------------------------------------------------------
# Read manifest
# ---------------------------------------------------------

records = []

with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        records.append(row)


print("\nSamples to extract:", len(records))


# ---------------------------------------------------------
# Create extraction directory
# ---------------------------------------------------------

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ---------------------------------------------------------
# Extract only required RGB and Thermal videos
# ---------------------------------------------------------

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    extracted = 0
    skipped = 0

    for index, record in enumerate(records, start=1):

        rgb_zip_path = (
            "ARIM_v1/videos/" +
            record["rgb_path"]
        )

        thermal_zip_path = (
            "ARIM_v1/videos/" +
            record["thermal_path"]
        )

        paths = [
            rgb_zip_path,
            thermal_zip_path
        ]

        for zip_path in paths:

            # Remove ARIM_v1/videos/
            relative_path = zip_path.replace(
                "ARIM_v1/videos/",
                ""
            )

            output_path = os.path.join(
                OUTPUT_ROOT,
                relative_path
            )

            output_dir = os.path.dirname(
                output_path
            )

            os.makedirs(
                output_dir,
                exist_ok=True
            )

            if os.path.exists(output_path):

                skipped += 1

            else:

                with z.open(zip_path) as source:
                    with open(
                        output_path,
                        "wb"
                    ) as target:

                        target.write(
                            source.read()
                        )

                extracted += 1

        if index % 100 == 0:
            print(
                f"Processed {index}/{len(records)} samples..."
            )


print("\n" + "=" * 70)
print("EXTRACTION COMPLETED")
print("=" * 70)

print("New files extracted:", extracted)
print("Existing files skipped:", skipped)

print("\nDataset location:")
print(os.path.abspath(OUTPUT_ROOT))

print("\nNext step:")
print("We will verify RGB/Thermal videos and inspect frame properties.")