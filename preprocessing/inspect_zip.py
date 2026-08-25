import zipfile
import os

ZIP_PATH = r"data\raw\darkact_datasets.zip"

print("=" * 70)
print("DARKACT ZIP STRUCTURE INSPECTION")
print("=" * 70)

if not os.path.exists(ZIP_PATH):
    print("ERROR: ZIP file not found!")
    print(ZIP_PATH)
    exit()

file_size = os.path.getsize(ZIP_PATH) / (1024 * 1024)

print(f"\nZIP file: {ZIP_PATH}")
print(f"ZIP size: {file_size:.2f} MB")

with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:

    files = zip_file.namelist()

    print(f"\nTotal entries in ZIP: {len(files)}")

    print("\nFirst 100 entries:")
    print("-" * 70)

    for item in files[:100]:
        print(item)

print("\n" + "=" * 70)
print("ZIP inspection completed.")
print("=" * 70)