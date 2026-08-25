import csv
import os
import cv2
import numpy as np


MANIFEST_PATH = r"data\splits\darkact_multimodal_manifest.csv"
DATASET_ROOT = r"data\processed\darkact"
OUTPUT_DIR = r"outputs\figures"

NUM_SAMPLES = 5


def read_frame(video_path, frame_number):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    success, frame = cap.read()

    cap.release()

    if not success:
        return None

    return frame


def resize_frame(frame):

    return cv2.resize(
        frame,
        (640, 480)
    )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    records = list(
        csv.DictReader(f)
    )


print("=" * 70)
print("CREATING RGB-THERMAL SAMPLE FRAMES")
print("=" * 70)


for index, record in enumerate(
    records[:NUM_SAMPLES],
    start=1
):

    rgb_path = os.path.join(
        DATASET_ROOT,
        record["rgb_path"]
    )

    thermal_path = os.path.join(
        DATASET_ROOT,
        record["thermal_path"]
    )

    # Get frame count from RGB video
    cap = cv2.VideoCapture(rgb_path)

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    # Select middle frame
    frame_number = max(
        0,
        frame_count // 2
    )

    rgb_frame = read_frame(
        rgb_path,
        frame_number
    )

    thermal_frame = read_frame(
        thermal_path,
        frame_number
    )

    if rgb_frame is None:
        print(
            f"Could not read RGB sample {index}"
        )
        continue

    if thermal_frame is None:
        print(
            f"Could not read thermal sample {index}"
        )
        continue

    rgb_frame = resize_frame(
        rgb_frame
    )

    thermal_frame = resize_frame(
        thermal_frame
    )

    # Add labels
    cv2.putText(
        rgb_frame,
        "RGB",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )

    cv2.putText(
        thermal_frame,
        "THERMAL",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )

    # Side-by-side
    combined = np.hstack(
        [rgb_frame, thermal_frame]
    )

    action = record["rgb_path"].split("/")[1]

    output_file = os.path.join(
        OUTPUT_DIR,
        f"sample_{index}_{action}.jpg"
    )

    cv2.imwrite(
        output_file,
        combined
    )

    print(
        f"Saved: {output_file}"
    )


print("\n" + "=" * 70)
print("SAMPLE FRAME CREATION COMPLETED")
print("=" * 70)