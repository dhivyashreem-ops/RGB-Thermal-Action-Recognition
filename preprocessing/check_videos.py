import csv
import os
import cv2


MANIFEST_PATH = r"data\splits\darkact_multimodal_manifest.csv"
DATASET_ROOT = r"data\processed\darkact"

NUM_SAMPLES = 5


def get_video_info(path):

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        return None

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0
    )

    # Try reading the first frame
    success, frame = cap.read()

    cap.release()

    return {
        "frames": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
        "readable": success,
        "frame_shape": (
            frame.shape
            if success
            else None
        )
    }


print("=" * 70)
print("DARKACT RGB-THERMAL VIDEO VERIFICATION")
print("=" * 70)


with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    records = list(reader)


print("\nTotal records:", len(records))
print(
    f"Checking first {NUM_SAMPLES} paired samples...\n"
)


for index, record in enumerate(
    records[:NUM_SAMPLES],
    start=1
):

    rgb_relative = record["rgb_path"]
    thermal_relative = record["thermal_path"]

    rgb_path = os.path.join(
        DATASET_ROOT,
        rgb_relative
    )

    thermal_path = os.path.join(
        DATASET_ROOT,
        thermal_relative
    )

    print("=" * 70)
    print(f"SAMPLE {index}")
    print("=" * 70)

    print("\nAction label:", record["label"])

    print("\nRGB:")
    print(rgb_relative)

    rgb_info = get_video_info(
        rgb_path
    )

    if rgb_info is None:
        print("ERROR: RGB video cannot be opened.")
    else:
        print("Frames:", rgb_info["frames"])
        print("FPS:", rgb_info["fps"])
        print(
            "Resolution:",
            rgb_info["width"],
            "x",
            rgb_info["height"]
        )
        print(
            "Duration:",
            round(rgb_info["duration"], 2),
            "seconds"
        )
        print(
            "First frame readable:",
            rgb_info["readable"]
        )
        print(
            "Frame shape:",
            rgb_info["frame_shape"]
        )

    print("\nThermal:")
    print(thermal_relative)

    thermal_info = get_video_info(
        thermal_path
    )

    if thermal_info is None:
        print(
            "ERROR: Thermal video cannot be opened."
        )
    else:
        print("Frames:", thermal_info["frames"])
        print("FPS:", thermal_info["fps"])
        print(
            "Resolution:",
            thermal_info["width"],
            "x",
            thermal_info["height"]
        )
        print(
            "Duration:",
            round(
                thermal_info["duration"],
                2
            ),
            "seconds"
        )
        print(
            "First frame readable:",
            thermal_info["readable"]
        )
        print(
            "Frame shape:",
            thermal_info["frame_shape"]
        )


print("\n" + "=" * 70)
print("VIDEO VERIFICATION COMPLETED")
print("=" * 70)