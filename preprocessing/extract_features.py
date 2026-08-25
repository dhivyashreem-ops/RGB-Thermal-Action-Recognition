import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)

from datasets.darkact_dataset import DarkActDataset
from models.feature_extractor import ResNetFeatureExtractor


def extract_features(
    modality,
    split,
    max_samples=None
):

    manifest = (
        r"data\splits"
        r"\darkact_multimodal_manifest.csv"
    )

    dataset_root = (
        r"data\processed\darkact"
    )

    output_dir = (
        r"data\processed\features"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    num_frames = 8
    image_size = 224
    batch_size = 2

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("DARKACT FEATURE EXTRACTION")
    print("=" * 70)

    print("\nModality:", modality)
    print("Split:", split)
    print("Device:", device)

    dataset = DarkActDataset(
        manifest_path=manifest,
        dataset_root=dataset_root,
        split=split,
        num_frames=num_frames,
        image_size=image_size,
        max_samples=max_samples
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = ResNetFeatureExtractor()

    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():

        for batch_index, batch in enumerate(
            loader,
            start=1
        ):

            data = batch[modality].to(
                device
            )

            labels = batch["label"]

            batch_size_actual = data.shape[0]
            time_steps = data.shape[1]

            data = data.reshape(
                batch_size_actual * time_steps,
                data.shape[2],
                data.shape[3],
                data.shape[4]
            )

            frame_features = model(
                data
            )

            frame_features = frame_features.reshape(
                batch_size_actual,
                time_steps,
                -1
            )

            video_features = frame_features.mean(
                dim=1
            )

            all_features.append(
                video_features.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

            processed = min(
                batch_index * batch_size,
                len(dataset)
            )

            print(
                f"Processed "
                f"{processed}/{len(dataset)}"
            )

    features = np.concatenate(
        all_features,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    suffix = (
        f"_{max_samples}"
        if max_samples is not None
        else ""
    )

    feature_file = os.path.join(
        output_dir,
        f"{modality}_{split}{suffix}_features.npy"
    )

    label_file = os.path.join(
        output_dir,
        f"{modality}_{split}{suffix}_labels.npy"
    )

    np.save(
        feature_file,
        features
    )

    np.save(
        label_file,
        labels
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETED")
    print("=" * 70)

    print("\nFeatures:", features.shape)
    print("Labels:", labels.shape)

    print("\nSaved:")
    print(feature_file)
    print(label_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        choices=[
            "rgb",
            "thermal"
        ],
        required=True
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "test"
        ],
        required=True
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None
    )

    args = parser.parse_args()

    extract_features(
        modality=args.modality,
        split=args.split,
        max_samples=args.max_samples
    )