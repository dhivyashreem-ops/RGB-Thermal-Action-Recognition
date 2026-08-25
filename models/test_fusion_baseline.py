import torch

from fusion_baseline import FusionBaseline


print("=" * 70)
print("RGB-THERMAL FUSION BASELINE TEST")
print("=" * 70)


model = FusionBaseline(
    rgb_dim=512,
    thermal_dim=512,
    num_classes=27
)


rgb = torch.randn(
    4,
    512
)

thermal = torch.randn(
    4,
    512
)


print("\nRGB feature shape:")
print(rgb.shape)


print("\nThermal feature shape:")
print(thermal.shape)


with torch.no_grad():

    output = model(
        rgb,
        thermal
    )


print("\nOutput shape:")
print(output.shape)


print("\nExpected:")
print("torch.Size([4, 27])")


print("\n" + "=" * 70)
print("FUSION BASELINE TEST COMPLETED")
print("=" * 70)