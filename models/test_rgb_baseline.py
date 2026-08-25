import torch

from rgb_baseline import RGBBaseline


print("=" * 70)
print("RGB BASELINE MODEL TEST")
print("=" * 70)


model = RGBBaseline(
    num_classes=27
)


x = torch.randn(
    2,
    16,
    3,
    224,
    224
)


print("\nInput shape:")
print(x.shape)


with torch.no_grad():

    output = model(x)


print("\nOutput shape:")
print(output.shape)


print("\nExpected:")
print("torch.Size([2, 27])")


print("\n" + "=" * 70)
print("RGB BASELINE TEST COMPLETED")
print("=" * 70)