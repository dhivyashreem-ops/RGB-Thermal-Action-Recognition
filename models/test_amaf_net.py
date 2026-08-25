import torch

from amaf_net import AMAFNet


print("=" * 70)
print("AMAF-NET ARCHITECTURE TEST")
print("=" * 70)


model = AMAFNet(
    rgb_dim=512,
    thermal_dim=512,
    hidden_dim=256,
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


print("\nRGB input:")
print(rgb.shape)


print("\nThermal input:")
print(thermal.shape)


with torch.no_grad():

    output, attention = model(
        rgb,
        thermal,
        return_attention=True
    )


print("\nClassification output:")
print(output.shape)


print("\nAttention weights:")
print(attention.shape)


print("\nAttention values:")
print(attention)


print("\nAttention row sums:")
print(attention.sum(dim=1))


print("\nExpected output:")
print("torch.Size([4, 27])")


print("\nExpected attention:")
print("torch.Size([4, 2])")


print("\n" + "=" * 70)
print("AMAF-NET TEST COMPLETED")
print("=" * 70)