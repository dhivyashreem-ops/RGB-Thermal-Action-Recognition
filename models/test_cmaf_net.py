import torch

from cmaf_net import CMAFNet


print("=" * 70)
print("CMAF-NET V2 ARCHITECTURE TEST")
print("=" * 70)


model = CMAFNet(
    rgb_dim=512,
    thermal_dim=512,
    hidden_dim=256,
    num_heads=4,
    num_classes=27
)

model.eval()


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

    (
        output,
        rgb_attention,
        thermal_attention
    ) = model(
        rgb,
        thermal,
        return_attention=True
    )


print("\nClassification output:")
print(output.shape)

print("\nRGB -> Thermal attention:")
print(rgb_attention.shape)

print("\nThermal -> RGB attention:")
print(thermal_attention.shape)


print("\nRGB -> Thermal attention values:")
print(rgb_attention)

print("\nThermal -> RGB attention values:")
print(thermal_attention)


print("\nNaN checks:")

print(
    "Output:",
    torch.isnan(output).any().item()
)

print(
    "RGB -> Thermal:",
    torch.isnan(rgb_attention).any().item()
)

print(
    "Thermal -> RGB:",
    torch.isnan(thermal_attention).any().item()
)


# Only verify the classification output.
# Do NOT assume a fixed attention tensor shape.

assert output.shape == (
    4,
    27
), f"Unexpected classification shape: {output.shape}"


assert not torch.isnan(
    output
).any(), "NaN detected in classification output"


assert not torch.isnan(
    rgb_attention
).any(), "NaN detected in RGB -> Thermal attention"


assert not torch.isnan(
    thermal_attention
).any(), "NaN detected in Thermal -> RGB attention"


print("\n" + "=" * 70)
print("ALL CMAF V2 CHECKS PASSED")
print("=" * 70)

print("\nActual RGB -> Thermal attention shape:")
print(rgb_attention.shape)

print("\nActual Thermal -> RGB attention shape:")
print(thermal_attention.shape)

print("\n" + "=" * 70)
print("CMAF-NET V2 ARCHITECTURE TEST COMPLETED")
print("=" * 70)