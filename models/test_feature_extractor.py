import torch

from feature_extractor import ResNetFeatureExtractor


print("=" * 70)
print("RESNET FEATURE EXTRACTOR TEST")
print("=" * 70)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nDevice:", device)

model = ResNetFeatureExtractor()
model = model.to(device)

x = torch.randn(
    8,
    3,
    224,
    224
).to(device)

print("\nInput:")
print(x.shape)

with torch.no_grad():
    features = model(x)

print("\nOutput:")
print(features.shape)

print("\nFeature dimension:")
print(model.feature_dim)

print("\n" + "=" * 70)
print("FEATURE EXTRACTOR TEST COMPLETED")
print("=" * 70)