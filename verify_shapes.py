"""Shape verification for STU-integrated encoder."""

import torch
import sys

sys.path.insert(0, "/Users/rsalik/Documents/Code/Hazan Lab/IW/tsuki")

from models.networks.encoder import EncoderNetwork

# Test with temporal sequence
batch, timesteps, C, H, W = 2, 16, 4, 96, 96  # 4 stacked frames, 96x96 Atari
x = torch.randn(batch, timesteps, C, H, W)

print(f"Input shape: {x.shape}")
print(f"Expected: [batch={batch}, timesteps={timesteps}, C={C}, H={H}, W={W}]")

encoder = EncoderNetwork(
    observation_shape=(C, H, W),
    num_blocks=2,
    num_channels=64,
    downsample=True,
    sequence_length=timesteps,
)

print(f"\nEncoder created successfully!")
print(
    f"Spatial dims after downsampling: H'={encoder.spatial_h}, W'={encoder.spatial_w}"
)
print(f"Spatial feature dim: {encoder.spatial_dim}")

# Forward pass
out = encoder(x)
print(f"\nOutput shape: {out.shape}")
print(
    f"Expected: [batch={batch}, timesteps={timesteps}, num_channels=64, H'={encoder.spatial_h}, W'={encoder.spatial_w}]"
)

# Verify shapes match
expected_shape = (batch, timesteps, 64, encoder.spatial_h, encoder.spatial_w)
assert out.shape == expected_shape, f"Shape mismatch: {out.shape} != {expected_shape}"
print("\n✓ All shape checks passed!")
