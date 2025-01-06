import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Autoencoder(nn.Module):
    def __init__(self, 
                 input_channels: int, 
                 hidden_dim_1: int, 
                 hidden_dim_2: int,
                 latent_dim: int):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim_1, hidden_dim_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim_2, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Decoder 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim_2, hidden_dim_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim_1, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() # Normalize to 0-1
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def test_autoencoder():
    # Define the parameters for the Autoencoder
    input_channels = 1
    hidden_dim_1 = 64
    hidden_dim_2 = 32
    latent_dim = 16
    input_size = (32, input_channels, 28, 28)  # Batch size of 1, 28x28 image

    # Create an instance of the Autoencoder
    model = Autoencoder(input_channels, hidden_dim_1, hidden_dim_2, latent_dim)

    # Create a random input tensor with the specified size
    input_tensor = torch.randn(input_size)

    # Pass the input tensor through the autoencoder
    reconstructed, latent = model(input_tensor)

    # Check the output dimensions
    assert reconstructed.shape == input_tensor.shape, f"Expected {input_tensor.shape}, but got {reconstructed.shape}"
    assert latent.shape[1] == latent_dim, f"Expected latent dimension {latent_dim}, but got {latent.shape[1]}"

    print(f"Input shape: {input_tensor.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")
    print("Autoencoder test passed!")

# Run the test
# test_autoencoder()