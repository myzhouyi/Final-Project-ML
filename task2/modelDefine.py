import torch
import torch.nn as nn
import torch.nn.functional as F



# Define the VAE model and its loss function

class VAE(nn.Module):
    def __init__(self, input_channel, height, width, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mean = nn.Linear(32 * (height // 4) * (width // 4), latent_dim)
        self.fc_log_var = nn.Linear(32 * (height // 4) * (width // 4), latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * (height // 4) * (width // 4)), # 32 * 8 * 8
            nn.Unflatten(1, (32, height // 4, width // 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channel, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        hidden = self.encoder(x) # 2048
        mean = self.fc_mean(hidden)
        log_var = self.fc_log_var(hidden)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var) # 20
        reconstruction = self.decode(z) # 3, 32, 32
        return reconstruction, mean, log_var

    
def vae_loss(reconstruction, x, mean, log_var):
    reconstruction_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_loss