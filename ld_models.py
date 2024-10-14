import torch

from torch import nn
from typing import List

from unet import UNet
from autoencoder import KLAutoencoder
from conditioning import CLIPTextEmbedder


class Text2ImgLDModel(nn.Module):

    def __init__(
            self,
            unet: UNet,
            autoencoder: KLAutoencoder,
            text_embedder: CLIPTextEmbedder,
            n_steps: int = 1000,
            beta_min: float = 1e-4,
            beta_max: float = 0.02,
            ):
        
        super().__init__()

        self.unet = unet
        self.autoencoder = autoencoder
        self.text_embedder = text_embedder
        self.n_steps = n_steps

        betas = torch.linspace(beta_min**0.5, beta_max**0.5, n_steps, dtype=torch.float32)**2
        alphas = 1.0 - betas
        alphas = torch.cumsum(alphas.log(), dim=0).exp()

        self.betas = nn.Parameter(betas, requires_grad=False)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        
        for params in self.autoencoder.parameters():
            params.requires_grad = False

        for params in self.text_embedder.parameters():
            params.requires_grad = False

    def forward(self, z: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        return self.unet(z, t, c)

    def text_conditioning(self, prompts: List[str]):
        return self.text_embedder(prompts)
    
    def encode(self, img: torch.Tensor):
        return self.autoencoder.encode(img)
    
    def decode(self, z: torch.Tensor):
        return self.autoencoder.decode(z)
    
    def add_noise(self, z: torch.Tensor, t: torch.Tensor):
        alphas_t = self.alphas[t][:, None, None, None]
        noise = torch.randn_like(z)
        z = z * torch.sqrt(alphas_t) + torch.sqrt(1.0 - alphas_t) * noise
        return z, noise

    @torch.inference_mode()
    def sample(self, z: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        betas_t = self.betas[t][:, None, None, None]
        alphas_t = self.alphas[t][:, None, None, None]
        inv_sqrt_alpha_t = 1.0 / (torch.sqrt(1.0 - betas_t))
        inv_sqrt_1m_alpha_bar_t = 1.0 / (torch.sqrt(1.0 - alphas_t))
        mean = inv_sqrt_alpha_t * (z - betas_t * inv_sqrt_1m_alpha_bar_t * self.unet(z, t, c))
        noise = torch.randn_like(z) * torch.sqrt(betas_t)
        return mean + noise

