import torch

from torch import nn
from typing import List

from unet import UNet
from autoencoder import AutoEncoder
from conditioning import CLIPTextEmbedder


class Text2ImgLDModel(nn.Module):

    def __init__(
            self,
            unet: UNet,
            autoencoder: AutoEncoder,
            text_embedder: CLIPTextEmbedder,
            latent_scaling: int,
            ):
        
        super().__init__()

        self.unet = unet
        self.autoencoder = autoencoder
        self.text_embedder = text_embedder
        self.latent_scaling = latent_scaling

    def text_conditioning(self, prompts: List[str]):
        return self.text_embedder(prompts)
    
    def encode(self, img: torch.Tensor):
        return self.latent_scaling * self.autoencoder.encode(img).sample()
    
    def decode(self, z: torch.Tensor):
        return self.autoencoder.decode(z / self.latent_scaling)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        return self.unet(x, t, c)
