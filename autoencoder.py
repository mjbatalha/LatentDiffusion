import torch

from torch import nn
from torch.nn import functional as F
from typing import List

from modules import ResBlock, AttnBlock, UpSample, DownSample, norm


class GaussianDistribution:
    """
    Represents a multivariate normal distribution with diagonal covariance matrix.

    :param parameters: A tensor of shape (batch_size, 2 * z_channels, h, w) containing the mean and
        log-variance of the distribution.
    """
    def __init__(self, parameters: torch.Tensor):
        """
        :param parameters: A tensor of shape (batch_size, 2 * z_channels, h, w) containing the mean and
            log-variance of the distribution.
        """
        self.mean, logvar = parameters.chunk(2, dim=1)
        self.logvar = torch.clamp(logvar, -30, 20)
        self.std = torch.exp(0.5 * self.logvar)
    def sample(self):
        return self.mean + torch.randn_like(self.mean) * self.std
    

class AutoEncoder(nn.Module):

    """
    The AutoEncoder class combines an encoder and a decoder with a moment-estimating
    convolutional layer and a decoding convolutional layer.

    The moment-estimating convolutional layer takes the output of the encoder and
    produces the moments of the latent distribution. The decoding convolutional layer
    de-quantizes the latent variable and produces the input to the decoder.
    """
    def __init__(self, encoder: "Encoder", decoder: "Decoder", z_channels: int, q_channels: int):
        """
        :param encoder: The encoder network.
        :param decoder: The decoder network.
        :param z_channels: The number of channels of latent space.
        :param q_channels: The number of channels of quantized latent space.
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.moment_conv = nn.Conv2d(2*z_channels, 2*q_channels, 1, stride=1, padding=0)
        self.decode_conv = nn.Conv2d(q_channels, z_channels, 1, stride=1, padding=0)

    def encode(self, x: torch.Tensor) -> GaussianDistribution:
        z = self.encoder(x)
        moments = self.moment_conv(z)
        return GaussianDistribution(moments)
    
    def decode(self, z: torch.Tensor):
        z = self.decode_conv(z)
        x = self.decoder(z)
        return x
    

class Encoder(nn.Module):

    """
    The Encoder class represents a neural network that takes input data and encodes it
    into a latent space.

    The neural network consists of a convolutional in-layer, a sequence of down-sampling
    stages and an output layer. Each down-sampling stage consists of a sequence of
    ResNet blocks, followed by a down-sampling layer. The output layer is a sequence of
    ResNet blocks, followed by an attention layer, another ResNet block and a
    convolutional layer that outputs the moments of the latent distribution.
    """
    def __init__(self, 
                 x_channels: int, 
                 h_channels: int, 
                 z_channels: int, 
                 n_resnet: int, 
                 ch_multipliers: List[int]):
        """
        :param x_channels: The number of channels in the input data.
        :param h_channels: The number of channels in the hidden layers.
        :param z_channels: The number of channels in the latent space.
        :param n_resnet: The number of ResNet blocks in each down-sampling stage.
        :param ch_multipliers: A list of integers that specifies the number of channels
            in each down-sampling stage. The number of channels in each stage is the
            number of channels in the previous stage multiplied by the corresponding
            integer in the list.
        """
        super().__init__()

        channels = [n * h_channels for n in [1] + ch_multipliers]

        self.in_layer = nn.Conv2d(x_channels, h_channels, 3, stride=1, padding=1)

        self.down = nn.ModuleList()
        for i in range(len(ch_multipliers)):
            resnets = nn.ModuleList()
            for _ in range(n_resnet):
                resnets.append(ResBlock(h_channels, channels[i + 1]))
                h_channels = channels[i + 1]
            down = nn.Module()
            down.block = resnets
            if i != len(ch_multipliers) - 1:
                down.downsample = DownSample(h_channels)
            else:
                down.downsample = nn.Identity()
            self.down.append(down)

        self.out_layer = nn.Sequential(
            ResBlock(h_channels),
            AttnBlock(h_channels),
            ResBlock(h_channels),
            norm(h_channels),
            nn.SiLU(),
            nn.Conv2d(h_channels, 2 * z_channels, 3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor):

        x = self.in_layer(x)
        for down in self.down:
            for block in down.block:
                x = block(x)
            x = down.downsample(x)

        return self.out_layer(x)
    

class Decoder(nn.Module):
    """
    The Decoder class is a neural network that takes a latent space sample as input and
    outputs a sample from the data space. It consists of an input layer, a sequence of
    up-sampling stages and an output layer. Each up-sampling stage consists of a sequence
    of ResNet blocks, followed by an up-sampling layer. The output layer is a sequence of
    normalization, activation and convolutional layers that output the final sample.
    """
    def __init__(self, 
                x_channels: int, 
                h_channels: int, 
                z_channels: int, 
                n_resnet: int, 
                ch_multipliers: List[int]):
        """
        :param x_channels: The number of channels in the input data.
        :param h_channels: The number of channels in the hidden layers.
        :param z_channels: The number of channels in the latent space.
        :param n_resnet: The number of ResNet blocks in each up-sampling stage.
        :param ch_multipliers: A list of integers that specifies the number of channels
            in each up-sampling stage. The number of channels in each stage is the
            number of channels in the previous stage multiplied by the corresponding
            integer in the list.
        """
        super().__init__()

        channels = [n * h_channels for n in ch_multipliers]
        h_channels = channels[-1]

        self.in_layer = nn.Sequential(
            nn.Conv2d(z_channels, h_channels, 3, stride=1, padding=1),
            ResBlock(h_channels),
            AttnBlock(h_channels),
            ResBlock(h_channels),
        )

        self.up = nn.ModuleList()
        for i in reversed(range(len(ch_multipliers))):
            resnets = nn.ModuleList()
            for _ in range(n_resnet + 1):
                resnets.append(ResBlock(h_channels, channels[i]))
                h_channels = channels[i]
            up = nn.Module()
            up.block = resnets
            if i != 0:
                up.upsample = UpSample(h_channels)
            else:
                up.upsample = nn.Identity()
            self.up.insert(0, up)
            
        self.out_layer = nn.Sequential(
            norm(h_channels),
            nn.SiLU(),
            nn.Conv2d(h_channels, x_channels, 3, stride=1, padding=1)
        )

    def forward(self, z: torch.Tensor):

        h = self.in_layer(z)
        for up in reversed(self.up):
            for block in up.block:
                h = block(h)
            h = up.upsample(h)

        return self.out_layer(h)
