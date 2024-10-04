import torch

from torch import nn
from torch.nn import functional as F


def norm(n_channels: int) -> nn.GroupNorm:
    """
    Group Normalization (paper: https://doi.org/10.48550/arXiv.1803.08494).
    
    The number of channels is divided by 32 and the result is used as the number of groups.
    The number of channels must be divisible by 32.
    
    Parameters
    ----------
    n_channels : int
        The number of channels in the input data.
    
    Returns
    -------
    nn.GroupNorm
        The GroupNorm module.
    """
    return nn.GroupNorm(32, n_channels, eps=1e-6)


class UpSample(nn.Module):
    def __init__(self, n_channels: int):
        """
        Module to upsample the input tensor by a factor of 2.

        Parameters
        ----------
        n_channels : int
            The number of channels in the input tensor.
        """
        super().__init__()

        self.conv = nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)

        return x
    

class DownSample(nn.Module):
    def __init__(self, n_channels: int):
        """
        Module to downsample the input tensor by a factor of 2.

        Parameters
        ----------
        n_channels : int
            The number of channels in the input tensor.
        """
        super().__init__()

        self.conv = nn.Conv2d(n_channels, n_channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):

        x  = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        x = self.conv(x)

        return x
    

class ResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int = None):
        """
        A residual block module.

        A residual block is a sequence of two convolutional layers with
        normalization and activation functions. The input is added to the
        output of the second layer, which allows the network to learn
        residual functions.

        Parameters
        ----------
        in_channels : int
            The number of channels in the input tensor.
        out_channels : int, optional
            The number of channels in the output tensor. If not specified,
            the number of channels is the same as the input tensor.
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels

        self.in_layer = nn.Sequential(
            norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )

        self.out_layer  = nn.Sequential(
            norm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        
        h = self.in_layer(x)
        h = self.out_layer(h)
        
        return h + self.skip_connection(x)


class AttnBlock(nn.Module):
    
    def __init__(self, n_channels: int):
        """
        Initialize the Attention Block module.

        The Attention Block is a module that implements the attention
        mechanism described in the paper "Attention is All You Need"
        (https://doi.org/10.48550/arXiv.1706.03762).

        Parameters
        ----------
        n_channels : int
            The number of channels in the input tensor.
        """
        super().__init__()

        self.norm = norm(n_channels)
        self.q = nn.Conv2d(n_channels, n_channels, 1, stride=1, padding=0)
        self.k = nn.Conv2d(n_channels, n_channels, 1, stride=1, padding=0)
        self.v = nn.Conv2d(n_channels, n_channels, 1, stride=1, padding=0)
        self.o = nn.Conv2d(n_channels, n_channels, 1, stride=1, padding=0)
        self.s = n_channels**-0.5

    def forward(self, x: torch.Tensor):

        x_norm = self.norm(x)

        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        attn = torch.einsum('bci,bcj->bij', q, k) * self.s
        attn = F.softmax(attn, dim=-1)
        attn = torch.einsum('bij,bcj->bci', attn, v)
        attn = attn.view(b, c, h, w)

        return self.o(attn) + x
