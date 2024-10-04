import torch

from torch import nn
from torch.nn import functional as F


def norm(n_channels: int) -> nn.GroupNorm:
    """
    Returns a GroupNorm normalization layer with 32 groups.

    (see: https://doi.org/10.48550/arXiv.1803.08494)

    :param n_channels: The number of channels in the input tensor.
    :return: A GroupNorm normalization layer.
    """
    
    return nn.GroupNorm(32, n_channels, eps=1e-6)


class UpSample(nn.Module):
    """
    Up-sampling module (x2) using nearest neighbor interpolation and a
    convolutional layer.
    """
    def __init__(self, n_channels: int):
        """
        :param n_channels: : The number of channels in the input tensor.
        """
        super().__init__()

        self.conv = nn.Conv2d(
            n_channels, n_channels, 3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor):

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)

        return x
    

class DownSample(nn.Module):
    """
    Down-sampling module (x2) using a convolutional layer.
    
    The output of the convolutional layer is down-sampled by a factor of 2.
    The input is padded with zeros on the right and bottom borders.
    """
    def __init__(self, n_channels: int):
        """
        :param n_channels: The number of channels in the input tensor.
        """
        super().__init__()

        self.conv = nn.Conv2d(n_channels, n_channels, 3, stride=2, padding=0)
        
    def forward(self, x: torch.Tensor):

        x  = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        x = self.conv(x)

        return x
    

class ResBlock(nn.Module):
    """
    Residual block with two convolutional layers and a shortcut connection.
    
    The input is processed by two convolutional layers with a SiLU activation
    function in between. The output of the second convolutional layer is added
    to the input (after a convolutional layer to match the number of channels)
    to produce the output of the block.
    """
    def __init__(self, in_channels: int, out_channels: int = None):
        """
        :param in_channels: The number of channels in the input tensor.
        :param out_channels: The number of channels in the output tensor.
            If None, the number of channels is the same as the input tensor.
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
    """
    Attention block with self-attention mechanism.
    
    The input is first normalized using GroupNorm. Then it is split into three
    parts: queries, keys and values. The queries and keys are used to compute
    the attention weights. The attention weights are then used to compute the
    output of the block by taking the dot product of the attention weights and
    the values.

    (see: Attention is all you need, https://doi.org/10.48550/arXiv.1706.03762)

    """
    def __init__(self, n_channels: int):
        """
        :param n_channels: The number of channels in the input tensor.
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
