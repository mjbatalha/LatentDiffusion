import torch

from flash_attn import flash_attn_qkvpacked_func
from torch import nn
from torch.nn import functional as F
from typing import Optional


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


class CondResBlock(nn.Module):
    """
    Conditional Residual block.
    """
    def __init__(self, t_dim: int, in_channels: int, out_channels: int = None, dp_rate: float = 0.1):
        """
        :param t_dim: The dimensionality the conditional input.
        :param in_channels: The number of channels in the input tensor.
        :param out_channels: The number of channels in the output tensor.
            If None, the number of channels is the same as the input tensor.
        :param dp_rate: The dropout rate.
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_layer = nn.Sequential(
            norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )

        self.cond_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, out_channels),
        )

        self.out_layer  = nn.Sequential(
            norm(out_channels),
            nn.SiLU(),
            nn.Dropout(dp_rate),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        
        h = self.in_layer(x)
        h = h + self.cond_emb(t)[:, :, None, None]
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
    

class CrossAttnBlock(nn.Module):
    """
    Cross-Attention block with self-attention mechanism.

    (see: Attention is all you need, https://doi.org/10.48550/arXiv.1706.03762)

    Setup flash attention: https://github.com/HazyResearch/flash-attention.

    """
    def __init__(self, model_dim: int, cond_dim: int, head_dim: int, n_heads: int):
        """
        :param model_dim: The model dimensionality.
        :param cond_dim: The conditioning dimensionality.
        :param head_dim: The attention head dimensionality.
        :param n_heads: The number of attention heads.
        """
        super().__init__()

        self.head_dim = head_dim
        self.n_heads = n_heads

        self.w_q = nn.Linear(model_dim, head_dim * n_heads, bias=False)
        self.w_k = nn.Linear(cond_dim, head_dim * n_heads, bias=False)
        self.w_v = nn.Linear(cond_dim, head_dim * n_heads, bias=False)

        self.w_o = nn.Linear(head_dim * n_heads, model_dim, bias=False)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        
        if cond is None:
            cond = x
        q = self.w_q(x)
        k = self.w_k(cond)
        v = self.w_v(cond)

        if cond is None:
            attn = self.flash_attention(q, k, v)
        else:
            attn = self.normal_attention(q, k, v)

        return attn
    
    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        
        b, l, _ = q.shape
        qkv = torch.stack((q, k, v), dim=2)
        qkv = qkv.view(b, l, 3, self.n_heads, self.head_dim)
        attn = flash_attn_qkvpacked_func(
            qkv.type(torch.float16), 
            softmax_scale=self.head_dim**-0.5
            ).type_as(qkv)
        attn = attn.view(b, l, self.n_heads * self.head_dim)
        
        return self.w_o(attn)
    
    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        
        q = q.view(*q.shape[:2], self.n_heads, self.head_dim)
        k = k.view(*k.shape[:2], self.n_heads, self.head_dim)
        v = v.view(*v.shape[:2], self.n_heads, self.head_dim)

        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.head_dim**-0.5
        attn = attn.softmax(dim=-1)
        attn = torch.einsum('bhij,bjhd->bihd', attn, v)
        attn = attn.reshape(*attn.shape[:2], -1)

        return self.w_o(attn)

class GeGLU(nn.Module):
    """
    GeGLU Activation (see: https://doi.org/10.48550/arXiv.2002.05202)

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """
    def __init__(self, in_dim: int, out_dim: int):
        """
        :param in_dim: The input dimensionality of the activation.
        :param out_dim: The output dimensionality of the activation.
        """
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.linear = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        xWb, xVc = self.linear(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return xWb * F.gelu(xVc)
    

class FFNGeGLU(nn.Module):
    """
    GeGLU Feed-Forward Network (see: https://doi.org/10.48550/arXiv.2002.05202)
    """
    def __init__(self, model_dim: int, multiplier: int = 4, dp_rate: float = 0.0):
        """
        :param model_dim: The input embedding dimensionality
        :param multiplier: The multiplicative factor for the hidden layer size
        :param dp_rate: The dropout rate
        """
        super().__init__()
        self.ffn_geglu = nn.Sequential(
            GeGLU(model_dim, model_dim * multiplier),
            nn.Dropout(dp_rate),
            nn.Linear(model_dim * multiplier, model_dim)
        )
    def forward(self, x: torch.Tensor):
        return self.ffn_geglu(x)


class TransformerBlock(nn.Module):
    """
    A single transformer block that consists of two cross-attention layers and
    a feed-forward network.

    The first cross-attention layer is self-attention, and the second layer is
    attention over the conditioning input. The feed-forward network is a
    GeGLU-activated network with a single hidden layer.
    """
    def __init__(self, 
                 model_dim: int, 
                 cond_dim: int, 
                 head_dim: int, 
                 n_heads: int, 
                 dp_rate: float = 0.0):
        """ 
        :param model_dim: The input embedding dimensionality
        :param cond_dim: The conditioning input dimensionality
        :param head_dim: The attention head dimensionality
        :param n_heads: The number of attention heads
        :param dp_rate: The dropout rate  
        """
        super().__init__()
        
        # Self-attention layer
        self.attn_1 = CrossAttnBlock(model_dim, model_dim, head_dim, n_heads)
        self.norm_1 = nn.LayerNorm(model_dim)

        # Cross-attention layer over the conditioning input
        self.attn_2 = CrossAttnBlock(model_dim, cond_dim, head_dim, n_heads)
        self.norm_2 = nn.LayerNorm(model_dim)

        # Feed-forward network
        self.ffn = FFNGeGLU(model_dim, dp_rate=dp_rate)
        self.norm_3 = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        
        x = self.attn_1(self.norm_1(x)) + x
        x = self.attn_2(self.norm_2(x), cond) + x
        x = self.ffn(self.norm_3(x)) + x
        
        return x
