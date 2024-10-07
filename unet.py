import torch

from torch import nn
from typing import List, Optional

from modules import CondResBlock, DownSample, TransformerBlock, UpSample, norm


class SpatialTransformer(nn.Module):
    """
    SpatialTransformer class
    """
    def __init__(self, n_channels: int, n_transformers: int, n_heads: int, c_dim: int):
        """
        :param n_channels: The number of channels in the input tensor.
        :param n_transformers: The number of transformer blocks to use.
        :param n_heads: The number of attention heads in each transformer block.
        :param c_dim: The dmensionality of the conditioning tensor.
        """
        super().__init__()

        self.norm = norm(n_channels)
        self.conv_in = nn.Conv2d(n_channels, n_channels, 1, stride=1, padding=0)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                model_dim=n_channels, 
                cond_dim=c_dim, 
                head_dim=n_channels // n_heads,
                n_heads=n_heads,
                ) for _ in range(n_transformers)
            ])
        
        self.conv_out = nn.Conv2d(n_channels, n_channels, 1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        
        x_in = x
        bs, ch, h, w = x.shape
        x = self.norm(x)
        x = self.conv_in(x)
        x = x.permute(0, 2, 3, 1).view(bs, h * w, ch)
        for block in self.transformer_blocks:
            x = block(x, c)
        x = x.view(bs, h, w, ch).permute(0, 3, 1, 2)
        x = self.conv_out(x)

        return x + x_in


class TimeEmbSeq(nn.Sequential):
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None):
        """
        :param x: The input tensor.
        :param t: The time embedding tensor.
        :param c: The conditioning tensor (optional).
        :return: The output tensor.
        """
        for layer in self:
            if isinstance(layer, CondResBlock):
                x = layer(x, t)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, c)
            else:
                x = layer(x)
        return x


class UNet(nn.Module):
    """
    UNet class
    
    (see: https://doi.org/10.48550/arXiv.1505.04597)
    
    """
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            h_channels: int,
            c_dim: int,
            n_heads: int,
            n_resnets: int, 
            n_transformers: int,
            ch_multipliers: List[int],
            attn_levels: List[int],
            ):
        """
        :param in_channels: The number of channels in the input data.
        :param out_channels: The number of channels in the output data.
        :param h_channels: The number of channels in the hidden layers.
        :param c_dim: The dimensionality of the conditioning input.
        :param n_heads: The number of attention heads.
        :param n_resnets: The number of ResNet blocks in each stage.
        :param n_transformers: The number of transformer blocks in each attention layer.
        :param ch_multipliers: A list of integers that specifies the number of channels
            in each stage. The number of channels in each stage is the number of channels
            in the previous stage multiplied by the corresponding integer in the list.
        :param attn_levels: A list of integers that specifies the stages that contain
            attention layers.
        """
        super().__init__()

        self.h_channels = h_channels

        # time embedding
        t_dim = h_channels * 4
        self.t_emb = nn.Sequential(
            nn.Linear(h_channels, t_dim),
            nn.SiLU(), 
            nn.Linear(t_dim, t_dim),
        )

        # input layers
        self.in_layers = nn.ModuleList()
        self.in_layers.append(TimeEmbSeq(
            nn.Conv2d(in_channels, h_channels, 3, stride=1, padding=1),
        ))
        in_channels = [h_channels]
        channels = [n * h_channels for n in ch_multipliers]
        for i in range(len(channels)):
            for _ in range(n_resnets):

                layers = [CondResBlock(t_dim, h_channels, channels[i])]
                h_channels = channels[i]

                if i in attn_levels:
                    layers.append(SpatialTransformer(h_channels, n_transformers, n_heads, c_dim))

                self.in_layers.append(TimeEmbSeq(*layers))
                in_channels.append(h_channels)

            if i != len(channels) - 1:
                self.in_layers.append(TimeEmbSeq(DownSample(h_channels)))
                in_channels.append(h_channels)

        # middle layers
        self.mid_layers = TimeEmbSeq(
            CondResBlock(t_dim, h_channels),
            SpatialTransformer(h_channels, n_transformers, n_heads, c_dim),
            CondResBlock(t_dim, h_channels),
        )

        # output layers
        self.out_layers = nn.ModuleList()
        for i in reversed(range(len(channels))):
            for j in range(n_resnets + 1):

                layers = [CondResBlock(t_dim, h_channels + in_channels.pop(), channels[i])]
                h_channels = channels[i]

                if i in attn_levels:
                    layers.append(SpatialTransformer(h_channels, n_transformers, n_heads, c_dim))

                if i != 0 and j == n_resnets:
                    layers.append(UpSample(h_channels))

                self.out_layers.append(TimeEmbSeq(*layers))

        self.out_layers.append(
            nn.Sequential(
                norm(h_channels),
                nn.SiLU(),
                nn.Conv2d(h_channels, out_channels, 3, stride=1, padding=1),
            )
        )
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):

        t = self.time_step_emb(t)
        t = self.t_emb(t)
        
        xs = []
        for layer in self.in_layers:
            x = layer(x, t, c)            
            xs.append(x)

        x = self.mid_layers(x, t, c)
        for layer in self.out_layers[:-1]:
            x = torch.cat([x, xs.pop()], dim=1)
            x = layer(x, t, c)

        return self.out_layers[-1](x)

    def time_step_emb(self, ts: torch.Tensor, max_period=10_000):

        freqs = torch.exp(
            -torch.log(torch.tensor(max_period)) * torch.arange(self.h_channels // 2) / 
            (self.h_channels // 2)
            ).to(device=ts.device)
        args = ts[:, None] * freqs[None]
                      
        return torch.cat((args.sin(), args.cos()), dim=-1)
