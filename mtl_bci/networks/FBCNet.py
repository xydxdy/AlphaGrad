import torch
import torch.nn as nn
from .. import layers


class FBCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    """
    FBNet with seperate variance for every 1s.
    The data input is in a form of batch x 1 x chan x time x filterBand
    """

    def __init__(
        self,
        n_channel,
        n_time,
        n_class=2,
        n_band=9,
        m=32,
        temporal_layer="LogVarLayer",
        stride=4,
        weight_norm=True,
        *args,
        **kwargs
    ):
        super(FBCNet, self).__init__()

        self.n_band = n_band
        self.m = m
        self.stride = stride

        # create all the parrallel SCBc
        self.scb = self.SCB(m, n_channel, self.n_band, weight_norm=weight_norm)

        # Formulate the temporal agreegator
        self.temporal_layer = layers.__dict__[temporal_layer](dim=3)

        # The final fully connected layer
        self.last_layer = self.last_block(
            self.m * self.n_band * self.stride, n_class, weight_norm=weight_norm
        )

    def SCB(self, m, n_channel, n_band, weight_norm=True, *args, **kwargs):
        """
        The spatial convolution block
        m : number of sptatial filters.
        n_band: number of bands in the data
        """
        return nn.Sequential(
            layers.Conv2dWithConstraint(
                n_band,
                m * n_band,
                (n_channel, 1),
                groups=n_band,
                max_norm=2,
                weight_norm=weight_norm,
                padding=0,
            ),
            nn.BatchNorm2d(m * n_band),
            layers.swish(),
        )

    def last_block(self, in_f, out_f, weight_norm=True, *args, **kwargs):
        return nn.Sequential(
            layers.LinearWithConstraint(
                in_f, out_f, max_norm=0.5, weight_norm=weight_norm, *args, **kwargs
            ),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        # x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.stride, int(x.shape[3] / self.stride)])
        x = self.temporal_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.last_layer(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)