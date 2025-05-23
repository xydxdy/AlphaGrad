import torch
import torch.nn as nn
from .. import layers

class FBMSNet(nn.Module):
    def __init__(
        self,
        n_band,
        n_channel,
        n_time,
        latent_dim=1152,
        n_class=2,
        temporalLayer="LogVarLayer",
        stride=4,
        dilatability=8,
        weight_norm=True,
        latent_modify=False,
        *args,
        **kwargs
    ):
        # input_size: channel x datapoint
        super(FBMSNet, self).__init__()
        self.n_band = n_band
        self.latent_dim = latent_dim
        self.kernel_size = [(1, 15), (1, 31), (1, 63), (1, 125)]
        self.n_feature = self.n_band * len(self.kernel_size)
        self.stride = stride
        self.dilatability = dilatability
        self.weight_norm = weight_norm
        self.latent_modify = latent_modify

        self.mixConv2d = nn.Sequential(
            layers.MixedConv2d(
                in_channels=self.n_band,
                out_channels=self.n_feature,
                kernel_size=self.kernel_size,
                stride=1,
                padding="",
                dilation=1,
                depthwise=False,
            ),
            nn.BatchNorm2d(self.n_feature),
        )
        self.scb = self.SCB(
            in_chan=self.n_feature,
            out_chan=self.n_feature * self.dilatability,
            n_channel=int(n_channel),
            weight_norm=self.weight_norm
        )

        # Formulate the temporal agreegator
        self.temporalLayer = layers.__dict__[temporalLayer](dim=3)

        if latent_modify:
            self.latent_layer = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(in_features=self.latent_dim, out_features=self.latent_dim),
            )
        else:
            self.latent_layer = nn.Flatten(start_dim=1)
        
        self.fc = self.last_block(self.latent_dim, n_class, weight_norm=self.weight_norm)

    def SCB(self, in_chan, out_chan, n_channel, weight_norm=True, *args, **kwargs):
        """
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        """
        return nn.Sequential(
            layers.Conv2dWithConstraint(
                in_chan,
                out_chan,
                (n_channel, 1),
                groups=in_chan,
                max_norm=2,
                weight_norm=weight_norm,
                padding=0,
            ),
            nn.BatchNorm2d(out_chan),
            layers.swish(),
        )

    def last_block(self, in_f, out_f, weight_norm=True, *args, **kwargs):
        return nn.Sequential(
            layers.LinearWithConstraint(
                in_f, out_f, max_norm=0.5, weight_norm=weight_norm, *args, **kwargs
            ),
            nn.LogSoftmax(dim=1),
        )

    def get_size(self, n_channel, n_time):
        with torch.no_grad():
            data = torch.ones((1, self.n_band, n_channel, n_time))
            x = self.mixConv2d(data)
            x = self.scb(x)
            x = x.reshape([*x.shape[0:2], self.stride, int(x.shape[3] / self.stride)])
            x = self.temporalLayer(x)
            x = torch.flatten(x, start_dim=1)
        return x.size()[1]
    
    def get_shared_layer(self):
        return self.latent_layer 

    def forward(self, x):
        y = self.mixConv2d(x)
        x = self.scb(y)
        x = x.reshape([*x.shape[0:2], self.stride, int(x.shape[3] / self.stride)])
        x = self.temporalLayer(x)
        f = self.latent_layer(x)
        c = self.fc(f)
        return f, c
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
