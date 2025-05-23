import torch
import torch.nn as nn
from .. import layers

class MIN2Net(nn.Module):

    """channels_first

    from torchsummary import summary
    n_channel, n_time, n_class = 22, 1000, 2
    input_size = (n_channel, n_time, 1)
    subsampling = 250
    net = MIN2Net(n_channel, n_time, n_class, subsampling).cuda()
    summary(net, input_size)

    """

    def __init__(
        self, n_channel, n_time, n_band=1, n_class=2, subsampling=100, *args, **kwargs
    ) -> None:
        super().__init__()
        ## params
        self.n_channel = n_channel
        self.n_time = n_time
        self.n_band = n_band
        self.input_shape = (n_channel, n_time, n_band)
        self.n_class = n_class
        self.subsampling = subsampling
        # self.C, self.T, self.D = self.input_shape
        self.pool_t1 = self.n_time // self.subsampling
        self.pool_d1 = 2 if (self.n_band // 2) != 0 else 1
        self.pool_size_1 = (self.pool_t1, self.pool_d1)
        self.pool_size_2 = (4, 1)
        self.filter_1 = self.n_channel
        self.filter_2 = self.n_channel // 2
        if self.pool_d1 == 2:
            self.flatten_size = (self.n_time // self.pool_size_1[0] // self.pool_size_2[0]) * (self.n_band//2)
        else:
            self.flatten_size = (self.n_time // self.pool_size_1[0] // self.pool_size_2[0])     
        self.latent_dim = int(self.n_channel * (self.n_class / 2) * self.n_band)
        self.weight_norm = True
        # self.loss_coefs = self.set_loss_coefs([1, 1, 1])
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

        ## models
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.classifier = self.build_classifier()

    def build_encoder(self):
        self.en_block1 = nn.Sequential(
            layers.Conv2dWithConstraint(
                in_channels=self.n_channel,
                out_channels=self.filter_1,
                kernel_size=(64, 1),
                padding="same",
                max_norm=2,
            ),
            nn.ELU(),
            nn.BatchNorm2d(self.filter_1),
            nn.AvgPool2d(self.pool_size_1),
        )
        self.en_block2 = nn.Sequential(
            layers.Conv2dWithConstraint(
                in_channels=self.filter_1,
                out_channels=self.filter_2,
                kernel_size=(32, 1),
                padding="same",
                max_norm=2,
            ),
            nn.ELU(),
            nn.BatchNorm2d(self.filter_2),
            nn.AvgPool2d(self.pool_size_2),
        )
        self.latent_layer = nn.Sequential(
            nn.Flatten(),
            layers.LinearWithConstraint(
                in_features=self.flatten_size * self.filter_2,
                out_features=self.latent_dim,
                max_norm=0.5,
                weight_norm=self.weight_norm,
            ),
        )

        return nn.Sequential(self.en_block1, self.en_block2, self.latent_layer)

    def build_decoder(self):
        self.de_input = layers.LinearWithConstraint(
            in_features=self.latent_dim,
            out_features=1 * self.flatten_size * self.filter_2,
            max_norm=0.5,
            weight_norm=self.weight_norm,
        )
        self.de_block2 = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(self.filter_2, self.flatten_size, 1)),
            layers.Conv2dWithConstraint(
                in_channels=self.filter_2,
                out_channels=self.filter_2,
                kernel_size=(64, 1),
                padding="same",
            ),
            nn.UpsamplingNearest2d(
                size=(self.flatten_size * self.pool_size_2[0], self.n_band//self.pool_d1), scale_factor=None
            ),
            nn.ELU(),
        )
        self.de_block1 = nn.Sequential(
            layers.Conv2dWithConstraint(
                in_channels=self.filter_2,
                out_channels=self.filter_1,
                kernel_size=(32, 1),
                padding="same",
            ),
            nn.UpsamplingNearest2d(size=(self.n_time, self.n_band), scale_factor=None),
            nn.ELU(),
        )
        return nn.Sequential(self.de_input, self.de_block2, self.de_block1)

    def build_classifier(self):
        self.last_layer = nn.Sequential(
            layers.LinearWithConstraint(
                self.latent_dim,
                self.n_class,
                max_norm=0.5,
                weight_norm=self.weight_norm,
            ),
            nn.Softmax(dim=1),
        )
        return self.last_layer

    def get_shared_layer(self):
        return self.latent_layer

    def forward(self, x):
        latent = self.encoder(x)
        train_xr = self.decoder(latent)
        z = self.classifier(latent)
        return latent, train_xr, z

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)