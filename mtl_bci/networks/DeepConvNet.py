import torch
import torch.nn as nn
from .. import layers

class DeepConvNet(nn.Module):
    
    def __init__(self, n_channel, n_time, n_class=2, dropout=0.5, *args, **kwargs):
        super(DeepConvNet, self).__init__()

        self.n_channel = n_channel
        self.n_time = n_time
        self.n_class = n_class
        self.dropout = dropout
        
        F1 = 25
        K1 = (1,5)
        in_channels_list = [25, 25, 50, 100]
        out_channels_list = [25, 50, 100, 200]
        kernal_size_list = [(self.n_channel, 1), (1, 5), (1, 5), (1, 5)]

        firstLayer = layers.Conv2dWithConstraint(1, F1, K1, padding=0, max_norm=2)
        middleLayers = nn.Sequential(*[self.conv_block(in_channels, out_channels, dropout, kernal_size)
            for in_channels, out_channels, kernal_size in zip(in_channels_list, out_channels_list, kernal_size_list)])

        self.sequential_conv_layers = nn.Sequential(firstLayer, middleLayers)

        self.flatten_size = self.cal_flat_size(self.sequential_conv_layers)
        self.last_layer = self.last_block(self.flatten_size, n_class)
        
    def conv_block(self, in_channels, out_channels, dropout, kernal_size,  *args, **kwargs):
        return nn.Sequential(
            layers.Conv2dWithConstraint(in_channels, out_channels, kernal_size, padding=0, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d((1,2), stride=(1,2)),
            nn.Dropout(p=dropout),
        )

    def last_block(self, in_features, out_features, *args, **kwargs):
        return nn.Sequential(
            nn.Flatten(),
            layers.LinearWithConstraint(
                in_features=in_features,
                out_features=out_features,
                max_norm=0.25,
                weight_norm=True,
            ),
            nn.Softmax(dim=1),
        )

    def cal_flat_size(self, model):
        '''
        Calculate the output based on input size.
        model is from nn.Module.
        '''
        data = torch.rand(1,1,self.n_channel, self.n_time)
        model.eval()
        out = model(data).shape
        return int(torch.Tensor([out[1:]]).prod().item())

    def forward(self, x):

        x = self.sequential_conv_layers(x)
        x = self.last_layer(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)