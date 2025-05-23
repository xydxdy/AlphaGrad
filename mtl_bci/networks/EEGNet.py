import torch
import torch.nn as nn
from .. import layers

class EEGNet(nn.Module):
    
    def __init__(self, n_channel, n_time, n_class=2,
                 dropout=0.5, F1=8, D=2,
                 C1=200, *args, **kwargs):
        super(EEGNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.n_time = n_time
        self.n_class = n_class
        self.n_channel = n_channel
        self.C1 = C1

        self.conv_layer = self.initial_block(dropout)
        self.flatten_size = self.cal_flat_size(self.conv_layer)
        self.flatten_layer = self.flatten_block(self.flatten_size, n_class)
        
    def initial_block(self, dropout, *args, **kwargs):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding='same', bias=False),
                nn.BatchNorm2d(self.F1),
                layers.Conv2dWithConstraint(self.F1, self.F1*self.D, (self.n_channel, 1),
                                     padding=0, bias=False, max_norm=1,
                                     groups=self.D),
                nn.BatchNorm2d(self.F1*self.D),
                nn.ELU(),
                nn.AvgPool2d((1,4)),
                nn.Dropout(p=dropout),
                layers.SeparableConv2d(self.F1*self.D, self.F2, (1, self.C1//4), padding='same', bias=False),
                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1,8)),
                nn.Dropout(p=dropout)
                )
        return block1

    def flatten_block(self, in_features, out_features, *args, **kwargs):
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
        x = self.conv_layer(x)
        x = self.flatten_layer(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)