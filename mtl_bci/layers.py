import torch
import torch.nn as nn

from typing import Tuple, Optional
import torch.nn.functional as F

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bias=False, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=padding, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias, **kwargs)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        
    def conv2d_same(self, 
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
        ih, iw = x.size()[-2:]
        kh, kw = weight.size()[-2:]
        pad_h = self._calc_same_pad(ih, kh, stride[0], dilation[0])
        pad_w = self._calc_same_pad(iw, kw, stride[1], dilation[1])
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
    
    def _calc_same_pad(self, i: int, k: int, s: int, d: int):
        return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        return self.conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = self._split_channels(in_channels, num_groups)
        out_splits = self._split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(
                str(idx),
                self.create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits
        
    def _split_channels(self, num_chan, num_groups):
        split = [num_chan // num_groups for _ in range(num_groups)]
        split[0] += num_chan - sum(split)
        return split

    def create_conv2d_pad(self, in_chs, out_chs, kernel_size, **kwargs):
        padding = kwargs.pop('padding', '')
        kwargs.setdefault('bias', False)
        padding, is_dynamic = self.get_padding_value(padding, kernel_size, **kwargs)
        if is_dynamic:
            return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
        else:
            if isinstance(kernel_size, tuple):
                padding = (0,padding)
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
        
    def get_padding_value(self, padding, kernel_size, **kwargs):
        dynamic = False
        if isinstance(padding, str):
            # for any string padding, the padding will be calculated for you, one of three ways
            padding = padding.lower()
            if padding == 'same':
                # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
                if self._is_static_pad(kernel_size, **kwargs):
                    # static case, no extra overhead
                    padding = self._get_padding(kernel_size, **kwargs)
                else:
                    # dynamic padding
                    padding = 0
                    dynamic = True
            elif padding == 'valid':
                # 'VALID' padding, same as padding=0
                padding = 0
            else:
                # Default to PyTorch style 'same'-ish symmetric padding
                padding = self._get_padding(kernel_size, **kwargs)
        return padding, dynamic
    
    def _is_static_pad(self, kernel_size, stride=1, dilation=1, **_):
        return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

    def _get_padding(self, kernel_size, stride=1, dilation=1, **_):
        if isinstance(kernel_size, tuple):
            kernel_size = max(kernel_size)
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        return padding

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x
    
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, weight_norm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.weight_norm = weight_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.weight_norm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, weight_norm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.weight_norm = weight_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.weight_norm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

#%% Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class StdLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim = self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma ,ima = x.max(dim = self.dim, keepdim=True)
        return ma

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)