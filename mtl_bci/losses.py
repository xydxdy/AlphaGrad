import torch
import torch.nn as nn
from torch.autograd.function import Function
import numpy as np
from pytorch_metric_learning import losses, miners

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)
        self.mining_func = miners.TripletMarginMiner(
            margin=self.margin, type_of_triplets="semihard"
        )

    def forward(self, embeddings, labels, **kwargs):
        indices_tuple = self.mining_func(embeddings, labels)
        loss = self.loss_func(embeddings, labels, indices_tuple)
        return loss

class CenterLoss(nn.Module):
    def __init__(self, num_classes=2, latent_dim=1152, size_average=True, reduction="mean", **kwargs):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.size_average = size_average
        self.reduction = reduction
        self.device = torch.cuda.current_device()
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
        
        self.centers = None
        self.centerlossfunc = CenterlossFunc.apply

    def forward(self, embeddings, labels):
        if self.centers is None:
            np.random.seed(19981127) 
            centers = np.random.randn(self.num_classes, self.latent_dim)
            self.device = embeddings.device
            self.centers = nn.Parameter(torch.from_numpy(centers)).to(embeddings.device)
        batch_size = embeddings.size(0)
        embeddings = embeddings.view(batch_size, -1)
        # To check the dim of centers and features
        if embeddings.size(1) != self.latent_dim:
            raise ValueError(
                "Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(
                    self.latent_dim, embeddings.size(1)
                )
            )
        batch_size_tensor = embeddings.new_empty(1).fill_(
            batch_size if self.size_average else 1
        )
        loss = self.centerlossfunc(embeddings.to(self.device), 
                                   labels.to(self.device), 
                                   self.centers.to(self.device), 
                                   batch_size_tensor.to(self.device))
        if self.reduction == "mean":
            loss = torch.mean(loss)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, embeddings, labels, centers, batch_size):
        ctx.save_for_backward(embeddings, labels, centers, batch_size)
        centers_batch = centers.index_select(0, labels.long())
        return (embeddings - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        embeddings, labels, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, labels.long())
        diff = centers_batch - embeddings
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(labels.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, labels.long(), ones)
        grad_centers.scatter_add_(
            0, labels.unsqueeze(1).expand(embeddings.size()).long(), diff
        )
        grad_centers = grad_centers / counts.view(-1, 1)
        return -grad_output * diff / batch_size, None, grad_centers / batch_size, None
