import numpy as np
import torch
import copy
from . import gradpolicy


class AlphaGrad():
    def __init__(self, loss_weights, lr=0.01, p="fro", device="cuda:0", **kwargs): #0.01
        self.device = device
        self.loss_weights = self.convert(loss_weights)
        self.lr = lr
        self.p = p # p='fro' : (Frobenius norm: L2-norm)
        self.T = len(loss_weights)
        self.reg = 1e-4
        self.eps = 1e-7
        self.history = [self.loss_weights.detach().tolist()]
        super(AlphaGrad, self).__init__(**kwargs)
        
    def convert(self, loss_weights):
        if isinstance(loss_weights, list):
            loss_weights = torch.nn.Parameter(torch.Tensor(loss_weights))
        if isinstance(loss_weights, torch.Tensor):
            loss_weights = loss_weights
        return loss_weights
    
    def backward(self, layer, task_loss):   
        gw = []
        for i in range(self.T):
            dl = torch.autograd.grad(task_loss[i]*self.loss_weights[i], layer.parameters(), create_graph=True)[0]
            gw.append(torch.norm(dl, p=self.p))
        self.gw = torch.stack(gw)
            
        self.grad  = self.gw / (torch.norm(self.gw, p=self.p) + self.eps)
            
    def step(self):
        self.loss_weights = self.loss_weights.detach().to(self.device) - (self.lr * self.grad)
        self.loss_weights = torch.clip(self.loss_weights, min=self.reg)
        self.loss_weights = (self.loss_weights / self.loss_weights.sum() * self.T).detach()
        self.history.append(self.loss_weights.detach().tolist())
    
    def get_loss_weights(self):
        return self.loss_weights


class GradNorm():
    def __init__(self, loss_weights, lr=0.1, device="cuda:0", **kwargs): #0.01
        self.device = device
        self.loss_weights = self.convert(loss_weights)
        self.lr = lr
        self.T = len(loss_weights)
        self.reg = 0.1
        self.eps = 1e-7
        self.history = [self.loss_weights.detach().tolist()]
        super(GradNorm, self).__init__(**kwargs)
        
    def convert(self, loss_weights):
        if isinstance(loss_weights, list):
            loss_weights = torch.nn.Parameter(torch.Tensor(loss_weights))
        if isinstance(loss_weights, torch.Tensor):
            loss_weights = loss_weights
        return loss_weights
    
    def backward(self, layer, task_loss):   
        gw = []
        for i in range(self.T):
            dl = torch.autograd.grad(task_loss[i]*self.loss_weights[i], layer.parameters(), create_graph=True)[0]
            gw.append(torch.mean(dl))
        self.gw = torch.stack(gw)
            
        self.grad  = self.gw / (torch.sum(torch.abs(self.gw)) + self.eps)
            
    def step(self):
        self.loss_weights = self.loss_weights.detach().to(self.device) - (self.lr * self.grad)
        self.loss_weights = torch.clip(self.loss_weights, min=self.reg)
        self.loss_weights = (self.loss_weights / self.loss_weights.sum() * self.T).detach()
        self.history.append(self.loss_weights.detach().tolist())
    
    def get_loss_weights(self):
        return self.loss_weights

class AdaMT():
    def __init__(self, loss_weights, device="cuda:0", **kwargs): 
        self.device = device
        self.loss_weights = self.convert(loss_weights)
        self.T = len(loss_weights)
        self.reg = 0.1
        self.eps = 1e-7
        self.history = [self.loss_weights.detach().tolist()]
        super(AdaMT, self).__init__(**kwargs)
        
    def convert(self, loss_weights):
        if isinstance(loss_weights, list):
            loss_weights = torch.nn.Parameter(torch.Tensor(loss_weights))
        if isinstance(loss_weights, torch.Tensor):
            loss_weights = loss_weights
        return loss_weights
    
    def backward(self, layer, task_loss):   
        gw = []
        for i in range(self.T):
            dl = torch.autograd.grad(task_loss[i]*self.loss_weights[i], layer.parameters(), create_graph=True)[0]
            gw.append(torch.mean(dl))
        self.gw = torch.stack(gw)
            
    def step(self):
        self.loss_weights = self.gw / (torch.sum(self.gw) + self.eps)
        # norm loss_weights
        self.loss_weights = torch.clip(self.loss_weights, min=self.reg)
        self.loss_weights = (self.loss_weights / self.loss_weights.sum() * self.T).detach()
        self.history.append(self.loss_weights.detach().tolist())
    
    def get_loss_weights(self):
        return self.loss_weights
    
    
class GradApprox(object):
    def __init__(
        self, loss_weights, policy="HistoricalTangentSlope", warmup_epoch=2, overlap=1, **kwargs
    ):
        self.loss_weights = self.convert(loss_weights)
        self.policy = gradpolicy.__dict__[policy]
        self.warmup_epoch = warmup_epoch
        self.epoch = 0
        self.T = len(self.loss_weights)
        self.history = [self.loss_weights]
        self.gradient = [self.policy(overlap=overlap) for i in range(self.T)]

    def convert(self,loss_weights):
        if isinstance(loss_weights, torch.Tensor):
            loss_weights = loss_weights.detach().tolist()
        return loss_weights

    def add_losses(self, train_losses, valid_losses):
        self.epoch += 1
        if len(train_losses) != self.T:
            raise Exception(
                "len of train_losses not match with `n`: ",
                len(train_losses),
                "!=",
                self.T,
            )
        for i, (tr_loss, va_loss) in enumerate(zip(train_losses, valid_losses)):
            self.gradient[i].add_losses(tr_loss, va_loss)

    def compute_weights(self):
        w = []
        for i in range(self.T):
            w_, g_, o_ = self.gradient[i].compute_weights()
            w.append(w_)
        return w

    def compute_adaptive_weights(self, to_tensor=True):
        weights = copy.deepcopy(self.loss_weights)
        if self.epoch > self.warmup_epoch:
            w_list = self.compute_weights()
            if isinstance(self.policy, gradpolicy.BlendingRatio):
                w_min = np.min(w_list)
                if w_min != 0.0:
                    zeros = [1e-5] * self.T
                    weights = np.array(list(map(max, w_min / w_list, zeros)))
                    weights = weights / weights.sum() * self.T
                else:
                    weights = self.loss_weights
            else:
                w_sum = np.sum(w_list)
                if w_sum != 0.0:
                    zeros = [1e-5] * self.T
                    weights = list(map(max, w_list / w_sum * self.T, zeros))
                else:
                    weights = self.loss_weights
        
        self.history.append(weights)

        if to_tensor:
            weights = torch.Tensor(weights)

        return weights