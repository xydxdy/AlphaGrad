
from matplotlib import cm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import os

from mtl_bci.grads import GradApprox
from mtl_bci.utils.tools import write_log
################################################################################
#
# Define the Optimization Problem
#
################################################################################
LOWER = 0.000005
ITER = 20000
ALPHA = 0.001
DEVICE = "cpu"
RUN_MEDTHODS =  ["GradApprox", "fixed", "AdaMT", "AlphaGrad"]
BATCHSIZE = 10 # FOR GradApprox
WARMUP = 20 # FOR GradApprox

X0 = [-8.5, 7.5]
X1 = [-10., -1.]
X2 = [9.,   9.]
        
class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()
        self.centers = torch.Tensor([
            [-3.0, 0],
            [3.0, 0]]).to(DEVICE)

    def forward(self, x, batch=False):
        x = x.to(DEVICE)
        
        if not batch:
            x1 = x[0]
            x2 = x[1]
        else:
            x1 = x[:, 0]
            x2 = x[:, 1]

        f1 = torch.clamp((0.5*(-x1-7)-torch.tanh(-x2)).abs(), LOWER).log() + 6
        f2 = torch.clamp((0.5*(-x1+3)+torch.tanh(-x2)+2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2*0.5), 0)

        f1_sq = ((-x1).pow(2) + 0.1*(-x2).pow(2)) / 10 - 20
        f2_sq = ((-x1).pow(2) + 0.1*(-x2).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2*0.5), 0)

        f1 = f1 * c1 + f1_sq * c2
        f2 = f2 * c1 + f2_sq * c2

        f = torch.stack([f1, f2]).to(DEVICE)
        return f
    
################################################################################
#
# Plot Utils
#
################################################################################

def plotme(F, all_traj=None, xl=11):
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.forward(Xs, batch=True)

    colormaps = {
        "AlphaGrad": "tab:red",
        "fixed": "tab:blue",
        "AdaMT": "tab:orange",
        "GradApprox": "tab:cyan"
    }

    plt.figure(figsize=(12, 5))
    plt.subplot(131)
    c = plt.contour(X, Y, Ys[0].view(n,n))
    if all_traj is not None:
        for i, (k, v) in enumerate(all_traj.items()):
            plt.plot(all_traj[k][:,0], all_traj[k][:,1], '--', c=colormaps[k], label=k)
    plt.title("L1(x)")

    plt.subplot(132)
    c = plt.contour(X, Y, Ys[1].view(n,n))
    if all_traj is not None:
        for i, (k, v) in enumerate(all_traj.items()):
            plt.plot(all_traj[k][:,0], all_traj[k][:,1], '--', c=colormaps[k], label=k)
    plt.title("L2(x)")

    plt.subplot(133)
    c = plt.contour(X, Y, Ys.mean(0).view(n,n))
    if all_traj is not None:
        for i, (k, v) in enumerate(all_traj.items()):
            plt.plot(all_traj[k][:,0], all_traj[k][:,1], '--', c=colormaps[k], label=k)
    plt.legend()
    plt.title("0.5*(L1(x)+L2(x))")

    plt.tight_layout()
    plt.savefig(f"toy_ct.png")

def plot3d(F, xl=11):
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.forward(Xs, batch=True)
    print(Ys)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    Yv = Ys.mean(0).view(n,n)
    surf = ax.plot_surface(X, Y, Yv.numpy(), cmap=cm.viridis)
    print(Ys.mean(1).min(), Ys.mean(1).max())

    ax.set_zticks([-16, -8, 0, 8])
    ax.set_zlim(-20, 10)

    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])

    ax.view_init(25)
    plt.tight_layout()
    plt.savefig(f"{MAIN_DIR}/3d-obj.png", dpi=1000)

def plot_contour(F, task=1, traj=None, xl=11, plotbar=False, name="tmp"): 
    n = 500
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    Xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    Ys = F.forward(Xs,  batch=True)

    cmap = cm.get_cmap('viridis')

    # yy = -8.3552
    yy = -5.8
    if task == 0:
        Yv = Ys.mean(0)
        plt.plot(X0[0], X0[1], marker='o', markersize=10, zorder=5, color='k')
        plt.plot(X1[0], X1[1], marker='o', markersize=10, zorder=5, color='k')
        plt.plot(X2[0], X2[1], marker='o', markersize=10, zorder=5, color='k')
        plt.plot([-7, 7], [yy, yy], linewidth=8.0, zorder=0, color='gray')
        plt.plot(0, yy, marker='*', markersize=15, zorder=5, color='k')
    elif task == 1:
        Yv = Ys[0]
        plt.plot(0, yy, marker='*', markersize=15, zorder=5, color='k')
    else:
        Yv = Ys[1]
        plt.plot(0, yy, marker='*', markersize=15, zorder=5, color='k')

    c = plt.contour(X, Y, Yv.view(n,n), cmap=cm.viridis, linewidths=4.0)
    # c = plt.contourf(X, Y, Yv.view(n, n), levels=100, cmap=cm.viridis)

    if traj is not None:
        for tt in traj:
            l = tt.shape[0]
            color_list = np.zeros((l,3))
            color_list[:,0] = 1.
            color_list[:,1] = np.linspace(0, 1, l)
            #color_list[:,2] = 1-np.linspace(0, 1, l)
            ax.scatter(tt[:,0], tt[:,1], color=color_list, s=6, zorder=10)

    if plotbar:
        cbar = fig.colorbar(c, ticks=[-15, -10, -5, 0, 5])
        cbar.ax.tick_params(labelsize=15)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.xticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.yticks([-10, -5, 0, 5, 10], fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=100)
    plt.close()

def smooth(x, n=20):
    l = len(x)
    y = []
    for i in range(l):
        ii = max(0, i-n)
        jj = min(i+n, l-1)
        v = np.array(x[ii:jj]).astype(np.float64)
        if i < 3:
            y.append(x[i])
        else:
            y.append(v.mean())
    return y

def plot_loss(trajs, name="tmp"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormaps = {
        "AlphaGrad": "tab:red",
        "fixed": "tab:blue",
        "AdaMT": "tab:orange",
        "GradApprox": "tab:cyan"
    }
    maps = {
        "AlphaGrad": "AlphaGrad",
        "fixed" : "Fixed Equal Weights",
        "AdaMT" : "AdaMT",
        "GradApprox": "GradApprox"
    }
    for method in RUN_MEDTHODS:
        traj = trajs[method][::100]
        Ys = F.forward(traj, batch=True)
        x = np.arange(traj.shape[0])
        #y = torch.cummin(Ys.mean(1), 0)[0]
        y = Ys.mean(0)

        ax.plot(x, smooth(list(y)),
                color=colormaps[method],
                linestyle='-',
                label=maps[method], linewidth=4.)

    plt.xticks([0, 40, 80, 120, 160, 200],
               ["0", "4K", "8K", "120K", "160K", "200K"],
               fontsize=15)

    plt.yticks(fontsize=15)
    ax.grid()
    plt.legend(fontsize=15)

    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.tight_layout()
    plt.savefig(f"{name}.png", dpi=100)
    plt.close()

################################################################################
#
# Multi-Objective Optimization Solver
#
################################################################################

def fixed_weights(grads, w, it):
    return w

def AlphaGrad(grads, w, it, p=1):
    reg = 1e-4
    eps = 1e-7
    g1 = torch.norm(grads[0], p=p) # grad task 1
    g2 = torch.norm(grads[1], p=p) # grad task 2
    gw = torch.stack([g1, g2])
    grad  = gw / (torch.norm(gw, p=p) + eps)
    w = w - (ALPHA * grad)
    w = torch.clip(w, min=reg)
    w = w / w.sum() * 2
    return w

def AdaMT(grads, w, it):
    reg = 0.1
    eps = 1e-7
    g1 = torch.mean(grads[0])
    g2 = torch.mean(grads[1])
    gw = torch.stack([g1, g2])
    w  = gw / (torch.sum(gw) + eps)
    w = torch.clip(w, min=reg)
    w = w / w.sum() * 2
    return w

### Define the problem ###
F = Toy()

maps = {
    "fixed": fixed_weights,
    "AlphaGrad": AlphaGrad,
    "AdaMT": AdaMT,
    "GradApprox": GradApprox,
}

### Start experiments ###

def run_all():
    
    all_traj = {}

    # the initial positions
    inits = [
        torch.Tensor(X0),
        torch.Tensor(X1),
        torch.Tensor(X2),
    ]
    val_inits = [
        torch.Tensor(X0)+0.35,
        torch.Tensor(X1)-0.25,
        torch.Tensor(X2)+0.25,
    ]
    w_inits = [
        torch.Tensor([1.0, 1.0]),
        torch.Tensor([1.0, 1.0]),
        torch.Tensor([1.0, 1.0]),
    ]

    for i, (w_init, init, v_init) in enumerate(zip(w_inits, inits, val_inits)):
        write_log(filepath=f'{MAIN_DIR}/logs', name=f'{MAIN_DIR}/toy_time_log_{i}.csv', data=["method", "iter", "time"], mode='w')
        for m in tqdm(RUN_MEDTHODS):
            all_traj[m] = None
            traj = []
            solver = maps[m]
            x = init.clone()
            x_val = v_init.clone()
            x.requires_grad = True
            
            w = w_init.clone()
            opt = torch.optim.Adam([x], lr=0.001)

            if solver.__name__ == "GradApprox":
                grad_approx = solver(loss_weights=w, warmup_epoch=WARMUP, overlap=BATCHSIZE)
            
            list_weighted_task_loss = []
            list_val_weighted_task_loss = []
            
            for it in range(ITER):
                
                time_start = time.time()
                
                traj.append(x.detach().numpy().copy())
                
                with torch.enable_grad():
                    opt.zero_grad()
                    task_loss = F(x)
                    # task_loss = torch.stack([loss[0], loss[1]]).to(DEVICE)
                    weighted_task_loss = torch.mul(
                        w.to(DEVICE), task_loss
                    ).to(DEVICE)

                    if solver.__name__ == "AdaMT" or solver.__name__ == "AlphaGrad":
                        g1 = torch.autograd.grad(weighted_task_loss[0], x, retain_graph=True)[0]
                        g2 = torch.autograd.grad(weighted_task_loss[1], x, retain_graph=True)[0]
                        w = solver([g1, g2], w, it)
                        
                    total_loss = torch.sum(weighted_task_loss).to(DEVICE)
                    total_loss.backward(retain_graph=True)
                
                    opt.step()
                
                if solver.__name__ == "GradApprox":
                    with torch.no_grad():
                        val_task_loss = F(x_val)
                        val_weighted_task_loss = torch.mul(
                            w.to(DEVICE), val_task_loss
                        ).to(DEVICE)
                        val_total_loss = torch.sum(val_weighted_task_loss).to(DEVICE)

                        if it % BATCHSIZE != 0:
                            list_weighted_task_loss.append(weighted_task_loss.tolist())
                            list_val_weighted_task_loss.append(val_weighted_task_loss.tolist())
                        else:
                            if it >= BATCHSIZE:
                                # print(list_weighted_task_loss)
                                grad_approx.add_losses(np.array(list_weighted_task_loss).swapaxes(0, 1), 
                                                       np.array(list_val_weighted_task_loss).swapaxes(0, 1))
                                w = grad_approx.compute_adaptive_weights()
                                print(w)
                            list_weighted_task_loss = []
                            list_val_weighted_task_loss = []
                
                time_end = time.time()
                write_log(filepath=f'{MAIN_DIR}/logs', name=f'{MAIN_DIR}/toy_time_log_{i}.csv', data=[solver.__name__, it, time_end-time_start], mode='a')       
            all_traj[m] = torch.tensor(traj)
        torch.save(all_traj, f"{MAIN_DIR}/toy{i}.pt")


def plot_results():
    plot3d(F)
    # plotme(F)
    plot_contour(F, 1, name=f"{MAIN_DIR}/toy_task_1")
    plot_contour(F, 2, name=f"{MAIN_DIR}/toy_task_2")
    t1 = torch.load(f"{MAIN_DIR}/toy0.pt")
    t2 = torch.load(f"{MAIN_DIR}/toy1.pt")
    t3 = torch.load(f"{MAIN_DIR}/toy2.pt")
    
    plot_loss(trajs=t1, name=f"{MAIN_DIR}/loss/toy_loss_t1")
    plot_loss(trajs=t2, name=f"{MAIN_DIR}/loss/toy_loss_t2")
    plot_loss(trajs=t3, name=f"{MAIN_DIR}/loss/toy_loss_t3")

    length = t1["fixed"].shape[0]

    for method in RUN_MEDTHODS:
        ranges = list(range(10, length, 1000))
        ranges.append(length-1)
        for t in tqdm(ranges):
            plot_contour(F,
                         task=0, # task == 0 meeas plot for both tasks
                         traj=[t1[method][:t],t2[method][:t],t3[method][:t]],
                         plotbar=(method == "AlphaGrad"),
                         name=f"{MAIN_DIR}/imgs/toy_{method}_{t}")


if __name__ == "__main__":
    
    MAIN_DIR = os.path.dirname(os.path.abspath(__file__)) + "/toys"
    
    os.makedirs(f"{MAIN_DIR}/imgs/", exist_ok=True)
    os.makedirs(f"{MAIN_DIR}/loss/", exist_ok=True)
    os.makedirs(f"{MAIN_DIR}/logs/", exist_ok=True)
    
    run_all()
    plot_results()