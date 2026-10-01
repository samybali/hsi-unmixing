import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

import core.utils as utils

def compute_metrics_and_plot(E_hat=None, A_hat=None, A_gt=None, E_gt=None):
    """
    Displays the predicted endmembers and abundances
    """
    n_graph = None

    sad = utils.SADLoss()
    mse = nn.MSELoss(reduction="sum")

    bg_colors = ["mediumpurple", "cornflowerblue", "indianred", "goldenrod", "mediumseagreen", "lightpink"]
    colors = ["thistle", "lavender", "mistyrose", "lightyellow", "lightblue", "lavenderblush"]

    if E_gt != None:
        E_gt = E_gt.detach().cpu()
        E_hat = E_hat.detach().cpu()
        A_hat = A_hat.detach().cpu()
        if E_gt.dim() == 3:
            E_gt = E_gt[0]

        E_hat, A_hat, indices = utils.order_endmembers(E_gt.T, E_hat.T , A_hat)
        E_hat = E_hat.T

    if E_hat != None:

        E_hat = E_hat.detach().cpu()
        if E_hat.dim() == 3:   
            E_hat = E_hat[0]

        c = E_hat.shape[1]
        n_graph = c // 2
        if c % 2 != 0: n_graph = n_graph + 1
            
        if E_gt != None:
            
            E_hat = utils.normalize(E_hat, is_endmember=True)
            E_gt = utils.normalize(E_gt, is_endmember=True)

            total_sad = sad(E_gt, E_hat)

            fig, axes = plt.subplots(2, n_graph, figsize=(7,5))
            axes = axes.flatten()
            for i in range(c):
                ax = axes[i]

                sad_val = sad(E_gt[:, i], E_hat[:, i])

                ax.plot(E_gt[:, i], 'r', linewidth=1.0, label='GT')
                ax.plot(E_hat[:, i], 'k-', linewidth=1.0, label='predict')
                ax.set_title(f"SAD = {format(sad_val, '.2f')}", fontsize=10, pad=10, backgroundcolor=bg_colors[indices[i].item()], color=colors[indices[i].item()])

                if i == 0:
                    ax.legend() 

            for j in range(i + 1, len(axes)):
                axes[j].axis('off')

            plt.subplots_adjust(hspace=0.5, wspace=0.4)

            E_title = f"Endmember estimation, SAD = {format(total_sad, '.3f')}"
                
            plt.suptitle(E_title)

        else:
            E_title = 'Endmember estimation'
            for i in range(c):
                ax = plt.subplot(2, n_graph, i + 1)
                plt.plot(E_hat[:, i].detach().cpu(), 'k-', linewidth=1.0, label='predict')
            plt.suptitle(E_title)

    if A_gt is not None:
        
        A_gt = A_gt.detach().cpu()
        A_hat = A_hat.detach().cpu()
        if A_hat.dim() == 4:
            A_hat = A_hat[0]
        
        if A_gt.dim() == 4:
            A_gt = A_gt[0]

        if n_graph is None:
            c = A_hat.shape[0]
            n_graph = c // 2
            if c % 2 != 0: n_graph = n_graph + 1

        A_hat = utils.normalize(A_hat)
        A_gt = utils.normalize(A_gt)
        
        total_mse = mse(A_gt, A_hat)/(torch.norm(A_gt)**2)

        fig, axes = plt.subplots(2, c, figsize=(10, 5))
        for i in range(c):
            pred = axes[0, i].imshow(A_hat[i].detach().cpu())

            mse_val = mse(A_gt[i], A_hat[i])/(torch.norm(A_gt[i])**2)

            axes[0, i].set_title(f"NMSE = {format(mse_val, '.2f')}", fontsize=10, pad=10, backgroundcolor=bg_colors[indices[i].item()], color=colors[indices[i].item()])
            axes[0, i].axis('off')

            gt = axes[1, i].imshow(A_gt[i].detach().cpu())
            axes[1, i].axis('off')

            fig.colorbar(pred, ax=axes[0, i], fraction=0.046, pad=0.04)
            fig.colorbar(gt, ax=axes[1, i], fraction=0.046, pad=0.04)
        fig.text(0.05, 0.7, 'prediction', va='center', ha='center', fontsize=12, rotation='vertical')
        fig.text(0.05, 0.4, 'gt', va='center', ha='center', fontsize=12, rotation='vertical')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.5, bottom=0.1, wspace=0.1, hspace=0.5)
        fig.tight_layout(rect=[0.05, 0.25, 0.95, 0.9])

        A_title = f"Abundance estimation, NMSE = {format(total_mse, '.3f')}"
        plt.suptitle(A_title)

    else:
        A_hat = A_hat.detach().cpu()
        if A_hat.dim() == 4:
            A_hat = A_hat[0]

        c = A_hat.shape[0]

        fig, axes = plt.subplots(1, c, figsize=(10, 5))
        for i in range(c):
            pred = axes[i].imshow(A_hat[i].detach().cpu())
            axes[i].axis('off')
            fig.colorbar(pred, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Adaptative placement of the title as a function of the number of endmembers
        offset_y = -(0.1/3) * c + 0.9
        plt.suptitle("Abundance estimation", y=offset_y)

    return total_sad, total_mse

def plot_hsi(Y, n_channels=4, rgb=False, title=None, channels=None, cmap="viridis"):
    """
    Displays n channels of the input HSI
    Must be of shape (batch, B, H, W) or (B, H, W)
    """
    if Y.dim() > 3:
        Y = Y.squeeze(0)
    B, _, _ = Y.shape

    if channels != None:
        fig, axes = plt.subplots(1, len(channels), figsize=(20,20))

        for idx, i in enumerate(channels):

            if rgb:
                assert i+3 <= B, "Can't display 3 channels if selected channel is the final channel"
                im = axes[idx].imshow(Y[i:i+3].permute(1,2,0).detach().cpu())
            else:
                im = axes[idx].imshow(Y[i].detach().cpu())
                fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            axes[idx].set_title(f"Channel {i} / {B}")
            
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])

    else:
        fig, axes = plt.subplots(1, n_channels, figsize=(20, 20))
        step = (B - 1) / (n_channels - 1)

        for k, idx in enumerate(range(n_channels)):
            i = int(k*step)

            if rgb:
                im = axes[idx].imshow(Y[i:i+3].permute(1,2,0).detach().cpu())
            else:
                im = axes[idx].imshow(Y[i].detach().cpu())
                fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            axes[idx].set_title(f"Channel {i} / {B}")
            
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
    
    if title != None:
        plt.suptitle(title, y=0.6)