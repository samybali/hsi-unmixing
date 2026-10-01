import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
import math
import scipy.io as io
import os
from PIL import Image

def load_hsi(name, normalize_Y=True):
    """
    Loads the HSI from the dataset folder as well as the unmixing ground truth and wavelength list
    
    Args:
        normalize_Y (bool, optional): if set to True, will scale Y to have max=1
    """

    data = io.loadmat(f"datasets/{name}.mat")
    Y_flat = torch.tensor(data["Y"], dtype=torch.float)
    A_flat = torch.tensor(data["A"], dtype=torch.float)
    E_init = torch.tensor(data["E"], dtype=torch.float)

    Y_init = oneD_to_2d(Y_flat)

    if normalize_Y:
        Y_init = Y_init/Y_init.max()

    A_init = oneD_to_2d(A_flat)
    Y_init = Y_init.unsqueeze(0)
    A_init = A_init.unsqueeze(0)

    wavelengths_path = f"datasets/{name}_wavelength.txt"
    with open(wavelengths_path, "r") as file:
        lines = file.readlines()
        wavelengths = [float(line.strip()) for line in lines if line.strip()]
    
    return Y_init, A_init, E_init, wavelengths

def oneD_to_2d(Y, H=None, W=None):
    """
    Reshapes 1D tensors to 2D image
    If H and W are not passed, assumes the image is square
    
    Args:
        Y: input tensor to be reshaped
    """
    is_batched = Y.dim()==3
    if is_batched:
        batch, B, N = Y.shape
        if H != None:
            return Y.reshape(batch, B, H, W)
        
        else:
            H = int(N**0.5)
            return Y.reshape(batch, B, H, H)
    else:
        B, N = Y.shape
        if H != None:
            return Y.reshape(B, H, W)
        
        else:
            H = int(N**0.5)
            return Y.reshape(B, H, H)

class TVLoss(nn.Module):
    def __init__(self, reduction=None):
        super(TVLoss,self).__init__()
        self.reduction = reduction

    def forward(self,x):
        """
        Expects input x to be of shape (batch, B, H, W)
        """
        batch = x.shape[0]

        diff1 = x[..., 1:, :] - x[..., :-1, :]
        diff2 = x[..., :, 1:] - x[..., :, :-1]

        res1 = diff1.abs().sum([1, 2, 3])
        res2 = diff2.abs().sum([1, 2, 3])
        score = res1 + res2

        if self.reduction == "mean":
            return score.sum() / batch
        elif self.reduction == "sum":
            return score.sum()
        elif self.reduction is None or batch == "none":
            return score[0]

class SADLoss(nn.Module):
    """
    SAD loss function for EndMember matrices. To use it on Abundances, transpose the two inputs. (Doesn't correct permutations)
    """
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, y_true, y_pred):
        if y_pred.dim() == 1:
            y_true = F.normalize(y_true, dim=0, p=2)
            y_pred = F.normalize(y_pred, dim=0, p=2)
        else:
            y_true = F.normalize(y_true, dim=1, p=2)
            y_pred = F.normalize(y_pred, dim=1, p=2)

        A = torch.mul(y_true, y_pred)

        if y_true.dim() == 1:
            A = torch.sum(A, dim=0)
        else:
            A = torch.sum(A, dim=1)
            
        sad = torch.acos(A)
        loss = torch.mean(sad)
        return loss

def SiVM(Y, c): 
    """ 
    SiVM endmember extractor based on UnDIP's repository 

    Args: 
        Y: input HSI to extract endmembers from (shape (B, N) or (B, H, W), no batch) 
        c (int): the number of endmembders to extract 
    """ 
    dev = Y.device

    if Y.dim()!= 2:
        if Y.dim() ==4:
            Y = Y[0]
        B, h, w = Y.shape
        N = h*w
        Y = Y.reshape(B, N)
    else:
        B, N = Y.shape
    
    Vh, S, U = torch.linalg.svd(Y, full_matrices=False)
    PC = torch.diag(S) @ U 
    Yp = Vh[:, :c] @ PC[:c, :] 
    d = torch.zeros((c, N), device=dev) # distance matrix 
    I = [] # endmembers indices 
    
    # First endmember: farthest from origin 
    d[0] = torch.sum(Y**2, dim=0) 
    I.append(torch.argmax(d[0, :])) 
    
    for v in range(1, c): 
        E = Yp[:, I] # Selected endmembers (shape: B x v) 
        P = E @ torch.linalg.pinv(E.T @ E) @ E.T 
        residual = Yp - P @ Yp 
        d[v] = torch.sum(residual**2, dim=0) # Squared orthogonal distance 
        d[v, I] = -torch.inf 
        I.append(torch.argmax(d[v])) 
    
    E = Yp[:, I] 
        
    return E

def order_endmembers(tensor_gt, tensor_hat, tensor2_hat=None):
    """
    Uses scipy linear_sum_assignement algorithm to reorder tensor_hat columns to match tensor_gt
    Tensors must be of shape (batch, D, X) or (D, X) where D is the axis along which to reorder
    tensor_2_hat is another tensor that can be reordered based on the tensor_hat reordering (for abundances)
    """
    is_batched = True
    if tensor_hat.dim() == 2:
        is_batched = False
        tensor_hat = tensor_hat.unsqueeze(0)
    if tensor_gt.dim() == 2:
        is_batched = False
        tensor_gt = tensor_gt.unsqueeze(0)

    tensor_hat_ordered = torch.zeros_like(tensor_hat)
        
    if tensor2_hat != None:
        if tensor2_hat.dim() < 4:
            tensor2_hat = tensor2_hat.unsqueeze(0)

        tensor2_hat_ordered = torch.zeros_like(tensor2_hat)

    for b in range(tensor_hat.size()[0]):

        # Normalize the tensors
        tensor_gt_norm = F.normalize(tensor_gt[b], p=2.0, dim=1)  # Normalize along reordered axis
        tensor_hat_norm = F.normalize(tensor_hat[b], p=2.0, dim=1)

        # Compute cost matrix (cosine distance)
        cost_matrix = torch.acos(torch.clamp(tensor_gt_norm @ tensor_hat_norm.T, -1.0, 1.0))
        cost_matrix_np = cost_matrix.cpu().numpy() 

        # Solve assignment problem
        _, col_ind = linear_sum_assignment(cost_matrix_np)

        # Reorder E_hat to match E_gt
        tensor_hat_ordered[b] = tensor_hat[b, col_ind]

        if tensor2_hat != None:
            tensor2_hat_ordered[b] = tensor2_hat[b, col_ind]

    if tensor2_hat != None:
        if is_batched:
            return tensor_hat_ordered, tensor2_hat_ordered, col_ind
        else:
            return tensor_hat_ordered[0], tensor2_hat_ordered[0], col_ind
    else:
        if is_batched:
            return tensor_hat_ordered, col_ind
        else:
            return tensor_hat_ordered[0], col_ind
   
def normalize(X, is_endmember=False):
    """
    Normalizes batched tensors

    If is_endmember -> set max to 1 for every endmember
    If is Y or A (must be of shape (batch, B/c, H, W)) -> divides by frobenius norm
    """

    if is_endmember:
        X_norm = X/torch.max(X, dim=0).values

    else:
        X_norm = X/torch.norm(X)

    return X_norm