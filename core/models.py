import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
import sys
from pathlib import Path
from timm.models.vision_transformer import Block

import core.utils as utils

cwd = Path.cwd()
global_path = cwd.parent

sys.path.append(f"{global_path}")
from spectral_earth.src.backbones import spec_vit

sys.path.append(f"{global_path}/DOFA")
from dofa.wave_dynamic_layer import Dynamic_MLP_OFA

class OFAViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, drop_rate=0.,
                 embed_dim=1024, depth=24, num_heads=16, out_indices=[23], wv_planes=128, num_classes=45,
                 global_pool=True, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)
        
        self.out_indices = out_indices

        self.patch_embed = Dynamic_MLP_OFA(wv_planes=wv_planes, kernel_size=patch_size, embed_dim=embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, wave_list):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist
        x, _ = self.patch_embed(x, self.waves)
        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Use every block output of out_indices
            # The final features will be (N*embed_dim, num_patches+1)
            # Where N is the number of outputs in out_indices
            if i in self.out_indices:
                features.append(x[:, 1:, :].squeeze(0).T)

        features = torch.cat(features, dim=0)

        return features

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, wave_list):
        x = self.forward_features(x, wave_list)
        x = self.forward_head(x)
        return x

def create_dofa(Y, A, path="path/to/checkpoints"):
    batch, B, H, _ = Y.shape
    device = Y.device

    if H < 224:
        Y = F.interpolate(Y, size=(224,224))
        A = F.interpolate(A, size=(224,224))
        new_H = 224
    else:
        new_H = 224
    Y = Y[:,:,:new_H, :new_H]
    A = A[:,:,:new_H, :new_H]

    check_point = torch.load(f"{path}/DOFA_ViT_large_e100.pth", map_location=device)
    fm = OFAViT(
        img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, out_indices=[23], mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    fm.load_state_dict(check_point, strict=False)

    return fm, Y, A, new_H

def create_model_dofa(path="path/to/checkpoints"):
    check_point = torch.load(f"{path}/DOFA_ViT_large_e100.pth", map_location="cpu")
    fm = OFAViT(
        img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, out_indices=[23], mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    fm.load_state_dict(check_point, strict=False)

    return fm

def extract_features(dofa, Y, wavelengths):
    batch, B, H, _ = Y.shape

    if H < 224:
        Y = F.interpolate(Y, size=(224,224))
        new_H = 224
    else:
        new_H = 224
    Y = Y[:,:,:new_H, :new_H]
    
    features = dofa.forward_features(Y, wavelengths)
    
    return utils.oneD_to_2d(features)

def create_spectralearth(device, path="path/to/checkpoints"):

    fm = spec_vit.SpecViTBase() # or small, large, huge...
    checkpoint = torch.load(f"{path}/spec_ViTb_mae.pth", map_location=device)

    fm.load_state_dict(checkpoint, strict=False)

    return fm

def spectral_adapter(spectralearth, Y):
    adapter = spectralearth.spectral_adapter
    Y_adapted = adapter(Y)

    return Y_adapted

def init_decoder_weights(model, Y, c):
    """
    Initializes the model's decoder weights with SiVM's extracted endmembers
    input Y must be of shape (B, N) or (B, H, W) -> no batch
    """

    init_em = utils.SiVM(Y, c)

    model_dict = model.decoder.state_dict() 
    model_dict["decoder.weight"][:,:,0,0] = init_em
    model.decoder.load_state_dict(model_dict)

    return model

class weightConstraint(object):
    def __init__(self):
        pass
    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.clamp_(min=0)

class Sum_to_one(nn.Module):
    def __init__(self, scale=1):
        super(Sum_to_one, self).__init__()
        self.scale = scale
    def forward(self, x):
        # print(x.max())
        x = F.softmax(self.scale * x, dim=1)
        return x

class Decoder(nn.Module):
    def __init__(self, c, B, kernel_size=1):
        super(Decoder, self).__init__()
        self.B = B
        self.c = c
    
        padding = kernel_size //2
        self.decoder = nn.Conv2d(in_channels=c, out_channels=B,
                                kernel_size=kernel_size,stride=1,
                                padding=padding, bias=False)
        self.relu = nn.ReLU()

    def forward(self, code):

        code = self.relu(self.decoder(code))
        
        return code

    def get_endmembers(self):
        return self.decoder.weight.data.squeeze([2, 3])

class Unmixing_from_features(nn.Module):
    def __init__(self, D, B, c, H=224, alpha=None):
        """
        Args:
            D (int): The embed_dim
            B (int): The number of spectral bands in the hsi
            c (int): The number of endmembers to extract
            H (int): The size of the input hsi
            alpha (int): The size of the features

        """
        super(Unmixing_from_features, self).__init__()
        self.D = D
        self.alpha = alpha
        self.B = B
        self.c = c
        self.H = H

        # Upsampling features
        self.upsample = nn.Sequential(
            nn.Linear(self.alpha**2, self.H**2)
        )

        # Features to abundances
        self.abundance_estimator = nn.Sequential(
            nn.Conv2d(D, c, kernel_size=1, bias=False),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(c),
            nn.Dropout(0.2)
        )

        self.sum_to_one = Sum_to_one()
        self.decoder = Decoder(B=B, c=c)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, W_sad=1, W_ab=0.6, W_tv_e=3e-5,W_mse=0.09):
        sad = utils.SADLoss()
        tv = utils.TVLoss(reduction="mean")
        mse = nn.MSELoss(reduction='sum')
        print("ici")
        if torch.isnan(Y_hat).any() or torch.isinf(Y_hat).any():
            print("Crash détecté : Y_hat contient des NaN ou Inf")
        loss_sad = W_sad * sad(Y_gt, Y_hat)

        loss_mse = W_mse * mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)

        """Abundances and endmembers regularisation"""
        
        # TV on endmembers (sum of difference between consecutive endmembers)
        loss_tv_e = W_tv_e * (torch.abs(E_hat[:, 1:] - E_hat[:, :-1]).sum())
        
        loss_ab = W_ab * torch.sqrt(A_hat).mean()

        loss = loss_sad + loss_ab + loss_tv_e + loss_mse

        return loss

    def get_abundances(self, features):

        """Upsample the features"""
        features = features.reshape(self.D, self.alpha*self.alpha)
        features_up = self.upsample(features)
        features_up = utils.oneD_to_2d(features_up)
        features_up = (features_up - features_up.mean())/ (1e-8 + features_up.std())

        """Estimate the abundances from the upsampled features"""
        A_hat = self.abundance_estimator(features_up.unsqueeze(0))
        A_hat = self.sum_to_one(A_hat)

        return A_hat
    
    def get_endmembers(self):
        return self.decoder.get_endmembers()
    
    def forward(self, features):
        A_hat = self.get_abundances(features)
        Y_hat = self.decoder(A_hat)
        E_hat = self.decoder.get_endmembers()

        return E_hat, A_hat, Y_hat