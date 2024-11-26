import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from models.simCLR.modules.identity import Identity

class simCLRrn50(nn.Module):
    """
    SimCLR implementation using ResNet-50 as the encoder.
    """

    def __init__(self, projection_dim, pretrained=False):
        super(simCLRrn50, self).__init__()

        # Load ResNet-50
        self.encoder = models.resnet50(pretrained=pretrained)

        # Number of features from the encoder's penultimate layer
        n_features = self.encoder.fc.in_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # Projection head: MLP with one hidden layer
        self.projector = nn.Sequential(
            nn.Linear(n_features, projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        # Obtain representations from the encoder
        h_i = self.encoder(x_i)  # Encoder outputs h_i
        h_j = self.encoder(x_j)

        # Pass representations through the projection head
        z_i = self.projector(h_i)  # Projected representations z_i
        z_j = self.projector(h_j)

        return h_i, h_j, z_i, z_j


class simCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features):
        super(simCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j
    
# resnet model
class ResBlk(nn.Module):
    """
    Basic residual block with convolutions 
    """
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        """Build core network layers

        Args:
            dim_in (int): input dimensionality 
            dim_out (int): output dimensionality 
        """
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        """Shortcut connection in the residual framework 

        Args:
            x (torch.Tensor): The input image 

        Returns:
            torch.Tensor: Input processed by the skip connection  
        """
        # If the shortcut is to be learned 
        if self.learned_sc:
            x = self.conv1x1(x)
        # If downsampling is to be performed 
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        """Residual connection 

        Args:
            x (torch.Tensor): The input image 

        Returns:
            torch.Tensor: Input processed by the residual connection  
        """
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        # Return output normalized to unit variance
        return x / math.sqrt(2)
    

class ImageEncoder(nn.Module):
    """Encoder from images to style vector 
    """
    def __init__(self, 
                 img_size=96,
                 style_dim=256, #  latent dim
                 max_conv_dim=512,
                 in_channels=3, 
                 dim_in=64, 
                #  single_style=True,
                 num_domains=None):
        
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(in_channels, dim_in, 3, 1, 1)]

        # For 96x96 image this downsamples till 3x3 spatial dimension 
        repeat_num = math.ceil(np.log2(img_size)) - 2
        final_conv_dim = img_size // (2**repeat_num)
        
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        # Downsamples to spatial dimensionality of 1 
        blocks += [nn.Conv2d(dim_out, dim_out, final_conv_dim, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.conv = torch.nn.Sequential(*blocks)
        
        # output layer
        self.linear = nn.Linear(dim_out, style_dim)
            
        # self.single_style = single_style

    def forward(self, x, y=None):
        # Apply shared layer and linearize 
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        
        # if not self.single_style:
        #     out = []
        #     for layer in self.unshared:
        #         out += [layer(h)]
        #     out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        #     idx = torch.LongTensor(range(y.size(0))).to(y.device)
        #     z_style = out[idx, y]  # (batch, style_dim)
        # else:
        z_style = self.linear(h)
        return z_style