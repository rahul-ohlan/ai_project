import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hflayers import Hopfield



class CLIP(nn.Module):
    def __init__(self,
                 image_encoder,
                 init_inv_tau: float = 14.3,
                 learnable_inv_tau: bool = True,
                 image_projection_network=None,
                 mol_projection_network=None,
                 compression_net = None,
                 attention = None,
                 **kwargs
                 ):
        super().__init__()

        # self.projection = nn.Linear(1024,256)
        self.image_encoder = image_encoder
        self.image_projection = image_projection_network
        self.mol_projection = mol_projection_network

        # Logit scales for the inner product in the InfoNCE loss
        self.logit_inv_tau = nn.Parameter(torch.ones([]) * np.log(init_inv_tau))
        self.logit_inv_tau.requires_grad = learnable_inv_tau

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            return self.visual.fc.weight.dtype

    
    def encode_image(self, image):
        
        latent_vector = self.image_encoder(image)

        if self.image_projection is not None:
            latent_vector = self.image_projection(latent_vector)


        return latent_vector
    
    def freeze_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = True

    def encode_mol(self, mol_embed): # chemical encoder (bs, 2048)

        # project through mol projection network

        projected_mol_embeds = self.mol_projection(mol_embed)

        return projected_mol_embeds

    def forward(self, image, mol_embed):  
        
        print('input mol embeds:', mol_embed.shape, mol_embed)
        image_features = self.encode_image(image)
        mol_features = self.encode_mol(mol_embed) # first encode mol_embeds through projection layer

        # gene_features_norm = gene_features / gene_features.norm(dim=-1, keepdim=True)
        # text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, mol_features, self.logit_inv_tau.exp()
    
    def infoNCELoss(self, gene_features, mol_features, inv_tau, device):
        # Normalize the features
        gene_features_norm = F.normalize(gene_features, p=2, dim=-1)
        mol_features_norm = F.normalize(mol_features, p=2, dim=-1)

        # Compute the similarity matrix
        logits_genes = torch.mm(gene_features_norm, mol_features_norm.t()) * inv_tau
        logits_mols = logits_genes.t()

        # Compute the InfoNCE loss
        batch_size = gene_features.size(0)
        labels = torch.arange(batch_size).to(device)

        # symmetric loss
        loss_genes = F.cross_entropy(logits_genes, labels, reduction="mean")
        loss_mols = F.cross_entropy(logits_mols, labels, reduction="mean")

        loss = (loss_genes + loss_mols) / 2

        return loss
    
    # https://github.com/ml-jku/cloob/blob/master/src/training/methods.py -- same method used in cloome paper
    def cloob(self, gene_features, mol_features, inv_tau, input_dim, scale, device):

        # initialize a hopfield layer
        hopfield_layer = Hopfield(input_size=input_dim, # embedding dim = 256
                              scaling=scale,
                              normalize_hopfield_space=False,
                              normalize_hopfield_space_affine=False,
                              normalize_pattern_projection=False,
                              normalize_pattern_projection_affine=False,
                              normalize_state_pattern_affine=False,
                              normalize_state_pattern=False,
                              normalize_stored_pattern_affine=False,
                              state_pattern_as_static=True,
                              pattern_projection_as_static=True,
                              stored_pattern_as_static=True,
                              disable_out_projection=True,
                              num_heads=1,
                              dropout=False).to(device)
        p_xx, p_yy, p_xy, p_yx = self.hopfield_retrieval(gene_features, mol_features, hopfield_layer)

        identity = torch.eye(p_xx.shape[0]) > 0.5
        i = identity.to(p_xx.device)
        # print("****loss_img******")
        loss_genes = self.infoLOOB_loss(p_xx, p_xy, i, inv_tau=inv_tau)
        # print(loss_img)
        # print("****loss_txt******")
        loss_mols = self.infoLOOB_loss(p_yy, p_yx, i, inv_tau=inv_tau)
        # print(loss_txt)
        return loss_genes + loss_mols
        
    def hopfield_retrieval(self,gene_features, mol_features, hopfield_layer):
        patterns_xx = self.hopfield(state_patterns=gene_features, stored_patterns=gene_features, hopfield_layer=hopfield_layer)
        patterns_yy = self.hopfield(state_patterns=mol_features, stored_patterns=mol_features, hopfield_layer=hopfield_layer)
        patterns_xy = self.hopfield(state_patterns=mol_features, stored_patterns=gene_features, hopfield_layer=hopfield_layer)
        patterns_yx = self.hopfield(state_patterns=gene_features, stored_patterns=mol_features, hopfield_layer=hopfield_layer)

        return patterns_xx, patterns_yy, patterns_xy, patterns_yx
    
    def hopfield(self,state_patterns, stored_patterns, hopfield_layer):
        retrieved_patterns = hopfield_layer.forward((stored_patterns.unsqueeze(0), state_patterns.unsqueeze(0), stored_patterns.unsqueeze(0))).squeeze()
        # Row vectors -> dim=1 to normalize the row vectors
        retrieved_patterns = retrieved_patterns / retrieved_patterns.norm(dim=1, keepdim=True)
        return retrieved_patterns
    
    def infoLOOB_loss(self,x, y, i, inv_tau):
        tau = 1 / inv_tau
        k = x @ y.T / tau
        positives = -torch.mean(torch.sum(k * i, dim=1))

        # For logsumexp the zero entries must be equal to a very large negative number
        large_neg = -1000.0
        arg_lse = k * torch.logical_not(i) + i * large_neg
        negatives = torch.mean(torch.logsumexp(arg_lse, dim=1))

        return tau * (positives + negatives)
    
    
class projection(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()

        self.layers = nn.ModuleList()

        for layer in range(len(hidden_sizes)):
            dim = input_dim if layer == 0 else hidden_sizes[layer-1]
            self.layers.append(nn.Sequential(
                               nn.Linear(dim, hidden_sizes[layer]),
                            #    nn.LayerNorm(hidden_sizes[layer]),
                            # nn.BatchNorm1d(hidden_sizes[layer]),
                            #    nn.LeakyReLU(0.2),
                            nn.ReLU())
                            #    nn.Dropout(0.2))
                               )

        self.layers.append(nn.Sequential(
                           nn.Linear(hidden_sizes[-1], output_dim))
                           )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class AttentivePooling(nn.Module):
    def __init__(self, mol_hidden_size, gene_hidden_size):
        super(AttentivePooling, self).__init__()
        self.mol_hidden_size = mol_hidden_size
        self.gene_hidden_size = gene_hidden_size
        # Learnable attention parameter
        self.param = nn.Parameter(torch.zeros(mol_hidden_size, gene_hidden_size))

    def forward(self, mol_embedding, gene_embedding):
        # Compute attention scores for molecule -> gene
        mol_to_gene_att = torch.matmul(gene_embedding.unsqueeze(1), self.param.transpose(0, 1).unsqueeze(0))
        score_mol_to_gene = F.softmax(mol_to_gene_att, dim=2)
        
        # Compute attention scores for gene -> molecule
        gene_to_mol_att = torch.matmul(mol_embedding.unsqueeze(1), self.param.unsqueeze(0))
        score_gene_to_mol = F.softmax(gene_to_mol_att, dim=2)
        
        # Weighted representations
        rep_mol = torch.sum(mol_embedding.unsqueeze(1) * score_mol_to_gene, dim=1)
        rep_gene = torch.sum(gene_embedding.unsqueeze(1) * score_gene_to_mol, dim=1)
        
        # Concatenate the attention-weighted representations
        combined_representation = torch.cat((rep_mol, rep_gene), dim=1)
        return combined_representation
    

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
