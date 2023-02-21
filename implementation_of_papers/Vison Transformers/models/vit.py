import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
import numpy as np
from functools import reduce
from typing import Union

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    

class MultiheadAttention(nn.Module):
    def __init__(self, dim:int, inner_dim:int, heads:int, dropout:float) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, inner_dim * 3, bias = False) # ! omit rearrange
        self.attn = nn.MultiheadAttention(embed_dim=inner_dim, num_heads=heads, batch_first=True, dropout=dropout)
        self.project = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim)
        )
        self.heads = heads
    def forward(self, x):
        q, k, v = self.linear(x).chunk(3, dim=-1)
        attn_output, _ = self.attn(query=q, key=k, value=v)
        ret = self.project(attn_output)
        return ret
    
        
        

class transformer_encoder(nn.Module):
    def __init__(self, num_layers:int, dim:int, heads:int, dim_head:int, mlp_dim:int, dropout:float) -> None:
        super().__init__()
        inner_dim = dim_head*heads
        self.layers = [[nn.Sequential(
                            nn.LayerNorm(dim),
                            MultiheadAttention(dim=dim, inner_dim=inner_dim, heads=heads, dropout=dropout)
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            MLP(dim=dim, hidden_dim=mlp_dim)
                        )] for _ in range(num_layers)] # ? not sure if directly use list would be less efficient or not
    
    def forward(self, x):
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, image_size: Union[list, int], patch_size: Union[list, int], num_classes:int, \
                dim:int, depth:int, heads:int, mlp_dim:int, pool='cls', channels=3, dim_head=64, \
                dropout=0., emb_dropout=0.) -> None:
        super().__init__()
        image_size = [image_size]*2 if type(image_size)==int else image_size
        patch_size = [patch_size]*2 if type(patch_size)==int else patch_size
        assert all(img%patch==0 for img, patch in zip(image_size, patch_size))
        
        patch_dim = channels * np.prod(patch_size)
        num_patches = (image_size[0]//patch_size[0])*(image_size[1]//patch_size[1])
        self.pool = pool
        self.dropout = nn.Dropout(emb_dropout)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size[0], p2 = patch_size[1]),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )   
        
        ## some learnable params in ViT
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.transformer_encoder = transformer_encoder(num_layers=depth, dim=dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        
        ## to inject position encoding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer_encoder(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

if __name__ == "__main__":
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=3,
        dim=1024,
        depth=3,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    img = torch.randn(1, 3, 256, 256)
    print(v(img).shape)
    