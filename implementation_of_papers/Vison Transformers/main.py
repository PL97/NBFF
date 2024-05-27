import torch
from vit_pytorch import ViT



if __name__ == "__main__":
    ## implement with torch.nn.MultiheadAttension
    
    
    ## an easy implementation using vit_pytorch
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=3,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(1, 3, 256, 256)
    preds = v(img)
    print(preds.shape)

