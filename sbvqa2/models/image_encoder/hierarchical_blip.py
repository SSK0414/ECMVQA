# sbvqa2/models/vision/hierarchical_blip.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...vit import VisionTransformer  # your existing vit implementation
from torchvision.transforms.functional import InterpolationMode

class HierarchicalImageEncoder(nn.Module):
    """
    Simple hierarchical multi-scale image encoder:
      - uses same ViT at several input sizes (shared weights)
      - returns concatenated CLS embeddings for scales (and optionally patch features)
    Intended to be light and easy to plug in.
    """
    def __init__(self, img_sizes=(224,), vit_variant='base', patch_size=16, out_dim=1024, pretrained=False):
        """
        img_sizes: tuple of ints (small->large)
        vit_variant: 'base' or 'large' -- maps to embed_dim inside create_vit in your repo
        out_dim: desired output dim after projection
        """
        super().__init__()
        assert len(img_sizes) >= 1
        self.img_sizes = list(img_sizes)
        # pick the largest img size for constructing ViT internals (so positional embeddings cover largest)
        max_img = max(img_sizes)
        # use the existing VisionTransformer; map vit_variant to args as in your create_vit
        if vit_variant == 'base':
            vision_width = 768
            depth = 12; num_heads = 12
        else:
            vision_width = 1024
            depth = 24; num_heads = 16

        # instantiate a single ViT (shared weights) sized for max_img
        self.vit = VisionTransformer(img_size=max_img, patch_size=patch_size, embed_dim=vision_width,
                                     depth=depth, num_heads=num_heads)

        # projection to out_dim from vit embed_dim (CLS token dim)
        self.proj = nn.Linear(vision_width * len(img_sizes), out_dim)

        # optional layernorm
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, images):
        """
        images: [B, C, H, W] (raw images)
        returns: feature vector [B, out_dim]
        Also returns internal per-scale cls tokens if needed: (list_of_cls) each [B, embed_dim]
        """
        B = images.size(0)
        cls_list = []
        for s in self.img_sizes:
            if s != images.size(-1):
                x = F.interpolate(images, size=(s, s), mode='bicubic', align_corners=False)
            else:
                x = images
            # feed into vit (expecting same interface as your VisionTransformer)
            # vit returns [B, num_patches+1, embed_dim]
            out = self.vit(x)
            cls = out[:, 0, :]  # CLS token
            cls_list.append(cls)

        # concat CLS tokens across scales
        cat = torch.cat(cls_list, dim=-1)  # [B, embed_dim * num_scales]
        projected = self.proj(cat)         # [B, out_dim]
        projected = self.ln(projected)
        return projected  # [B, out_dim]
