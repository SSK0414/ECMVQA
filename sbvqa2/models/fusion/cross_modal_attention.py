# sbvqa2/models/fusion/cross_modal_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion:
      - audio -> vision cross attention (audio queries attend to visual keys)
      - vision -> audio cross attention (vision queries attend to audio keys)
      - outputs are pooled and combined via MLP.
    Keeps last attention weights in `last_attn` for visualization.
    """
    def __init__(self, dim_audio, dim_visual, hidden_dim=1024, n_heads=8, dropout=0.1):
        super().__init__()
        # project dims to common internal dim
        self.common_dim = hidden_dim
        self.a_proj = nn.Linear(dim_audio, hidden_dim)
        self.v_proj = nn.Linear(dim_visual, hidden_dim)

        # cross-attention modules (PyTorch MultiheadAttention with batch_first)
        self.attn_a2v = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.attn_v2a = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout)

        # final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # classifier / projection optionally outside
        self.norm = nn.LayerNorm(hidden_dim)
        self.last_attn = None

    def forward(self, audio_feats, visual_feats, audio_mask=None, visual_mask=None):
        """
        audio_feats: [B, T, dim_audio]
        visual_feats: [B, N, dim_visual]
        returns: fused embedding [B, hidden_dim]
        """
        # project
        a = self.a_proj(audio_feats)   # [B, T, H]
        v = self.v_proj(visual_feats)  # [B, N, H]

        # normalize (stabilizes attention)
        a = F.normalize(a, dim=-1)
        v = F.normalize(v, dim=-1)

        # audio queries attend to visual keys/values -> audio-aware-vision
        a2v_out, a2v_w = self.attn_a2v(query=a, key=v, value=v, key_padding_mask=visual_mask)
        # vision queries attend to audio keys/values -> vision-aware-audio
        v2a_out, v2a_w = self.attn_v2a(query=v, key=a, value=a, key_padding_mask=audio_mask)

        # pool temporal/spatial dims (mean pooling)
        a_pooled = a2v_out.mean(dim=1)  # [B, H]
        v_pooled = v2a_out.mean(dim=1)  # [B, H]

        # concat and fuse
        cat = torch.cat([a_pooled, v_pooled], dim=-1)  # [B, 2H]
        fused = self.fusion_mlp(cat)                   # [B, H]
        fused = self.norm(fused)

        # store for visualization
        # keep both attention maps (shape: [B, heads, T, N] depending or [B, T, N] averaged)
        self.last_attn = {'a2v': a2v_w.detach().cpu() if a2v_w is not None else None,
                          'v2a': v2a_w.detach().cpu() if v2a_w is not None else None}
        return fused
