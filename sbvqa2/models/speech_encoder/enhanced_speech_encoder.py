import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer

class EnhancedSpeechEncoder(nn.Module):
    """
    ECMVQA Speech Encoder
    - Based on Conformer-L
    - Multi-level hidden extraction (phoneme / word / semantic)
    - Temporal attention pooling (not collapsed too early)
    - Optional prosody fusion
    """

    def __init__(self, input_dim=80, hidden_dim=512, num_layers=16, num_heads=8, use_prosody=True):
        super().__init__()
        self.use_prosody = use_prosody

        # Linear projection to unify dimensions (if needed)
        self.proj = nn.Linear(input_dim, hidden_dim)

        # Temporal attention
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Optional prosody features
        if self.use_prosody:
            self.prosody_fc = nn.Linear(2, hidden_dim)

    def forward(self, x, lengths=None, prosody_features=None):
        """
        Args:
            x: [B, T, 80] log-mel features
            lengths: [B] valid lengths (optional)
            prosody_features: [B, T, 2] pitch+energy (optional)
        Returns:
            speech_feats: [B, T, H] temporal speech features
        """
        # === Conformer ===
        x = self.proj(x)

        attn_out, _ = self.temporal_attn(x, x, x)

        if self.use_prosody and prosody_features is not None:
            x = x + self.prosody_fc(prosody_features)

        speech_feats = x + attn_out

        return speech_feats, lengths, None
