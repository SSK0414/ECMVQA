# multitask_head.py
# A simple multitask head: classification + optional auxiliary tasks (e.g., answer type prediction).
# Keeps the core classification path but can produce multiple outputs for multi-objective training.

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskHead(nn.Module):
    """
    Multi-task head:
      - base_dim: shared representation dim
      - num_classes: primary classification outputs
      - aux_dims: dict of {aux_name: aux_out_dim} for auxiliary tasks
    """
    def __init__(self, base_dim, num_classes, aux_dims=None, dropout=0.2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(base_dim, num_classes)
        self.aux_heads = nn.ModuleDict()
        if aux_dims:
            for name, dim in aux_dims.items():
                self.aux_heads[name] = nn.Linear(base_dim, dim)

    def forward(self, x):
        """
        x: [B, base_dim] shared representation
        returns: dict with keys:
            - 'logits' : primary logits [B, num_classes]
            - 'aux' : dict of auxiliary outputs (each is [B, dim])
        """
        h = self.shared(x)
        logits = self.classifier(h)
        aux_out = {name: head(h) for name, head in self.aux_heads.items()}
        return {'logits': logits, 'aux': aux_out}
