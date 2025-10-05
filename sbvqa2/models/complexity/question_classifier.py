# question_classifier.py
# A small configurable classifier used for question / audio embedding -> answer logits.
# It gives you control over width/depth and can be used as a plug-in classifier
# for experiments (complexity-aware or small/large variants).

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuestionClassifier(nn.Module):
    """
    A small MLP classifier for question / audio embeddings.
    - in_dim: input embedding dimension (e.g., output of q-encoder)
    - hidden_dims: list of hidden sizes (e.g., [1024])
    - out_dim: number of classes
    - activation: nonlinearity
    - dropout: probability for dropout
    """
    def __init__(self, in_dim, hidden_dims, out_dim, activation='relu', dropout=0.2):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'gelu':
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            cur = h
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [B, C] or [B, T, C] (if seq provided, will average temporal dim)
        returns logits [B, out_dim]
        """
        if x.dim() == 3:
            # temporal average pool if sequence passed in
            x = x.mean(dim=1)
        return self.net(x)
