# multitask_loss.py
# Combine main classification BCE/CrossEntropy with auxiliary losses.
# Designed to work with MultiTaskHead outputs.

import torch
import torch.nn.functional as F

def multitask_loss(outputs, targets, aux_targets=None, aux_weights=None):
    """
    outputs: dict from MultiTaskHead.forward: {'logits': [B, C], 'aux': {name: tensor}}
    targets: ground-truth labels for main task. for VQA-style multi-hot labels we expect [B, C] floats (same as existing code)
    aux_targets: dict of {name: tensor} with ground-truth for auxiliary tasks (optional)
    aux_weights: dict of {name: float} giving relative weights for auxiliary losses
    Returns: total_loss, details_dict
    """
    details = {}
    logits = outputs['logits']
    main_loss = F.binary_cross_entropy_with_logits(logits, targets) * targets.size(1)
    details['main_loss'] = main_loss.item() if isinstance(main_loss, torch.Tensor) else float(main_loss)

    total = main_loss

    if aux_targets is None:
        aux_targets = {}
    if aux_weights is None:
        aux_weights = {}

    for name, head_out in outputs.get('aux', {}).items():
        if name in aux_targets and aux_targets[name] is not None:
            # default: assume classification and use CE if targets are int labels, or MSE if floats
            t = aux_targets[name]
            weight = aux_weights.get(name, 1.0)
            if t.dtype in (torch.long, torch.int):
                l = F.cross_entropy(head_out, t)
            else:
                l = F.mse_loss(head_out, t.float())
            total = total + weight * l
            details[f'aux_{name}'] = l.item()
        else:
            details[f'aux_{name}'] = None

    details['total_loss'] = total.item() if isinstance(total, torch.Tensor) else float(total)
    return total, details
