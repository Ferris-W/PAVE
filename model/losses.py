"""Forecast supervision and full-to-partial relational distillation."""
import math
import torch
from torch.nn import functional as F


def spatial_relation_loss(student, teacher, target_ids, temperature=1.):
    """KL(T || S) over visible supports; average distributions across heads.

    Logits: [batch, heads, history, query nodes, key nodes], before road bias.
    Only target queries and non-target keys enter the loss. Each network may
    have a different head ordering; no head-to-head correspondence is imposed.
    """
    if (student.ndim != 5 or teacher.ndim != 5 or student.shape[0] != teacher.shape[0]
            or student.shape[2:] != teacher.shape[2:] or student.shape[-2] != student.shape[-1]
            or min(student.shape) < 1 or min(teacher.shape) < 1):
        raise ValueError("Expected compatible [B,heads,H,N,N] content logits")
    n = student.shape[-1]
    if (target_ids.ndim != 1 or target_ids.dtype != torch.long or not 0 < target_ids.numel() < n
            or target_ids.unique().numel() != target_ids.numel()
            or (target_ids < 0).any() or (target_ids >= n).any()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Need distinct target nodes, visible supports, and positive temperature")
    supports = torch.ones(n, dtype=torch.bool, device=student.device)
    supports[target_ids] = False
    support_ids = supports.nonzero(as_tuple=True)[0]

    def log_distribution(logits):
        selected = logits.index_select(-2, target_ids).index_select(-1, support_ids).float() / temperature
        per_head = F.log_softmax(selected, dim=-1)
        return torch.logsumexp(per_head, dim=1) - math.log(logits.shape[1])

    s, t = log_distribution(student), log_distribution(teacher.detach())
    return ((t.exp() * (t - s)).sum(-1).mean() * temperature**2).clamp_min(0.)


def subset_relation_loss(student, teacher, student_nodes, local_targets, temperature):
    """Teacher forwards on all seen nodes BEFORE restriction to common query/key IDs.

    The shared loss normalizes each head over visible non-target supports only,
    then averages heads. Never compare distributions with different key sets.
    """
    if (teacher.ndim != 5 or student.ndim != 5 or student_nodes.ndim != 1
            or student_nodes.dtype != torch.long or student_nodes.numel() != student.shape[-1]
            or student_nodes.unique().numel() != student_nodes.numel()
            or (student_nodes < 0).any() or (student_nodes >= teacher.shape[-1]).any()
            or teacher.shape[-2] != teacher.shape[-1]):
        raise ValueError("Student-to-teacher node mapping is invalid")
    aligned = teacher.detach().index_select(-2, student_nodes).index_select(-1, student_nodes)
    return spatial_relation_loss(student, aligned, local_targets, temperature)


def prediction_losses(pred, truth, target_ids, std):
    errors = (pred - truth).abs() * std
    supports = torch.ones(pred.shape[2], dtype=torch.bool, device=pred.device)
    supports[target_ids] = False
    target = errors[:, :, target_ids].mean()
    support = errors[:, :, supports].mean()
    return (target + support) / 2., target, support
