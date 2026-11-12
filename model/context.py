"""Target-preserving context perturbation and support sampling."""
import math
import numpy as np
import torch


def road_mixed_supports(shuffled_candidates, targets, proximity, count, fraction):
    candidates, targets = np.asarray(shuffled_candidates), np.asarray(targets)
    proximity = np.asarray(proximity)
    if (proximity.ndim != 2 or proximity.shape[0] != proximity.shape[1]
            or not np.isfinite(proximity).all() or not 0 <= fraction <= 1
            or not 0 < count <= len(candidates) or len(targets) == 0
            or len(np.unique(candidates)) != len(candidates)
            or np.intersect1d(candidates, targets).size):
        raise ValueError("Invalid road support sampling inputs")
    if min(candidates.min(), targets.min()) < 0 or max(candidates.max(), targets.max()) >= len(proximity):
        raise ValueError("Road support index out of range")
    # Closeness to ANY target; road features contain no traffic measurements.
    score = proximity[np.ix_(targets, candidates)].max(axis=0)
    near_count = min(count, round(count * fraction), int((score > 0).sum()))
    # Stable ties use the random candidate order, not fixed node IDs.
    ranked = np.argsort(-score, kind="stable")
    near = ranked[:near_count]
    random_rest = np.flatnonzero(~np.isin(np.arange(len(candidates)), near))[:count - near_count]
    return candidates[np.concatenate((near, random_rest))]


def context_count(nodes, targets, keep_ratio):
    if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
        raise ValueError("Context keep ratio must be in (0,1]")
    count = round(nodes * keep_ratio)
    if not targets < count <= nodes:
        raise ValueError("Student context must retain all targets and at least one support")
    return count


def sample_student_nodes(nodes, target_ids, count, probability, generator, road_proximity=None, road_fraction=0.):
    """Keep every query; vary the support set. One node set for the whole batch.

    Only this independent generator is consumed, in both KD and no-KD runs.
    Returning every node means a full-context batch, not a second forward.
    """
    if (target_ids.ndim != 1 or target_ids.dtype != torch.long or target_ids.device.type != "cpu"
            or target_ids.unique().numel() != len(target_ids)
            or not 0 < len(target_ids) < count <= nodes
            or (target_ids < 0).any() or (target_ids >= nodes).any()
            or not math.isfinite(probability) or not 0 < probability <= 1):
        raise ValueError("Invalid context targets, size, or subset probability")
    if not math.isfinite(road_fraction) or not 0 <= road_fraction <= 1:
        raise ValueError("Road fraction must be in [0,1]")
    if road_fraction > 0 and (road_proximity is None or road_proximity.shape != (nodes, nodes)):
        raise ValueError("Road support sampling requires seen-node road proximity")
    if count == nodes or (probability < 1 and torch.rand((), generator=generator).item() >= probability):
        selected = torch.arange(nodes)
    else:
        support_mask = torch.ones(nodes, dtype=torch.bool)
        support_mask[target_ids] = False
        candidates = support_mask.nonzero(as_tuple=True)[0]
        shuffled = candidates[torch.randperm(len(candidates), generator=generator)]
        support_ids = shuffled[:count - len(target_ids)]
        if road_fraction > 0:
            support_ids = torch.from_numpy(road_mixed_supports(shuffled.numpy(), target_ids.numpy(),
                road_proximity, count - len(target_ids), road_fraction))
        selected = torch.cat((target_ids, support_ids)).sort().values
    inverse = torch.full((nodes,), -1, dtype=torch.long)
    inverse[selected] = torch.arange(len(selected))
    return selected, inverse[target_ids]


def sample_relation_targets(batch, steps, nodes, target_ratio, generator):
    """Sample query nodes; the boolean tensor identifies KD positions, not hidden inputs."""
    if nodes < 2 or not 0 < target_ratio < 1 or min(batch, steps) < 1:
        raise ValueError("Need nonempty target/support sets")
    count = min(nodes - 1, max(1, round(nodes * target_ratio)))
    ids = torch.randperm(nodes, generator=generator)[:count].sort().values
    positions = torch.zeros((batch, steps, nodes), dtype=torch.bool)
    positions[:, :, ids] = True
    return ids, positions


def sample_kd_positions(positions, fraction, generator):
    """Select half-batch KD samples with the original independent RNG sequence."""
    if not 0 < fraction < 1:
        raise ValueError("KD sample fraction must be in (0,1)")
    b = positions.shape[0]
    if b == 1:
        selected = torch.rand(1, generator=generator) < fraction
    else:
        count = min(b - 1, max(1, round(b * fraction)))
        selected = torch.zeros(b, dtype=torch.bool)
        selected[torch.randperm(b, generator=generator)[:count]] = True
    return positions & selected.to(positions.device)[:, None, None]
