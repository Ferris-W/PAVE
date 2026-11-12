"""Road relations, query-chunked inference and checkpoint loading."""
import csv
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from .forecaster import ARCHITECTURE_VERSION


def road_relations(csv_path, count, method="auto"):
    if method not in ("auto", "dense", "sparse"):
        raise ValueError("Unknown shortest-path method")
    distance = np.full((count, count), np.inf, dtype=np.float64)
    hops = np.full_like(distance, np.inf)
    np.fill_diagonal(distance, 0.)
    np.fill_diagonal(hops, 0.)
    costs = []
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        distance_key = "cost" if "cost" in fields else "distance"
        if not {"from", "to", distance_key}.issubset(fields):
            raise ValueError("Road CSV must have from,to and cost/distance columns using raw-data node indices")
        for row in reader:
            a, b, cost = int(row["from"]), int(row["to"]), float(row[distance_key])
            if not 0 <= a < count or not 0 <= b < count or not np.isfinite(cost) or cost < 0:
                raise ValueError("Invalid road endpoint/distance")
            if a == b:
                continue
            distance[a, b] = distance[b, a] = min(distance[a, b], cost)
            hops[a, b] = hops[b, a] = 1.
            if cost > 0:
                costs.append(cost)
    if not costs:
        raise ValueError("Road graph has no positive-distance edges")
    direct = (hops == 1).astype(np.float32)
    # Topology for all sensors may be known; unseen traffic never enters this function.
    if method == "sparse" or (method == "auto" and count > 2048):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        rows, cols = np.nonzero(direct)
        sparse = csr_matrix((distance[rows, cols], (rows, cols)), shape=(count, count))
        distance = dijkstra(sparse, directed=False)
        hops = dijkstra(sparse, directed=False, unweighted=True)
    else:
        for k in range(count):
            distance = np.minimum(distance, distance[:, k:k+1] + distance[k:k+1, :])
            hops = np.minimum(hops, hops[:, k:k+1] + hops[k:k+1, :])
    scale = float(np.median(costs))
    features = np.empty((count, count, 4), dtype=np.float32)
    features[..., 0] = 1. / (1. + distance / scale)
    features[..., 1] = 1. / (1. + hops)
    features[..., 2] = direct
    features[..., 3] = np.eye(count, dtype=np.float32)
    return features, dict(source="undirected road CSV", channels=["distance_proximity", "hop_proximity", "direct_edge", "self"],
                          distance_scale=scale, disconnected_pairs=int(np.isinf(distance).sum()))


def _chunked_layer_forward(layer, states, dim=-2, spatial_prior=None, return_scores=False, *, chunk):
    if layer.training or torch.is_grad_enabled() or return_scores:
        raise ValueError("Query chunking requires evaluation without gradients or relation scores")
    states = states.transpose(dim, -2)
    batch_size = states.shape[0]
    generator = layer.structure_generator
    propagation = layer.graph_propagation
    queries = generator.target_projection(states)
    keys = generator.support_projection(states)
    queries = torch.cat(torch.split(queries, generator.channel_dim, dim=-1), dim=0)
    keys = torch.cat(torch.split(keys, generator.channel_dim, dim=-1), dim=0).transpose(-1, -2)
    values = propagation.project_states(states)
    chunks = []
    for start in range(0, queries.shape[-2], chunk):
        stop = start + chunk
        scores = (queries[..., start:stop, :] @ keys) / generator.channel_dim**0.5
        if spatial_prior is not None:
            scores = scores + spatial_prior[..., start:stop, :]
        chunks.append(torch.softmax(scores, dim=-1) @ values)
    aggregated = torch.cat(chunks, dim=-2)
    aggregated = torch.cat(torch.split(aggregated, batch_size, dim=0), dim=-1)
    propagated = propagation.channel_fusion(aggregated)
    states = layer.propagation_norm(states + layer.propagation_dropout(propagated))
    states = layer.update_norm(states + layer.update_dropout(layer.feature_update(states)))
    return states.transpose(dim, -2)


@contextmanager
def spatial_query_chunks(model, chunk=0):
    if chunk < 0:
        raise ValueError("Query chunk must be nonnegative")
    if not chunk:
        yield
        return
    if model.training or torch.is_grad_enabled():
        raise ValueError("Enter query chunking under eval() and no_grad()")
    saved = []
    try:
        for layer in model.structure_learner.layers:
            saved.append((layer, layer.__dict__.get("forward")))
            def forward(self, *args, **kwargs):
                return _chunked_layer_forward(self, *args, chunk=chunk, **kwargs)
            layer.forward = MethodType(forward, layer)
        yield
    finally:
        for layer, original in saved:
            if original is None:
                del layer.forward
            else:
                layer.forward = original


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state", "model_args", "protocol", "mean", "std", "metadata", "epoch"}
    if not required.issubset(state):
        raise ValueError(f"Incomplete PAVE checkpoint: {sorted(required - state.keys())}")
    if state.get("format_version") != 1 or state["metadata"].get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError("Unsupported PAVE checkpoint format or architecture")
    if state["metadata"]["dataset"] != state["protocol"]["dataset"]:
        raise ValueError("Checkpoint dataset and protocol differ")
    return state
