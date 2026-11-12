"""PAVE temporal encoding, spatial structure learning and forecasting."""
import torch
from torch import nn

ARCHITECTURE_VERSION = "pave-structure-learning-1"


class TemporalAttention(nn.Module):
    """Multi-channel attention over the history of an individual node."""

    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.channel_dim = model_dim // num_heads
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.output = nn.Linear(model_dim, model_dim)

    def forward(self, states):
        batch_size = states.shape[0]
        query, key, value = self.query(states), self.key(states), self.value(states)
        query = torch.cat(torch.split(query, self.channel_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.channel_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.channel_dim, dim=-1), dim=0)
        scores = (query @ key.transpose(-1, -2)) / self.channel_dim**0.5
        encoded = torch.softmax(scores, dim=-1) @ value
        encoded = torch.cat(torch.split(encoded, batch_size, dim=0), dim=-1)
        return self.output(encoded)


class TemporalEncodingLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim, num_heads, dropout):
        super().__init__()
        self.temporal_attention = TemporalAttention(model_dim, num_heads)
        self.feature_update = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim), nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.attention_norm = nn.LayerNorm(model_dim)
        self.update_norm = nn.LayerNorm(model_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.update_dropout = nn.Dropout(dropout)

    def forward(self, states):
        states = states.transpose(1, -2)
        states = self.attention_norm(states + self.attention_dropout(self.temporal_attention(states)))
        states = self.update_norm(states + self.update_dropout(self.feature_update(states)))
        return states.transpose(1, -2)


class TemporalEncoder(nn.Module):
    def __init__(self, in_steps, steps_per_day, model_dim, feed_forward_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.steps_per_day = steps_per_day
        self.value_embedding = nn.Linear(1, model_dim)
        self.time_of_day_embedding = nn.Embedding(steps_per_day, model_dim)
        self.day_of_week_embedding = nn.Embedding(7, model_dim)
        self.position_embedding = nn.Parameter(torch.empty(in_steps, model_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.embedding_norm = nn.LayerNorm(model_dim)
        self.layers = nn.ModuleList([
            TemporalEncodingLayer(model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, raw, visible, history_mask, mask_embedding):
        values = self.value_embedding(visible[..., None])
        values = torch.where(history_mask[..., None], mask_embedding, values)
        time_of_day = (raw[..., 1] * self.steps_per_day).round().long().clamp(0, self.steps_per_day - 1)
        states = self.embedding_norm(values + self.time_of_day_embedding(time_of_day)
                                     + self.day_of_week_embedding(raw[..., 2].long())
                                     + self.position_embedding[None, :, None, :])
        for layer in self.layers:
            states = layer(states)
        return states


class StateAdaptiveStructure(nn.Module):
    """Generate channel-specific node-pair scores from the current states."""

    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.channel_dim = model_dim // num_heads
        self.target_projection = nn.Linear(model_dim, model_dim)
        self.support_projection = nn.Linear(model_dim, model_dim)

    def forward(self, states):
        targets = self.target_projection(states)
        supports = self.support_projection(states)
        targets = torch.cat(torch.split(targets, self.channel_dim, dim=-1), dim=0)
        supports = torch.cat(torch.split(supports, self.channel_dim, dim=-1), dim=0)
        return (targets @ supports.transpose(-1, -2)) / self.channel_dim**0.5


class GraphPropagation(nn.Module):
    """Aggregate node states on the learned structures and fuse channels."""

    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.channel_dim = model_dim // num_heads
        self.state_projection = nn.Linear(model_dim, model_dim)
        self.channel_fusion = nn.Linear(model_dim, model_dim)

    def project_states(self, states):
        states = self.state_projection(states)
        return torch.cat(torch.split(states, self.channel_dim, dim=-1), dim=0)

    def forward(self, structures, projected_states, batch_size):
        aggregated = structures @ projected_states
        aggregated = torch.cat(torch.split(aggregated, batch_size, dim=0), dim=-1)
        return self.channel_fusion(aggregated)


class StructureLearningLayer(nn.Module):
    """Generate structures, propagate states, and apply a residual update."""

    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0):
        super().__init__()
        self.num_heads = num_heads
        self.structure_generator = StateAdaptiveStructure(model_dim, num_heads)
        self.graph_propagation = GraphPropagation(model_dim, num_heads)
        self.feature_update = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.propagation_norm = nn.LayerNorm(model_dim)
        self.update_norm = nn.LayerNorm(model_dim)
        self.propagation_dropout = nn.Dropout(dropout)
        self.update_dropout = nn.Dropout(dropout)

    def forward(self, states, dim=-2, spatial_prior=None, return_scores=False):
        states = states.transpose(dim, -2)
        batch_size = states.shape[0]
        scores = self.structure_generator(states)
        values = self.graph_propagation.project_states(states)
        content_scores = None
        if return_scores:
            content_scores = scores.reshape(self.num_heads, batch_size, *scores.shape[1:]).movedim(0, 1)
        if spatial_prior is not None:
            scores = scores + spatial_prior
        structures = torch.softmax(scores, dim=-1)
        propagated = self.graph_propagation(structures, values, batch_size)
        states = self.propagation_norm(states + self.propagation_dropout(propagated))
        updated = self.feature_update(states)
        states = self.update_norm(states + self.update_dropout(updated))
        states = states.transpose(dim, -2)
        return (states, content_scores) if return_scores else states


class IterativeStructureLearner(nn.Module):
    """Refine structures using node states updated by preceding graph layers."""

    def __init__(self, model_dim, feed_forward_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            StructureLearningLayer(model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, states, spatial_prior, return_scores=False):
        final_scores = None
        for index, layer in enumerate(self.layers):
            if return_scores and index == len(self.layers) - 1:
                states, final_scores = layer(states, dim=2, spatial_prior=spatial_prior, return_scores=True)
            else:
                states = layer(states, dim=2, spatial_prior=spatial_prior)
        return states, final_scores


class SpatialPrior(nn.Module):
    """Map four spatial relation features to a bias for each relation channel."""

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.relation_weights = nn.Parameter(torch.tensor([1., 1., .5, 0.]).repeat(num_heads, 1))

    def forward(self, relations, batch_size, nodes):
        if relations is None or relations.shape != (nodes, nodes, 4):
            raise ValueError("Expected spatial relations with shape [N,N,4]")
        pair_bias = torch.einsum("ijc,kc->kij", relations, self.relation_weights)
        return pair_bias[:, None, None].expand(-1, batch_size, 1, -1, -1).reshape(
            self.num_heads * batch_size, 1, nodes, nodes)


class ForecastHead(nn.Module):
    """Predict future changes relative to the latest visible observation."""

    def __init__(self, in_steps, model_dim, out_steps):
        super().__init__()
        self.projection = nn.Linear(in_steps * model_dim, out_steps)

    def forward(self, states, visible, history_mask):
        batch_size, history, nodes = visible.shape
        prediction = self.projection(states.transpose(1, 2).reshape(batch_size, nodes, -1)).transpose(1, 2)[..., None]
        time_ids = torch.arange(history, device=visible.device)[None, :, None].expand(batch_size, history, nodes)
        last_visible = time_ids.masked_fill(history_mask, -1).amax(dim=1)
        residual = visible.gather(1, last_visible.clamp_min(0)[:, None, :])
        residual = residual.masked_fill(last_visible[:, None, :] < 0, 0.)[..., None]
        return prediction + residual


class PAVE(nn.Module):
    """Node-shared forecaster for varying sets of observed nodes."""

    def __init__(self, in_steps=12, out_steps=12, steps_per_day=288,
                 model_dim=96, num_heads=4, num_layers=2, feed_forward_dim=192,
                 dropout=0.1, use_graph=True, use_adapter=False, prediction_residual=True):
        super().__init__()
        if model_dim % num_heads or min(in_steps, out_steps, num_layers) < 1:
            raise ValueError("Invalid model dimensions")
        if not use_graph or use_adapter or not prediction_residual:
            raise ValueError("PAVE requires spatial relations and a residual forecast without a feature adapter")
        self.in_steps, self.out_steps = in_steps, out_steps
        self.temporal_encoder = TemporalEncoder(in_steps, steps_per_day, model_dim, feed_forward_dim,
                                                num_heads, num_layers, dropout)
        self.structure_learner = IterativeStructureLearner(model_dim, feed_forward_dim, num_heads, num_layers, dropout)
        self.forecast_head = ForecastHead(in_steps, model_dim, out_steps)
        self.mask_embedding = nn.Parameter(torch.zeros(model_dim))
        self.spatial_prior = SpatialPrior(num_heads)

    def forward(self, raw, relations=None, history_mask=None, return_features=False, return_relations=False):
        if raw.ndim != 4 or raw.shape[1] != self.in_steps or raw.shape[-1] != 4:
            raise ValueError("Expected [B,L,N,flow+3 calendar fields]")
        batch_size, history, nodes, _ = raw.shape
        if nodes < 1:
            raise ValueError("At least one node is required")
        if history_mask is None:
            history_mask = torch.zeros((batch_size, history, nodes), dtype=torch.bool, device=raw.device)
        if history_mask.shape != (batch_size, history, nodes) or history_mask.dtype != torch.bool:
            raise ValueError("history_mask must be boolean [B,L,N]")
        visible = raw[..., 0].masked_fill(history_mask, 0.)
        states = self.temporal_encoder(raw, visible, history_mask, self.mask_embedding)
        prior = self.spatial_prior(relations, batch_size, nodes)
        states, scores = self.structure_learner(states, prior, return_scores=return_relations)
        prediction = self.forecast_head(states, visible, history_mask)
        if return_relations:
            return prediction, states, scores
        return (prediction, states) if return_features else prediction


def build_model(config, **overrides):
    """Construct a forecaster from the model_args section of a configuration."""
    return PAVE(**{**config["model_args"], **overrides})
