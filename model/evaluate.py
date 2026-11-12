"""Evaluate a self-contained PAVE checkpoint on the original task."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.data import make_loaders
from lib.metrics import evaluate_groups
from lib.protocol import make_manifest
from model.forecaster import PAVE
from model.utils import road_relations, spatial_query_chunks, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-chunk", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing files are never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evaluation output already exists; choose a new path")
    checkpoint = load_checkpoint(args.checkpoint)
    saved = checkpoint["protocol"]
    protocol, raw, _ = make_manifest(args.data_root, saved["dataset"], node_seed=saved["node_seed"],
        node_ratio=saved["train_node_ratio"], split_ratio=saved["split_ratio"], node_sampling=saved["node_sampling"],
        in_steps=saved["in_steps"], out_steps=saved["out_steps"], steps_per_day=saved["steps_per_day"],
        target_channel=saved["target_channel"])
    for key in ("seen_nodes", "unseen_nodes", "raw_shape", "window_counts"):
        if saved[key] != protocol[key]:
            raise ValueError(f"Saved task differs from supplied data: {key}")
    loaders, mean_array, std_array = make_loaders(raw, protocol, args.batch_size, 0)
    mean, std = float(mean_array[0]), float(std_array[0])
    if (mean, std) != (checkpoint["mean"], checkpoint["std"]):
        raise ValueError("Saved scaler differs from supplied training data")
    graph, _ = road_relations(args.data_root / saved["dataset"] / f"{saved['dataset']}.csv", raw.shape[1])
    model = PAVE(**checkpoint["model_args"]).to(args.device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    graph = torch.from_numpy(graph).to(args.device)
    truth, prediction = [], []
    with torch.no_grad(), spatial_query_chunks(model, args.query_chunk):
        for x, y in loaders["test"]:
            prediction.append((model(x.to(args.device), graph) * std + mean)[..., 0].cpu().numpy())
            truth.append((y * std + mean)[..., 0].numpy())
    metrics = evaluate_groups(np.concatenate(truth), np.concatenate(prediction), saved["seen_nodes"], saved["unseen_nodes"])
    result = dict(dataset=saved["dataset"], checkpoint=str(args.checkpoint),
                  best_epoch=checkpoint["epoch"], batch_size=args.batch_size, query_chunk=args.query_chunk,
                  test_windows=len(loaders["test"].dataset), metrics=metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
