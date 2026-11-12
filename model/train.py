"""Train the road model with full or sampled node context, with/without relation KD."""
import argparse
import datetime
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.protocol import add_protocol_arguments, manifest_from_args, comparison_settings
from lib.data import make_loaders
from lib.metrics import evaluate_groups
from lib.utils import seed_everything
from model import (PAVE, ARCHITECTURE_VERSION, sample_relation_targets,
                   sample_kd_positions, spatial_relation_loss, prediction_losses,
                   context_count, sample_student_nodes, subset_relation_loss)
from model.utils import road_relations, spatial_query_chunks, load_checkpoint



def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def atomic_save(path, state):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def has_converged(epoch, wait, lr, cfg):
    return (epoch >= cfg["min_epochs"] and wait >= cfg["early_stop"]
            and lr <= cfg["min_lr"] * 1.001)


def frozen_teacher(path, model, model_args, protocol, mean, std, smoke):
    state = load_checkpoint(path)
    meta = state["metadata"]
    if (meta.get("architecture_version") != ARCHITECTURE_VERSION
            or not meta.get("prediction_residual") or state["model_args"] != model_args):
        raise ValueError("Teacher architecture must match the PAVE student")
    if meta.get("variant") not in ("PAVE-teacher",):
        raise ValueError("Use the corresponding full-history no-KD teacher")
    recipe = meta["training"]
    if (recipe.get("kd_weight", 0) != 0 or recipe.get("relation_weight", 0) != 0
            or recipe.get("teacher") is not None or recipe.get("mask_mode") != "none"):
        raise ValueError("Teacher must be supervised without KD or history masking")
    if meta.get("smoke") and not smoke:
        raise ValueError("Smoke teachers cannot be used for formal training")
    saved = state["protocol"]
    for key in ("dataset", "seen_nodes", "raw_shape", "split_ratio", "target_channel", "in_steps", "out_steps", "steps_per_day"):
        if saved[key] != protocol[key]:
            raise ValueError(f"Teacher task/training pool differs: {key}")
    if (state["mean"], state["std"]) != (mean, std):
        raise ValueError("Teacher normalization statistics differ")
    with torch.random.fork_rng(devices=[]):
        teacher = PAVE(**state["model_args"])
    teacher = teacher.to(next(model.parameters()).device).eval().requires_grad_(False)
    teacher.load_state_dict(state["model_state"])
    return teacher, dict(variant=meta["variant"], best_epoch=state["epoch"],
                         val_seen_mae=state["val_seen_mae"], pretraining_mask_mode=recipe["mask_mode"],
                         input_mode="full_history", frozen=True)


@torch.no_grad()
def validate(model, loader, graph, device, std, limit=None):
    model.eval()
    total, count = 0., 0
    for i, (x, y) in enumerate(loader):
        if limit is not None and i >= limit:
            break
        error = (model(x.to(device), graph) - y.to(device)).abs() * std
        if not torch.isfinite(error).all():
            raise FloatingPointError("Nonfinite validation")
        total += error.sum().item()
        count += error.numel()
    return total / count


def save_rng(loader, mask_rng, device, task_rng=None):
    result = dict(cpu=torch.get_rng_state(), loader=loader.generator.get_state(), mask=mask_rng.get_state(),
                  python=random.getstate())
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device)
    if task_rng is not None:
        result["task"] = task_rng.get_state()
    return result


def restore_rng(state, loader, mask_rng, device, task_rng=None):
    torch.set_rng_state(state["cpu"])
    loader.generator.set_state(state["loader"])
    mask_rng.set_state(state["mask"])
    random.setstate(state["python"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    if task_rng is not None:
        task_rng.set_state(state["task"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="PEMS08")
    add_protocol_arguments(p)
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--config", type=Path, help="Defaults to configs/<DATASET>.json")
    p.add_argument("--variant", choices=("no_kd", "relation_kd"), default="no_kd")
    p.add_argument("--teacher-checkpoint", type=Path)
    p.add_argument("--context-keep-ratio", type=float, default=1., help="1 preserves the stable full-context task")
    p.add_argument("--context-subset-probability", type=float, default=1., help="Fraction of batches using a sampled subgraph")
    p.add_argument("--context-road-fraction", type=float, default=0., help="Fraction of supports selected by road proximity; remainder random")
    p.add_argument("--eval-batch-size", type=int, help="Test batch size only; training and validation retain their existing batch size")
    p.add_argument("--eval-query-chunk", type=int, default=0, help="Inference-only spatial query block size; every key is retained")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--resume", type=Path, help="Existing run directory; restores optimizer, scheduler, and RNG")
    p.add_argument("--output", type=Path, default=ROOT / "runs/train")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.out_steps != 12:
        p.error("Only the full 12-step forecast is evaluated")
    args.dataset = args.dataset.upper()
    args.config = args.config or ROOT / "configs" / f"{args.dataset}.json"
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if not 0 < cfg["target_ratio"] < 1 or not 0 < cfg["kd_sample_ratio"] < 1:
        p.error("Target and KD sample ratios must be in (0,1)")
    if min(cfg["max_epochs"], cfg["min_epochs"], cfg["early_stop"], cfg["batch_size"], cfg["lr_patience"], cfg["kd_ramp_epochs"]) < 1:
        p.error("Invalid training budget")
    if (not all(np.isfinite(cfg[k]) for k in ("relation_weight", "relation_temperature", "lr", "weight_decay"))
            or cfg["lr"] <= 0 or cfg["weight_decay"] < 0 or cfg["relation_weight"] < 0 or cfg["relation_temperature"] <= 0
            or cfg["kd_warmup_epochs"] < 0 or not 0 < cfg["lr_factor"] < 1 or cfg["min_lr"] <= 0 or cfg["min_delta"] < 0):
        p.error("Invalid loss/learning rate configuration")
    use_kd = args.variant == "relation_kd"
    if cfg["model_args"]["num_layers"] < 2 or (use_kd and cfg["relation_weight"] <= 0):
        p.error("Relation KD requires at least two spatial layers and positive weight")
    if use_kd != (args.teacher_checkpoint is not None):
        p.error("Only relation_kd requires --teacher-checkpoint")
    if (not np.isfinite(args.context_subset_probability) or not 0 < args.context_subset_probability <= 1
            or not np.isfinite(args.context_keep_ratio) or not 0 < args.context_keep_ratio <= 1):
        p.error("Context keep ratio and subset probability must be in (0,1]")
    use_context = args.context_keep_ratio < 1.
    if (not np.isfinite(args.context_road_fraction) or not 0 <= args.context_road_fraction <= 1
            or (args.context_road_fraction > 0 and not use_context)
            or (args.eval_batch_size is not None and args.eval_batch_size < 1) or args.eval_query_chunk < 0):
        p.error("Invalid support sampling or inference options")
    family = "student" if use_context else "teacher"
    epochs = 1 if args.smoke else (args.max_epochs if args.max_epochs is not None else cfg["max_epochs"])
    if not 1 <= epochs <= 400:
        p.error("Training is limited to 1–400 epochs (including resumed epochs)")
    device = torch.device(args.device)
    seed_everything(args.seed)
    protocol, raw, adjacency = manifest_from_args(args, args.data_root)
    del adjacency
    if len(protocol["seen_nodes"]) < 2:
        p.error("Retained-target training needs at least two seen nodes")
    loaders, mean_arr, std_arr = make_loaders(raw, protocol, cfg["batch_size"], args.seed, args.eval_batch_size)
    mean, std = float(mean_arr[0]), float(std_arr[0])
    graph, graph_info = road_relations(args.data_root / args.dataset / f"{args.dataset}.csv", raw.shape[1])
    seen = protocol["seen_nodes"]
    target_count = min(len(seen) - 1, max(1, round(len(seen) * cfg["target_ratio"])))
    retained_nodes = context_count(len(seen), target_count, args.context_keep_ratio)
    if use_context and retained_nodes == len(seen):
        p.error("Context keep ratio must remove at least one node")
    train_graph = torch.from_numpy(graph[np.ix_(seen, seen)]).to(device)
    road_proximity = graph[np.ix_(seen, seen)][..., 0] if args.context_road_fraction > 0 else None
    test_graph = torch.from_numpy(graph).to(device)
    protocol.pop("lape_dim", None)
    protocol.pop("n_clusters", None)
    model_args = dict(cfg["model_args"], use_graph=True, use_adapter=False,
                      prediction_residual=True, in_steps=args.in_steps,
                      out_steps=args.out_steps, steps_per_day=args.steps_per_day)
    model = PAVE(**model_args).to(device)
    teacher, teacher_info = None, None
    if use_kd:
        teacher, teacher_info = frozen_teacher(args.teacher_checkpoint, model, model_args, protocol,
                                                mean, std, args.smoke)
    run_dir = args.resume or args.output / f"{args.dataset}-{family}-{args.variant}-s{args.seed}-{datetime.datetime.now():%Y%m%d-%H%M%S-%f}"
    if not args.resume:
        run_dir.mkdir(parents=True, exist_ok=False)
    hyper = dict(lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper["lr"], weight_decay=hyper["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg["lr_factor"],
        patience=cfg["lr_patience"], threshold=cfg["min_delta"], threshold_mode="abs", min_lr=cfg["min_lr"])
    training = dict(max_epochs=epochs, min_epochs=cfg["min_epochs"], early_stop=cfg["early_stop"],
        min_delta=cfg["min_delta"], batch_size=cfg["batch_size"], scheduler="ReduceLROnPlateau",
        lr_factor=cfg["lr_factor"], lr_patience=cfg["lr_patience"], min_lr=cfg["min_lr"],
        target_ratio=cfg["target_ratio"],
        mask_mode="none", supervised_loss="0.5 target MAE + 0.5 support MAE",
        kd_mode="relation" if use_kd else "none", student_task="clean",
        kd_sample_ratio=cfg["kd_sample_ratio"],
        relation_weight=cfg["relation_weight"] if use_kd else 0.,
        relation_temperature=cfg["relation_temperature"],
        relation_source="last spatial layer content logits; target-to-support; mean head probabilities" if use_kd else None,
        kd_warmup_epochs=cfg["kd_warmup_epochs"], kd_ramp_epochs=cfg["kd_ramp_epochs"],
        teacher=teacher_info, student_initialization="random", **hyper)
    if use_context:
        training["student_task"] = "sampled_node_context"
        training["context"] = dict(version="context-20260918", keep_ratio=args.context_keep_ratio,
            subset_probability=args.context_subset_probability, full_nodes=len(seen), subset_nodes=retained_nodes,
            target_nodes=target_count, retain_target_history=True, support_sampling="uniform without replacement",
            teacher_input="all seen nodes", validation_input="all seen nodes", test_input="all nodes")
        if args.context_road_fraction > 0:
            training["context"].update(support_sampling="road-near plus uniform without replacement",
                                        road_fraction=args.context_road_fraction)
    metadata = dict(dataset=args.dataset, variant="PAVE" if use_kd else ("PAVE-teacher" if not use_context else "PAVE-no-kd"), seed=args.seed,
        node_seed=protocol["node_seed"], node_sampling=protocol["node_sampling"], comparison=comparison_settings(protocol),
        model_args=model_args, training=training, configuration=cfg, pseudo_weight=0., smoke=args.smoke,
        architecture_version=ARCHITECTURE_VERSION, protocol=protocol,
        prediction_residual=True, diagnostics_enabled=False,
        graph_info=graph_info,
        topology_used_by_predictor=True, evaluation_horizons=[12],
        num_parameters=sum(v.numel() for v in model.parameters()))
    if args.eval_batch_size is not None or args.eval_query_chunk:
        metadata["inference"] = dict(batch_size=args.eval_batch_size or cfg["batch_size"],
                                     query_chunk=args.eval_query_chunk, key_context="all nodes")
    seed_everything(args.seed)
    mask_rng = torch.Generator().manual_seed(args.seed + 1729)
    task_rng = torch.Generator().manual_seed(args.seed + 2718)
    context_rng = torch.Generator().manual_seed(args.seed + 31415) if use_context else None
    start, best, significant_best, wait, history = 0, float("inf"), float("inf"), 0, []
    elapsed_before = 0.
    code = [f for folder in ("model", "lib") for f in (ROOT / folder).glob("*.py")]
    if args.resume:
        last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        old = last["metadata"]
        for key in ("variant", "dataset", "seed", "node_seed", "model_args", "comparison", "configuration", "smoke"):
            if old[key] != metadata[key]:
                raise ValueError(f"Resume settings/source differ: {key}")
        if {k:v for k,v in old["training"].items() if k != "max_epochs"} != {k:v for k,v in training.items() if k != "max_epochs"}:
            raise ValueError("Resume teacher or training task differs")
        if json.loads((run_dir / "protocol.json").read_text())["seen_nodes"] != seen:
            raise ValueError("Resume training pool differs")
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        restore_rng(last["rng"], loaders["train"], mask_rng, device, task_rng)
        if use_context:
            context_rng.set_state(last["rng"]["context"])
        start, best, significant_best, wait = last["epoch"], last["best"], last["significant_best"], last["wait"]
        history, elapsed_before = last["history"], last["elapsed_seconds"]
        if epochs <= start:
            p.error("Resume --max-epochs must exceed completed epochs")
        # A partial/resumed run must not remain eligible for result aggregation.
        if (run_dir / "metrics.json").exists():
            old_report = json.loads((run_dir / "metrics.json").read_text())
            old_report["status"] = "resuming"
            write_json(run_dir / "metrics.json", old_report)
    else:
        write_json(run_dir / "protocol.json", protocol)
        np.savez(run_dir / "scaler.npz", mean=mean_arr, std=std_arr)
        np.savez_compressed(run_dir / "road_relations.npz", relations=graph)
        shutil.copy2(args.config, run_dir / "config.json")
        for f in code:
            dest = run_dir / "source" / f.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
    write_json(run_dir / "run.json", metadata)
    limit = 2 if args.smoke else None
    warmup = 0 if args.smoke else cfg["kd_warmup_epochs"]
    ramp = 1 if args.smoke else cfg["kd_ramp_epochs"]
    begun, stopped_by = time.monotonic(), "max_epochs"
    for epoch in range(start, epochs):
        model.train()
        totals, samples = np.zeros(6), 0
        kd_samples = 0
        subset_samples, node_total = 0, 0
        smallest_context, largest_context = len(seen), 0
        schedule = min(1., max(0., (epoch + 1 - warmup) / ramp))
        beta = (cfg["relation_weight"] if use_kd else 0.) * schedule
        lr_used = optimizer.param_groups[0]["lr"]
        for i, (x, y) in enumerate(loaders["train"]):
            if limit is not None and i >= limit:
                break
            ids, positions = sample_relation_targets(len(x), x.shape[1], x.shape[2], cfg["target_ratio"], mask_rng)
            if use_kd:
                positions = sample_kd_positions(positions, cfg["kd_sample_ratio"], task_rng)
            if use_context:
                student_nodes, local_targets = sample_student_nodes(len(seen), ids, retained_nodes,
                    args.context_subset_probability, context_rng, road_proximity, args.context_road_fraction)
            x, y, ids, positions = x.to(device), y.to(device), ids.to(device), positions.to(device)
            full_x = x
            student_graph = train_graph
            if use_context:
                student_nodes, ids = student_nodes.to(device), local_targets.to(device)
                x, y = x.index_select(2, student_nodes), y.index_select(2, student_nodes)
                student_graph = train_graph.index_select(0, student_nodes).index_select(1, student_nodes)
                subset_samples += len(x) * int(len(student_nodes) < len(seen))
                node_total += len(x) * len(student_nodes)
                smallest_context = min(smallest_context, len(student_nodes))
                largest_context = max(largest_context, len(student_nodes))
            kd_rows = positions.any(dim=2).any(dim=1)
            active_kd = teacher is not None and beta > 0 and bool(kd_rows.any())
            kd_fraction = float(kd_rows.float().mean().item())
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, student_graph, return_features=True, return_relations=active_kd)
            pred = outputs[0]
            loss, target_loss, support_loss = prediction_losses(pred, y, ids, std)
            relation_kd = loss.new_zeros(())
            if active_kd:
                kd_samples += int(kd_rows.sum().item())
                with torch.no_grad():
                    teacher_outputs = teacher(full_x[kd_rows], train_graph, return_features=True, return_relations=beta > 0)
                if use_context:
                    relation_kd = subset_relation_loss(outputs[2][kd_rows], teacher_outputs[2], student_nodes,
                        ids, cfg["relation_temperature"]) * kd_fraction
                else:
                    relation_kd = spatial_relation_loss(outputs[2][kd_rows], teacher_outputs[2], ids, cfg["relation_temperature"]) * kd_fraction
                loss = loss + std * (beta * relation_kd)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            totals += np.array([target_loss.item(), support_loss.item(), 0., loss.item(),
                                relation_kd.item(), std * beta * relation_kd.item()]) * len(x)
            samples += len(x)
        val = validate(model, loaders["val"], train_graph, device, std, limit)
        scheduler.step(val)
        if val < significant_best - cfg["min_delta"]:
            significant_best, wait = val, 0
        else:
            wait += 1
        # History is also saved inside last.pt. Python scalars keep it readable
        # by torch.load(weights_only=True), including older server PyTorch versions.
        row = dict(epoch=epoch + 1, target_mae=float(totals[0] / samples), support_mae=float(totals[1] / samples),
                   target_nodes=len(ids), support_nodes=x.shape[2] - len(ids),
                   masked_samples=0, clean_samples=samples, kd_samples=kd_samples,
                   relation_kl_loss=float(totals[4] / samples) if beta > 0 else None,
                   weighted_relation_kd=float(beta * std * totals[4] / samples),
                   weighted_aux=float(totals[5] / samples), train_loss=float(totals[3] / samples),
                   val_seen_mae=val, lr=lr_used, next_lr=optimizer.param_groups[0]["lr"],
                   epochs_without_significant_improvement=wait)
        history.append(row)
        if use_context:
            row.update(subset_samples=subset_samples, full_context_samples=samples - subset_samples,
                       mean_student_nodes=node_total / samples, min_student_nodes=smallest_context,
                       max_student_nodes=largest_context)
        print(json.dumps(row), flush=True)
        write_json(run_dir / "history.json", history)
        if val < best:
            best = val
            atomic_save(run_dir / "best.pt", dict(model_state=model.state_dict(), model_args=model_args,
                        format_version=1, protocol=protocol, epoch=epoch + 1, val_seen_mae=best, mean=mean, std=std, metadata=metadata))
        elapsed = elapsed_before + time.monotonic() - begun
        rng_state = save_rng(loaders["train"], mask_rng, device, task_rng)
        if use_context:
            rng_state["context"] = context_rng.get_state()
        atomic_save(run_dir / "last.pt", dict(model_state=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), rng=rng_state, epoch=epoch + 1,
            best=best, significant_best=significant_best, wait=wait, history=history,
            elapsed_seconds=elapsed, metadata=metadata))
        if not args.smoke and has_converged(epoch + 1, wait, optimizer.param_groups[0]["lr"], cfg):
            stopped_by = "converged_early_stop"
            break
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    actual, predictions = [], []
    with torch.no_grad(), spatial_query_chunks(model, args.eval_query_chunk):
        for i, (x, y) in enumerate(loaders["test"]):
            if limit is not None and i >= limit:
                break
            pred = model(x.to(device), test_graph)
            actual.append((y * std + mean)[..., 0].numpy())
            predictions.append((pred * std + mean)[..., 0].cpu().numpy())
    convergence = dict(stopped_by="smoke" if args.smoke else stopped_by, completed_epochs=len(history),
        best_epoch=checkpoint["epoch"], epochs_since_best=len(history) - checkpoint["epoch"],
        epochs_without_significant_improvement=wait, final_lr=optimizer.param_groups[0]["lr"],
        converged=stopped_by == "converged_early_stop")
    report = dict(metadata, status="complete", best_epoch=checkpoint["epoch"], val_seen_mae=best,
        elapsed_seconds=elapsed_before + time.monotonic() - begun, convergence=convergence,
        metrics=evaluate_groups(np.concatenate(actual), np.concatenate(predictions), seen, protocol["unseen_nodes"]))
    write_json(run_dir / "metrics.json", report)
    print(f"Results: {run_dir}", flush=True)
    group = "unseen" if protocol["unseen_nodes"] else "all"
    print(json.dumps(dict(evaluation_group=group, metrics=report["metrics"][group]["prefix"]["12"], convergence=convergence)), flush=True)


if __name__ == "__main__":
    main()
