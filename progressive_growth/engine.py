from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from .growth import GrowthConfig, GrowthMaskController
from .hessian import HessianConfig, compute_hessian_metrics
from .transition_metrics import clone_state_dict_to_cpu, compute_transition_metrics
from .utils import ensure_dir, save_checkpoint, save_json


@dataclass
class OptimConfig:
    name: str = "sgd"
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    nesterov: bool = False


@dataclass
class TrainConfig:
    epochs: int = 100
    label_smoothing: float = 0.0
    gradient_clip_norm: float = 0.0
    log_every: int = 50
    amp: bool = False
    barrier_threshold: float = 0.1
    history_flush_every: int = 5
    show_batch_progress: bool = True

    transition_curve_points: int = 11
    transition_eval_batches: int = 2

    force_stage_schedule_for_train_loss: bool = False
    force_stage_schedule_for_all_growth_runs: bool = False
    force_stage_schedule_weighting: str = "stage_index"
    force_stage_schedule_gamma: float = 2.0
    min_epochs_per_stage: int = 3

    eval_test_every_epoch: bool = False


@dataclass
class RunSpec:
    dataset: str
    arch: str
    seed: int
    run_name: str
    output_dir: str
    criterion: str
    growth: GrowthConfig
    force_stage_schedule_override: Optional[bool] = None



def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().item())



def evaluate(model: nn.Module, loader, device: torch.device, loss_fn) -> Dict[str, float]:
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_correct += float((logits.argmax(dim=1) == y).sum().item())
            total_n += bs

    if was_training:
        model.train()

    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
    }



def train_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    loss_fn,
    growth_controller: Optional[GrowthMaskController] = None,
    gradient_clip_norm: float = 0.0,
    progress_bar=None,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_n = 0

    use_amp = scaler is not None and device.type == "cuda" and scaler.is_enabled()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()

            if gradient_clip_norm and gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            if gradient_clip_norm and gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            optimizer.step()

        if growth_controller is not None:
            growth_controller.apply_masks_(optimizer)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += float((logits.argmax(dim=1) == y).sum().item())
        total_n += bs

        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(total_correct / max(1, total_n)):.3f}")

    return {
        "loss": total_loss / max(1, total_n),
        "acc": total_correct / max(1, total_n),
    }



def _build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    name = cfg.name.lower()

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov,
        )

    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    raise ValueError(f"Unsupported optimizer: {cfg.name}")



def _metric_for_criterion(history_row: Dict[str, float], criterion: str) -> float:
    c = criterion.lower()
    if c == "val_acc":
        return float(history_row["val_acc"])
    if c == "train_loss":
        return -float(history_row["train_loss"])
    if c in {"lambda_max", "trace", "logdet"}:
        return -float(history_row[c])
    if c in {"active_lambda_max", "active_trace"}:
        return -float(history_row[c])
    raise ValueError(f"Unknown criterion: {criterion}")



def _need_hessian(criterion: str) -> bool:
    return criterion.lower() in {"lambda_max", "trace", "logdet", "active_lambda_max", "active_trace"}



def _snapshot_model(model: nn.Module, stage_idx: int, epoch: int, metric_value: float) -> Dict:
    return {
        "model_state": clone_state_dict_to_cpu(model.state_dict()),
        "stage_idx": int(stage_idx),
        "epoch": int(epoch),
        "metric_value": float(metric_value),
    }



def _allocate_integer_budget(total: int, weights: List[float], minimum: int) -> List[int]:
    n = len(weights)
    if n <= 0:
        return []

    total = int(total)
    minimum = int(max(0, minimum))
    if total < n * minimum:
        raise ValueError(f"Cannot allocate total={total} across {n} stages with minimum={minimum} each.")

    w = torch.tensor(weights, dtype=torch.float64)
    w = torch.clamp(w, min=1e-12)
    w = w / w.sum()

    remaining = total - n * minimum
    raw_extra = remaining * w

    extra_floor = torch.floor(raw_extra).to(torch.int64)
    lengths = torch.full((n,), minimum, dtype=torch.int64) + extra_floor

    leftover = int(total - int(lengths.sum().item()))
    if leftover > 0:
        frac = raw_extra - torch.floor(raw_extra)
        order = torch.argsort(frac, descending=True)
        for i in range(leftover):
            lengths[order[i]] += 1

    out = [int(x) for x in lengths.tolist()]
    assert sum(out) == total
    assert all(x >= minimum for x in out)
    return out

def _make_weighted_stage_lengths(
    total_epochs: int,
    stages: int,
    weighting: str = "stage_index",
    gamma: float = 2.0,
    min_epochs_per_stage: int = 3,
) -> List[int]:
    total_epochs = int(total_epochs)
    stages = int(stages)
    if stages <= 0:
        return []
    if stages == 1:
        return [total_epochs]

    gamma = float(max(0.0, gamma))

    if weighting == "stage_index":
        weights = [float((t + 1) ** gamma) for t in range(stages)]
    else:
        raise ValueError(f"Unknown force_stage_schedule_weighting={weighting!r}")

    return _allocate_integer_budget(total=total_epochs, weights=weights, minimum=int(min_epochs_per_stage))



def _make_stage_deadlines_weighted(
    total_epochs: int,
    stages: int,
    weighting: str = "stage_index",
    gamma: float = 2.0,
    min_epochs_per_stage: int = 3,
) -> Tuple[List[int], List[int]]:
    lengths = _make_weighted_stage_lengths(
        total_epochs=total_epochs,
        stages=stages,
        weighting=weighting,
        gamma=gamma,
        min_epochs_per_stage=min_epochs_per_stage,
    )

    deadlines: List[int] = []
    csum = 0
    for i in range(len(lengths) - 1):
        csum += int(lengths[i])
        deadlines.append(csum)

    return deadlines, lengths



def _should_use_forced_schedule(train_cfg: TrainConfig, run_spec: RunSpec) -> bool:
    if not run_spec.growth.enabled:
        return False
    if run_spec.force_stage_schedule_override is not None:
        return bool(run_spec.force_stage_schedule_override)
    if train_cfg.force_stage_schedule_for_all_growth_runs:
        return True
    if train_cfg.force_stage_schedule_for_train_loss and run_spec.criterion.lower() == "train_loss":
        return True
    return False



def run_training(
    model: nn.Module,
    loaders,
    device: torch.device,
    optim_cfg: OptimConfig,
    train_cfg: TrainConfig,
    hessian_cfg: HessianConfig,
    run_spec: RunSpec,
) -> Dict:
    out_dir = ensure_dir(run_spec.output_dir)
    model = model.to(device)

    init_snapshot = _snapshot_model(model=model, stage_idx=0, epoch=0, metric_value=0.0)
    save_checkpoint(init_snapshot, out_dir / "checkpoint_init.pt")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    optimizer = _build_optimizer(model, optim_cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, train_cfg.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=(train_cfg.amp and device.type == "cuda"))

    growth = GrowthMaskController(model, run_spec.growth)
    if run_spec.growth.enabled:
        growth.apply_masks_(optimizer)

    if "hessian" not in loaders:
        raise KeyError("loaders must contain a 'hessian' loader")
    hessian_batch = next(iter(loaders["hessian"]))

    transition_loader = loaders.get("transition", loaders.get("val", None))
    if transition_loader is None:
        raise KeyError("loaders must contain either 'transition' or 'val'")

    history_rows: List[Dict] = []
    stage_transition_rows: List[Dict] = []

    best_val_acc = -1.0
    best_val_snapshot = _snapshot_model(model=model, stage_idx=growth.stage_index(), epoch=0, metric_value=-1.0)

    stage_metric_best = float("-inf")
    stage_epochs_since_improve = 0
    best_stage_snapshot = _snapshot_model(model=model, stage_idx=growth.stage_index(), epoch=0, metric_value=float("-inf"))

    pending_transition: Optional[Dict] = None
    last_hessian_values: Dict[str, float] = {}
    last_test_stats: Dict[str, float] = {"loss": float("nan"), "acc": float("nan")}

    use_forced_schedule = _should_use_forced_schedule(train_cfg=train_cfg, run_spec=run_spec)
    stage_deadlines: List[int] = []
    stage_lengths: List[int] = []
    if use_forced_schedule:
        stage_deadlines, stage_lengths = _make_stage_deadlines_weighted(
            total_epochs=train_cfg.epochs,
            stages=run_spec.growth.stages,
            weighting=train_cfg.force_stage_schedule_weighting,
            gamma=train_cfg.force_stage_schedule_gamma,
            min_epochs_per_stage=train_cfg.min_epochs_per_stage,
        )

    if run_spec.growth.enabled and len(stage_deadlines) > 0:
        print(f"[{run_spec.run_name}] forced stage lengths   = {stage_lengths}")
        print(f"[{run_spec.run_name}] forced stage deadlines = {stage_deadlines}")

    def flush_history() -> None:
        pd.DataFrame(history_rows).to_csv(out_dir / "history.csv", index=False)

    def flush_transitions() -> None:
        pd.DataFrame(stage_transition_rows).to_csv(out_dir / "stage_transitions.csv", index=False)

    def finalize_stage(current_epoch: int, force_advance: bool, advance_reason: str = "unknown") -> None:
        nonlocal pending_transition, best_stage_snapshot, stage_metric_best, stage_epochs_since_improve

        model.load_state_dict(best_stage_snapshot["model_state"], strict=True)
        growth.set_stage(int(best_stage_snapshot["stage_idx"]))
        growth.apply_masks_(optimizer)

        if pending_transition is not None:
            active_mask_dict = growth.param_mask_dict(device)
            new_mask_dict = growth.newly_released_mask_dict(
                int(pending_transition["from_stage"]),
                int(best_stage_snapshot["stage_idx"]),
                device,
            )

            tm = compute_transition_metrics(
                model=model,
                pre_state=pending_transition["pre_state"],
                post_state=best_stage_snapshot["model_state"],
                eval_loader=transition_loader,
                hessian_batch=hessian_batch,
                device=device,
                hessian_cfg=hessian_cfg,
                active_mask_dict=active_mask_dict,
                new_mask_dict=new_mask_dict,
                barrier_threshold=train_cfg.barrier_threshold,
                interpolation_points=int(train_cfg.transition_curve_points),
                max_eval_batches=int(train_cfg.transition_eval_batches),
                label_smoothing=float(train_cfg.label_smoothing),
            )

            row = {
                "from_stage": int(pending_transition["from_stage"]),
                "to_stage": int(best_stage_snapshot["stage_idx"]),
                "from_epoch": int(pending_transition["epoch"]),
                "to_epoch": int(best_stage_snapshot["epoch"]),
                "advance_reason": str(pending_transition.get("advance_reason", "unknown")),
            }
            row.update(tm)
            stage_transition_rows.append(row)
            flush_transitions()
            pending_transition = None

        if force_advance and run_spec.growth.enabled and (not growth.is_last_stage()):
            pre_state = clone_state_dict_to_cpu(model.state_dict())
            prev_stage = growth.stage_index()

            pending_transition = {
                "from_stage": int(prev_stage),
                "epoch": int(best_stage_snapshot["epoch"]),
                "pre_state": pre_state,
                "advance_reason": str(advance_reason),
            }

            growth.advance_stage()
            growth.apply_masks_(optimizer)

            stage_metric_best = float("-inf")
            stage_epochs_since_improve = 0
            best_stage_snapshot = _snapshot_model(
                model=model,
                stage_idx=growth.stage_index(),
                epoch=current_epoch,
                metric_value=float("-inf"),
            )
        else:
            stage_metric_best = best_stage_snapshot["metric_value"]
            stage_epochs_since_improve = 0

    epoch_bar = tqdm(range(1, train_cfg.epochs + 1), desc=f"{run_spec.run_name} | seed={run_spec.seed}", leave=True)
    
    for epoch in epoch_bar:
        batch_bar = None
        if train_cfg.show_batch_progress:
            batch_bar = tqdm(total=len(loaders["train"]), desc=f"epoch {epoch:03d} | stage {growth.stage_index()}", leave=False)

        train_stats = train_one_epoch(
            model=model,
            loader=loaders["train"],
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            growth_controller=(growth if run_spec.growth.enabled else None),
            gradient_clip_norm=train_cfg.gradient_clip_norm,
            progress_bar=batch_bar,
        )

        if batch_bar is not None:
            batch_bar.close()

        scheduler.step()

        val_stats = evaluate(model, loaders["val"], device=device, loss_fn=loss_fn)
        if train_cfg.eval_test_every_epoch:
            last_test_stats = evaluate(model, loaders["test"], device=device, loss_fn=loss_fn)

        row: Dict[str, float | int | str] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "stage": int(growth.stage_index()),
            "train_loss": float(train_stats["loss"]),
            "train_acc": float(train_stats["acc"]),
            "val_loss": float(val_stats["loss"]),
            "val_acc": float(val_stats["acc"]),
        }
        if train_cfg.eval_test_every_epoch:
            row["test_loss"] = float(last_test_stats["loss"])
            row["test_acc"] = float(last_test_stats["acc"])

        need_hessian_this_epoch = (
            _need_hessian(run_spec.criterion)
            and hessian_cfg.enabled
            and (epoch % max(1, int(hessian_cfg.eval_every)) == 0)
        )
        if need_hessian_this_epoch:
            mask_dict = growth.param_mask_dict(device) if run_spec.growth.enabled else None

            crit = run_spec.criterion.lower()
            base_metrics = set()
            if crit.startswith("active_"):
                base_metrics.add(crit.replace("active_", ""))
            else:
                base_metrics.add(crit)

            hess = compute_hessian_metrics(
                model=model,
                batch=hessian_batch,
                device=device,
                cfg=hessian_cfg,
                mask_dict=mask_dict,
                metrics=tuple(sorted(base_metrics)),
            )

            for k, v in hess.items():
                key = f"active_{k}" if crit.startswith("active_") else k
                row[key] = float(v)
                last_hessian_values[key] = float(v)
        else:
            row.update({k: float(v) for k, v in last_hessian_values.items()})

        metric_value = _metric_for_criterion(row, run_spec.criterion)
        improved = metric_value > stage_metric_best
        if improved:
            stage_metric_best = metric_value
            stage_epochs_since_improve = 0
            best_stage_snapshot = _snapshot_model(
                model=model,
                stage_idx=growth.stage_index(),
                epoch=epoch,
                metric_value=metric_value,
            )
            save_checkpoint(best_stage_snapshot, out_dir / "checkpoint_best_stage.pt")
        else:
            stage_epochs_since_improve += 1

        if float(val_stats["acc"]) > best_val_acc:
            best_val_acc = float(val_stats["acc"])
            best_val_snapshot = _snapshot_model(
                model=model,
                stage_idx=growth.stage_index(),
                epoch=epoch,
                metric_value=best_val_acc,
            )
            save_checkpoint(best_val_snapshot, out_dir / "checkpoint_best_val.pt")

        history_rows.append(row)

        epoch_bar.set_postfix(
            stage=int(growth.stage_index()),
            train_loss=f"{row['train_loss']:.4f}",
            val_acc=f"{row['val_acc']:.4f}",
            best_stage=f"{stage_metric_best:.4f}" if stage_metric_best != float("-inf") else "nan",
        )

        if epoch % max(1, train_cfg.history_flush_every) == 0:
            flush_history()

        can_advance_stage = run_spec.growth.enabled and (not growth.is_last_stage())

        must_end_stage_by_patience = (
            can_advance_stage
            and (not use_forced_schedule)
            and (stage_epochs_since_improve >= max(1, run_spec.growth.patience))
        )

        must_end_stage_by_schedule = (
            can_advance_stage
            and use_forced_schedule
            and (growth.stage_index() < len(stage_deadlines))
            and (epoch >= stage_deadlines[growth.stage_index()])
        )

        if must_end_stage_by_patience or must_end_stage_by_schedule:
            advance_reason = "patience" if must_end_stage_by_patience else "scheduled"
            finalize_stage(epoch, force_advance=True, advance_reason=advance_reason)

    finalize_stage(train_cfg.epochs, force_advance=False, advance_reason="finalize")

    model.load_state_dict(best_val_snapshot["model_state"], strict=True)
    growth.set_stage(int(best_val_snapshot["stage_idx"]))
    growth.apply_masks_(optimizer)

    final_val = evaluate(model, loaders["val"], device=device, loss_fn=loss_fn)
    final_test = evaluate(model, loaders["test"], device=device, loss_fn=loss_fn)

    final_active_hessian = (
        compute_hessian_metrics(
            model=model,
            batch=hessian_batch,
            device=device,
            cfg=hessian_cfg,
            mask_dict=(growth.param_mask_dict(device) if run_spec.growth.enabled else None),
            metrics=("lambda_max", "trace"),
        )
        if hessian_cfg.enabled
        else {"lambda_max": 0.0, "trace": 0.0}
    )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(out_dir / "history.csv", index=False)

    trans_df = pd.DataFrame(stage_transition_rows)
    trans_df.to_csv(out_dir / "stage_transitions.csv", index=False)

    save_checkpoint(
        {
            "model_state": clone_state_dict_to_cpu(model.state_dict()),
            "run_spec": run_spec.__dict__,
            "selected_for_reporting": "best_val",
            "stage_idx": int(best_val_snapshot["stage_idx"]),
            "epoch": int(best_val_snapshot["epoch"]),
        },
        out_dir / "checkpoint_report_model.pt",
    )

    summary = {
        "dataset": run_spec.dataset,
        "arch": run_spec.arch,
        "seed": run_spec.seed,
        "run_name": run_spec.run_name,
        "criterion": run_spec.criterion,
        "growth_enabled": bool(run_spec.growth.enabled),
        "stages": int(run_spec.growth.stages),
        "patience": int(run_spec.growth.patience),
        "forced_stage_schedule_used": bool(use_forced_schedule),
        "force_stage_schedule_override": (None if run_spec.force_stage_schedule_override is None else bool(run_spec.force_stage_schedule_override)),
        "stage_deadlines": [int(x) for x in stage_deadlines],
        "stage_lengths": [int(x) for x in stage_lengths],
        "best_val_epoch": int(best_val_snapshot["epoch"]),
        "best_val_stage": int(best_val_snapshot["stage_idx"]),
        "final_val_acc": float(final_val["acc"]),
        "final_val_loss": float(final_val["loss"]),
        "final_test_acc": float(final_test["acc"]),
        "final_test_loss": float(final_test["loss"]),
        "final_active_lambda_max": float(final_active_hessian["lambda_max"]),
        "final_active_trace": float(final_active_hessian["trace"]),
        "num_stage_transitions": int(len(stage_transition_rows)),
        "growth_mode": str(run_spec.growth.mode),
        "growth_start_stage": int(run_spec.growth.start_stage),
    }
    
    if len(stage_transition_rows) > 0:
        trans = pd.DataFrame(stage_transition_rows).copy()

        last_row = trans.iloc[-1]
        idx_max_barrier = int(trans["barrier"].astype(float).idxmax())
        max_barrier_row = trans.loc[idx_max_barrier]

        summary.update(
            {
                "transition_barrier_mean": float(trans["barrier"].mean()),
                "transition_barrier_std": float(trans["barrier"].std(ddof=0)),
                "retention_mean": float(trans["retained"].mean()),
                "leakage_mean": float(trans["leakage"].mean()),
                "transition_active_lambda_max_mean": float(trans["active_lambda_max"].mean()),
                "transition_active_trace_mean": float(trans["active_trace"].mean()),
                "transition_new_lambda_max_mean": float(trans["new_lambda_max"].mean()),
                "transition_new_trace_mean": float(trans["new_trace"].mean()),
                "transition_endpoint_gap_mean": float(trans["endpoint_gap"].mean()),
                "transition_endpoint_gap_std": float(trans["endpoint_gap"].std(ddof=0)),
                "transition_endpoint_abs_gap_mean": float(trans["endpoint_abs_gap"].mean()),
                "transition_endpoint_abs_gap_std": float(trans["endpoint_abs_gap"].std(ddof=0)),
                "transition_endpoint_improvement_mean": float(trans["endpoint_improvement"].mean()),
                "transition_endpoint_degradation_mean": float(trans["endpoint_degradation"].mean()),
                "last_transition_barrier": float(last_row["barrier"]),
                "last_transition_retained": float(last_row["retained"]),
                "last_transition_leakage": float(last_row["leakage"]),
                "last_transition_endpoint_gap": float(last_row["endpoint_gap"]),
                "last_transition_endpoint_abs_gap": float(last_row["endpoint_abs_gap"]),
                "last_transition_curve_argmax_alpha": float(last_row["curve_argmax_alpha"]),
                "last_transition_advance_reason": str(last_row.get("advance_reason", "unknown")),
                "max_transition_barrier": float(max_barrier_row["barrier"]),
                "argmax_transition_barrier_from_stage": int(max_barrier_row["from_stage"]),
                "argmax_transition_barrier_to_stage": int(max_barrier_row["to_stage"]),
                "argmax_transition_barrier_epoch_from": int(max_barrier_row["from_epoch"]),
                "argmax_transition_barrier_epoch_to": int(max_barrier_row["to_epoch"]),
                "argmax_transition_barrier_endpoint_gap": float(max_barrier_row["endpoint_gap"]),
                "argmax_transition_barrier_curve_argmax_alpha": float(max_barrier_row["curve_argmax_alpha"]),
                "argmax_transition_barrier_advance_reason": str(max_barrier_row.get("advance_reason", "unknown")),
            }
        )

        if "active_dim" in trans.columns:
            summary["transition_active_dim_mean"] = float(trans["active_dim"].mean())
            summary["transition_active_dim_min"] = float(trans["active_dim"].min())
            summary["transition_active_dim_max"] = float(trans["active_dim"].max())

        if "new_dim" in trans.columns:
            summary["transition_new_dim_mean"] = float(trans["new_dim"].mean())
            summary["transition_new_dim_min"] = float(trans["new_dim"].min())
            summary["transition_new_dim_max"] = float(trans["new_dim"].max())

        if "active_trace_per_param" in trans.columns:
            summary["transition_active_trace_per_param_mean"] = float(trans["active_trace_per_param"].mean())

        if "new_trace_per_param" in trans.columns:
            summary["transition_new_trace_per_param_mean"] = float(trans["new_trace_per_param"].mean())

        if "grad_norm_active" in trans.columns:
            summary["transition_grad_norm_active_mean"] = float(trans["grad_norm_active"].mean())
        if "grad_norm_new" in trans.columns:
            summary["transition_grad_norm_new_mean"] = float(trans["grad_norm_new"].mean())
        if "grad_ratio_new" in trans.columns:
            summary["transition_grad_ratio_new_mean"] = float(trans["grad_ratio_new"].mean())

    save_json(
        {
            "summary": summary,
            "optim": optim_cfg.__dict__,
            "train": train_cfg.__dict__,
            "hessian": hessian_cfg.__dict__,
            "growth": run_spec.growth.__dict__,
        },
        out_dir / "summary.json",
    )

    return {
        "summary": summary,
        "history": history_df,
        "stage_transitions": trans_df,
    }