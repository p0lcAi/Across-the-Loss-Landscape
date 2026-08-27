from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import yaml

from progressive_growth.data import DataConfig, build_dataloaders
from progressive_growth.engine import HessianConfig, OptimConfig, RunSpec, TrainConfig, run_training
from progressive_growth.growth import GrowthConfig
from progressive_growth.models import ModelConfig, build_model
from progressive_growth.utils import ensure_dir, get_device, save_json, seed_everything, serialize_config


def _get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _first_not_none(*values, default=None):
    for v in values:
        if v is not None:
            return v
    return default


def _run_growth_get(cfg: Dict[str, Any], run_cfg: Dict[str, Any], key: str, default=None):
    growth = run_cfg.get("growth", {})
    if isinstance(growth, dict) and key in growth:
        return growth[key]

    legacy_key = {
        "enabled": "growth_enabled",
        "stages": "stages",
        "patience": "patience",
        "mode": "growth_mode",
        "start_stage": "start_stage",
    }.get(key)

    if legacy_key is not None and legacy_key in run_cfg:
        return run_cfg[legacy_key]

    if key in run_cfg:
        return run_cfg[key]

    global_value = _get(cfg, "growth", key, default=None)
    if global_value is not None:
        return global_value

    return default

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def build_run_list(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "runs" in cfg:
        return cfg["runs"]

    if isinstance(cfg.get("experiment"), dict) and "runs" in cfg["experiment"]:
        return cfg["experiment"]["runs"]

    criteria = [str(c) for c in _get(cfg, "experiment", "criteria", default=["val_acc"])]
    stages = [int(s) for s in _get(cfg, "experiment", "stages", default=[1, 5])]
    patience = int(_get(cfg, "experiment", "patience", default=2))

    out: List[Dict[str, Any]] = []

    baseline_enabled = bool(_get(cfg, "experiment", "include_baseline", default=True))
    baseline_criterion = str(
        _get(
            cfg,
            "experiment",
            "baseline_criterion",
            default=(criteria[0] if len(criteria) > 0 else "val_acc"),
        )
    )

    if baseline_enabled:
        out.append(
            {
                "name": "baseline_fullmodel",
                "growth_enabled": False,
                "stages": 1,
                "patience": patience,
                "criterion": baseline_criterion,
                "force_stage_schedule": False,
            }
        )

    for criterion in criteria:
        for s in stages:
            if int(s) <= 1:
                continue
            out.append(
                {
                    "name": f"growth_s{int(s)}_{criterion}",
                    "growth_enabled": True,
                    "stages": int(s),
                    "patience": patience,
                    "criterion": str(criterion),
                    "force_stage_schedule": _get(cfg, "experiment", "force_stage_schedule", default=None),
                }
            )

    return out



def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive growth experiment suite")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = ensure_dir(args.output_root)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    data_cfg = DataConfig(
        dataset=_get(cfg, "data", "dataset", default="cifar100"),
        root=_first_not_none(
            _get(cfg, "data", "root", default=None),
            _get(cfg, "data", "data_dir", default=None),
            default="./data",
        ),
        batch_size=_get(cfg, "data", "batch_size", default=128),
        eval_batch_size=_get(cfg, "data", "eval_batch_size", default=256),
        num_workers=_get(cfg, "data", "num_workers", default=4),
        val_fraction=_get(cfg, "data", "val_fraction", default=0.1),
        image_size=_get(cfg, "data", "image_size", default=32),
        augmentation=_get(cfg, "data", "augmentation", default="cifar"),
        pin_memory=bool(_get(cfg, "data", "pin_memory", default=True)),
        persistent_workers=bool(_get(cfg, "data", "persistent_workers", default=True)),
        seed=_get(cfg, "data", "seed", default=0),
        hessian_batch_size=_get(cfg, "data", "hessian_batch_size", default=128),
        n_train=_get(cfg, "data", "n_train", default=4000),
        n_test=_get(cfg, "data", "n_test", default=2000),
        synthetic_noise=_get(cfg, "data", "synthetic_noise", default=0.15),
    )
    
    model_cfg = ModelConfig(
        arch=_get(cfg, "model", "arch", default="resnet18"),
        num_classes=_get(cfg, "model", "num_classes", default=100),
        pretrained=_get(cfg, "model", "pretrained", default=False),
        vit_name=_get(cfg, "model", "vit_name", default="vit_tiny_patch16_224"),
        drop_path_rate=_get(cfg, "model", "drop_path_rate", default=0.0),
        img_size=_first_not_none(
            _get(cfg, "model", "img_size", default=None),
            _get(cfg, "model", "image_size", default=None),
            _get(cfg, "data", "image_size", default=None),
            default=32,
        ),
        input_dim=_get(cfg, "model", "input_dim", default=2),
        hidden_dim=_get(cfg, "model", "hidden_dim", default=512),
        depth=_get(cfg, "model", "depth", default=4),
    )

    optim_cfg = OptimConfig(
        name=_get(cfg, "optim", "name", default="sgd"),
        lr=_get(cfg, "optim", "lr", default=0.1),
        weight_decay=_get(cfg, "optim", "weight_decay", default=5e-4),
        momentum=_get(cfg, "optim", "momentum", default=0.9),
        nesterov=_get(cfg, "optim", "nesterov", default=False),
    )

    train_cfg = TrainConfig(
        epochs=_get(cfg, "train", "epochs", default=100),
        label_smoothing=_get(cfg, "train", "label_smoothing", default=0.0),
        gradient_clip_norm=_first_not_none(
            _get(cfg, "train", "gradient_clip_norm", default=None),
            _get(cfg, "train", "grad_clip_norm", default=None),
            default=0.0,
        ),
        log_every=_get(cfg, "train", "log_every", default=50),
        amp=_get(cfg, "train", "amp", default=False),
        barrier_threshold=_get(cfg, "train", "barrier_threshold", default=0.1),
        history_flush_every=_get(cfg, "train", "history_flush_every", default=5),
        show_batch_progress=_get(cfg, "train", "show_batch_progress", default=True),
        transition_curve_points=_first_not_none(
            _get(cfg, "transition_metrics", "curve_points", default=None),
            _get(cfg, "hessian", "transition_curve_points", default=None),
            default=11,
        ),
        transition_eval_batches=_first_not_none(
            _get(cfg, "transition_metrics", "eval_batches", default=None),
            _get(cfg, "hessian", "transition_eval_batches", default=None),
            default=2,
        ),
        force_stage_schedule_for_train_loss=_get(cfg, "train", "force_stage_schedule_for_train_loss", default=False),
        force_stage_schedule_for_all_growth_runs=_get(cfg, "train", "force_stage_schedule_for_all_growth_runs", default=False),
        force_stage_schedule_weighting=_first_not_none(
            _get(cfg, "train", "force_stage_schedule_weighting", default=None),
            _get(cfg, "train", "stage_schedule", "weight_by", default=None),
            default="stage_index",
        ),
        force_stage_schedule_gamma=_first_not_none(
            _get(cfg, "train", "force_stage_schedule_gamma", default=None),
            _get(cfg, "train", "stage_schedule", "weight_gamma", default=None),
            default=2.0,
        ),
        min_epochs_per_stage=_first_not_none(
            _get(cfg, "train", "min_epochs_per_stage", default=None),
            _get(cfg, "train", "stage_schedule", "min_epochs_per_stage", default=None),
            default=3,
        ),
        eval_test_every_epoch=_get(cfg, "train", "eval_test_every_epoch", default=False),
    )

    hessian_cfg = HessianConfig(
        enabled=_get(cfg, "hessian", "enabled", default=True),
        eval_every=_get(cfg, "hessian", "eval_every", default=_get(cfg, "train", "hessian_every", default=1)),
        max_power_iters=_first_not_none(
            _get(cfg, "hessian", "max_power_iters", default=None),
            _get(cfg, "hessian", "power_iters", default=None),
            default=15,
        ),
        trace_samples=_first_not_none(
            _get(cfg, "hessian", "trace_samples", default=None),
            _get(cfg, "hessian", "trace_probes", default=None),
            default=6,
        ),
        logdet_samples=_first_not_none(
            _get(cfg, "hessian", "logdet_samples", default=None),
            _get(cfg, "hessian", "logdet_probes", default=None),
            default=4,
        ),
        lanczos_steps=_get(cfg, "hessian", "lanczos_steps", default=10),
        damping=_get(cfg, "hessian", "damping", default=1e-3),
        transition_curve_points=_first_not_none(
            _get(cfg, "transition_metrics", "curve_points", default=None),
            _get(cfg, "hessian", "transition_curve_points", default=None),
            default=11,
        ),
        transition_eval_batches=_first_not_none(
            _get(cfg, "transition_metrics", "eval_batches", default=None),
            _get(cfg, "hessian", "transition_eval_batches", default=None),
            default=2,
        ),
    )

    seeds = [int(s) for s in _get(cfg, "experiment", "seeds", default=[0])]
    runs = build_run_list(cfg)

    loaders, num_classes = build_dataloaders(data_cfg)
    model_cfg.num_classes = num_classes

    aggregate_rows: List[Dict[str, Any]] = []
    save_json(
        {
            "raw_config": cfg,
            "resolved": {
                "data": serialize_config(data_cfg),
                "model": serialize_config(model_cfg),
                "optim": serialize_config(optim_cfg),
                "train": serialize_config(train_cfg),
                "hessian": serialize_config(hessian_cfg),
                "device": str(device),
                "runs": runs,
                "seeds": seeds,
            },
        },
        output_root / "resolved_config.json",
    )

    for run_cfg in runs:
        run_seeds = [int(run_cfg["seed"])] if "seed" in run_cfg else seeds

        for seed in run_seeds:
            base_run_name = str(run_cfg["name"])
            seed_suffix = f"_seed{seed}"
            run_name = base_run_name if base_run_name.endswith(seed_suffix) else f"{base_run_name}{seed_suffix}"
            run_dir = ensure_dir(output_root / run_name)
            seed_everything(int(seed))
            model = build_model(model_cfg)
            growth_cfg = GrowthConfig(
                enabled=bool(_run_growth_get(cfg, run_cfg, "enabled", default=True)),
                stages=int(_run_growth_get(cfg, run_cfg, "stages", default=5)),
                patience=int(_run_growth_get(cfg, run_cfg, "patience", default=2)),
                criterion=str(run_cfg.get("criterion", _get(cfg, "growth", "criterion", default="val_acc"))),
                random_seed=int(_run_growth_get(cfg, run_cfg, "random_seed", default=seed)),
                mode=str(_run_growth_get(cfg, run_cfg, "mode", default="progressive")),
                start_stage=int(_run_growth_get(cfg, run_cfg, "start_stage", default=0)),
            )
            spec = RunSpec(
                dataset=data_cfg.dataset,
                arch=model_cfg.arch,
                seed=int(seed),
                run_name=run_name,
                output_dir=str(run_dir),
                criterion=growth_cfg.criterion,
                growth=growth_cfg,
                force_stage_schedule_override=(
                    None
                    if run_cfg.get("force_stage_schedule", None) is None
                    else bool(run_cfg.get("force_stage_schedule"))
                ),
            )
            print(f"\n=== Running {run_name} on {device} ===")
            res = run_training(
                model=model,
                loaders=loaders,
                device=device,
                optim_cfg=copy.deepcopy(optim_cfg),
                train_cfg=copy.deepcopy(train_cfg),
                hessian_cfg=copy.deepcopy(hessian_cfg),
                run_spec=spec,
            )
            row = dict(res["summary"])
            row.update(
                {
                    "run_group": run_cfg["name"],
                    "seed": int(seed),
                }
            )
            aggregate_rows.append(row)

    agg = pd.DataFrame(aggregate_rows)
    agg.to_csv(output_root / "all_runs_summary.csv", index=False)
    group_cols = ["run_group", "criterion", "growth_enabled", "stages", "patience", "dataset", "arch"]
    if len(agg) > 0:
        numeric = agg.select_dtypes(include="number").columns.tolist()
        grouped = agg.groupby(group_cols, as_index=False)[numeric].mean()
        grouped.to_csv(output_root / "grouped_summary_mean.csv", index=False)
    print(f"\nSaved outputs to {output_root.resolve()}")


if __name__ == "__main__":
    main()