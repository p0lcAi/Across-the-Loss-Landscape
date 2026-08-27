from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hessian import HessianConfig, compute_hessian_metrics


# ============================================================
# State dict helpers
# ============================================================

def clone_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def interpolate_state_dicts(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """
    Linear interpolation of two state_dicts, robust to non-float buffers.

    For floating/complex tensors: (1-alpha)*a + alpha*b
    For non-floating tensors (e.g. num_batches_tracked): choose nearest endpoint.
    """
    alpha = float(alpha)
    out: Dict[str, torch.Tensor] = {}
    for k in state_a.keys():
        va = state_a[k]
        vb = state_b[k]
        if torch.is_floating_point(va) or torch.is_complex(va):
            out[k] = (1.0 - alpha) * va + alpha * vb
        else:
            out[k] = va.clone() if alpha < 0.5 else vb.clone()
    return out


# ============================================================
# Loss evaluation (used for barrier/leakage)
# ============================================================

def evaluate_loss_on_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
    label_smoothing: float = 0.0,
) -> float:
    """
    Average CE loss over a loader (optionally limited to max_batches).
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_n = 0

    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_n += bs

    if was_training:
        model.train()

    return total_loss / max(1, total_n)


# ============================================================
# Hessian helpers
# ============================================================

def _safe_hessian_metrics(
    model: nn.Module,
    batch,
    device: torch.device,
    hessian_cfg: HessianConfig,
    mask_dict,
) -> Dict[str, float]:
    if not hessian_cfg.enabled:
        return {"lambda_max": 0.0, "trace": 0.0}

    out = compute_hessian_metrics(
        model=model,
        batch=batch,
        device=device,
        cfg=hessian_cfg,
        mask_dict=mask_dict,
        metrics=("lambda_max", "trace"),
    )

    return {
        "lambda_max": float(out.get("lambda_max", 0.0)),
        "trace": float(out.get("trace", 0.0)),
    }


def _mask_dim(mask_dict: Optional[Dict[str, torch.Tensor]]) -> int:
    """
    Dimension (count of active parameters) for a 0/1 mask dict.
    Robust to dtype/bool/device.
    """
    if not mask_dict:
        return 0
    total = 0
    for m in mask_dict.values():
        if m is None:
            continue
        total += int(m.detach().to(dtype=torch.float32).sum().item())
    return int(total)


def _masked_grad_norm_from_batch(
    model: nn.Module,
    batch,
    device: torch.device,
    mask_dict: Optional[Dict[str, torch.Tensor]],
) -> float:
    """
    Compute || P_mask grad || on a single batch (CE loss).
    Uses current model parameters (no state_dict loads here).
    """
    if not mask_dict:
        return 0.0

    was_training = model.training
    model.train(False)

    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]

    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)

    sq = 0.0
    for (name, p), g in zip(named_params, grads):
        if g is None:
            continue
        m = mask_dict.get(name, None)
        if m is None:
            continue  # if not present, treat as frozen for this projected norm
        # project and accumulate squared norm
        gm = g * m.to(device=g.device, dtype=g.dtype)
        sq += float(gm.detach().to(dtype=torch.float32).pow(2).sum().item())

    if was_training:
        model.train(True)

    return float(math.sqrt(max(0.0, sq)))


# ============================================================
# Main: transition metrics
# ============================================================

def compute_transition_metrics(
    model: nn.Module,
    pre_state: Dict[str, torch.Tensor],
    post_state: Dict[str, torch.Tensor],
    eval_loader,
    hessian_batch,
    device: torch.device,
    hessian_cfg: HessianConfig,
    active_mask_dict,
    new_mask_dict,
    barrier_threshold: float = 0.1,
    interpolation_points: int = 11,
    max_eval_batches: Optional[int] = None,
    label_smoothing: float = 0.0,
    compute_grad_norms: bool = False,
) -> Dict[str, object]:
    """
    Compute transition metrics between two checkpoints (theta^- -> theta^+):

    Core (theory-aligned):
      - interpolation barrier (max loss along straight line) and retained/leakage
      - restricted curvature on active and newly released subspaces (lambda_max, trace)
      - dimensions of masks (active_dim/new_dim) and per-parameter trace normalization

    Extras:
      - endpoint gap / improvement / degradation
      - curve argmax alpha
      - optional projected gradient norms on hessian_batch
    """

    if interpolation_points < 2:
        interpolation_points = 2

    # Keep current state to restore later (avoid side-effects on training loop)
    current_state = clone_state_dict_to_cpu(model.state_dict())

    try:
        # ---- Interpolation curve on eval loader ----
        alphas = np.linspace(0.0, 1.0, int(interpolation_points))
        curve_losses: List[float] = []

        for alpha in alphas:
            interp_state = interpolate_state_dicts(pre_state, post_state, float(alpha))
            model.load_state_dict(interp_state, strict=True)

            loss_val = evaluate_loss_on_loader(
                model=model,
                loader=eval_loader,
                device=device,
                max_batches=max_eval_batches,
                label_smoothing=label_smoothing,
            )
            curve_losses.append(float(loss_val))

        # Endpoints and barrier
        endpoint_a_loss = float(curve_losses[0])
        endpoint_b_loss = float(curve_losses[-1])
        curve_max_loss = float(max(curve_losses))

        barrier = float(curve_max_loss - max(endpoint_a_loss, endpoint_b_loss))
        if barrier < 0.0:
            barrier = 0.0

        retained = int(barrier <= float(barrier_threshold))
        leakage = int(1 - retained)

        curve_argmax_idx = int(np.argmax(curve_losses))
        curve_argmax_alpha = float(curve_argmax_idx / max(1, len(curve_losses) - 1))

        # Endpoint differences
        endpoint_gap = float(endpoint_b_loss - endpoint_a_loss)
        endpoint_abs_gap = float(abs(endpoint_gap))
        endpoint_improvement = float(max(0.0, endpoint_a_loss - endpoint_b_loss))
        endpoint_degradation = float(max(0.0, endpoint_b_loss - endpoint_a_loss))

        # ---- Restricted Hessian metrics at post_state ----
        model.load_state_dict(post_state, strict=True)

        active_hessian = _safe_hessian_metrics(
            model=model,
            batch=hessian_batch,
            device=device,
            hessian_cfg=hessian_cfg,
            mask_dict=active_mask_dict,
        )

        new_hessian = _safe_hessian_metrics(
            model=model,
            batch=hessian_batch,
            device=device,
            hessian_cfg=hessian_cfg,
            mask_dict=new_mask_dict,
        )

        # ---- Mask dimensions and per-param normalizations ----
        active_dim = _mask_dim(active_mask_dict)
        new_dim = _mask_dim(new_mask_dict)

        active_trace_pp = float(active_hessian["trace"]) / float(max(1, active_dim))
        new_trace_pp = float(new_hessian["trace"]) / float(max(1, new_dim))

        out: Dict[str, object] = {
            # Core transition quantities
            "barrier": float(barrier),
            "retained": int(retained),
            "leakage": int(leakage),

            # Curvature (restricted)
            "active_lambda_max": float(active_hessian["lambda_max"]),
            "active_trace": float(active_hessian["trace"]),
            "new_lambda_max": float(new_hessian["lambda_max"]),
            "new_trace": float(new_hessian["trace"]),

            # Dimensions + normalized curvature (IMPORTANT for comparisons across S)
            "active_dim": int(active_dim),
            "new_dim": int(new_dim),
            "active_trace_per_param": float(active_trace_pp),
            "new_trace_per_param": float(new_trace_pp),

            # Curve diagnostics
            "curve_losses": [float(v) for v in curve_losses],
            "endpoint_a_loss": float(endpoint_a_loss),
            "endpoint_b_loss": float(endpoint_b_loss),
            "curve_max_loss": float(curve_max_loss),

            # Endpoint delta diagnostics
            "endpoint_gap": float(endpoint_gap),
            "endpoint_abs_gap": float(endpoint_abs_gap),
            "endpoint_improvement": float(endpoint_improvement),
            "endpoint_degradation": float(endpoint_degradation),

            # Argmax location on curve
            "curve_argmax_idx": int(curve_argmax_idx),
            "curve_argmax_alpha": float(curve_argmax_alpha),
        }

        # ---- Optional projected gradient norms (single batch) ----
        if compute_grad_norms:
            gn_active = _masked_grad_norm_from_batch(
                model=model,
                batch=hessian_batch,
                device=device,
                mask_dict=active_mask_dict,
            )
            gn_new = _masked_grad_norm_from_batch(
                model=model,
                batch=hessian_batch,
                device=device,
                mask_dict=new_mask_dict,
            )
            out["grad_norm_active"] = float(gn_active)
            out["grad_norm_new"] = float(gn_new)
            out["grad_ratio_new"] = float(gn_new / (gn_active + 1e-12))

        return out

    finally:
        # Always restore previous model state
        model.load_state_dict(current_state, strict=True)