from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class HessianConfig:
    enabled: bool = True
    eval_every: int = 1
    max_power_iters: int = 20
    trace_samples: int = 10
    logdet_samples: int = 8
    lanczos_steps: int = 16
    damping: float = 1e-3
    transition_curve_points: int = 11
    transition_eval_batches: int = 2



def _named_params(model) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]



def _flatten(ts: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(ts) == 0:
        return torch.empty(0)
    return torch.cat([t.reshape(-1) for t in ts], dim=0)



def _mask_list_from_dict(named_params: Sequence[Tuple[str, torch.nn.Parameter]], mask_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for name, p in named_params:
        if name in mask_dict:
            out.append(mask_dict[name].to(device=p.device, dtype=p.dtype))
        else:
            out.append(torch.ones_like(p))
    return out



def _loss_from_batch(model, batch, device: torch.device) -> torch.Tensor:
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    logits = model(x)
    return F.cross_entropy(logits, y)


class RestrictedHessianOperator:
    def __init__(self, model, batch, device: torch.device, mask_dict: Optional[Dict[str, torch.Tensor]] = None):
        self.model = model
        self.batch = batch
        self.device = device
        self.named_params = _named_params(model)
        self.params = [p for _, p in self.named_params]
        self.mask_list = _mask_list_from_dict(self.named_params, mask_dict or {})
        self.mask_flat = _flatten([m.to(dtype=torch.float32) for m in self.mask_list]).to(device)
        self.dim = int(self.mask_flat.numel())

    def hvp(self, vec: torch.Tensor) -> torch.Tensor:
        vec = vec.to(self.device)
        masked_vec = vec * self.mask_flat
        self.model.zero_grad(set_to_none=True)
        loss = _loss_from_batch(self.model, self.batch, self.device)
        grads = torch.autograd.grad(loss, self.params, create_graph=True, allow_unused=True)
        flat_grads = _flatten([
            (g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, self.params)
        ])
        dot = torch.dot(flat_grads * self.mask_flat, masked_vec)
        hv = torch.autograd.grad(dot, self.params, retain_graph=False, create_graph=False, allow_unused=True)
        flat_hv = _flatten([
            (h if h is not None else torch.zeros_like(p)).reshape(-1) for h, p in zip(hv, self.params)
        ])
        return flat_hv * self.mask_flat

    def random_rademacher(self) -> torch.Tensor:
        z = torch.randint(0, 2, (self.dim,), device=self.device, dtype=torch.int64).to(torch.float32)
        z = 2.0 * z - 1.0
        return z * self.mask_flat



def power_iteration(op: RestrictedHessianOperator, num_iters: int = 20) -> float:
    if op.mask_flat.sum().item() < 1:
        return 0.0
    v = op.random_rademacher()
    v = v / (v.norm() + 1e-12)
    eig = 0.0
    for _ in range(max(1, int(num_iters))):
        w = op.hvp(v)
        nrm = w.norm() + 1e-12
        v = w / nrm
        eig = float(torch.dot(v, op.hvp(v)).item())
    return eig



def hutchinson_trace(op: RestrictedHessianOperator, num_samples: int = 10) -> float:
    if op.mask_flat.sum().item() < 1:
        return 0.0
    vals = []
    for _ in range(max(1, int(num_samples))):
        z = op.random_rademacher()
        hz = op.hvp(z)
        vals.append(float(torch.dot(z, hz).item()))
    return float(sum(vals) / len(vals))



def _lanczos(op: RestrictedHessianOperator, q0: torch.Tensor, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    q = q0 / (q0.norm() + 1e-12)
    alphas: List[torch.Tensor] = []
    betas: List[torch.Tensor] = []
    q_prev = torch.zeros_like(q)
    beta_prev = torch.tensor(0.0, device=q.device)
    for _ in range(steps):
        z = op.hvp(q)
        if len(alphas) > 0:
            z = z - beta_prev * q_prev
        alpha = torch.dot(q, z)
        z = z - alpha * q
        beta = z.norm()
        alphas.append(alpha)
        betas.append(beta)
        if beta.item() < 1e-10:
            break
        q_prev = q
        q = z / beta
        beta_prev = beta
    a = torch.stack(alphas)
    if len(alphas) <= 1:
        b = torch.zeros(0, device=q0.device)
    else:
        b = torch.stack(betas[:-1])
    return a, b



def stochastic_logdet(op: RestrictedHessianOperator, num_samples: int = 8, lanczos_steps: int = 16, damping: float = 1e-3) -> float:
    if op.mask_flat.sum().item() < 1:
        return 0.0
    estimates: List[float] = []
    active_dim = max(1.0, float(op.mask_flat.sum().item()))
    for _ in range(max(1, int(num_samples))):
        q0 = op.random_rademacher()
        q0 = q0 / (q0.norm() + 1e-12)
        alphas, betas = _lanczos(op, q0, steps=max(2, int(lanczos_steps)))
        k = alphas.numel()
        T = torch.diag(alphas)
        if betas.numel() > 0:
            T = T + torch.diag(betas, diagonal=1) + torch.diag(betas, diagonal=-1)
        T = T + damping * torch.eye(k, device=T.device)
        evals, evecs = torch.linalg.eigh(T)
        evals = torch.clamp(evals, min=1e-12)
        weights = evecs[0, :] ** 2
        estimates.append(float(active_dim * torch.sum(weights * torch.log(evals)).item()))
    return float(sum(estimates) / len(estimates))



def compute_hessian_metrics(
    model,
    batch,
    device: torch.device,
    cfg: HessianConfig,
    mask_dict: Optional[Dict[str, torch.Tensor]] = None,
    metrics: Sequence[str] = ("lambda_max", "trace"),
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    op = RestrictedHessianOperator(model, batch=batch, device=device, mask_dict=mask_dict)
    out: Dict[str, float] = {}
    if "lambda_max" in metrics:
        out["lambda_max"] = power_iteration(op, num_iters=cfg.max_power_iters)
    if "trace" in metrics:
        out["trace"] = hutchinson_trace(op, num_samples=cfg.trace_samples)
    if "logdet" in metrics:
        out["logdet"] = stochastic_logdet(op, num_samples=cfg.logdet_samples, lanczos_steps=cfg.lanczos_steps, damping=cfg.damping)
    if was_training:
        model.train()
    return out
