"""
Theorem-compatible growth toy (CPU-only, robust) code.

Goal
----
Implement a toy that matches the *theoretical* mechanism:

- Basins are local quadratic ellipsoids.
- Growth = nested affine slices: at stage t, only K_t coordinates are free,
  the complement is frozen at the initialization offset.
- The "selection" is not done by running GD (which introduces optimization artifacts),
  but by *probabilistic weighting proportional to intersection volumes* of the ellipsoid
  with the stage slice.
- Across stages, we multiply these slice-volume factors (entropic amplification).

This isolates the theorem mechanism:
    "Progressive growth amplifies selection toward flatter basins via a volume effect."

Implementation details:
    We use a Hessian family:
        H_i = I + U_i U_i^T   (SPD, low rank r)
    which makes Schur-complement computations cheap (Woodbury).

    For a given stage with free dimension K:
    - partition coordinates into free (x) of size K and frozen (y) of size d-K,
    where y is fixed to the initialization offset in those coordinates.
    - the minimal quadratic value on the slice is:
        q_min(K) = y^T (A_yy - A_yx A_xx^{-1} A_xy) y
    where A is H in the stage coordinate system.
    - the slice volume proxy is:
        vol_i(K) \propto slack(K)^{K/2} / sqrt(det(A_xx)),
    slack(K) = 2*eps - q_min(K).
    - log-vol proxy:
        logvol_i(K) = (K/2) log(slack) - (1/2) logdet(A_xx),
    if slack>0 else -inf.

    Across a schedule K_1,...,K_S:
        logw_i = Σ_t logvol_i(K_t)
        p_i \propto exp(logw_i)

    We report (averaged across random trials):
    - P(flat-family) and P(top-k flattest)
    - E[logdet(H_selected)] and E[lambda_max(H_selected)]
    - family frequency curves

    CPU-only: numpy + matplotlib. No torch, no cuda.

Usage
-----
Quick sanity:
  python toy.py --scenario multiplicity --preset quick

Paper-like:
  python toy.py --scenario multiplicity --preset paper
  python toy.py --scenario tradeoff     --preset paper

You can override:
  --stage-counts "1,5,10,20,50,100"
  --n-trials 500
  --d 100 --rank 5
  --eps 1.0
  --sigma 0.12
  --K0 1
  --topk 5
  --out-dir outputs/myrun
"""

from __future__ import annotations

import argparse
import yaml
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# Basic utilities

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def parse_stage_counts(s: str) -> List[int]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty stage-count list")
    if any(v < 1 for v in vals):
        raise ValueError("All stage counts must be >= 1")
    return vals

def softmax_from_logw(logw: np.ndarray) -> np.ndarray:
    """
    Robust softmax. If all entries are -inf (no feasible basin), return uniform.
    """
    logw = np.asarray(logw, dtype=float)
    mx = np.max(logw)
    if not np.isfinite(mx):
        return np.ones_like(logw) / float(logw.size)
    w = np.exp(logw - mx)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        return np.ones_like(logw) / float(logw.size)
    return w / s

def mean_ci95(x: np.ndarray) -> Tuple[float, float]:
    """
    Return (mean, half-width) of approx 95% CI using 1.96*SE.
    """
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    sd = float(np.std(x))
    n = max(1, x.size)
    hw = 1.96 * sd / math.sqrt(n)
    return m, hw

def save_current_fig(out_dir: Path, stem: str, dpi: int = 220) -> None:
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")


# Growth schedule

def make_K_schedule(d: int, n_stages: int, K0: int = 1) -> np.ndarray:
    """
    Non-decreasing K schedule of length n_stages ending at d.
    """
    d = int(d)
    n_stages = int(n_stages)
    if n_stages <= 1:
        return np.array([d], dtype=int)

    K0 = int(max(1, min(d, K0)))
    raw = np.linspace(K0, d, num=n_stages)
    Ks = np.rint(raw).astype(int)
    Ks = np.clip(Ks, 1, d)
    for i in range(1, Ks.size):
        Ks[i] = max(Ks[i], Ks[i - 1])
    Ks[-1] = d
    for i in range(1, Ks.size):
        Ks[i] = max(Ks[i], Ks[i - 1])
    return Ks


# Scenario definitions

@dataclass(frozen=True)
class FamilyParams:
    name: str
    count: int
    # curvature via low-rank scale s in H = I + U U^T, with U = sqrt(s)*G/sqrt(d)
    scale_range: Tuple[float, float]
    # optional center radius (tradeoff scenario); multiplicity sets these to 0
    center_radius_range: Tuple[float, float] = (0.0, 0.0)

@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    families: List[FamilyParams]

def default_scenarios() -> Dict[str, ScenarioSpec]:
    # Pure theorem story: same location (mu=0), different curvature, multiplicity imbalance
    multiplicity = ScenarioSpec(
        name="multiplicity",
        description="Many flat basins vs fewer sharp basins. Same location (mu=0). Only curvature/volume matters.",
        families=[
            FamilyParams("flat",  25, (0.00, 0.60), (0.0, 0.0)),
            FamilyParams("mid",   15, (0.60, 1.20), (0.0, 0.0)),
            FamilyParams("sharp", 10, (1.20, 2.00), (0.0, 0.0)),
        ],
    )

    # Add a distance/energy accessibility component (NOT needed for the pure theorem mechanism)
    tradeoff = ScenarioSpec(
        name="tradeoff",
        description="Tradeoff: sharp basins closer, flat basins farther (accessibility vs volume).",
        families=[
            FamilyParams("flat",  15, (0.00, 0.60), (1.4, 2.0)),
            FamilyParams("mid",   20, (0.60, 1.20), (0.9, 1.3)),
            FamilyParams("sharp", 15, (1.20, 2.00), (0.4, 0.8)),
        ],
    )

    return {"multiplicity": multiplicity, "tradeoff": tradeoff}

# Basin construction: H_i = I + U_i U_i^T (low-rank SPD)

@dataclass
class BuiltToy:
    U: np.ndarray                 # (m, d, r)
    mu: np.ndarray                # (m, d)
    family_of: np.ndarray         # (m,)
    family_names: List[str]
    logdetH: np.ndarray           # (m,)
    lmaxH: np.ndarray             # (m,)
    flat_mask: np.ndarray         # (m,)
    topk_mask: np.ndarray         # (m,)

def _random_unit_vector(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def build_toy(spec: ScenarioSpec, d: int, rank: int, seed_toy: int, topk: int) -> BuiltToy:
    rng = rng_from_seed(seed_toy)
    families = spec.families
    family_names = [f.name for f in families]
    m = sum(f.count for f in families)

    U = np.zeros((m, d, rank), dtype=float)
    mu = np.zeros((m, d), dtype=float)
    family_of = np.zeros(m, dtype=int)

    idx = 0
    for fid, fam in enumerate(families):
        for _ in range(fam.count):
            s = float(rng.uniform(*fam.scale_range))
            G = rng.standard_normal((d, rank)) / math.sqrt(d)
            U[idx] = math.sqrt(s) * G

            rlo, rhi = fam.center_radius_range
            if rhi > 0.0 or rlo > 0.0:
                rad = float(rng.uniform(rlo, rhi))
                mu[idx] = rad * _random_unit_vector(rng, d)
            else:
                mu[idx] = 0.0

            family_of[idx] = fid
            idx += 1

    # Precompute logdet(H) and lambda_max(H) using small rank matrices:
    # det(I + U U^T) = det(I + U^T U), eigenvalues(H) are 1 + eig(U^T U) plus ones.
    logdetH = np.zeros(m, dtype=float)
    lmaxH = np.zeros(m, dtype=float)
    for i in range(m):
        UtU = U[i].T @ U[i]              # (r,r)
        M = np.eye(rank) + UtU
        sign, ld = np.linalg.slogdet(M)
        if sign <= 0:
            # Should not happen, but keep robust
            ld = float("nan")
        logdetH[i] = float(ld)
        eig = np.linalg.eigvalsh(UtU)
        lmaxH[i] = float(1.0 + np.max(eig))

    # "flat family" = family named "flat" if present
    flat_fid = family_names.index("flat") if "flat" in family_names else 0
    flat_mask = (family_of == flat_fid)

    # top-k flattest by logdet(H) (lower = flatter)
    k = max(1, min(int(topk), m))
    order = np.argsort(logdetH)
    topk_mask = np.zeros(m, dtype=bool)
    topk_mask[order[:k]] = True

    return BuiltToy(
        U=U, mu=mu, family_of=family_of, family_names=family_names,
        logdetH=logdetH, lmaxH=lmaxH, flat_mask=flat_mask, topk_mask=topk_mask
    )


# Core theorem-compatible log-volume computation

def _chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve (L L^T) x = b given lower-triangular Cholesky factor L.
    """
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)

def compute_fvals_for_basin(
    Uperm: np.ndarray,   # (d,r) rows permuted into stage coordinate order
    y: np.ndarray,       # (d,) = (theta0 - mu)_perm
    eps: float,
) -> np.ndarray:
    """
    Compute fvals[K] = logvol proxy for all K=0..d:
      logvol(K) = (K/2) log(slack(K)) - 0.5 logdet(I + G(K))
      where G(K) = sum_{j< K} u_j u_j^T,
            uy(K) = sum_{j>=K} u_j y_j,
            y2(K) = sum_{j>=K} y_j^2,
            quad(K) = y2 + ||uy||^2 - uy^T G uy + (G uy)^T (I+G)^{-1} (G uy),
            slack(K) = 2eps - quad(K).

    Returns fvals shape (d+1,), with fvals[0]=0.
    """
    d, r = Uperm.shape
    fvals = np.full(d + 1, -np.inf, dtype=float)
    fvals[0] = 0.0

    # prefixG[K] = sum_{j< K} u_j u_j^T
    # Compute per-row outer products then cumulative sum
    outer = np.einsum("ir,is->irs", Uperm, Uperm)         # (d,r,r)
    prefixG = np.zeros((d + 1, r, r), dtype=float)
    prefixG[1:] = np.cumsum(outer, axis=0)

    # suffixUy[K] = sum_{j>=K} u_j y_j
    Uyvec = Uperm * y[:, None]                            # (d,r)
    suffixUy = np.zeros((d + 1, r), dtype=float)
    suffixUy[:-1] = np.cumsum(Uyvec[::-1], axis=0)[::-1]

    # suffixY2[K] = sum_{j>=K} y_j^2
    y2 = np.zeros(d + 1, dtype=float)
    y2[:-1] = np.cumsum((y * y)[::-1])[::-1]

    I = np.eye(r, dtype=float)

    for K in range(1, d + 1):
        G = prefixG[K]                 # (r,r)
        M = I + G                      # (r,r)

        # Cholesky for logdet and solve
        # Add tiny jitter if needed for numerical stability
        jitter = 0.0
        for _ in range(3):
            try:
                L = np.linalg.cholesky(M + jitter * I)
                break
            except np.linalg.LinAlgError:
                jitter = 1e-10 if jitter == 0.0 else 10.0 * jitter
        else:
            # Should not happen
            fvals[K] = -np.inf
            continue

        logdetM = 2.0 * float(np.sum(np.log(np.diag(L))))

        uy = suffixUy[K]               # (r,)
        t = G @ uy                     # (r,)
        bnorm2 = float(uy @ t)         # uy^T G uy
        Minv_t = _chol_solve(L, t)     # (I+G)^{-1} (G uy)
        tMinv_t = float(t @ Minv_t)

        quad = float(y2[K] + (uy @ uy) - bnorm2 + tMinv_t)
        slack = 2.0 * float(eps) - quad
        if slack <= 1e-12 or not np.isfinite(slack):
            fvals[K] = -np.inf
            continue

        fvals[K] = (K / 2.0) * math.log(slack) - 0.5 * logdetM

    return fvals


# Experiment runner (expectations, not GD)

@dataclass
class RunConfig:
    d: int = 100
    rank: int = 5
    eps: float = 1.0
    sigma: Optional[float] = None
    K0: int = 1
    stage_counts: List[int] = None
    n_trials: int = 200
    topk: int = 5
    seed_toy: int = 0
    seed_trials: int = 1
    out_dir: str = "toy_outputs"
    dpi: int = 220

    def __post_init__(self):
        if self.stage_counts is None:
            self.stage_counts = [1, 5, 10, 20, 50, 100]
        if self.sigma is None:
            # default: scale so that E[||y_frozen||^2] is O(eps)
            # for K small, frozen dims ~ d, so sigma^2 ~ 2 eps / d (up to a factor)
            self.sigma = 0.9 * math.sqrt(2.0 * self.eps / float(self.d))

def preset_defaults(preset: str) -> Dict:
    if preset == "quick":
        return dict(n_trials=120, dpi=180)
    if preset == "paper":
        return dict(n_trials=500, dpi=220)
    raise ValueError(f"Unknown preset={preset!r}")

def run_theorem_toy(spec: ScenarioSpec, cfg: RunConfig) -> Dict:
    built = build_toy(spec, d=cfg.d, rank=cfg.rank, seed_toy=cfg.seed_toy, topk=cfg.topk)
    m = built.U.shape[0]
    fam_names = built.family_names
    n_fam = len(fam_names)

    print("=" * 88)
    print(f"[Theorem-toy] scenario={spec.name}")
    print(f"  {spec.description}")
    print(f"  d={cfg.d}, rank={cfg.rank}, m={m}, eps={cfg.eps}, sigma={cfg.sigma}")
    print(f"  trials={cfg.n_trials}, stage_counts={cfg.stage_counts}, K0={cfg.K0}, topk={cfg.topk}")
    print()

    # Baseline "full-space" distribution: p_i \propto det(H_i)^(-1/2)
    logw_full = -0.5 * built.logdetH
    p_full = softmax_from_logw(logw_full)

    # Precompute K schedules
    schedules = {int(S): make_K_schedule(cfg.d, int(S), cfg.K0) for S in cfg.stage_counts}

    # Collect per-trial expectations (growth depends on trial; full is constant)
    metrics = {
        "growth": {int(S): {"p_flat": [], "p_topk": [], "elogdet": [], "elmax": [], "famfreq": []} for S in cfg.stage_counts},
        "full":   {int(S): {"p_flat": [], "p_topk": [], "elogdet": [], "elmax": [], "famfreq": []} for S in cfg.stage_counts},
        "_meta":  {"family_names": fam_names, "flat_mask": built.flat_mask, "topk_mask": built.topk_mask, "schedules": schedules},
    }

    # full metrics are constant across S and trials, but we store arrays for plotting CI=0
    pflat_full = float(np.sum(p_full[built.flat_mask]))
    ptopk_full = float(np.sum(p_full[built.topk_mask]))
    elogdet_full = float(p_full @ built.logdetH)
    elmax_full = float(p_full @ built.lmaxH)
    famfreq_full = np.zeros(n_fam, dtype=float)
    for fid in range(n_fam):
        famfreq_full[fid] = float(np.sum(p_full[built.family_of == fid]))

    rng = rng_from_seed(cfg.seed_trials)

    n_no_feasible = 0

    for t in range(cfg.n_trials):
        theta0 = rng.normal(0.0, float(cfg.sigma), size=cfg.d)
        perm = rng.permutation(cfg.d)

        # For each basin, compute fvals[K] once; then stage-count weights are sums along the schedule.
        # In multiplicity scenario, mu=0 so y is shared across basins; in tradeoff, mu differs.
        fvals_mat = np.full((m, cfg.d + 1), -np.inf, dtype=float)

        for i in range(m):
            y = (theta0 - built.mu[i])[perm]            # (d,)
            Uperm = built.U[i][perm, :]                 # (d,rank)
            fvals_mat[i] = compute_fvals_for_basin(Uperm, y, cfg.eps)

        for S in cfg.stage_counts:
            S = int(S)
            Ks = schedules[S]
            logw = np.sum(fvals_mat[:, Ks], axis=1)     # (m,)
            p = softmax_from_logw(logw)

            # If everything was infeasible -> softmax returned uniform; count it
            if not np.isfinite(np.max(logw)):
                n_no_feasible += 1

            pflat = float(np.sum(p[built.flat_mask]))
            ptopk = float(np.sum(p[built.topk_mask]))
            elogdet = float(p @ built.logdetH)
            elmax = float(p @ built.lmaxH)

            famfreq = np.zeros(n_fam, dtype=float)
            for fid in range(n_fam):
                famfreq[fid] = float(np.sum(p[built.family_of == fid]))

            metrics["growth"][S]["p_flat"].append(pflat)
            metrics["growth"][S]["p_topk"].append(ptopk)
            metrics["growth"][S]["elogdet"].append(elogdet)
            metrics["growth"][S]["elmax"].append(elmax)
            metrics["growth"][S]["famfreq"].append(famfreq)

            # full (constant)
            metrics["full"][S]["p_flat"].append(pflat_full)
            metrics["full"][S]["p_topk"].append(ptopk_full)
            metrics["full"][S]["elogdet"].append(elogdet_full)
            metrics["full"][S]["elmax"].append(elmax_full)
            metrics["full"][S]["famfreq"].append(famfreq_full)

    if n_no_feasible > 0:
        print(f"  [warn] {n_no_feasible} / {cfg.n_trials * len(cfg.stage_counts)} (trial,stage) had no feasible basin "
              f"(fell back to uniform). Consider increasing eps or decreasing sigma.\n")

    # Print quick summary
    print("  Summary (means over trials):")
    for S in cfg.stage_counts:
        S = int(S)
        m_pflat, hw_pflat = mean_ci95(np.array(metrics["growth"][S]["p_flat"]))
        m_elog, hw_elog = mean_ci95(np.array(metrics["growth"][S]["elogdet"]))
        m_lmax, hw_lmax = mean_ci95(np.array(metrics["growth"][S]["elmax"]))
        print(f"    S={S:>3d}: P(flat)={m_pflat:.3f}±{hw_pflat:.3f}, "
              f"E[logdet]={m_elog:.3f}±{hw_elog:.3f}, E[lmax]={m_lmax:.3f}±{hw_lmax:.3f}")
    print()

    return metrics


# Plotting

def plot_results(spec: ScenarioSpec, cfg: RunConfig, metrics: Dict) -> None:
    out_dir = ensure_dir(Path(cfg.out_dir) / spec.name)
    xs = np.array(sorted(int(s) for s in cfg.stage_counts), dtype=int)
    fam_names = metrics["_meta"]["family_names"]
    n_fam = len(fam_names)

    suffix = f"scn{spec.name}_d{cfg.d}_m{sum(f.count for f in spec.families)}_r{cfg.rank}_eps{cfg.eps}_tr{cfg.n_trials}"

    def curve(method: str, key: str) -> Tuple[np.ndarray, np.ndarray]:
        means = []
        hws = []
        for S in xs:
            arr = np.array(metrics[method][int(S)][key], dtype=float)
            m, hw = mean_ci95(arr)
            means.append(m)
            hws.append(hw)
        return np.array(means), np.array(hws)

    # 1) P(flat family)
    plt.figure()
    for method, label in [("growth", "growth (theorem toy)"), ("full", "full-space baseline")]:
        y, hw = curve(method, "p_flat")
        plt.errorbar(xs, y, yerr=hw, marker="o", capsize=4, label=label)
    plt.xscale("log")
    plt.xticks(xs, [str(int(s)) for s in xs])
    plt.ylim(0.0, 1.02)
    plt.xlabel("number of stages (log scale)")
    plt.ylabel("P(selected basin in flat family)")
    plt.title("Theorem-toy: flat-family selection probability")
    plt.legend()
    save_current_fig(out_dir, f"pflat_family_vs_stages_{suffix}", cfg.dpi)

    # 2) P(top-k flattest)
    plt.figure()
    for method, label in [("growth", "growth (theorem toy)"), ("full", "full-space baseline")]:
        y, hw = curve(method, "p_topk")
        plt.errorbar(xs, y, yerr=hw, marker="o", capsize=4, label=label)
    plt.xscale("log")
    plt.xticks(xs, [str(int(s)) for s in xs])
    plt.ylim(0.0, 1.02)
    plt.xlabel("number of stages (log scale)")
    plt.ylabel(f"P(selected basin in top-{cfg.topk} flattest)")
    plt.title("Theorem-toy: top-k flattest selection probability")
    plt.legend()
    save_current_fig(out_dir, f"ptopk_flat_vs_stages_{suffix}", cfg.dpi)

    # 3) E[logdet]
    plt.figure()
    for method, label in [("growth", "growth (theorem toy)"), ("full", "full-space baseline")]:
        y, hw = curve(method, "elogdet")
        plt.errorbar(xs, y, yerr=hw, marker="o", capsize=4, label=label)
    plt.xscale("log")
    plt.xticks(xs, [str(int(s)) for s in xs])
    plt.xlabel("number of stages (log scale)")
    plt.ylabel("E[logdet(H_selected)]  (lower = flatter)")
    plt.title("Theorem-toy: selected-basin logdet")
    plt.legend()
    save_current_fig(out_dir, f"elogdet_selected_vs_stages_{suffix}", cfg.dpi)

    # 4) E[lambda_max]
    plt.figure()
    for method, label in [("growth", "growth (theorem toy)"), ("full", "full-space baseline")]:
        y, hw = curve(method, "elmax")
        plt.errorbar(xs, y, yerr=hw, marker="o", capsize=4, label=label)
    plt.xscale("log")
    plt.xticks(xs, [str(int(s)) for s in xs])
    plt.xlabel("number of stages (log scale)")
    plt.ylabel("E[lambda_max(H_selected)]  (lower = flatter)")
    plt.title("Theorem-toy: selected-basin lambda_max")
    plt.legend()
    save_current_fig(out_dir, f"elmax_selected_vs_stages_{suffix}", cfg.dpi)

    # 5) Family frequencies (stacked per S) — growth
    plt.figure()
    width = 0.8 / len(xs)
    xbase = np.arange(n_fam)
    for j, S in enumerate(xs):
        famfreqs = np.mean(np.stack(metrics["growth"][int(S)]["famfreq"], axis=0), axis=0)
        plt.bar(xbase + j * width, famfreqs, width, label=f"S={int(S)}")
    plt.xticks(xbase + 0.5 * width * (len(xs) - 1), fam_names)
    plt.ylim(0.0, 1.0)
    plt.ylabel("expected selection frequency")
    plt.title("Theorem-toy: family selection frequencies (growth)")
    plt.legend(ncol=2)
    save_current_fig(out_dir, f"family_freqs_growth_{suffix}", cfg.dpi)

    # 6) Family frequencies — full
    plt.figure()
    for j, S in enumerate(xs):
        famfreqs = np.mean(np.stack(metrics["full"][int(S)]["famfreq"], axis=0), axis=0)
        plt.bar(xbase + j * width, famfreqs, width, label=f"S={int(S)}")
    plt.xticks(xbase + 0.5 * width * (len(xs) - 1), fam_names)
    plt.ylim(0.0, 1.0)
    plt.ylabel("expected selection frequency")
    plt.title("Theorem-toy: family selection frequencies (full baseline)")
    plt.legend(ncol=2)
    save_current_fig(out_dir, f"family_freqs_full_{suffix}", cfg.dpi)

    print(f"[saved] {out_dir.resolve()}\n")



def _load_yaml_defaults(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else dict(data)


def _stage_counts_default(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(v)) for v in value)
    if value is None:
        return "1,5,10,20,50,100"
    return str(value)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = _load_yaml_defaults(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Theorem-compatible growth toy (CPU-only)",
        parents=[pre_parser],
    )
    parser.add_argument("--scenario", choices=["multiplicity", "tradeoff"], default=defaults.get("scenario", "multiplicity"))
    parser.add_argument("--preset", choices=["quick", "paper"], default=defaults.get("preset", "quick"))
    parser.add_argument("--d", type=int, default=int(defaults.get("d", 100)))
    parser.add_argument("--rank", type=int, default=int(defaults.get("rank", 5)))
    parser.add_argument("--eps", type=float, default=float(defaults.get("eps", 1.0)))
    parser.add_argument("--sigma", type=float, default=defaults.get("sigma", None), help="init offset std; default ~ 0.9*sqrt(2*eps/d)")
    parser.add_argument("--K0", type=int, default=int(defaults.get("K0", 1)))
    parser.add_argument("--stage-counts", type=str, default=_stage_counts_default(defaults.get("stage_counts", None)))
    parser.add_argument("--n-trials", type=int, default=defaults.get("n_trials", None))
    parser.add_argument("--topk", type=int, default=int(defaults.get("topk", 5)))
    parser.add_argument("--seed-toy", type=int, default=int(defaults.get("seed_toy", 0)))
    parser.add_argument("--seed-trials", type=int, default=int(defaults.get("seed_trials", 1)))
    parser.add_argument("--out-dir", type=str, default=str(defaults.get("out_dir", "toy_outputs")))
    parser.add_argument("--dpi", type=int, default=defaults.get("dpi", None))
    args = parser.parse_args()

    preset = preset_defaults(args.preset)
    stage_counts = parse_stage_counts(args.stage_counts)

    n_trials = int(args.n_trials if args.n_trials is not None else preset["n_trials"])
    dpi = int(args.dpi if args.dpi is not None else preset["dpi"])

    scenarios = default_scenarios()
    spec = scenarios[args.scenario]

    cfg = RunConfig(
        d=int(args.d),
        rank=int(args.rank),
        eps=float(args.eps),
        sigma=(None if args.sigma is None else float(args.sigma)),
        K0=int(args.K0),
        stage_counts=stage_counts,
        n_trials=n_trials,
        topk=int(args.topk),
        seed_toy=int(args.seed_toy),
        seed_trials=int(args.seed_trials),
        out_dir=str(args.out_dir),
        dpi=dpi,
    )

    metrics = run_theorem_toy(spec, cfg)
    plot_results(spec, cfg, metrics)

if __name__ == "__main__":
    main()