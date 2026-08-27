from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

@dataclass
class GrowthConfig:
    enabled: bool = True
    stages: int = 5
    patience: int = 2
    criterion: str = "val_acc"
    random_seed: int = 0

    mode: str = "progressive"
    start_stage: int = 0


class GrowthMaskController:
    def __init__(self, model: nn.Module, config: GrowthConfig):
        self.model = model
        self.config = config
        self.enabled = bool(config.enabled and config.stages > 1)
        self.total_stages = max(1, int(config.stages))
        self.mode = str(config.mode).lower()
        if self.mode not in {"progressive", "fixed_subspace", "one_shot"}:
            raise ValueError(
                f"Unknown growth mode {self.mode!r}. "
                "Expected one of: progressive, fixed_subspace, one_shot."
            )

        self.current_stage = int(max(0, min(self.total_stages - 1, int(config.start_stage))))

        self._gen = torch.Generator().manual_seed(int(config.random_seed))
        self.param_groups: Dict[str, torch.Tensor] = {}
        self.buffer_groups: Dict[str, torch.Tensor] = {}
        self.init_params: Dict[str, torch.Tensor] = {}
        self.init_buffers: Dict[str, torch.Tensor] = {}
        self.param_order: List[str] = []
        self.buffer_order: List[str] = []
        self._params_by_name = dict(model.named_parameters())
        self._buffers_by_name = dict(model.named_buffers())
        self._build_groups()
        self._snapshot_init_state()

    def _rand_groups(self, n: int) -> torch.Tensor:
        if n <= 0:
            return torch.zeros(0, dtype=torch.long)
        perm = torch.randperm(n, generator=self._gen)
        groups = torch.empty(n, dtype=torch.long)
        chunk = math.ceil(n / self.total_stages)
        for s in range(self.total_stages):
            lo = s * chunk
            hi = min(n, (s + 1) * chunk)
            if lo >= hi:
                break
            groups[perm[lo:hi]] = s
        return groups

    @staticmethod
    def _expand_groups(base: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        if len(target_shape) == 0:
            return torch.zeros(target_shape, dtype=torch.long)
        if len(target_shape) == 1:
            return base.clone()
        view_shape = [target_shape[0]] + [1] * (len(target_shape) - 1)
        return base.view(*view_shape).expand(*target_shape).clone()

    def _make_group_tensor(self, tensor: torch.Tensor, groups_1d: Optional[torch.Tensor] = None, force_active=False) -> torch.Tensor:
        if force_active:
            return torch.zeros_like(tensor, dtype=torch.long)
        if tensor.ndim == 0:
            return torch.zeros_like(tensor, dtype=torch.long)
        if groups_1d is None:
            groups_1d = self._rand_groups(int(tensor.shape[0]))
        return self._expand_groups(groups_1d.to(dtype=torch.long), tensor.shape)

    def _build_groups(self) -> None:
        recent_groups_by_size: Dict[int, torch.Tensor] = {}
        for module_name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue

            classifier_like = module_name.endswith("fc") or module_name.endswith("head") or module_name == "model.fc"

            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight_name = f"{module_name}.weight" if module_name else "weight"
                if weight_name in self._params_by_name:
                    p = self._params_by_name[weight_name]
                    groups_1d = torch.zeros(p.shape[0], dtype=torch.long) if classifier_like else self._rand_groups(int(p.shape[0]))
                    self.param_groups[weight_name] = self._make_group_tensor(p.data, groups_1d=groups_1d, force_active=classifier_like)
                    if p.shape[0] > 0:
                        recent_groups_by_size[int(p.shape[0])] = groups_1d
                bias_name = f"{module_name}.bias" if module_name else "bias"
                if getattr(module, "bias", None) is not None and bias_name in self._params_by_name:
                    b = self._params_by_name[bias_name]
                    groups_1d = torch.zeros(b.shape[0], dtype=torch.long) if classifier_like else recent_groups_by_size.get(int(b.shape[0]), self._rand_groups(int(b.shape[0])))
                    self.param_groups[bias_name] = self._make_group_tensor(b.data, groups_1d=groups_1d, force_active=classifier_like)
                continue

            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                n = int(module.num_features)
                groups_1d = recent_groups_by_size.get(n, self._rand_groups(n))
                for suffix in ["weight", "bias"]:
                    name = f"{module_name}.{suffix}" if module_name else suffix
                    if name in self._params_by_name:
                        self.param_groups[name] = self._make_group_tensor(self._params_by_name[name].data, groups_1d=groups_1d)
                for suffix in ["running_mean", "running_var"]:
                    name = f"{module_name}.{suffix}" if module_name else suffix
                    if name in self._buffers_by_name:
                        self.buffer_groups[name] = self._make_group_tensor(self._buffers_by_name[name].data, groups_1d=groups_1d)
                continue

            if isinstance(module, nn.LayerNorm):
                n = int(module.weight.numel()) if getattr(module, "weight", None) is not None else int(module.bias.numel())
                groups_1d = recent_groups_by_size.get(n, self._rand_groups(n))
                for suffix in ["weight", "bias"]:
                    name = f"{module_name}.{suffix}" if module_name else suffix
                    if name in self._params_by_name:
                        self.param_groups[name] = self._make_group_tensor(self._params_by_name[name].data, groups_1d=groups_1d)
                continue

        for name, p in self.model.named_parameters():
            if name not in self.param_groups:
                self.param_groups[name] = self._make_group_tensor(p.data)
            self.param_order.append(name)
        for name, b in self.model.named_buffers():
            if name not in self.buffer_groups:
                self.buffer_groups[name] = self._make_group_tensor(b.data)
            self.buffer_order.append(name)

    def _snapshot_init_state(self) -> None:
        self.init_params = {name: p.detach().cpu().clone() for name, p in self.model.named_parameters()}
        self.init_buffers = {name: b.detach().cpu().clone() for name, b in self.model.named_buffers()}

    def stage_fraction(self, stage_idx: Optional[int] = None) -> float:
        if not self.enabled:
            return 1.0
        if stage_idx is None:
            stage_idx = self.current_stage
        return float(stage_idx + 1) / float(self.total_stages)

    def _active_from_groups(self, groups: torch.Tensor, stage_idx: int) -> torch.Tensor:
        if not self.enabled:
            return torch.ones_like(groups, dtype=torch.bool)
        return groups <= int(stage_idx)

    def stage_index(self) -> int:
        return int(self.current_stage)

    def is_last_stage(self) -> bool:
        if not self.enabled:
            return True
        if self.mode == "fixed_subspace":
            return True
        return self.current_stage >= self.total_stages - 1

    def set_stage(self, stage_idx: int) -> None:
        self.current_stage = int(max(0, min(self.total_stages - 1, stage_idx)))

    def advance_stage(self) -> bool:
        if self.is_last_stage():
            return False

        if self.mode == "one_shot":
            self.current_stage = self.total_stages - 1
        else:
            self.current_stage += 1

        return True

    def param_mask_dict_for_stage(self, stage_idx: int, device: torch.device) -> Dict[str, torch.Tensor]:
        out = {}
        for name in self.param_order:
            out[name] = self._active_from_groups(self.param_groups[name], stage_idx).to(device=device)
        return out

    def buffer_mask_dict_for_stage(self, stage_idx: int, device: torch.device) -> Dict[str, torch.Tensor]:
        out = {}
        for name in self.buffer_order:
            out[name] = self._active_from_groups(self.buffer_groups[name], stage_idx).to(device=device)
        return out

    def param_mask_dict(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return self.param_mask_dict_for_stage(self.current_stage, device)

    def buffer_mask_dict(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return self.buffer_mask_dict_for_stage(self.current_stage, device)

    def flat_mask_for_stage(self, stage_idx: int, device: torch.device) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        pdict = self.param_mask_dict_for_stage(stage_idx, device)
        for name in self.param_order:
            pieces.append(pdict[name].reshape(-1).to(dtype=torch.float32))
        return torch.cat(pieces, dim=0)

    def flat_mask(self, device: torch.device) -> torch.Tensor:
        return self.flat_mask_for_stage(self.current_stage, device)

    def active_mask(self, device: torch.device) -> torch.Tensor:
        return self.flat_mask(device)

    def frozen_mask(self, device: torch.device) -> torch.Tensor:
        return 1.0 - self.flat_mask(device)

    def newly_released_mask_dict(self, prev_stage: int, next_stage: int, device: torch.device) -> Dict[str, torch.Tensor]:
        prev_dict = self.param_mask_dict_for_stage(prev_stage, device)
        next_dict = self.param_mask_dict_for_stage(next_stage, device)
        out: Dict[str, torch.Tensor] = {}
        for name in self.param_order:
            out[name] = torch.clamp(next_dict[name].to(torch.float32) - prev_dict[name].to(torch.float32), min=0.0, max=1.0)
        return out

    def newly_released_mask(self, prev_stage: int, next_stage: int, device: torch.device) -> torch.Tensor:
        prev_mask = self.flat_mask_for_stage(prev_stage, device)
        next_mask = self.flat_mask_for_stage(next_stage, device)
        return torch.clamp(next_mask - prev_mask, min=0.0, max=1.0)

    def apply_masks_(self, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        device = next(self.model.parameters()).device
        param_masks = self.param_mask_dict(device)
        buffer_masks = self.buffer_mask_dict(device)

        with torch.no_grad():
            # Parameters
            for name, p in self.model.named_parameters():
                mask_bool = param_masks[name].to(device=device, dtype=torch.bool)
                mask = mask_bool.to(dtype=p.dtype)
                init = self.init_params[name].to(device=device, dtype=p.dtype)

                p.data.mul_(mask).add_(init * (1.0 - mask))

                if p.grad is not None:
                    p.grad.mul_(mask)

            # Buffers
            for name, b in self.model.named_buffers():
                if name not in buffer_masks:
                    continue

                mask_bool = buffer_masks[name].to(device=device, dtype=torch.bool)
                init = self.init_buffers[name].to(device=device, dtype=b.dtype)

                if torch.is_floating_point(b) or torch.is_complex(b):
                    mask = mask_bool.to(dtype=b.dtype)
                    b.data.mul_(mask).add_(init * (1.0 - mask))
                else:
                    # integer / bool buffers, e.g. BatchNorm.num_batches_tracked
                    b.data.copy_(torch.where(mask_bool, b.data, init))

        if optimizer is None:
            return

        for group in optimizer.param_groups:
            for p in group["params"]:
                name = None
                for k, v in self._params_by_name.items():
                    if v is p:
                        name = k
                        break
                if name is None:
                    continue

                mask = param_masks[name].to(device=p.device, dtype=p.dtype)
                state = optimizer.state.get(p, {})

                for key, value in state.items():
                    if torch.is_tensor(value):
                        if value.shape == p.shape:
                            value.mul_(mask)
                        elif value.ndim == 0:
                            continue
    def stage_state(self) -> Dict[str, int | bool | str]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "current_stage": int(self.current_stage),
            "total_stages": int(self.total_stages),
            "is_last_stage": self.is_last_stage(),
        }

    def make_snapshot(self) -> Dict:
        return {
            "model_state": copy.deepcopy(self.model.state_dict()),
            "current_stage": int(self.current_stage),
        }

    def load_snapshot(self, snapshot: Dict) -> None:
        self.model.load_state_dict(snapshot["model_state"])
        self.set_stage(int(snapshot.get("current_stage", self.current_stage)))
