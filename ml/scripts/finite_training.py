"""Finite-state gates for the certified FP32 training path (no sanitization)."""
from __future__ import annotations

import math
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


def require_finite(value, stage: str):
    """Report only stage, never patient values."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite state: {stage}")
    elif isinstance(value, dict):
        for item in value.values():
            require_finite(item, stage)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite(item, stage)
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"Nonfinite state: {stage}")


def gradient_norm(parameters):
    """Float64 accumulation prevents a finite FP32 norm overflowing during clip."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0., dtype=torch.float64)
    require_finite(gradients, "gradient")
    norm = torch.stack([torch.linalg.vector_norm(g.detach().double()) for g in gradients]).norm()
    require_finite(norm, "gradient norm")
    return norm


def optimizer_finite(optimizer):
    optimizer = getattr(optimizer, "optimizer", optimizer)
    for group in optimizer.param_groups:
        require_finite(group["params"], "parameter after optimizer step")
    require_finite(optimizer.state, "optimizer state")


class FiniteMetric:
    """Keep PF finite arithmetic/reduction; reject before its 1e9 fallback.

    Only documented padding is replaced before computing loss. Valid values are
    never replaced, clamped, dropped or repaired.
    """
    def update(self, y_pred, target):
        weight = None
        if isinstance(target, (tuple, list)):
            target, weight = target
        packed = isinstance(target, PackedSequence)
        if packed:
            dense, lengths = pad_packed_sequence(target, batch_first=True)
            valid = torch.arange(dense.size(1), device=dense.device)[None] < lengths.to(dense.device)[:, None]
            require_finite(y_pred[valid], "prediction on valid target")
            require_finite(dense[valid], "valid target")
            y_pred = y_pred.masked_fill(~valid.unsqueeze(-1), 0.)
            if weight is not None:
                require_finite(weight[valid], "valid sample weight")
                weight = weight.masked_fill(~valid, 0.)
        else:
            require_finite(target, "target")
            require_finite(y_pred, "prediction")
            require_finite(weight, "sample weight")
        return super().update(y_pred, (target, weight))

    def _update_losses_and_lengths(self, losses, lengths):
        valid = torch.arange(losses.size(1), device=losses.device)[None] < lengths[:, None]
        require_finite(losses[valid], "per-position loss")
        if torch.any(lengths <= 0):
            raise ValueError("Loss requires positive valid decoder lengths")
        if self.reduction != "none":
            total = self.mask_losses(losses, lengths).sum()
            require_finite(total, "loss reduction sum")
            require_finite(self.losses + total, "loss accumulator")
            count = self.lengths + lengths.sum()
            require_finite(count, "loss denominator")
            if torch.any(count <= 0):
                raise FloatingPointError("Invalid loss denominator")
        return super()._update_losses_and_lengths(losses, lengths)

    def compute(self):
        result = super().compute()
        if self.reduction != "none":
            require_finite(result, "reduced loss")
        return result


class FiniteModel:
    def forward(self, x, *args, **kwargs):
        require_finite(x, "model input")
        if torch.any(x["target_scale"][..., 1] <= 0):
            raise ValueError("Target normalizer scale must be positive")
        out = super().forward(x, *args, **kwargs)
        require_finite(out["prediction"], "transformed prediction")
        return out

    def step(self, x, y, batch_idx, **kwargs):
        self._valid_positions_for_log = int(x["decoder_lengths"].sum())
        try:
            result = super().step(x, y, batch_idx, **kwargs)
            require_finite(result[0]["loss"], "batch loss")
            return result
        finally:
            self._valid_positions_for_log = None

    def log(self, name, value, *args, **kwargs):
        positions = getattr(self, "_valid_positions_for_log", None)
        if positions is not None and name in ("train_loss", "val_loss", "test_loss"):
            kwargs["batch_size"] = positions
        return super().log(name, value, *args, **kwargs)

    def on_after_backward(self):
        gradient_norm(self.parameters())
        return super().on_after_backward()

    def on_before_optimizer_step(self, optimizer):
        gradient_norm(self.parameters())
        return super().on_before_optimizer_step(optimizer)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        norm = gradient_norm(parameters)
        if gradient_clip_val and gradient_clip_val > 0:
            if gradient_clip_algorithm not in (None, "norm"):
                raise ValueError("Baseline supports norm clipping only")
            factor = min(1., float(gradient_clip_val) / (float(norm) + 1e-6))
            for p in parameters:
                if p.grad is not None:
                    p.grad.mul_(factor)
        gradient_norm(parameters)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        result = super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        optimizer_finite(optimizer)
        return result

    def on_save_checkpoint(self, checkpoint):
        require_finite(self.state_dict(), "checkpoint model")
        if self._trainer is not None:
            for optimizer in self.trainer.optimizers:
                optimizer_finite(optimizer)
        return super().on_save_checkpoint(checkpoint)


class CheckedAdamW(torch.optim.AdamW):
    """Run Lightning's closure (backward + clip), then gate the actual AdamW call."""
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        parameters = [p for group in self.param_groups for p in group["params"]]
        gradient_norm(parameters)
        result = super().step()
        optimizer_finite(self)
        return loss if closure is not None else result
