from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.scale = float(alpha) / rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.lora_down.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_up.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_up(self.lora_down(inputs)) * self.scale


def inject_vision_lora(clip_model, layer_indices: Iterable[int], rank: int, alpha: float) -> None:
    layers = clip_model.vision_model.encoder.layers
    for index in layer_indices:
        if index < 0 or index >= len(layers):
            raise IndexError(f"vision layer index {index} is out of range")
        attention = layers[index].self_attn
        for name in ("q_proj", "v_proj"):
            module = getattr(attention, name)
            if not isinstance(module, LoRALinear):
                setattr(attention, name, LoRALinear(module, rank, alpha))


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [parameter for name, parameter in module.named_parameters() if ".lora_" in name]


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if ".lora_" in name
    }


def load_lora_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    result = module.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unexpected LoRA checkpoint keys: {result.unexpected_keys}")
    missing = [name for name in state if name not in module.state_dict()]
    if missing:
        raise ValueError(f"missing LoRA checkpoint keys: {missing}")
