"""Fail-closed PEFT export checks for the attention-only MVP."""

import json
import re
from pathlib import Path

from .jobs import file_sha256


def validate_artifact(directory, spec):
    import torch
    from safetensors import safe_open

    directory = Path(directory).resolve(strict=True)
    config_path = directory / "adapter_config.json"
    weights_path = directory / "adapter_model.safetensors"
    config = json.loads(config_path.read_text())
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("expected causal-LM PEFT LoRA")
    if any(
        config.get(key)
        for key in (
            "use_dora",
            "use_rslora",
            "use_qalora",
            "lora_bias",
            "fan_in_fan_out",
            "modules_to_save",
            "target_parameters",
            "alora_invocation_tokens",
            "layer_replication",
            "trainable_token_indices",
        )
    ):
        raise ValueError("unsupported export extensions")
    if (
        config.get("bias", "none") != "none"
        or config.get("rank_pattern")
        or config.get("alpha_pattern")
    ):
        raise ValueError("MVP requires bias=none and uniform rank/alpha")
    if config.get("r") != spec.rank or config.get("lora_alpha") != spec.alpha:
        raise ValueError("export rank/alpha does not match job")
    if set(config.get("target_modules", [])) != set(spec.target_modules):
        raise ValueError("export targets do not match job")
    if (
        Path(config.get("base_model_name_or_path", "")).resolve()
        != Path(spec.model_path).resolve()
    ):
        raise ValueError("export base model does not match job")
    model_config = json.loads((Path(spec.model_path) / "config.json").read_text())
    hidden = model_config["hidden_size"]
    head_dim = model_config.get(
        "head_dim", hidden // model_config["num_attention_heads"]
    )
    shapes = {}
    with safe_open(weights_path, framework="pt", device="cpu") as stream:
        for name in stream.keys():
            match = re.fullmatch(
                r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_(A|B)\.weight",
                name,
            )
            if not match or int(match[1]) >= model_config["num_hidden_layers"]:
                raise ValueError(f"unexpected LoRA tensor {name}")
            _, module, factor = match.groups()
            output_size = (
                head_dim
                * model_config[
                    (
                        "num_attention_heads"
                        if module == "q_proj"
                        else "num_key_value_heads"
                    )
                ]
            )
            expected = (
                (spec.rank, hidden) if factor == "A" else (output_size, spec.rank)
            )
            tensor = stream.get_tensor(name)
            if (
                tuple(tensor.shape) != expected
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all().item()
            ):
                raise ValueError(f"invalid shape or nonfinite tensor {name}")
            shapes[name] = list(tensor.shape)
    expected_count = (
        model_config["num_hidden_layers"] * len(set(spec.target_modules)) * 2
    )
    if len(shapes) != expected_count:
        raise ValueError("export must include both LoRA factors for every target/layer")
    return {
        "path": str(directory),
        "files": {
            "adapter_config.json": file_sha256(config_path),
            "adapter_model.safetensors": file_sha256(weights_path),
        },
        "tensor_count": len(shapes),
        "rank": spec.rank,
        "targets": sorted(set(spec.target_modules)),
        "base_model_path": str(Path(spec.model_path).resolve()),
    }
