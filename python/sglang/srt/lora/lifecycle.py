"""Control-plane aggregation for immutable LoRA adapter generations."""

from typing import List

from sglang.srt.managers.io_struct import LoRAAdapterStatusOutput, LoRAUpdateOutput


def merge_lora_update_results(results: List[LoRAUpdateOutput]) -> LoRAUpdateOutput:
    """Every worker must succeed; preserve the first reply for legacy LRU callers."""
    if not results:
        return LoRAUpdateOutput(success=False, error_message="No worker replies")
    failed = [r for r in results if not r.success]
    if not failed:
        return results[0]
    messages = list(dict.fromkeys(r.error_message for r in failed if r.error_message))
    return LoRAUpdateOutput(
        success=False,
        error_message=" | ".join(messages),
        loaded_adapters=failed[0].loaded_adapters,
    )


def merge_adapter_status(results, snapshot, lora_name=None):
    workers = [worker for result in results for worker in result.workers]
    errors = [
        r.error_message or "Worker status failed" for r in results if not r.success
    ]
    if not workers:
        errors.append("No worker status replies")
    for worker in workers:
        for name in worker["adapters"]:
            if name not in snapshot:
                errors.append(f"Worker contains unregistered adapter {name}")
    adapters = {}
    for name, (ref, count) in snapshot.items():
        if lora_name is not None and name != lora_name:
            continue
        states = [w["adapters"].get(name) for w in workers]
        loaded = bool(states) and all(
            state is not None and state["lora_id"] == ref.lora_id for state in states
        )
        adapters[name] = {
            "lora_id": ref.lora_id,
            "lora_path": ref.lora_path,
            "pinned": bool(ref.pinned),
            "active_requests": count,
            "loaded": loaded,
            "resident": loaded and all(s["resident"] for s in states),
        }
        if not loaded:
            errors.append(f"Adapter {name} is inconsistent across workers")
    if lora_name is not None and lora_name not in adapters:
        errors.append(f"LoRA adapter {lora_name} is not registered")
    return LoRAAdapterStatusOutput(
        success=not errors,
        error_message=" | ".join(dict.fromkeys(errors)) or None,
        adapters=adapters,
        workers=workers,
    )
