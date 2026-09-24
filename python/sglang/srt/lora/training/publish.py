"""Publish readiness, never business/user routing. Cleanup only our new ID."""

import json
import urllib.error
import urllib.parse
import urllib.request

from .artifacts import validate_artifact
from .jobs import TrainingSpec


class InferenceClient:
    def __init__(self, endpoint, timeout=300, admin_key=None):
        parsed = urllib.parse.urlparse(endpoint)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
            or parsed.username
        ):
            raise ValueError(
                "endpoint must be an HTTP(S) service URL without credentials/query"
            )
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.admin_key = admin_key
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, method, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.admin_key:
            headers["Authorization"] = "Bearer " + self.admin_key
        request = urllib.request.Request(
            self.endpoint + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
            method=method,
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(
                result.get("error_message") or "inference operation failed"
            )
        return result

    def status(self, name=None):
        query = (
            "" if name is None else "?" + urllib.parse.urlencode({"lora_name": name})
        )
        return self.call("GET", "/get_lora_adapter_status" + query)


def publish_job(store, job_id, client, input_ids, canary_text):
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(type(token) is not int or token < 0 for token in input_ids)
        or not canary_text.strip()
    ):
        raise ValueError("publication requires valid prompt IDs and a canary prompt")
    with store.lock(job_id):
        job = store.read(job_id)
        spec = TrainingSpec(**job["spec"])
        if job["state"] not in ("trained", "publish_failed"):
            raise ValueError("only a completed training job can be published")
        store.transition(
            job_id, "publishing", error=None, cleanup=None, reconcile_required=False
        )
        loaded_id = None
        load_attempted = False
        try:
            artifact = validate_artifact(store.directory(job_id) / "v2", spec)
            if artifact["files"] != job["artifact"]["files"]:
                raise ValueError("immutable artifact changed after training")
            metadata = client.call("GET", "/get_model_info")
            from pathlib import Path

            if (
                Path(metadata.get("model_path", "")).resolve()
                != Path(spec.model_path).resolve()
            ):
                raise ValueError(
                    "inference base model differs from training base model"
                )
            status = client.status()
            if spec.adapter_name in status["adapters"]:
                raise ValueError("adapter name already exists; refusing to replace it")
            load_attempted = True
            loaded = client.call(
                "POST",
                "/load_lora_adapter",
                {
                    "lora_name": spec.adapter_name,
                    "lora_path": artifact["path"],
                    "pinned": False,
                },
            )
            loaded_id = loaded.get("lora_id")
            if not loaded_id:
                raise RuntimeError("load reply lacks runtime identity")
            warmed = client.call(
                "POST",
                "/prewarm_lora_adapter",
                {
                    "lora_name": spec.adapter_name,
                    "input_ids": input_ids,
                },
            )
            adapter = warmed["adapters"].get(spec.adapter_name, {})
            if (
                not warmed.get("probe_completed")
                or not adapter.get("loaded")
                or not adapter.get("resident")
                or adapter.get("lora_id") != loaded_id
            ):
                raise RuntimeError(
                    "prewarm did not confirm loaded/resident matching identity"
                )
            canary = client.call(
                "POST",
                "/generate",
                {
                    "text": canary_text,
                    "lora_path": spec.adapter_name,
                    "sampling_params": {
                        "max_new_tokens": 16,
                        "temperature": 0,
                        "ignore_eos": True,
                    },
                },
            )
            reason = canary.get("meta_info", {}).get("finish_reason", {})
            if (
                not reason
                or reason.get("type") == "abort"
                or canary.get("meta_info", {}).get("completion_tokens", 0) < 1
            ):
                raise RuntimeError("normal canary request did not complete")
            final = client.status(spec.adapter_name)["adapters"][spec.adapter_name]
            if not final.get("loaded") or final.get("lora_id") != loaded_id:
                raise RuntimeError("adapter identity changed during canary")
            return store.transition(
                job_id,
                "inference_ready",
                publication={
                    "endpoint": client.endpoint,
                    "adapter_name": spec.adapter_name,
                    "lora_id": loaded_id,
                    "status": final,
                    "canary": canary,
                    "business_routing_changed": False,
                    "quality_evaluation": "not_performed",
                },
            )
        except Exception as error:
            cleanup = {"attempted": False, "success": None}
            if loaded_id:
                cleanup["attempted"] = True
                try:
                    current = client.status()["adapters"].get(spec.adapter_name)
                    if current is not None and current["lora_id"] != loaded_id:
                        raise RuntimeError(
                            "identity changed; refusing to unload another adapter"
                        )
                    client.call(
                        "POST",
                        "/unload_lora_adapter",
                        {
                            "lora_name": spec.adapter_name,
                            "missing_ok": True,
                        },
                    )
                    cleanup["success"] = True
                except Exception as cleanup_error:
                    cleanup.update(success=False, error=str(cleanup_error))
            store.transition(
                job_id,
                "publish_failed",
                error=str(error),
                cleanup=cleanup,
                reconcile_required=load_attempted
                and (not loaded_id or cleanup["success"] is not True),
            )
            raise
