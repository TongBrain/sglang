# LoRA lifecycle and inference-priority training

This experimental branch extends the Paged MoE LoRA implementation at
`443fff832f3e7176bf7f41f247d3ff691df43e07` with runtime lifecycle APIs and a
single-GPU LoRA training execution layer. It is a development handoff, not a
production MaaS deployment or an upstream-merged feature.

## Included capabilities

- Query adapter identity, active requests, worker residency and page usage.
- Prewarm adapters through the normal inference scheduling path.
- Drain existing requests before unloading an adapter; preserve other versions
  when loading or rollback fails.
- Train or continue compatible attention LoRA adapters, validate artifacts and
  publish them to an existing local SGLang instance.
- Save complete single-worker checkpoints and resume optimizer, RNG and data
  position at an optimizer-step boundary.
- Pace training using inference throughput and p99 TTFT/TPOT windows.
- Gate each new training step on an existing `/v1/loads?include=core` busy signal.

The platform still owns user-to-version routing, business quality evaluation,
deployment policy, global GPU allocation and cross-instance coordination.
Training and inference currently load separate base models. Shared base storage,
distributed checkpointing and automatic crash recovery are not implemented.

## Code and documentation

| Component | Location |
| --- | --- |
| Lifecycle aggregation and API semantics | `python/sglang/srt/lora/lifecycle.py` and [API contract](test/manual/lora/lifecycle-api.md) |
| Training, job storage and publication | `python/sglang/srt/lora/training/` and [train/serve guide](test/manual/lora/train-serve-loop.md) |
| Checkpoint and resume | `python/sglang/srt/lora/training/checkpoint.py` and [checkpoint guide](test/manual/lora/training-checkpoint.md) |
| Inference-window policy | `python/sglang/srt/lora/training/governor.py` |
| Step permit | `python/sglang/srt/lora/training/permit.py` |
| Comparison gate | `python/sglang/benchmark/colocation_gate.py` |
| Live window experiment | `test/manual/lora/run_colocation_governor.py` |

The internal CLI provides `train`, `status`, `pause`, `resume` and `publish`:

```bash
# Run from this checkout with a compatible SGLang/Transformers/PEFT environment.
export PYTHONPATH="$PWD/python"
python -m sglang.srt.lora.training.cli --help
```

`initial_adapter` starts a new optimizer from existing adapter weights.
`resume` restores a complete checkpoint; these are different operations.
`status` reads persisted job state, not a live GPU reservation or a guarantee that
an earlier `inference_ready` publication is still serving.

## Configure manual GPU experiments

Manual scripts discover the repository from their own location and use the
current Python interpreter. They do not contain a developer's server paths.
Provide absolute paths to existing local assets and select an idle physical GPU:

```bash
export PYTHONPATH="$PWD/python"
export SGLANG_LORA_TEST_MODEL=/path/to/Qwen1.5-MoE-A2.7B
export SGLANG_LORA_TEST_GPU=0

# For colocation experiments: a compatible q_proj/v_proj attention LoRA r8
# adapter and the local dataset path used by the serving benchmark.
export SGLANG_LORA_TEST_ADAPTER=/path/to/attention-lora-r8
export SGLANG_LORA_TEST_DATASET=/path/to/benchmark-dataset.json

# For the standalone lifecycle test: r8_000, r8_001 and r16_000 fixtures.
export SGLANG_LORA_TEST_CATALOG=/path/to/structural-adapter-catalog

# Optional: another checkout whose core files should stay unchanged.
# Defaults to the checkout containing the manual script.
export SGLANG_LORA_TEST_ORIGINAL_REPO=/path/to/reference-checkout
```

No models, adapters, datasets, checkpoints, private handoff documents or raw
profiling files are included in this branch. `/path/to/...` defaults must be
replaced with actual local assets before running live tests.

The Governor experiment measures its own inference baseline and then runs
candidate windows. After confirming GPU and port availability:

```bash
python test/manual/lora/run_colocation_governor.py \
  --gpu "$SGLANG_LORA_TEST_GPU" --port 13820 \
  --window-seconds 30 --candidate-windows 3 \
  --initial-step-interval 2.0 \
  --step-permit --permit-idle-seconds 0.5 --permit-poll-seconds 0.1
```

This starts a real server and trainer and writes a timestamped directory under
`work/`. A "30 second" window means 400 fixed requests at concurrency 16 with
256/128 input/output tokens; actual duration is measured, not time-stopped.
The script checkpoints the trainer and cleans up only its own processes.

For the older short candidate runner, first run `run_colocation_baseline.py`
and point `SGLANG_LORA_TEST_BASELINE_RESULT` at that run's `baseline.jsonl`.
Do not reuse another machine's historical baseline.

## Validation and limits

Historical GPU validation used a single H800 and Qwen1.5-MoE-A2.7B with BF16
attention LoRA (`q_proj/v_proj`, rank 8, alpha 16), batch 1 and synthetic training
samples. The recorded environment used PyTorch 2.13.0+cu130, Transformers 5.8.1
and PEFT 0.19.1. Other models and software versions need their own validation.

- Lifecycle tests covered decode CUDA Graph disabled/enabled, concurrent old/new
  versions, draining and failed-load cleanup.
- Continuous five-step training matched a one-step pause plus four-step resume
  for LoRA factors, AdamW, losses, RNG and sample position in that configuration.
- With fixed 2.0-second pacing and no permit, a longer inference window exceeded
  the p99 TTFT budget by 33.872% relative to its baseline.
- With step permits and 2.0-second pacing, three candidate windows completed
  1,200 requests with no errors; maximum p99 TTFT/TPOT degradation was
  0.276%/0.021%. Training reached step 22 during idle gaps, with 713 busy
  observations and 22 grants. This does not establish training throughput under
  continuously saturated inference.

The experimental budgets are at most 5% output-throughput degradation and 10%
p99 TTFT/TPOT degradation. The permit requires all observed DP ranks to remain
idle for 0.5 seconds, polls every 0.1 seconds, and fails closed on repeated load
query errors. An already-started training step cannot be preempted. Bursty
requests arriving after a permit, multi-GPU execution, production routing and
shared-base operation remain unvalidated.

CPU regression entry points, with GPU visibility disabled:

```bash
CUDA_VISIBLE_DEVICES=999 PYTHONPATH="$PWD/python" python -m pytest -q \
  test/registered/unit/lora/test_lora_lifecycle.py \
  test/registered/unit/lora/test_train_serve_loop.py \
  test/registered/unit/lora/test_training_checkpoint.py \
  test/registered/unit/lora/test_training_governor.py \
  test/registered/unit/lora/test_training_permit.py \
  test/registered/bench_fn/test_colocation_gate.py \
  test/manual/lora/test_colocation_governor_runner.py
```

Manual GPU experiments are not part of that command. The publication preparation
checks CPU regressions and portable configuration only; historical GPU results
do not constitute a fresh GPU validation of the publication snapshot.
