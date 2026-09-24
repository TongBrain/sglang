# 单 GPU LoRA 训练—推理闭环 MVP

本模块是独立训练执行层，不放入 SGLang scheduler，不修改 MaaS 平台。内部 CLI 可作为后续平台调用边界。没有实现自动切流或集群调度。完整 checkpoint 协作暂停/恢复已实现，参见 [checkpoint 说明](training-checkpoint.md)。

## 后端与支持范围

- 当前后端：现有 PyTorch + Transformers + PEFT；目标服务器未找到 MaaS 配置的 LLaMA-Factory 环境，因此未调用该后端，也没有安装/升级任何共享环境依赖。
- 仅本地模型、本地 JSONL，单 GPU BF16、batch size=1、attention q_proj/v_proj、uniform LoRA rank/alpha、bias=none。验证模型为 Qwen1.5-MoE-A2.7B。
- Base model 冻结，优化器只能包含 LoRA A/B。prompt labels=-100，response/EOS 参与 causal-LM loss。数据使用明确的原始 prompt/response 字符串，不自动套模型聊天模板。
- 可新建 LoRA，或通过 initial_adapter 从兼容 attention LoRA 继续优化；后者不是恢复优化器状态，训练使用新的 AdamW。
- 已有 PEFT 0.19.1 + Transformers 5.8.1 的 PeftModel.from_pretrained 在本次 MoE 加载中触发 WeightConverter(distributed_operation) 兼容错误。因此本实现先建立标准 q/v LoRA，再严格核对名称、形状和张量集合，直接复制已校验的 A/B 参数；没有修改 PEFT/Transformers，也不把这个窄范围装载器用于专家层或任意适配器。
- 保存不可变 v1 起点和 v2 结果，不 merge base model。导出仍是标准 adapter_config.json + adapter_model.safetensors。
- 不支持 MoE 专家目标、全参数训练、DoRA、QLoRA、多 GPU、梯度累积或真实业务效果评估。不能把本次 smoke 当完整 SFT 产品。

## 任务状态与产物

任务状态：queued → running → trained → publishing → inference_ready。

失败分别为 training_failed 或 publish_failed；publish_failed 可在完成状态核对/清理后重试。训练失败不覆盖产物，需新建 job_id。inference_ready 表示本次检查时可服务，非永久在线承诺，也不是业务已发布。

每个 job_id 只能创建一次，在 job root 下建立目录。job.json 持久记录 spec、数据 SHA-256、初始适配器指纹（若有）、训练进程 GPU 观测、结果和发布状态；通过原子替换及 fsync 保存状态。训练进度、日志和进程记录分别保存在 progress.json、train.log、process.json。

worker 退出且适配器完整校验通过、LoRA 参数确实变化，才标 trained。校验包含基础模型路径、rank/alpha/targets、每层 A/B 形状、有限值、完整张量集合和资产 SHA-256。

本 MVP 没有自动处理控制器崩溃留下的 running/publishing 状态；不能仅根据旧 PID 文件杀进程。需要核对进程身份、资产和后端状态后处理。

## GPU 使用边界

- 用户提供一个物理 GPU 索引。启动前检查没有已有 compute job，且空闲显存满足明确预算。
- 同一个 job root 下用 flock 协作 GPU lease，同时对同一 job 加独占锁。它不是跨 job root 的全局 GPU 预约，也不是 Kubernetes 的卡权，其他用户进程不会遵守此锁。
- 本次 live 验证在同一 GPU 上先训练、释放，再启动推理，不是同卡并行。
- 超时/中断仅终止自己创建的新进程组，不用 GPU PID 全量清理，不执行全局 pgrep/kill。
- 训练超时是失败/终止，不是无损抢占。v2 只是适配器，不含 optimizer、scheduler、RNG 和采样进度的完整恢复状态。

## 内部 CLI

从独立开发目录导入源码；以下命令须在 Linux 服务器上执行，PYTHONPATH 指向实际独立分支的 python 目录，使用已核验训练环境。

```bash
PYTHONPATH=/path/to/sglang/python \
python \
  -m sglang.srt.lora.training.cli --job-root /path/to/shared-job-root \
  train --spec /path/to/spec.json
```

spec 示例（路径和 GPU 需真实确认，不能照抄即占用资源）：

```json
{
  "job_id": "userA-train-001",
  "adapter_name": "userA@v2",
  "model_path": "/path/to/local/base-model",
  "dataset_path": "/path/to/local/data.jsonl",
  "gpu": 4,
  "initial_adapter": null,
  "rank": 8,
  "alpha": 16,
  "target_modules": ["q_proj", "v_proj"],
  "max_steps": 3,
  "max_length": 96,
  "learning_rate": 0.0002,
  "seed": 42,
  "timeout_seconds": 1200,
  "minimum_free_mib": 50000
}
```

JSONL 每行格式：

```json
{"prompt":"Question: What does LoRA train?\nAnswer:","response":" Low-rank adapter matrices."}
```

训练完成后，发布到**已获授权且基础模型一致**的服务。服务必须能访问同一资产路径（当前是本机共享文件系统，不提供上传），并预先配置支持相应 targets/rank 的 LoRA pool。

```bash
PYTHONPATH=/path/to/sglang/python \
python \
  -m sglang.srt.lora.training.cli --job-root /path/to/shared-job-root \
  publish --job-id userA-train-001 --endpoint http://127.0.0.1:13818 \
  --input-ids '[123,456]' --canary-text 'Explain LoRA briefly.'
```

input IDs 必须来自目标模型 tokenizer。管理凭证通过 LORA_ADMIN_KEY 环境变量提供，不写入 job spec/日志/命令参数；本次 loopback 测试没有配置业务凭证，生产鉴权未验证。

status 子命令只读取持久任务记录，不是实时 GPU 或在线驻留查询：

```bash
PYTHONPATH=/path/to/sglang/python \
python \
  -m sglang.srt.lora.training.cli --job-root /path/to/shared-job-root \
  status --job-id userA-train-001
```

## 发布语义

重新核对不可变产物 SHA-256 → 查询基础模型 → 拒绝已存在同名 → load → 核对 runtime ID → prewarm → loaded/resident/ID 一致 → 正常 canary → 再查身份 → inference_ready。

失败时仅在拿到本次 runtime ID 后尝试清理新名称；先检查身份，若已变为别人的 ID 则拒绝卸载。若 load 的网络结果不确定、未拿到 ID 或清理失败，reconcile_required=true，必须对账，不能盲目重试或宣称原子事务。此协议假定生命周期管理只有一个权威写入者；身份检查与卸载不是服务端 compare-and-swap，多写入者竞争仍需另设计。

canary 仅检查推理完成，不评估回答质量、不证明训练提升，不修改用户路由。旧版磁盘名称自动重载行为仍保留，平台切流后需停止旧名称的新请求。

## 可重复验证

test/manual/lora/test_train_serve_live.py 为显式 opt-in，不由普通 CPU 测试自动启动 GPU。GPU 和模型路径通过环境变量配置，port 默认为 13818；配置方式见仓库根目录 LORA_LIFECYCLE.md，执行前协调卡权。默认三步从新建 attention LoRA 训练；--initial-adapter 提供兼容已有版本，--steps 调整 smoke 步数。

验证：三条合成样例 → 真实训练 → GPU 释放 → 临时 SGLang 服务 → v1 长请求 → v2 加载/预热/canary → 等旧请求完成再卸载 → 页回收 → 停止自己的服务 → 核对原项目六文件指纹与 GPU 空闲。

后续已加入完整 checkpoint、Governor 和 step permit；当前能力与边界见仓库根目录 LORA_LIFECYCLE.md。MaaS API 接入和生产化仍待完成。
