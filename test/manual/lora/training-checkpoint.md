# 单卡训练安全暂停与 checkpoint 恢复

本模块是内部、可信本地目录上的单 worker 协作暂停，不是集群调度器、强制抢占或任意训练后端的 checkpoint 转换工具。只支持此前的 attention q_proj/v_proj LoRA、batch=1、固定循环数据、无 scheduler 的 AdamW。

## 调用链与状态

`train → running → pause 请求文件 → 当前 optimizer.step 完成 → checkpoint 提交 → worker 正常退出 → GPU 无 compute process → paused → resume → running → trained`

pause CLI 成功只表示请求已写入；必须等待 job.state=paused 且 gpu_released=true 才能计划资源交接。pause 请求绑定运行 UUID，每次恢复生成新的 UUID，遗留请求不影响后续运行。末步已经完成时训练优先完成，不保证迟到请求能暂停。

暂停文件不是信号杀进程。训练长 step、加载模型和 checkpoint I/O 都可能延迟暂停，没有毫秒级抢占 SLA。超时仍然清理自身进程组并标记失败，不声称超时/kill/crash 可以自动恢复。

GPU released 是检查时观察值，不是全局预留。lease 只约束同 job root 的协作执行器；其他用户程序和不同 root 不遵守。仍需一个全局 GPU 管理权威，不能据此直接让 MaaS 与本模块分别分卡。

## 内部 CLI

先使用已有 train 命令启动任务。另一个终端可以调用：

```bash
python -m sglang.srt.lora.training.cli --job-root /ABS/jobs pause --job-id JOB
python -m sglang.srt.lora.training.cli --job-root /ABS/jobs status --job-id JOB
# 确认 paused 后执行。resume 同步等待当前训练 segment 退出。
python -m sglang.srt.lora.training.cli --job-root /ABS/jobs resume --job-id JOB
```

resume 固定原始 spec 和相同物理 GPU，不支持修改总步数、迁卡或改变优化器超参。GPU 被占用时拒绝启动并保留 paused 状态，释放后可重试。数据内容、模型 config、初始 adapter 文件摘要与库版本不匹配时拒绝恢复。基座权重本身仍要求不可变，没有全量模型内容寻址。

JobStore 的 completed_steps 是已确认的暂停/完成进度，不是运行中的实时进度；运行时参考 progress.json。worker 的 elapsed_seconds、peak_memory_mib 描述当前进程/segment，不是暂停前后累计耗时或性能对照。

## checkpoint 内容与提交

包含 LoRA trainable factors、参数顺序、完整 AdamW state_dict（step/exp_avg/exp_avg_sq/param_groups）、Python/NumPy/Torch CPU/单卡 CUDA RNG、完成 step、loss 历史、下一个样本位置、训练身份和库版本。scheduler=None，因为当前 worker 没有 scheduler；未来加入 scheduler 后必须升级格式并保存/恢复其状态。

仅在完整 step 后保存，因此不包含半步梯度、梯度累积或分布式 rank 状态。恢复先建立模型及优化器，验证张量和优化器结构，再装载 factors/moments，最后恢复 RNG，避免构建模型消耗随机数而改变后续训练。

`checkpoints/step-N-UUID.pending/state.pt → fsync → commit.json（SHA-256） → 重命名不可变目录 → 原子 latest.json 指针 + 目录 fsync`。

只选择已完整提交且摘要一致的目录，忽略未引用 pending 目录；损坏或身份不一致不静默回退旧 checkpoint。state.pt 用 weights_only=True 读取，仍只接受可信本地存储；checksum 是完整性检查，不是防恶意替换的签名。最后一步也保留 checkpoint 作为对照证据，但 trained 状态不允许以新 max_steps 继续执行。

推理 v2 仍是独立标准 PEFT adapter 导出，不携带 optimizer/RNG；暂停时不创建 v2，不允许发布。恢复只完成剩余步数，v1 起点快照保持不变。

## 验证与边界

- CPU 单测：含 dropout 和 Python/NumPy/Torch 随机输入，对比连续 5 步与 2+3 步恢复，要求 weights/optimizer/loss/RNG 精确相等；另覆盖损坏提交、身份变化、优化器配置/结构、暂停请求隔离、拒绝提前交卡等。
- opt-in GPU 脚本 test_training_checkpoint_live.py：通过环境变量选择空闲 GPU 和模型；历史验证使用 Qwen1.5-MoE-A2.7B、三条合成样本，连续 5 步与 1+4 步恢复，要求精确相等。间隙运行短 CUDA 小任务证明可被其他计算任务使用；不是启动推理服务或同卡同时训推实测。
- 独立分支运行，不修改受保护原项目和已有六个推理核心源码；不安装共享依赖，不操作其他进程。
- GPU 精确等价只在实测模型/精度/软件/输入下报告；不同 CUDA 内核、硬件、分布式训练和随机采样器需要重新验证。
- 控制器崩溃对账、自动重启、共享文件系统多节点锁、训练质量、MaaS HTTP/租户鉴权均未实现。后续实验 Governor 可以根据指标触发协作暂停，但不是生产抢占服务。

历史开发环境的基线备份不随本分支发布；回退前需自行保全当前修改。回退只恢复本轮四个修改文件并撤销本轮新增文件；必须先检查这些文件之后是否又有用户修改，不能 git reset --hard 或删除整个 training 包。
