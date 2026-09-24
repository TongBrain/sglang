# 单卡训推共置：推理基线与 5% 门禁

## 本轮目标

先冻结可复现的“纯推理”参照物，再开始并发训练。现有
`sglang.benchmark.serving` 负责发请求和采集原始指标，本轮只新增结果判定，
不另造一套计时逻辑。

## 已确认的门禁

- 主指标：三次相同负载实验的 `output_throughput` 中位数。
- 候选共置方案必须达到纯推理基线的 95% 及以上。
- 所有请求必须成功完成；超时或缺失请求直接失败。
- p99 TTFT 和 p99 TPOT 的中位数相对基线均不得上升超过 10%。
- 基线波动范围会报告，不把单次偶然值当结论。

## 首轮建议工作负载（尚待项目方确认）

这是工程对比负载，不冒充生产流量：固定模型、LoRA、输入/输出长度、并发、随机
种子，预热后重复三次。基线与共置候选必须使用完全相同的服务参数和请求文件。

建议先用 128 个请求、固定输入 256 token、固定输出 128 token、最大并发 16；
`random-range-ratio` 必须设为 `1.0` 才表示固定长度。随后再根据真实线上长度分布和
到达率替换。正式运行前应再次确认 GPU 空闲，且全程只使用一张卡。

## 判定示例

```bash
python -m sglang.benchmark.colocation_gate \
  --baseline work/colocation/baseline.jsonl \
  --candidate work/colocation/candidate.jsonl \
  --expected-prompts-per-run 128 \
  --max-throughput-drop-pct 5 \
  --max-p99-ttft-increase-pct 10 \
  --max-p99-tpot-increase-pct 10 \
  --output work/colocation/comparison.json
```

进程退出码为 0 表示通过，1 表示未通过。输出 JSON 保留基线、候选、吞吐门禁、
错误检查以及 TTFT/TPOT 相对变化，便于后续接入自动降载控制器。
