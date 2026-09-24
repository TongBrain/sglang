# 推理侧 LoRA 生命周期接口约定

## 修改位置与边界

- 分支：`feat/lora-lifecycle-api`。
- 独立工作目录：`/path/to/sglang`。
- 基于 `feat/paged-moe-lora` 提交 `443fff832f3e7176bf7f41f247d3ff691df43e07`。
- 原项目 `/path/to/reference-checkout` 的源码、当前分支和已有工作区改动保持不变。
- 本轮只支持 Python HTTP server；没有给嵌入式 Rust HTTP server 增加新路由。

MaaS 平台负责产物校验、用户绑定、版本发布、灰度、切流、回滚、多实例协调和恢复。SGLang 不保存用户信息，也不提供平台级 activate/version-alias 接口。

适配器名称应不可变，例如 `user_A@v2`。请求中的 `lora_path` 填已注册的名称，不是训练输出目录。平台必须在请求进入推理之前确定具体版本。

## 接口清单

| 接口 | 方法 | 作用 |
| --- | --- | --- |
| `/load_lora_adapter` | POST | 从推理服务器可访问的目录或已有支持的模型源加载 |
| `/load_lora_adapter_from_tensors` | POST | 使用现有的按 TP rank 序列化 tensor 协议加载 |
| `/get_lora_adapter_status` | GET | 查询注册身份、活跃请求数、各分片显存驻留情况 |
| `/prewarm_lora_adapter` | POST | 通过正常调度执行单 token 探测，之后返回状态快照 |
| `/unload_lora_adapter` | POST | 停止注册入口，等待已保留的请求结束，再清理权重与页面 |

路径不带 `/v1`。生命周期接口统一标记为 `ADMIN_OPTIONAL`，遵循已有 API key/admin API key 配置；未配置任何 key 时仍保留原有开放行为，因此部署时应配置管理鉴权和内网访问控制。

操作失败返回 HTTP 400 和 `success: false`；请求结构不合法由框架返回 422。不能只依据 HTTP 200 判断平台发布完成，必须检查响应状态并完成目标实例的验证。

## 1. 路径加载

```json
{
  "lora_name": "user_A@v2",
  "lora_path": "/adapters/user_A/v2",
  "pinned": false
}
```

成功响应保留 `success`、`error_message`、`loaded_adapters`，新增 `lora_id`。该 ID 是本次运行时注册的身份；卸载后重新加载会产生新 ID，不是持久版本号。

同名加载会明确失败，不能用原名称覆盖权重。本轮不把“同一路径”视为幂等证据，因为目录内容可能被修改。网络超时后，平台应查询状态、核对自己保存的产物信息，而不是直接覆盖重试。

动态更新结果在请求广播的 CPU group 内汇总所有分片，并在 tokenizer 层汇总所有返回的副本结果。任一分片失败都会返回失败，并尝试清理同一次新 `lora_id` 在所有目标副本上的状态；若清理失败，错误信息会明确包含 rollback failed，平台不得切流。

路径产物的 checksum、base model revision、parent version 等仍由平台校验。本轮没有新增磁盘产物 manifest/checksum 协议。

## 2. 状态查询

```text
GET /get_lora_adapter_status?lora_name=user_A%40v2
```

不传 `lora_name` 时列出当前注册的适配器；指定未注册名称时返回失败。

响应字段：

- `adapters[name].lora_id`：tokenizer 注册身份。
- `lora_path`、`pinned`：已注册的路径与加载属性。
- `active_requests`：tokenizer 已保留、尚未释放的请求计数，仅为快照。
- `loaded`：所有查询到的分片都存在相同 `lora_id`。
- `resident`：身份一致，且所有查询到的分片均报告完整显存驻留。
- `workers`：每个分片的 `tp_rank`、`dp_rank`、`pp_rank` 和适配器元数据。
- `workers[].adapters[name]`：rank、target_modules、驻留状态、required_pages、resident_pages。
- 分页池信息：paged、total_pages、free_pages、page_rank_size。非分页池相关页面字段为 null。
- `probe_completed`：状态查询默认 false；预热全部探测完成后为 true。

查询不加载或换入权重，不改变 LRU。显存驻留是某一时刻的观察，不是保留承诺，也不是独立的质量验收。后续请求可能触发换出；平台应通过预热/探测确认可执行性。

新状态和预热接口要求 `pp_size=1`，且 `dp_size=1` 或启用 DP attention。未支持的配置会明确返回失败，不声称已验证所有流水线分片。

## 3. 预热与探测

```json
{
  "lora_name": "user_A@v2",
  "input_ids": [1, 2, 3]
}
```

必须提供适用于当前基础模型的合法、非空 prompt token ID；示例 token 不应直接用于所有模型。

推理侧在当前各个路由副本上，依次通过正常生成调度执行 `max_new_tokens=1`、`temperature=0` 的请求，再查询各分片状态。正常调度复用现有活跃页面保护，不增加绕过调度的直接显存抢占路径。

成功执行时 `probe_completed=true`。平台仍应检查 `success` 和目标版本身份；探测被 abort、无完成结果或分片状态不一致都会失败。该探测只验证执行链路，不评价 LoRA 质量；业务 canary 和离线效果门禁由平台提供。

预热期间串行化 LoRA 加载/卸载，但不暂停正常推理。它会受队列、暂停状态和服务负载影响，本轮未提供独立的预热超时/SLA 或显存驻留租约。

## 4. 推理请求

通过已有 `/generate` 请求：

```json
{
  "text": "请简要解释什么是 LoRA。",
  "lora_path": "user_A@v2",
  "sampling_params": {"max_new_tokens": 32, "temperature": 0}
}
```

平台先解析具体版本，再发给引擎。推理侧没有用户到适配器的映射，也不会自动替平台切换 v1/v2。

本轮修复注册表中的并发窗口：名称解析和请求计数增加位于同一个注册锁内，防止卸载在 lookup 与 retain 之间删除计数或提前释放权重。旧请求继续使用自己保留的 ID。

## 5. 卸载与回收

```json
{"lora_name": "user_A@v1"}
```

平台应先停止把新请求路由到 v1，再发起卸载。推理侧注销名称，等待已保留的请求归零后清理运行时资源。

新增 `missing_ok=true` 可把已不在注册表中的适配器视为完成卸载；内部也用它清理失败加载时尚未注册的分片状态。它不代表检查或删除了磁盘产物。

保留原有的 `lora_ref_cache` 自动重载行为：曾加载的磁盘适配器，在后续收到旧名称请求时仍可能被隐式重新加载。因此卸载不是永久禁止旧版本访问；平台必须控制过期版本路由，本轮未增加永久注销或 tombstone 接口。Tensor 适配器的旧缓存路径不具备从磁盘自动恢复的能力，平台负责重新推送。

## 平台推荐发布流程

1. 生成不可变目录和版本名，校验产物与基础模型兼容性。
2. 对每个目标实例加载新版本，检查成功结果和运行时 ID。
3. 预热，再执行平台业务 canary。
4. 仅当目标副本满足平台门禁后，原子切换用户/版本绑定。
5. 失败时保留旧绑定，清理新版本；如回滚清理失败，隔离异常实例。
6. 不再路由旧版本后发起卸载，确认完成，再按平台策略处理产物。

没有提供跨实例原子事务。进程重启后注册 ID 和运行时状态需要由平台重新协调。
