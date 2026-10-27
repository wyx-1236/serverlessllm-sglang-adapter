# 当前问题、原因与实验影响

更新日期：2026-10-27。本文记录当前已归档候选的问题，不表示已完成修复。真实运行证据来自 2026-10-16 的服务器实验；本次仅整理材料，没有重跑 GPU 实验。

**已测 Qwen3-0.6B、Qwen3.5-9B 的 Store 恢复生成已经通过有限配置验证。当前不能将停止/重放错误解释成“权重加载失败”，也不能将两模型通过扩大为任意模型可用。** 后续指定模型的加载验证可以继续准备；完整 SLLM 服务、弹性伸缩及请求迁移实验仍受下述缺口影响。

## 阅读证据的约定

- **实测失败**：请求结果直接记录失败，并有测试步骤、相关日志或源码可以交叉核对。
- **源码缺口**：实现明确缺失或受限，但没有对应配置下的真实失败日志；不补造报错。
- **实测风险**：日志/采样观察到了现象，尚未证明一定导致故障或已确定全部原因。
- **未验收**：没有完成目标范围验证，不能标为失败，也不能标为支持。

原始文件的[公开副本目录](evidence/2026-10-16-sglang-backend/README.md)已经包含日志、完整请求结果和实际执行脚本。仅替换服务器账号路径、物理 GPU UUID，保留全部行；[MANIFEST](evidence/2026-10-16-sglang-backend/MANIFEST.json)记录原件和公开副本摘要，本地原件未修改。

## 问题总览

| 编号 | 当前问题 | 证据性质 | 对下一阶段实验的影响 |
|---|---|---|---|
| P1 | 活动 token 跟踪、恢复/重放入口未实现 | 实测未实现 + 源码占位 | 不阻断独立加载/普通生成；阻断源请求快照与完整迁移验证 |
| P2 | stop 终止 worker，未正常排空活动请求 | 实测失败 + worker 日志 + 源码 | 不否定已完成的加载；阻断排空、缩容等行为验收 |
| P3 | 活动请求 shutdown 缺少正常取消终态 | 实测失败 + worker 日志 + 源码 | 影响请求关闭及异常回收验收；反复加载实验也必须落实进程组清理 |
| P4 | Qwen3.5 研究副本尚未统一到根目录候选 | 源码差异，非新增运行报错 | 使用根目录直接扩展模型可能先被准入检查拒绝；必须先选定统一候选 |
| P5 | Store 加载路径存在额外显存容量风险 | 实测采样 + 加载实现；本轮未观察到 OOM | 可能限制更大模型；需按目标模型测加载峰值，不能用已测结果推断所有模型 |
| P6 | 生成期间仍发生内核编译 | worker 日志明确警告；延迟影响尚未定量 | 影响首次请求和性能口径；不能未经预热条件核对就比较时延 |
| P7 | SG 自动准备/导出、完整服务及更多模型能力未完成 | 源码缺口 / 未验收 | 已有 Store 可支持有限验证；不能直接开展完整公平 Baseline 实验 |

## P1：活动 token 跟踪和重放没有实现

**原因属于我们的适配层。** [后端 `get_current_tokens()` 和 `resume_kv_cache()`](../research/backend-contract/candidate/sllm/backends/sglang_backend.py#L155-L159)直接抛出 `NotImplementedError`。尚未把 SG 的流式输出、请求状态与 token 输入组织成 SLLM 的活动快照、缓存预热和跨实例续写协议。

证据：

- [结果：活动请求已输出首 chunk，但 token_trace 为 not_implemented](evidence/2026-10-16-sglang-backend/result.json#L181-L196)。
- [结果：resume_token_replay 为 not_implemented](evidence/2026-10-16-sglang-backend/result.json#L207-L208)。
- [测试脚本：实际调用和异常捕获](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L99-L113)。

这是测试程序捕获异常后的结果记录，不存在对应的未捕获异常堆栈。恢复调用使用固定 prompt，并未完成一次真正的源到目标迁移。不能因此认定 SG 引擎缺少所有必要基础能力，也没有证据要求逐模型重写执行算子。

推进条件：先实现适配方法并做 CPU 状态机验证，再用真实源快照检查目标续写、输出连续性、剩余预算及缓存行为。原样重跑占位方法不会解决问题。

## P2：stop 没有等待活动请求正常完成

**原因是当前停止策略。** [适配源码](../research/backend-contract/candidate/sllm/backends/sglang_backend.py#L128-L150)中 `stop()` 直接调用 `shutdown()`，后者向所属进程组发出 SIGTERM。请求还在生成时 worker 已退出，连接读到 EOF 而未收到最终结果；适配层因此返回缺失终态错误。

证据：

- [首 chunk、停止耗时及请求失败结果](evidence/2026-10-16-sglang-backend/result.json#L181-L204)：`ValueError: Engine disconnected without a terminal result`，`drained_normal_response: false`。
- [worker1.log 第17–20行](evidence/2026-10-16-sglang-backend/worker1.log#L17-L20)：收到 SIGTERM，scheduler 以 `-15` 退出，随后清理。
- [脚本 stop 调用与结果判定](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L103-L107)。

`-15` 与主动终止信号一致，不应写成“SG 自发崩溃”。日志中虽然出现 `Draining requests and shutting down`，但请求结果没有正常终态，不能据此认领排空通过。

推进条件：停止接收新请求、维护活动请求集合、等待正常完成后再关闭。模型加载可行性验证可单独推进；正式缩容/排空实验需修复后重测。

## P3：shutdown 没有给活动请求返回约定的取消终态

当前实现先关闭整个 worker，尚未完成所有活动请求的请求级取消、结果收集和终态映射。它与 P2 共用进程终止路径，但目标行为不同：stop 应排空，shutdown 的取消策略则需明确并返回可识别的终态。

证据：

- [结果：先收到首 chunk，shutdown 后返回缺失终态错误](evidence/2026-10-16-sglang-backend/result.json#L211-L230)。
- [worker2.log 第8–11行](evidence/2026-10-16-sglang-backend/worker2.log#L8-L11)：收到 SIGTERM，scheduler `-15` 退出。
- [测试脚本：另一请求上的 shutdown 观察](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L114-L118)。

[请求传输模块](../research/backend-contract/candidate/sllm_sglang/runtime.py)已有消费者断连时调用原生 abort 的路径，但不代表整体 shutdown 已正确组合该能力。此处是适配层生命周期缺口，不能解释成 SG 没有 abort 接口。

推进条件：关闭前处理活动请求及终态，并验证重复关闭、超时、异常启动和进程回收。已有最终清理归零记录包含外部监督清理，不等于所有资源都由后端 API 自动释放。新的加载实验也应具备可靠外部清理，不把残留显存带入下一轮。

## P4：Qwen3.5 支持仍局限于独立研究副本

这项是已确认的源码差异，没有新增运行失败日志。

- [根目录架构门禁](../sllm_sglang/store_loading.py#L19-L24)只允许 Qwen2/Qwen3 且拒绝量化；[根目录 dtype 检查](../sllm_sglang/store_loader.py#L39-L43)要求 BF16。
- [真实后端研究副本门禁](../research/backend-contract/candidate/sllm_sglang/store_loading.py#L19-L24)针对 Qwen3.5；其 loader 按精确 checkpoint 的 dtype 集合检查。
- 根目录 context 为 [4160](../sllm_sglang/config.py#L16-L24)，研究副本为 [512](../research/backend-contract/candidate/sllm_sglang/config.py#L16-L24)。

因此，“Qwen3.5 研究配置已跑通”与“根目录可直接部署 Qwen3.5”是两件事。先统一代码、准入规则、环境和目标模型规格，再做下一模型验证；不要直接把三个快照拼接成一个已通过的版本。

## P5：权重加载已通过，但大模型容量仍有风险

**没有证据表明本次 stop/shutdown 报错由权重加载失败导致。** 同轮 [worker1.log 第2–7行](evidence/2026-10-16-sglang-backend/worker1.log#L2-L7)记录 Store 恢复和确认完成；[完整请求结果](evidence/2026-10-16-sglang-backend/result.json#L7-L179)记录普通生成输出及与自身原生参考一致。

容量风险的依据是：

- [104 条资源采样](evidence/2026-10-16-sglang-backend/gpu-samples.json)中，GPU 显存采样最大值为 **37,482 MiB（约 36.6 GiB）**，不是精确瞬时峰值。
- [Store 读取代码](../sllm_sglang/store_loading.py#L28-L36)先获得 CUDA 张量状态，再由 loader 将其交给原生模型加载。这一实现支持“源权重与目标参数可能共存”的判断，但采样本身不能精确拆分每部分占用。
- [历史原生/Store 对照摘要](historical-validation-summary.json)记录 Qwen3.5 四组 token/logprob 对照通过；不是全部模型的加载资格证明。

本轮没有记录 OOM，不能将容量风险写成已发生 OOM。后续更大模型、MoE、量化及 TP 多卡仍需逐项验证；可能需要降低源权重共存峰值或调整准备/分片流程。这是后续模型扩展的关键工作，优先级高于与首批加载实验无关的完整迁移功能。

## P6：首轮生成期间存在编译开销

[worker1.log 第10–16行](evidence/2026-10-16-sglang-backend/worker1.log#L10-L16)多次记录 Triton 内核在服务开始后编译，单次约 1.01–1.35 秒，并提示可能阻塞引擎。这是实测日志中的性能警告，没有导致本轮普通生成正确性失败。

需要核实编译/预热完成边界，再比较首请求或稳态性能。当前没有完整时间线证明负责人此前全部服务等待时间都来自这些编译，也不能据此归因 Ceph、Store 或 SG 的整体性能。

## P7：自动准备、完整服务和模型覆盖仍未验收

- [StoreManager 的 SG 分支](../sllm/store_manager.py#L427-L436)明确仅接收已存在并校验的 Store 文件，没有自动下载/导出分支。原版 vLLM 处理后的 rank 布局不能直接当成当前 HF 布局输入。
- [结果 scope](evidence/2026-10-16-sglang-backend/result.json#L1-L5)明确本轮未启动 Controller/Ray/HTTP；普通后端对象通过不能替代完整服务资格。
- Embedding、Chat 模板通用接入、LoRA、长上下文、更多架构、量化、多模态和 TP 多卡没有完成对应验收，不能笼统写为“模型不支持”或“已支持”。

另有历史材料记录原生 Store writer 不完整、使用 CPU 参考 writer 完成研究对照；本仓库当前未提供该 writer 失败的完整原始日志链。该问题仍列在[适配报告](sglang-adaptation-report.md)的准备流程缺口中，但不以这里的 stop/shutdown 日志充当其证据，也不把它算成 SG 特有故障。补齐日志前不得标为已修复。

## 哪些日志不能当作故障结论

[experiment-process.log](evidence/2026-10-16-sglang-backend/experiment-process.log)中的 `Timeout (0:00:45)!` 来自[脚本的定时线程栈打印](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L30)，不能单独认定为请求超时；`Finished with result: success` 只表示观察程序退出成功，不能证明每项能力通过。

## 对下一阶段的建议

先统一首批模型的候选版本，补齐 CPU 侧准备、Store 文件校验和可靠退出清理，再申请短时 GPU 验证原生/Store 一致性和加载容量。P1 的完整迁移能力可按实验目标安排，但 P2/P3 未完成时，不进行依赖排空或迁移语义的正式对照。修复后必须回归验证，不能仅修改文档即认领完成。
