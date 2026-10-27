# SGLang 适配当前问题：原因、技术难点与证据

更新日期：2026-10-27。实验日期：2026-10-16。P1–P7 尚未完成修复或对应验收。

## 当前状态

**已测模型已完成 Store 权重恢复与正确生成验证；当前主要缺口为 SLLM 后端接口实现不完整，以及研究代码版本尚未统一。** 对更多模型的权重加载，仍需检查格式、容量和并行布局，现有两模型结果不能替代这些验证。

| 层次 | 已验证结果 | 未验证范围 |
|---|---|---|
| 权重与生成 | 已测 Qwen3-0.6B、Qwen3.5-9B 在限定配置下完成原生/Store 对照；本轮后端普通请求与自身原生参考一致 | 更大模型、MoE、量化、多卡均能通过 |
| 请求管理 | 已观察到真实请求开始输出，也记录了停止时的失败 | 跟踪、恢复、排空停止和主动取消已保持 SLLM 原有行为 |
| 完整系统 | 本轮测试真实后端对象、worker 和 Store | Controller、Ray 调度、HTTP 全链和自动模型准备已完成 |

权重加载成功有[worker 加载日志](evidence/2026-10-16-sglang-backend/worker1.log#L2-L7)及[普通请求输出一致结果](evidence/2026-10-16-sglang-backend/result.json#L168-L178)支持；完整系统未启动在[实验 scope](evidence/2026-10-16-sglang-backend/result.json#L1-L5)中明确记录。

## 组件职责与接口来源

- **SLLM** 管理模型实例和请求，决定何时启动、停止或迁移。
- **Store** 保存和传输权重；**SGLang** 使用权重执行推理。
- **适配层**是本次新增的 `SglangBackend` 与 `sllm_sglang/`，负责让 SLLM 能控制 SG。
- **worker** 是运行 SG 的独立进程；适配层通过私有 socket 与它交换请求、输出和结束结果。
- **最终结果/终态**指请求明确结束：正常完成、取消或失败；收到部分 token 不代表请求已结束。

`stop()`、`shutdown()`、`get_current_tokens()`、`resume_kv_cache()` **都是原版 SLLM 已定义的后端接口**，见[后端基类](../sllm/backends/backend_utils.py#L49-L63)。本次新增的是这些接口在 SG 后端的实现。接口行为对照[原版 vLLM 后端实现](../sllm/backends/vllm_backend.py#L265-L322)。

所有 P1–P3 源码链接均指向当次真实后端实验快照 `research/backend-contract/candidate/`。日志、请求结果和测试脚本的[公开副本](evidence/2026-10-16-sglang-backend/README.md)仅替换账号路径、物理 GPU UUID；[摘要清单](evidence/2026-10-16-sglang-backend/MANIFEST.json)可核对原件与公开副本。

## P1：SLLM 无法取得活动请求的 token，也无法调用恢复预热

### 预期行为与用途

例如请求输入 20 个 token、已经输出 30 个，SLLM 想把它转到另一实例，需要取得这 50 个 token，让目标模型处理已有内容，为继续生成做准备。

原版 vLLM 后端的[进度查询](../sllm/backends/vllm_backend.py#L295-L308)返回输入与已生成 token；[恢复预热](../sllm/backends/vllm_backend.py#L310-L322)把已有 token 再作为输入执行一次短生成。这里的“恢复”是重算已有 token 来建立缓存，不是直接把源 GPU 的 KV 内存复制到目标。

### 实测结果

本次 SG 适配方法直接抛出 `NotImplementedError`。因此请求可以生成，但 SLLM 查询进度、调用预热时没有实际功能可执行。

| 可直接核查的证据 | 内容及其含义 |
|---|---|
| [SG 后端第155–159行](../research/backend-contract/candidate/sllm/backends/sglang_backend.py#L155-L159) | 两个方法均为未实现占位；这是代码缺失的直接原因 |
| [结果第181–196行](evidence/2026-10-16-sglang-backend/result.json#L181-L196) | 请求已输出首 token，随后查询结果为 `token_trace: not_implemented` |
| [结果第207–208行](evidence/2026-10-16-sglang-backend/result.json#L207-L208) | 调用恢复方法得到 `result: not_implemented` |
| [脚本第99–113行](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L99-L113) | 可看到实际调用、捕获异常并保存结果的步骤；恢复输入为固定 prompt |

### 技术难点

| 待实现功能 | 现有代码缺口 | 实现要求 |
|---|---|---|
| 可查询的请求记录 | [runtime 第41–48行](../research/backend-contract/candidate/sllm_sglang/runtime.py#L41-L48)只有活动 ID 集合，输出保存在每个请求自己的 `StreamReceipt` 中 | SLLM 后端在父进程，需要一个可访问的记录/查询通道；当前局部对象不是父进程可查询的请求表 |
| 输出持续更新和清理 | [runtime 第58–71行](../research/backend-contract/candidate/sllm_sglang/runtime.py#L58-L71)会处理每个 chunk，[第101–102行](../research/backend-contract/candidate/sllm_sglang/runtime.py#L101-L102)结束时移除 ID | 需要明确完成、取消、错误时如何更新，防止查询到已结束请求或漏掉仍在执行的请求 |
| 一致的源快照 | 测试只验证“方法未实现”，没有完成真实源到目标接续 | 源端仍在生成时复制 token，快照可能很快过期；接续边界和最终源输出必须一致，否则可能重复或遗漏内容。这是需设计/验证的风险，尚非实测故障 |
| 目标预热及接续 | 原接口只完成预热；完整接续还涉及路由请求与输出预算 | 预热产生的额外 token 不能直接混入用户输出；还需验证缓存实际复用、剩余输出预算和源目标异常处理 |

**技术阻碍：** 适配层缺少进度查询和恢复实现。已有流式 token 输出可作为基础；进度跟踪需要请求状态管理，完整跨实例接续还需要源目标协议、输出预算和并发一致性验证。

**影响和验收：** 不阻断独立权重加载/普通生成；阻断依赖快照的迁移实验。完成条件是能查询真实活动 token，再用源快照在目标接续，验证无重复/丢失及预算正确。原样重跑未实现方法不能消除该缺口。

## P2：stop 直接结束 worker，没有等待已有请求完成

### 预期行为与实际差异

`stop()` 用于撤掉实例前的正常停止：不再接收新请求，等已经接收的请求结束，再关闭。原版 vLLM [stop 实现](../sllm/backends/vllm_backend.py#L283-L293)明确设置 STOPPING 并等待活动请求数归零。

当前 SG 实现实际执行：

```text
stop() → shutdown() → _terminate_child() → 向 worker 进程组发 SIGTERM
```

**缺的是等待过程。** 请求还在生成，输出连接已经被关闭。

### 证据链：调用、实现、日志、结果

| 顺序 | 证据 | 验证结果 |
|---|---|---|
| 1 | [脚本第99–107行](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L99-L107) | 发起 64 token 请求，观察首 chunk 后调用 stop，再检查请求结果 |
| 2 | [SG 源码第127–150行](../research/backend-contract/candidate/sllm/backends/sglang_backend.py#L127-L150) | stop 调用 shutdown，后者终止进程组，没有活动请求归零等待 |
| 3 | [worker1 日志第17–20行](evidence/2026-10-16-sglang-backend/worker1.log#L17-L20) | 19:47:25 收到 SIGTERM，scheduler 以 -15 退出，随后清理 |
| 4 | [结果第181–204行](evidence/2026-10-16-sglang-backend/result.json#L181-L204) | 先收到首 chunk，stop 用时约 5.52 秒，最终返回断连错误，`drained_normal_response: false` |

原始错误为：

```text
ValueError: Engine disconnected without a terminal result
```

这段错误由[后端读取连接的代码](../research/backend-contract/candidate/sllm/backends/sglang_backend.py#L99-L116)在没有读到最终结果、连接已结束时抛出。日志的终止信号与源码主动关闭路径对应，原因不在权重加载。

### 技术难点

| 待解决项 | 源码依据与风险 | 验收要求 |
|---|---|---|
| 停止接收与记录活动请求必须协调 | [generate 第83–95行](../research/backend-contract/candidate/sllm/backends/sglang_backend.py#L83-L95)检查 RUNNING，而 shutdown 使用独立生命周期锁；当前没有覆盖接收/登记/排空的完整同步流程 | stop 开始后无新请求漏入；stop 前接收的请求全部计入等待 |
| 等待条件必须反映真实请求结束 | worker [活动集合](../research/backend-contract/candidate/sllm_sglang/runtime.py#L41-L48)和[清理](../research/backend-contract/candidate/sllm_sglang/runtime.py#L93-L102)在子进程，父进程没有排空查询流程 | 正常结束、异常和取消都能让等待正确收敛，而不提前关闭或永久等待 |
| 排空超时的处理必须明确 | 当前代码没有单独的排空流程 | 请求长时间不结束时，转入可识别的取消/失败路径，再释放进程 |

**判断与阻碍：** 属于后端生命周期适配，核心是跨进程请求计数和并发顺序，并非模型执行代码问题。需要补正常排空及超时兜底；已知代码路径足以说明当前失败原因，无需先原样重跑定位。

**影响和验收：** 普通加载/生成已通过的结果仍成立；依赖实例缩容、排空停止的实验不能认领通过。修复后需在生成中调用 stop，确认完整输出、正常结束、新请求拒绝及最终资源释放。

## P3：shutdown 关闭实例时，没有先让活动请求得到明确取消结果

### 接口原本就有，SG 实现尚未保留原有行为

`shutdown()` 是原版 SLLM 已定义的接口。[后端基类](../sllm/backends/backend_utils.py#L49-L51)要求每种后端提供关闭实现。原版 vLLM [实现第265–280行](../sllm/backends/vllm_backend.py#L265-L280)先取得活动请求 ID、逐个 abort，再清理引擎。

**P2 要求活动请求正常完成后关闭；P3 要求中止活动请求并向 SLLM 返回明确的取消或失败结果。** 请求重试策略和 HTTP 响应由上层处理，完整 HTTP 行为尚未验证。

### 实测现象与直接原因

当前 [SG shutdown](../research/backend-contract/candidate/sllm/backends/sglang_backend.py#L127-L147)设置删除状态后结束 worker，没有先遍历活动请求执行取消并收集结束结果。

| 证据 | 内容及结论 |
|---|---|
| [脚本第114–118行](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L114-L118) | 重新发起请求，看到首 chunk 后调用 shutdown |
| [结果第211–230行](evidence/2026-10-16-sglang-backend/result.json#L211-L230) | 请求已输出，约 5.74 秒关闭后得到相同的 `Engine disconnected without a terminal result` |
| [worker2 日志第8–11行](evidence/2026-10-16-sglang-backend/worker2.log#L8-L11) | 19:47:54 收到 SIGTERM，scheduler -15 退出，随后清理 |

### 请求取消与关闭流程的技术难点

| 技术困难 | 已有代码的具体依据 | 必须补齐/验证的行为 |
|---|---|---|
| 关闭端需要知道该取消哪些请求 | [runtime 第41–48行](../research/backend-contract/candidate/sllm_sglang/runtime.py#L41-L48)中 caller ID、原生 ID 和 receipt 是请求内部变量；活动集合仅保存 caller ID | 让整体关闭流程能够定位每个活动请求的原生 ID，且不取消错请求 |
| 已有 abort 路径没有接成整体关闭流程 | [runtime 第84–102行](../research/backend-contract/candidate/sllm_sglang/runtime.py#L84-L102)在断连/异常时会调用 `abort_request(native_id)`，随后做任务清理 | 明确主动关闭时如何逐个取消、等待结果并通知父进程；发送取消指令不等于调用方已收到结果 |
| 当前结果转换不接受取消类型 | [protocol 第78–89行](../research/backend-contract/candidate/sllm_sglang/protocol.py#L78-L89)要求有已生成输出，结束原因只能是 `stop` 或 `length` | 定义取消响应和部分输出处理；即使引擎取消成功，现有转换也不能直接认领兼容；未输出首 token 就取消也需覆盖 |
| 取消与正常完成可能同时发生 | 生成任务、断连监测通过 [asyncio.wait](../research/backend-contract/candidate/sllm_sglang/runtime.py#L73-L83)并行推进 | 每个请求只能有一个确定结局，不能重复返回、遗漏通知或留下等待任务；这是待验证并发风险 |
| 引擎无响应时不能永远等待 | [进程终止实现](../research/backend-contract/candidate/sllm/backends/sglang_backend.py#L127-L138)已有 SIGTERM/超时 SIGKILL 兜底 | 保留强制清理能力，同时使等待方得到明确错误；验证重复关闭及子进程/显存释放 |

**技术阻碍：** 请求登记、取消控制、跨进程结果返回和异常回收尚未形成完整关闭流程。现有代码已接入 SG 请求级 abort；完整主动关闭路径仍缺少真实运行成功记录。

**影响和验收：** 不否定已完成的权重加载；会阻碍活动请求关闭和异常资源回收验收。修复后至少覆盖：首 token 前关闭、生成中关闭、正常完成同时关闭、引擎不响应、重复关闭。每个请求应有明确结局，最终进程和资源应可核对回收。

## P4：Qwen3.5 的成功来自研究副本，根目录版本尚未统一

**具体问题：** 根目录候选只放行 Qwen2/Qwen3；Qwen3.5 在研究副本放行架构和实际 dtype 后才完成实测。根目录版本的模型准入范围与研究副本不同。

| 证据 | 根目录候选 | 真实后端研究副本 |
|---|---|---|
| 架构检查 | [只放行 Qwen2/Qwen3](../sllm_sglang/store_loading.py#L19-L24) | [只放行此次 Qwen3.5 架构](../research/backend-contract/candidate/sllm_sglang/store_loading.py#L19-L24) |
| dtype 检查 | [统一检查 BF16](../sllm_sglang/store_loader.py#L39-L43) | [检查固定 checkpoint 的允许集合](../research/backend-contract/candidate/sllm_sglang/store_loader.py#L39-L43) |
| context | [4160](../sllm_sglang/config.py#L16-L24) | [512](../research/backend-contract/candidate/sllm_sglang/config.py#L16-L24) |

**技术难点/阻碍：** 要把“某次实验允许的固定配置”整理成可维护的模型准入规则，明确不同 checkpoint 的架构、张量类型和容量；不能直接去掉检查来宣称通用支持。当前是源码差异，不是本次新增运行报错，没有相应失败日志。

**影响和验收：** 下一模型实验前先统一版本和参数，冻结同一份代码做 CPU 检查及真实加载验证；各快照的验证结果仅适用于各自代码和配置，统一版本需要重新验收。

## P5：当前 Store 加载方案对更大模型存在显存容量风险

### 已观察事实与原因判断分开看

- **已观察：** 本轮 Qwen3.5 普通请求生成成功；[104 条采样](evidence/2026-10-16-sglang-backend/gpu-samples.json)中显存最大值为 37,482 MiB（约 36.6 GiB）。本轮未记录 OOM。
- **源码依据：** [Store 读取](../research/backend-contract/candidate/sllm_sglang/store_loading.py#L28-L36)先得到一份 CUDA 张量状态，再由 [loader](../research/backend-contract/candidate/sllm_sglang/store_loader.py#L35-L43)逐项交给原生加载流程。源权重在消费过程中可能与目标参数共存。
- **工程判断：** 额外权重共存可能增加加载需求，但现有采样不能精确拆分源权重、目标参数、KV 缓存等占用，尚未证明它是全部差值的原因。

原生与 Store 四组请求一致性的公开数据在[历史验证摘要](historical-validation-summary.json)中。这里的容量风险不能写成权重恢复失败，也不能等同于 SG 固有显存开销。

**技术难点/阻碍：** 降低加载峰值需要验证分块供给和释放源张量的可行性，并保持 Store 传输完成约束及模型权重融合、布局处理的正确性。现有 Store 入口返回完整状态，外层迭代方式的变化不足以证明 GPU 占用下降。多卡加载还需实现目标参数分片并独立验收；分块和多卡优化尚未完成验证。

**影响和验收：** 这是后续大模型加载优先要测的风险。针对目标模型记录加载过程、驻留阶段和退出后的占用，检查完整张量与输出一致性；当前 1 秒采样不是精确峰值，不应据此直接认定更大模型一定放得下或放不下。

## P6：首次请求期间仍有内核编译，性能测量条件未统一

**现象证据：** [worker1 日志第10–16行](evidence/2026-10-16-sglang-backend/worker1.log#L10-L16)明确记录服务开始后 Triton 内核编译，单次约 1.01–1.35 秒。普通请求仍成功，这是性能警告，不是生成正确性失败。

**技术难点/阻碍：** 需要识别目标模型/输入形状实际会触发哪些编译，划清引擎就绪、预热完成、首请求和稳态测量边界。仅有这几条日志，不能量化全部端到端等待时间，更不能将原生路径和 Store 路径的不同预热状态直接比较。

**影响和验收：** 可以继续做加载正确性验证；性能实验需记录编译/缓存条件和完整时间线，再比较首次请求及稳态指标。完整服务等待时间的各阶段占比尚未测量。

## P7：现有文件能恢复，不等于自动准备和完整服务已经接通

| 具体缺口 | 证据 | 技术工作与验收阻碍 |
|---|---|---|
| SG 没有自动下载/导出准备分支 | [StoreManager 第427–436行](../sllm/store_manager.py#L427-L436)只检查已有 Store 文件 | 需要接通 HF 布局文件生成、索引/张量校验、注册与失败回收；旧 vLLM 处理后的 rank 布局不能直接复用为当前输入 |
| 本轮没有启动完整服务链 | [result scope](evidence/2026-10-16-sglang-backend/result.json#L1-L5)明确未启动 Controller/Ray/HTTP | 需验证真实调度、后端启动、HTTP 流式、断连和退出，不能用内部 chunk 正确替代 |
| 更多功能/模型没有对应资格记录 | [候选协议](../sllm_sglang/protocol.py#L16-L31)限定 raw IDs、贪心和输出预算，[架构检查](../sllm_sglang/store_loading.py#L19-L24)范围有限 | Chat/Embedding、LoRA、MoE、量化、多模态、长上下文、多卡应按目标需要逐项接入和验证；缺记录不等于全部已失败 |

历史报告还记载原生 Store writer 文件不完整、研究阶段使用 CPU 参考 writer。本仓库尚未提供该失败的完整原始日志链，因此只能列为待补证据的准备流程问题，不能把 stop/shutdown 日志当作它的证据，也不能认领已修复。详见[代码与验证报告](sglang-adaptation-report.md)。

## 日志字段含义

`NotImplementedError` 被[测试脚本](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L99-L113)捕获并保存为 `not_implemented`，因此没有对应未捕获异常堆栈。这不影响 P1“方法尚未实现”的源码和运行证据。

[experiment-process.log](evidence/2026-10-16-sglang-backend/experiment-process.log)中的 `Timeout (0:00:45)!` 来自[第30行定时线程栈打印](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L30)，不是请求超时判定。末尾 `Finished with result: success` 只表示观察程序完成，具体能力是否通过需看每个请求结果。

## 后续工作安排

若下一步目标是更多模型的权重加载，优先统一候选版本（P4）、落实 Store 准备（P7）、检查加载容量（P5），并保证实验退出清理可靠。P1 的完整迁移可以按研究需求排序；P2/P3 未修复前不能验收依赖停止、缩容或取消语义的服务实验。

修复验收包括 CPU 状态与异常路径检查，以及获批后的真实 GPU 回归。全部模型覆盖和总工期尚未确定。
