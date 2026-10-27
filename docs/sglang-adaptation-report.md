# SGLang 适配改动与验证说明

本文面向路线讨论和代码审阅，说明本仓库相对原版 ServerlessLLM 的代码变化、已有验证及剩余工作。整理日期：2026-10-27。

**当前结论：Store → SGLang 生成路线已在有限模型和配置下证明可行；尚未完成保持原版 ServerlessLLM 全部目标行为的正式迁移。** 已有请求级取消实现，但不能据此认为排空停止、活动请求关闭和跨实例续写已完成。

## 1. 对照基准与快照范围

代码对照基准为原版 ServerlessLLM 提交 [`9f50241baa5386e06a9321c51f19a9ef5f964c2b`](https://github.com/ServerlessLLM/ServerlessLLM/tree/9f50241baa5386e06a9321c51f19a9ef5f964c2b)，不是浮动的最新版。逐文件比较按文本内容进行，忽略换行符差异。

| 代码范围 | 用途与边界 |
|---|---|
| 根目录 `sllm/`、`sllm_sglang/`、`tests/` | candidate12 准备候选；保留已有 CPU 测试，不代表该整份快照通过 GPU 或全链验收 |
| [`research/backend-contract/candidate/`](../research/backend-contract/candidate/) | 2026-10-16 真实后端行为观察时保存的源码，包含针对该次实验的配置调整 |
| [`research/qwen35-store/sllm_sglang/`](../research/qwen35-store/sllm_sglang/) | Qwen3.5 原生/Store 对照使用的加载器研究副本 |

根目录候选限制为 Qwen2/Qwen3、BF16、TP1，配置中 context 为 4160、RAM 门槛为 16 GiB；真实后端研究副本将 context 调为 512、RAM 门槛调为 64 GiB，并调整 worker 监督检查。Qwen3.5 研究 loader 只放行 `Qwen3_5ForConditionalGeneration`，dtype 按固定源 checkpoint 的允许集合检查。因此，**Qwen3.5 实测通过不意味着根目录代码已经直接支持 Qwen3.5**，更不能将不同副本的能力相加认作完整版本。

源码来源和逐文件 SHA256 见 [`SOURCE-MANIFEST.json`](../SOURCE-MANIFEST.json)。该清单覆盖 Python 快照，不是完整依赖环境清单。

## 2. 相较原版具体修改了哪些代码

### 2.1 新增 SGLang 执行链

原版由 SLLM 后端直接接入 vLLM 等引擎；新增路线为：

```text
SLLM / Ray 后端对象
  → SglangBackend
  → 现代 Python 子进程 + 私有 Unix socket
  → SGLang Engine / 原生模型实现
  → 自定义 Store loader
  → SLLM Store 的 HF 布局 CUDA 张量
```

控制面与新版引擎使用不同解释器，以隔离依赖。新增代码没有重写 Qwen3/Qwen3.5 的模型执行源码；本仓库也没有修改 SGLang 上游模型文件。

| 新增文件 | 实际实现 | 当前限制 |
|---|---|---|
| [`sllm/backends/sglang_backend.py`](../sllm/backends/sglang_backend.py) | 实现 `SglangBackend`：启动/检查 worker、转发请求、接收 chunk 和最终响应、超时处理、终止所属进程组 | `stop()` 直接调用 `shutdown()`；`encode()`、`get_current_tokens()`、`resume_kv_cache()` 显式未实现 |
| [`sllm_sglang/config.py`](../sllm_sglang/config.py) | 构造 Engine 参数，固定 TP1/DP1、BF16、关闭图执行；要求显式 KV 容量和前缀缓存策略；检查运行期限与 GPU 绑定 | 含实验资源门槛，尚未成为通用部署配置 |
| [`sllm_sglang/worker.py`](../sllm_sglang/worker.py) | 检查解释器、依赖版本、文件摘要、cgroup 和私有目录；构造 Engine，运行 socket 服务，退出时关闭 Engine | 依赖 Linux、Python 3.12 和外部监督记录；根目录仍含 12 GiB 磁盘保护线，该数值不是引擎固有需求 |
| [`sllm_sglang/runtime.py`](../sllm_sglang/runtime.py) | 使用 `async_generate()` 转发原生流，连接对应唯一原生请求 ID；断连/超时时调用原生请求级 abort | 该路径的取消不等同于后端 `stop`/`shutdown` 已正确处理所有活动请求 |
| [`sllm_sglang/protocol.py`](../sllm_sglang/protocol.py) | 校验原始 token ID、EOS/PAD、贪心生成参数；累计输出转增量，检查 token/usage/终态，形成响应 | 输入最多 4096 token、输出 1..64；仅接受指定字段；记录 chunk 观察时间，不宣称每 token 延迟 |
| [`sllm_sglang/http_stream.py`](../sllm_sglang/http_stream.py) | 用 Ray Queue 将生成事件转换为 HTTP SSE，传递文本增量、最终响应和错误，清理请求队列 | 候选专用协议；完整真实 HTTP 流式和断连回收未验收 |
| [`sllm_sglang/store_loader.py`](../sllm_sglang/store_loader.py) | 继承 `DefaultModelLoader`，由 Store 提供张量迭代器，交给 SGLang 原生权重处理；禁止 HF 权重回退 | 源 GPU 权重与目标模型可能共存；额外权重、量化、多卡不在当前准入范围 |
| [`sllm_sglang/store_loading.py`](../sllm_sglang/store_loading.py) | 校验 Store 路径、索引摘要与架构，确认 Store 传输完成，拒绝空状态或 CPU 张量 | 根目录只放行 Qwen2/Qwen3；Qwen3.5 调整仅在研究副本中 |
| [`tests/`](../tests/) | 5 个测试文件、13 项 CPU 测试，覆盖配置、协议、Store 确认、后端拒绝启动和模拟流/取消 | 不创建真实 GPU 引擎，不覆盖完整服务链 |

`sllm_sglang/__init__.py` 为新增包入口。

### 2.2 修改原版已有文件

根目录 `sllm/` 共 32 个 Python 文件：对照基准提交，**9 个已有文件有文本修改、2 个文件新增、21 个文件文本内容相同**。新增文件是 `sglang_backend.py` 和下节说明的 `baseline_request.py`；这不是说全部 9 个已有文件都是 SGLang 迁移必需改动。

| 修改文件 | 相对原版的具体变化 | 归属 |
|---|---|---|
| [`sllm/backends/__init__.py`](../sllm/backends/__init__.py) | 增加 `SglangBackend`；将后端直接导入改为 `__getattr__` 延迟导入，减少无关引擎依赖在导入时触发 | SGLang 接入 |
| [`sllm/inference_instance.py`](../sllm/inference_instance.py) | `start_instance()` 增加 `backend == "sglang"` 分支 | SGLang 接入 |
| [`sllm/store_manager.py`](../sllm/store_manager.py) | 将 SG 映射至 `transformers/<model>` Store 布局；注册单份 HF 布局；准备阶段仅接受已校验 Store 文件，不走自动下载/导出 | SGLang 接入 |
| [`sllm/app_lib.py`](../sllm/app_lib.py) | `/v1/chat/completions` 新增候选协议校验和 SSE 分支，转到 `sllm_sglang.http_stream` | HTTP 实验接入 |
| [`sllm/controller.py`](../sllm/controller.py) | 部分控制 Actor 显式 `num_cpus=0`、保留自定义资源请求；等待 router 启动；模型删除时先 shutdown 后 kill router | 前期控制面/生命周期修改 |
| [`sllm/routers/roundrobin_router.py`](../sllm/routers/roundrobin_router.py) | 增加副本状态快照；请求计数归还置于 `finally`；分配循环处理已完成/取消 future，必要时归还实例计数 | 前期路由及回收修改 |
| [`sllm/routers/migration_router.py`](../sllm/routers/migration_router.py) | 对齐实例池引用和 LoRA 参数；补请求计数回收、迁移失败记录、目标实例回收；调整目标就绪与源 token 预热的迁移过程 | 前期迁移路由修改，不能证明 SG 迁移已完成 |
| [`sllm/schedulers/fcfs_scheduler.py`](../sllm/schedulers/fcfs_scheduler.py) | 遍历节点前按节点 ID 排序，使首次选择顺序稳定 | 前期实验调度修改 |
| [`sllm/backends/transformers_backend.py`](../sllm/backends/transformers_backend.py) | 增加副本物理 GPU/Actor 身份检查、候选原始 token 请求分支与锁；显式 thinking 参数；加强生成状态清理；调整 KV 预热和续写剩余预算处理 | 前期 Transformers 基线修改，不是 SG 模型执行实现 |

### 2.3 保留的历史代码和未打包依赖

新增的 [`sllm/backends/baseline_request.py`](../sllm/backends/baseline_request.py) 是 Transformers 实验请求辅助模块，负责显式贪心生成、token 计量及流式辅助。它引用 `generation_policy`、`chat_policy`、`token_stream04` 等外部实验模块，**这些模块未包含在本仓库中**。因此保留该文件不代表 Transformers 实验分支可独立运行。

对照基准，根目录 `sllm/backends/vllm_backend.py` 和 `sllm/model_downloader.py` 的文本内容未修改；研究归档中的同名文件必须单独看待。仓库未包含 SLLM Store 原生服务端/写入器源码改动、完整打包元数据和锁定环境。本仓库保存的是适配源码及研究快照，不是原版 ServerlessLLM 的完整可安装替代发行版。

## 3. 已完成哪些验证

下表区分真实 GPU、真实后端对象及模拟 CPU 测试。历史实测日期为 2026-10-16；本次文档整理没有申请或使用服务器 GPU。

历史结果的公开字段摘录见 [`historical-validation-summary.json`](historical-validation-summary.json)，其中记录原始本地归档文件名和 SHA256。摘录移除了服务器目录、GPU UUID 等环境身份信息，不是完整原始日志，也不能单靠摘录完整复现实验。

| 层次 | 验证内容 | 已观察到的结果 | 证据范围 |
|---|---|---|---|
| Store 基础链 | 旧 Store 服务与现代客户端的真实 CUDA IPC | 311 个源张量逐项一致 | 两路线共享基础验证，不能证明全部模型格式均适用 |
| Qwen3-0.6B 引擎/Store | SG 原生与真实 Store 恢复生成、重复请求 | 两次输出 ID 一致，记录的 token logprob 差值为 0 | BF16、TP1、单个短原始 prompt |
| Qwen3.5-9B 引擎/Store | SG 原生与 Store 四组请求对照 | 输入长度 5/11/121/5，每次输出 8 token；token、top5 集合和已记录 logprobs 一致，最大绝对误差 0 | SGLang 0.5.21、A100 40GB、BF16、TP1、context 512、文本、关闭图执行；研究 loader |
| 真实 SLLM 后端对象 | 普通生成与内部流结果核对 | 输出与自身原生参考一致，usage 正确 | 未启动 Controller/Ray/HTTP 全链 |
| 真实后端停止/恢复 | 活动请求期间 stop、shutdown、恢复方法 | **未通过**：stop/shutdown 造成缺失正常终态的断连；恢复未实现 | 明确记录失败，不能列为已适配 |
| 最终进程组清理 | 退出并执行清理后观察 GPU | 最终无 CUDA 客户端，显存归零 | 包含外部清理，不能宣称全部由后端 API 主动释放 |
| CPU 单元测试 | `python -m unittest discover -s tests -v` | 本次归档时及文档整理时复核：13 项通过 | 模拟引擎/张量；有真实断连取消协议的模拟测试，无真实 GPU |
| 语法与归档完整性 | compileall、来源清单校验 | 64 个 Python 源码文件 SHA256 与清单一致 | 证明语法/归档一致，不证明运行环境齐备 |

Qwen3.5 Store 阶段 GPU 显存的 1 秒采样最大值为 **37,482 MiB（约 36.6 GiB）**，原生阶段为 26,368 MiB。该观测说明当前加载路线存在容量压力，但两种路径条件不同，不能据此做公平性能排名或认定为 SGLang 固有开销。

恢复验证还受到 Store 文件准备路径的限制：历史原生 writer 曾产生不完整文件，研究阶段用 CPU 参考 writer 完成后续验证。**Store 恢复成功不能扩大为原版快速写入/导出路径已修复。** CPU 参考 writer 未随本次适配源码归档提供。

### 3.1 停止、跟踪与重放问题的逐项原始证据

公开的[证据目录与文件说明](evidence/2026-10-16-sglang-backend/README.md)包含同一次实验的完整文件副本，不再只有摘要。仅对服务器账号路径和物理 GPU UUID 做明确替换；时间、请求、错误、测试步骤和行数保持原样，原件及公开副本 SHA256 见 [MANIFEST](evidence/2026-10-16-sglang-backend/MANIFEST.json)。以下链接可在 GitHub 直接定位行号。

**① 活动 token 跟踪：确实未实现，不是推测。**

[测试脚本第99–102行](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L99-L102)先发起最多 64 token 的请求，等到首个 chunk 后调用 `get_current_tokens()`；捕获 `NotImplementedError` 后写入 `not_implemented`。[请求结果第181–196行](evidence/2026-10-16-sglang-backend/result.json#L181-L196)既记录了首 chunk（说明请求已在执行），也记录了 `"token_trace": "not_implemented"`。对应[后端方法第155–156行](../research/backend-contract/candidate/sllm/backends/sglang_backend.py#L155-L156)直接抛出该异常。

这证明当前适配层没有向 SLLM 提供迁移所需的活动 token 快照。它不证明 SGLang 引擎缺少实现这种适配所需的基础接口；普通流已经能提供 token chunk。

**② stop 未保持排空语义：worker 被终止，活动请求没有正常终态。**

[脚本第103–107行](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L103-L107)在活动请求期间调用 `stop()`，随后读取该请求结果。[结果第197–204行](evidence/2026-10-16-sglang-backend/result.json#L197-L204)记录：

```json
"response": {
  "error": "ValueError: Engine disconnected without a terminal result"
},
"drained_normal_response": false
```

[worker1 日志第17–20行](evidence/2026-10-16-sglang-backend/worker1.log#L17-L20)记录 `SIGTERM received`，继而 scheduler 以 `-15` 退出并触发清理。对应[后端第128–150行](../research/backend-contract/candidate/sllm/backends/sglang_backend.py#L128-L150)中，`stop()` 调用 `shutdown()`，后者终止所属进程组；不是等待请求自然完成后再关闭。

上述源码、调用步骤和结果共同支持“当前 stop 未保留原方案排空语义”的结论。日志中的 `Draining requests and shutting down` 是引擎收到信号后的日志文字，不能替代请求成功排空的结果证据；`generation_finished_on_stop_return: true` 也只表示任务已结束，错误结束仍算结束。

**③ 重放/预热：候选方法直接未实现，没有发生一次成功重放。**

[脚本第109–113行](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L109-L113)重新启动后端，调用 `resume_kv_cache([input_ids])` 并捕获未实现异常。[结果第207–208行](evidence/2026-10-16-sglang-backend/result.json#L207-L208)记录 `"op": "resume_token_replay"`、`"result": "not_implemented"`，与[后端第158–159行](../research/backend-contract/candidate/sllm/backends/sglang_backend.py#L158-L159)一致。

这里输入是固定 prompt，不是源实例的活动快照。因此证据能说明候选恢复入口缺失，不能把它描述成“完整跨实例迁移跑到中途失败”，也不能据此判断真实缓存命中。

**④ 活动请求 shutdown：同样没有返回约定的请求终态。**

[脚本第114–118行](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L114-L118)发起另一活动请求并确认首 chunk，然后调用 `shutdown()`。[结果第211–230行](evidence/2026-10-16-sglang-backend/result.json#L211-L230)记录该请求已开始输出，随后返回同样的缺失终态错误。[worker2 日志第8–11行](evidence/2026-10-16-sglang-backend/worker2.log#L8-L11)记录终止信号和 `-15` 退出。

这证明当前实现没有给活动请求正常的取消终态，不能据此认定 SG 没有请求级 abort 接口；[请求传输实现](../research/backend-contract/candidate/sllm_sglang/runtime.py)另有断连时调用原生 abort 的路径，它与后端整体 shutdown 的语义不同。

**⑤ 观察程序成功退出，不代表上述功能通过。**

[实验进程日志](evidence/2026-10-16-sglang-backend/experiment-process.log)保留线程栈及最终退出记录。其中 `Timeout (0:00:45)!` 来自[脚本第30行](evidence/2026-10-16-sglang-backend/backend_gpu_probe_v4.py#L30)的 `faulthandler.dump_traceback_later(45, repeat=True)`，是定时诊断，不是本次断连的错误根因。末尾 `Finished with result: success` 和[结果状态](evidence/2026-10-16-sglang-backend/result.json#L234)只表示观察流程返回并保存结果；测试程序会把不支持的能力和错误响应作为观察数据保存。

[普通请求结果](evidence/2026-10-16-sglang-backend/result.json#L7-L179)与这些失败记录位于同一文件，避免只展示失败片段。另有 [Store 日志](evidence/2026-10-16-sglang-backend/store-server.log)和 [104 条资源采样](evidence/2026-10-16-sglang-backend/gpu-samples.json)用于核对运行过程；本轮仍未启动 Controller/Ray/HTTP 全链。上述脚本作为历史实验材料保存，脱敏后不能直接执行复现，具体缺失环境见证据目录说明。

## 4. 尚未完成的适配及建议验收条件

以下是后续实施工作，不是已经承诺实现的能力或工期。

| 未完成工作 | 当前缺口 | 可检查的完成条件 |
|---|---|---|
| 合并研究改动为正式候选 | 根目录与实测副本架构、dtype、容量和监督门槛不同 | 选择统一配置入口，冻结版本和源码，在同一提交完成准备与真实运行验证 |
| 自动模型准备与注册 | SG 目前只接收预先存在的 HF 布局 Store 文件；旧 vLLM rank 布局不能直接复用 | 自动准备、索引/张量校验、Store 注册和恢复生成全部贯通；失败产物可识别和回收 |
| Store 写入和加载容量 | 原生 writer 问题未修复；源权重/目标模型共存占显存 | 验证写入完整性；降低或明确加载峰值，并用目标模型测量，不能只靠估算 |
| 请求级生命周期 | 有断连 abort 基础代码；stop 仍结束 worker；缺活动快照及正常取消终态 | 拒绝新请求、活动 token 跟踪、排空完成、主动 abort、重复关闭分别通过真实请求验证 |
| 原版迁移/恢复 | `get_current_tokens()`、`resume_kv_cache()` 未实现，SG 未接上 router 期待的跨实例续写协议 | 源端真实活动快照送往目标；输出无重复/丢失，剩余预算、终态和缓存预热行为可核对 |
| 完整服务与 HTTP | 只有后端对象/内部 chunk 及模拟测试证据 | Controller → Ray → Store → SG → HTTP 全链通过；覆盖 SSE、客户端断连、超时和异常启动 |
| 通用请求和模型功能 | SG 当前仅 raw IDs/贪心子集；Embedding、Chat 模板、LoRA、多模态等未完成 | 根据首批真实需求逐项定义输入/输出语义并回归；不能将旧后端接口直接等同于已支持 |
| 模型与并行范围 | MoE、量化、TP 多卡、长上下文和更多架构未验收 | 按固定 checkpoint、dtype、TP、上下文、输入类型验证原生/Store 一致性及容量 |
| 资源监督与可部署性 | 仍带实验专用门槛和外部依赖；MPS、全异常回收不足 | 提供可构建依赖和配置，验证启动失败、并发、取消、重启后的进程/显存/Store 资源释放 |
| 性能和负责人反馈闭环 | 缺完整启动时间分解、可信 HTTP 首输出/生成时延、本地盘/Ceph配对、观测开销对照 | 固定模型/缓存/资源/预热条件，记录完整时间线和用户端指标，再做公平基线比较 |

迁移路由代码存在，不等于任意引擎已实现完整迁移。对照基准中的旧 vLLM 后端自身也未完整衔接 router 所期待的 `resume_generate` 等协议，后续必须区分原方案目标、旧代码实际行为与新候选已验证行为。

## 5. 负责人可以据此作出的判断

已有证据足以支持继续投入 SGLang 通用后端和 Store 适配，没有显示需要逐模型重写执行算子。但它还不足以替代正式 Baseline，也不足以给出总工期或性能优劣结论。

下一步宜先确定实验是否必须统一 SGLang，以及首批模型和运行规格；再按上表补齐统一候选、自动准备、请求生命周期和全链验收。扩大模型或开展性能对照之前，应先完成这些直接影响结果可信度的工作。
