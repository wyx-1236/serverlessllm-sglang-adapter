# ServerlessLLM SGLang Adapter

保存 ServerlessLLM 接入 SGLang 的研究适配代码和服务器实测使用的代码快照。当前是研究原型，完整迁移尚未验收。

## 目录

- `sllm/`：包含 SGLang 后端接入的 SLLM 源码快照。保留相关原版模块，使调用关系可追溯。
- `sllm_sglang/`：worker、进程通信、请求转换、流式适配、运行配置及自定义 Store loader。
- `tests/`：candidate12 已有的 CPU 单元测试。
- `research/backend-contract/candidate/`：2026-10-16 真实后端行为实测归档中的源码副本，独立保存，不覆盖 candidate12。
- `research/qwen35-store/sllm_sglang/`：Qwen3.5 Store 恢复生成实测使用的 loader 副本。
- `SOURCE-MANIFEST.json`：每个 Python 文件的来源快照及 SHA256，所有源文件按原字节保存。
- `CANDIDATE-NOTES.md`：准备候选原始说明，反映当时阶段状态；其中外部笔记和结果路径未包含在本仓库。

三个快照代表不同阶段，不能合并解释成一个已经通过全部验证的版本。研究目录中的同名模块请在独立 Python 路径下使用，避免与根目录模块混用。

## 已有实测与边界

历史评估使用 SGLang 0.5.21，在 A100 40GB、BF16、TP1、context 512、文本输入和关闭图执行条件下，完成 Qwen3-0.6B 和 Qwen3.5-9B 的原生加载及真实 SLLM Store 恢复生成。Qwen3.5 四组短请求的原生/Store token 和 top5 logprobs 一致。未修改这两个模型的执行源码。

这些是历史有限配置的结果，本仓库上传时没有重跑 GPU 实验。原始实验日志、模型权重、运行环境及凭据不包含在仓库中。

已测的真实后端普通生成和计量正确；活动 token 跟踪、重放/恢复、排空停止和活动请求关闭语义尚未完整适配。Controller → Ray → Engine → HTTP 全链、自动模型准备注册、多卡、量化、MoE、多模态及公平性能对照仍未完成验收。Store loader 当前可能同时保留源 GPU 权重和目标模型，存在额外显存压力。

## CPU 检查

在仓库根目录执行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q sllm_sglang sllm research
```

这些测试使用模拟引擎/张量检查协议及失败处理，不能证明 GPU 推理或完整服务已通过。

## 运行依赖与集成

本仓库是源码归档，不是可直接安装的一体化发行包；没有提供经过验收的锁定环境或自动部署入口。完整运行还需 SLLM 控制面环境、现代 SGLang/Torch 环境、与其 ABI 匹配的 SLLM Store 客户端、已有 Store 文件以及外部进程监督。两套解释器通过子进程和私有 socket 通信。

Store 文件采用 `transformers/<model>` 的 HF 原始权重布局，不是旧 vLLM 处理后的 rank 布局。代码通过 SGLang 原生 `load_weights()` 处理权重，不允许静默退回 HF 权重加载。

运行配置需要显式提供模型/Store 路径、张量索引摘要、解释器及环境、GPU 绑定、运行期限和监督记录等。仓库不携带历史运行授权；运行前应重新落实资源授权与配置。

## 来源与许可证

SLLM 代码来源于 [ServerlessLLM](https://github.com/ServerlessLLM/ServerlessLLM)，参考本地上游提交 `9f50241baa5386e06a9321c51f19a9ef5f964c2b`，并包含研究阶段的后端接入修改和新增模块。保留原始版权声明及 Apache-2.0 `LICENSE`。原始候选和实测副本均未因本次归档修改执行逻辑。
