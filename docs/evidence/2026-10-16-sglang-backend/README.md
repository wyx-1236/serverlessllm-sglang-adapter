# SGLang 后端停止、跟踪与重放：服务器实验记录

实验日期：2026-10-16。证据分析：[停止、跟踪与重放](../../sglang-adaptation-report.md#31-停止跟踪与重放问题的逐项原始证据)。

## 文件清单

| 文件 | 用途 |
|---|---|
| [result.json](result.json) | 完整请求结果记录：普通生成、活动跟踪与 stop、重放、活动 shutdown，包含首 chunk、返回错误和输入 token |
| [worker1.log](worker1.log) | 第一个 SG worker 的 Store 加载、生成和 stop 期间退出日志 |
| [worker2.log](worker2.log) | 重新启动的第二个 SG worker 的加载和 shutdown 期间退出日志 |
| [store-server.log](store-server.log) | 同一轮 Store 服务日志，辅助核对权重加载与服务退出过程 |
| [experiment-process.log](experiment-process.log) | 实验进程的定时线程栈、最终状态及 service 退出记录；原文件名见 MANIFEST |
| [gpu-samples.json](gpu-samples.json) | 104 条周期资源采样；`epoch` 为采样时间，`value` 为 GPU 显存 MiB 与利用率百分比，`memory_current` 为 cgroup 内存字节数；不是瞬时峰值 |
| [backend_gpu_probe_v4.py](backend_gpu_probe_v4.py) | 实际使用的观察脚本：请求构造、活动状态确认、stop/shutdown 调用及异常捕获 |
| [MANIFEST.json](MANIFEST.json) | 来源相对路径、原件/公开副本 SHA256、文件字节数和脱敏替换次数 |

对应的[后端源码快照](../../../research/backend-contract/candidate/sllm/backends/sglang_backend.py)已在仓库中，可与日志和脚本对照。

## 原始性与公开处理

本地原件保持不动。公开副本保留每个文件的全部行、顺序、时间、错误文本、请求输出、PID 和结果字段，只替换两类环境身份信息：服务器账号路径替换为 `/home/LAB/research-user`，物理 GPU UUID 替换为 `GPU-00000000-0000-0000-0000-000000000000`。因此有替换的文件不是原件的字节相同副本；MANIFEST 分别给出两种摘要。未替换文件的两种摘要相同。

归档文件包含文本日志、JSON 结果和脚本，无截图。`NotImplementedError` 被脚本捕获后记为 `not_implemented`，对应位置没有完整未捕获异常堆栈。线程栈日志中的 `Timeout` 来自周期诊断，不等于单个请求已超时。

## 脚本使用边界

脚本保留当次实验步骤；独立运行所需材料尚未全部打包。环境路径和物理设备标识已脱敏，原生参考结果、Store 权重、现代解释器/客户端和当时的资源监督配置也未全部打包。没有上传历史 GPU 授权文件；历史授权不能用于新的 GPU 运行。

结果状态 `backend_observation_returned`、进程日志中的 `Finished with result: success` 表示观察程序完成并保存结果，不表示所有适配能力通过。具体失败必须看 `operations` 各项。
