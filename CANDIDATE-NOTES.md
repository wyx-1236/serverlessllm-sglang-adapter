# ServerlessLLM SGLang candidate12

Development candidate, not a qualified GPU baseline. Source and frozen historical
results are preserved in candidate11 and earlier directories.

Implemented: SGLang backend selection, a Ray-owned modern interpreter child,
private socket streaming and cancellation, raw-token HTTP adapter, explicit cache
and capacity configuration, and a fail-closed custom Store loader. The Store file
layout remains `transformers/<model>`; execution is native SGLang. Store registration
admits existing verified files only. No HF weight-loading fallback is allowed.

Current scope is Qwen2/Qwen3, BF16, TP1, explicit raw IDs, greedy sampling and
bounded 1..64 output tokens. Text chunks and token chunks are separate; no token
timing is synthesized. Chat templates, embeddings, LoRA, migration, TP2 and KV
resume are not qualified. The HTTP adapter is specific to this candidate.

The actor runs in the old control-plane interpreter and the child in the same
SGLang environment as CrossPool. Runtime configuration must provide the fresh
parent interval, registered unit, exact GPU binding/deadline, private socket/log
directory, runtime interpreter/environment/Python paths, package versions and
source hashes, explicit Store endpoint/root/index identity, EOS/PAD and capacities.
The external supervisor must bound the entire process tree and audit final CUDA
and private MPS release. Missing admission never launches an Engine.

CPU checks:

```text
python -m unittest discover -s tests -v
python -m compileall -q sllm_sglang sllm
```

Synthetic Engine/socket tests qualify protocol/lifecycle behavior only. Real
SGLang constructor/loader-factory checks and the modern Store client build/import
have separate remote CPU evidence in the parent results directory. GPU correctness,
Store IPC, full controller/HTTP path, resource sampler and release supervisor still
require their stated acceptance. No current GPU authorization exists.

Detailed phase goals and acceptance are maintained in the research vault note
`2026-10-15 - ServerlessLLM反馈优先执行与阶段验收计划.md`; transient execution
state is `../results/SGLANG-PHASED-IMPLEMENTATION-STATE-20261015.json`. Do not treat
historical Transformers comparison data as the new SGLang baseline.
