# flash_attn → vLLM 内核 shim 验证报告（结论：不安装）

日期：2026-07-19　venv：`/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/.venv-vllm023`
（torch 2.11.0+cu130, py3.12, vllm 0.23.0, transformers 5.12.1）

## 一句话结论

**验证失败 —— shim 未安装，venv 保持原样（`import flash_attn` 仍报 ModuleNotFoundError，
`is_flash_attn_2_available()==False`，verl actor 继续走 sdpa+fla，安全）。**

前向数值**全部通过**（bf16，max|diff| ≤ 1.07e-2 < 2e-2 容差），但 **反向传播 FAIL**：
vLLM 装的是**只有前向、没有反向**的 flash-attn 编译版（`_vllm_fa2_C.abi3.so` 内含字符串
`"This flash attention build does not support backward."`，且只注册了 `varlen_fwd`，
没有任何 `*_bwd` op）。用它做 actor 训练的注意力前向，反向时梯度会**静默变成 None/NaN**，
比 sdpa 更糟。因此按任务要求**不留下这个错误 shim**。

## 做了什么

在 scratch 里（**从未写进 site-packages**）构建了一个纯 Python `flash_attn` 包：
- `__version__="2.8.3"` + 完整 `flash-attn-2.8.3.dist-info`（METADATA `Name: flash-attn` /
  `Version: 2.8.3` / `top_level.txt=flash_attn`），使 transformers 的版本门通过。
- `flash_attn_varlen_func(...)`：**真 flash-attn 的参数签名**，把参数按**名字**转发给
  `vllm.vllm_flash_attn.flash_attn_varlen_func`（注意 vLLM 位置序是
  `q,k,v,max_seqlen_q,cu_seqlens_q,max_seqlen_k,cu_seqlens_k,...`，与真 FA 不同，全部走 keyword）。
  `window_size (-1,-1)→None`（真 FA 用 (-1,-1) 表示全注意力，vLLM 用 None）。默认返回单个 `out`。
- `flash_attn_func(...)`：vLLM 无此函数；由 (batch,seqlen) 构造 cu_seqlens 后委托 varlen。
- `flash_attn_with_kvcache`：占位（transformers FA2 分支会无条件 import 此名，调用即 raise）。
- `bert_padding`（index_first_axis/pad_input/unpad_input/rearrange）与
  `ops.triton.cross_entropy.cross_entropy_loss`：直接从 tarball
  `/scratch/gpfs/CHIJ/bohan/fs/wheelhouse/flash_attn-2.8.3.post1.tar.gz` 拷贝（纯 torch/纯 triton，无需编译）。

签名设计与 transformers 的 `supports_mapping` 机制对齐：`modeling_flash_attention_utils.py`
的 `_lazy_define_process_function` 会 `inspect.signature(flash_varlen_fn)` 决定往下传哪些 kwarg，
因此 shim 显式暴露了 `dropout_p/softmax_scale/causal/window_size/softcap/deterministic/
max_seqlen_q/max_seqlen_k/cu_seqlens_q/cu_seqlens_k`（否则 max_seqlen 等会被丢弃）。

源码留存（惰性、**不在任何 import 路径上**）：
`/scratch/gpfs/CHIJ/bohan/fs/innovation_prior/experiments/flash_attn_shim_forward_only_DO_NOT_INSTALL/`
（含 `shim/` 与 `gpu_test.py`）。

## 验证数值（H200，bf16，headdim=128，容差 2e-2）

| 用例 | causal | Hq/Hk | max&#124;diff&#124; vs sdpa | 结论 |
|---|---|---|---|---|
| varlen MHA | False | 8/8 | 7.88e-3 | PASS |
| varlen GQA | False | 8/2 | 4.29e-3 | PASS |
| varlen MHA | True  | 8/8 | 8.47e-3 | PASS |
| varlen GQA | True  | 8/2 | 8.49e-3 | PASS |
| flash_attn_func | False | 8/8 | 4.05e-3 | PASS |
| flash_attn_func | True  | 8/8 | 7.77e-3 | PASS |
| padding round-trip (unpad→varlen→pad, 仅有效 token) | True | 8/8 | 1.07e-2 | PASS |
| **反向 backward（varlen，requires_grad）** | — | — | q.grad = **None/NaN** | **FAIL** |
| transformers `is_flash_attn_2_available()` | — | — | **True** | PASS |
| transformers `_flash_attention_forward` 反向 | — | — | qs.grad **非有限** | **FAIL** |

反向时 PyTorch 明确告警：
`_vllm_fa2_C::varlen_fwd: an autograd kernel was not registered ... trying to backprop
through it. This may lead to silently incorrect behavior.` —— 即梯度不回传（actor 更新会被静默破坏）。

## 根因

- `vllm/vllm_flash_attn/flash_attn_interface.py` 的 `flash_attn_varlen_func` 只调用
  `torch.ops._vllm_fa2_C.varlen_fwd(...)`（推理引擎，前向 only）。`_vllm_fa2_C` 命名空间实测只有
  `varlen_fwd`，无 `varlen_bwd/bwd/mha_bwd`；`.so` 里 `mha_bwd` 是抛 "does not support backward" 的桩。
- 真 flash-attn（tarball `flash_attn/flash_attn_interface.py`）把 fwd+bwd 包在
  `torch.autograd.Function`（`FlashAttnVarlenFunc`，`backward` 调
  `torch.ops.flash_attn._flash_attn_varlen_backward`）里 —— 这个反向 CUDA 内核 vLLM 根本没编。
  无法只靠 Python shim 补出来。

## 为什么绝不能留着这个 shim（安全告警）

`scripts/run_verl_grpo_frontiercs_qwen35_9b.sh:176-186` 有**自动探测**：
```
if python -c "import flash_attn" >/dev/null 2>&1; then ACTOR_ATTN_IMPL="flash_attention_2"
else ACTOR_ATTN_IMPL="sdpa"; fi
```
一旦 shim 可 import，这个脚本会**自动**把 actor 切到 `flash_attention_2` → 训练梯度静默清零。
其余 RL 脚本（`run_verl_grpo_frontiercs_qwen35_27b.sh:83`、`..._9b_hardtest.sh:103` 等）
均硬编码 `attn_implementation=sdpa`。这更加坐实：forward-only shim 必须不安装。

## 想开启 flash_attention_2 的那一行 verl flag（仅当装上"真"内核后才可用）

verl actor 走：
```
+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2
```
替换当前的 `...=sdpa`（例如 `scripts/run_verl_grpo_frontiercs_qwen35_9b.sh:225` 用变量
`$ACTOR_ATTN_IMPL`，其余脚本第 83/103/107/... 行硬编码 sdpa）。
**当前 venv 下不要改**：真内核未就位前改成 flash_attention_2 会破坏训练。

## 干净路径（推荐）

1. **离线源码编译真 flash-attn**（唯一能同时给出前向+反向的正道）：仓库已有
   `scripts/build_flash_attn_cpu.sh`（`sbatch`，从 tarball `--no-index` 装，`TORCH_CUDA_ARCH_LIST=9.0`
   给 H200 交叉编译）。其注释指出 torch2.11+cu130 无预编译 wheel、且历史上 ABI 编译可能失败；
   若成功，上面的自动探测会自然选中 flash_attention_2，Ulysses-SP+rmpad 全链路即恢复。
   （wheelhouse 里的 `flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311` 预编译 wheel
   **不可用**：cp311≠cp312、torch2.8≠2.11、cu12≠cu13。）
2. **或维持现状 sdpa + fla 0.5.1**（已装、可反向、正确，只是慢）——零风险。

## 关键文件引用

- vLLM 前向 only：`.venv-vllm023/.../vllm/vllm_flash_attn/flash_attn_interface.py`（`varlen_fwd` 调用），
  `_vllm_fa2_C.abi3.so`（字符串 "This flash attention build does not support backward."）。
- transformers 版本门 / 调用约定：`.venv-vllm023/.../transformers/modeling_flash_attention_utils.py`
  （`_lazy_imports:163` import `flash_attn_func/varlen/with_kvcache`；`_flash_attention_forward` 以
  keyword 传 `cu_seqlens_q/cu_seqlens_k/max_seqlen_q/max_seqlen_k`），
  `transformers/utils/import_utils.py`（`is_flash_attn_2_available` 需 `packages_distributions()`
  能把 `flash_attn`→`flash-attn` 且 metadata 版本 ≥2.3.3）。
- verl import 面：`verl/utils/attention_utils.py:31`、`verl/utils/torch_functional.py:34`（cross_entropy）、
  `verl/models/transformers/{qwen2_vl,glm4v}.py:46`（`flash_attn_func/varlen`）。
- 自动探测开关：`scripts/run_verl_grpo_frontiercs_qwen35_9b.sh:176-186`；源码构建：`scripts/build_flash_attn_cpu.sh`。
