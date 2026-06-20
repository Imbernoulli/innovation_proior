What this record changed: replaced the materialized-score attention with **cuDNN Flash Attention** (the
frontend `sdpa` op), fusing QKᵀ → causal softmax → ·V into one tensor-core kernel that never materializes the
(B,NH,T,T) score matrix in HBM (forward or backward — backward recomputes scores from Q,K,V and the small saved
softmax `stats`). BF16 I/O, FP32 intermediate/compute; the graph is shape-keyed and cached because building is
slow.

Measured / documented numbers (the repo's own statements):

| record | figure | source |
|---|---|---|
| flash attention source | "As of May 1, 2024 we use the **Flash Attention from cuDNN**" | `README.md` (flash attention section) |
| compile-time cost (why it's opt-in) | "cuDNN bloats the compile time from a few seconds to **~minute**" | `README.md` |
| enable flag | `make train_gpt2cu USE_CUDNN=1` | `README.md`, `Makefile` |
| precision | default **BF16** (FP16 selectable), FP32 unsupported for cuDNN | `llmc/cudnn_att.cpp` (`CUDNN_16BIT`, `static_assert(false, "cuDNN is not supported in FP32 mode.")`) |
| backward avoids materialization | output (B,T,NH,HS) + stats (B,NH,T) only; no (B,NH,T,T) tensor | `llmc/cudnn_att.cpp` (graph outputs) |

The repo states cuDNN Flash Attention is the attention path used in the mainline mixed-precision trainer (as of
May 1, 2024) and documents it as a compile-time option (`USE_CUDNN=1`) precisely because of the ~1-minute
compile cost of the heavy dependency. The repo does not publish an isolated end-to-end tokens/sec delta for
toggling flash attention alone on the 124M run; the per-rung evidence is (a) that the mainline adopts the cuDNN
flash kernel, and (b) the rejected hand-rolled minimal flash attempt being ~3× slower than naive
(`dev/cuda/attention_forward.cu` version 2), which is why the vendor kernel is used. The math is exact
scaled-dot-product causal attention, so the 3.29 target is held.
