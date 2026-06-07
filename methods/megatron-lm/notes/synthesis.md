# Megatron-LM synthesis (Phase 1.5)

## Verified
- arXiv 1909.08053, "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism". Title verified from source.
- Canonical impl: NVIDIA/Megatron-LM core/tensor_parallel — mappings.py (_CopyToModelParallelRegion = f operator: identity fwd / all-reduce bwd; _ReduceFromModelParallelRegion = g operator: all-reduce fwd / identity bwd) and layers.py (ColumnParallelLinear, RowParallelLinear). Confirms paper's f/g code exactly.

## Pain point
- Models (BERT, GPT-2) exceed single-accelerator memory. Adam stores momentum+variance per param (2x extra memory), activation checkpointing helps but not enough. Need to put weights + optimizer state across multiple GPUs.
- Data parallelism (split minibatch) doesn't help: requires the whole model to fit on ONE worker.
- Existing model-parallel frameworks: GPipe (pipeline parallelism — bubbles, optimizer changes), Mesh-TensorFlow (needs a new language + compiler), parameter-server pipeline (consistency issues). All require rewriting the model / custom compilers.
- GOAL: intra-layer ("tensor") model parallelism that's SIMPLE — a few synchronization primitives inserted into existing PyTorch, no compiler, no model rewrite. Orthogonal to (composable with) pipeline + data parallelism.

## Key structural fact exploited
- A transformer layer = self-attention block + 2-layer MLP. Both are dominated by GEMMs (matrix multiplies). Partition the GEMMs across P GPUs so each holds a slice of the weights, with minimal communication.

## MLP block derivation (DERIVE both options, show why column-then-row wins)
- MLP: Y = GeLU(X A), then Z = dropout(Y B). X is [b*s, H]; A is [H, 4H]; B is [4H, H].
- Option 1 — split A along ROWS, X along COLUMNS: X=[X1,X2], A=[[A1],[A2]]. Then XA = X1A1 + X2A2.
  But GeLU is nonlinear: GeLU(X1A1 + X2A2) ≠ GeLU(X1A1) + GeLU(X2A2). So you must all-reduce (sum the partial products) BEFORE GeLU → a synchronization point in the middle of the block. Bad.
- Option 2 — split A along COLUMNS: A=[A1,A2]. Then XA = [XA1, XA2], and GeLU applies elementwise/independently per column block:
  [Y1, Y2] = [GeLU(XA1), GeLU(XA2)]. No communication before GeLU. Each GPU i computes Yi = GeLU(X Ai) entirely locally (X is replicated). 
- Then second GEMM B split along ROWS: B = [[B1],[B2]]. Each GPU computes Yi Bi locally (Yi is the local GeLU output, no comm needed). Output Z_i = Yi Bi; the full output is Z = Y1B1 + Y2B2 = sum_i Yi Bi → ONE all-reduce to sum across GPUs, then dropout.
- Net: 1 all-reduce in forward (g operator, sum the Yi Bi), 1 all-reduce in backward (f operator, sum gradient of X). The "split first GEMM column-wise, second GEMM row-wise" fusion eliminates the intermediate sync.

## Self-attention block derivation
- Multi-head attention is inherently parallel ACROSS HEADS. Partition Q,K,V projections COLUMN-wise so that each GPU holds a subset of the heads (whole heads, not split within a head). Then the per-head attention (softmax(QK^T/sqrt(d))V) for those heads is computed entirely locally — no communication needed to complete self-attention.
- The output projection (linear after attention) is split ROW-wise, taking the per-GPU attention output directly; the full output = sum of per-GPU contributions → ONE all-reduce. Same column-then-row structure as MLP.
- Per layer: 2 all-reduces forward (one after attention block's output proj = g, one after MLP's = g) + 2 all-reduces backward (the two f's). Total 4 communication ops per layer.

## The f / g conjugate operators (the core abstraction — DERIVE the conjugacy)
- Define two operators wrapping the model-parallel region:
  - f: forward = identity; backward = all-reduce. (Sits at the INPUT of a column-parallel region: forward just copies the replicated input X to each GPU; backward must SUM the gradients dL/dX flowing back from all P column-shards, since X feeds all of them.)
  - g: forward = all-reduce; backward = identity. (Sits at the OUTPUT of a row-parallel region: forward SUMS the partial outputs Yi Bi across GPUs; backward just copies the gradient to each GPU since each shard's output is the same summed Z.)
- They are conjugates: f does (identity fwd, all-reduce bwd), g does (all-reduce fwd, identity bwd) — mirror images. In code, f is a torch.autograd.Function: forward returns x; backward all-reduces the gradient.
- A model-parallel region is bracketed: X → f → [column-parallel GEMM, local nonlinearity, row-parallel GEMM] → g → output. f handles backward sync, g handles forward sync.

## Output/input embedding parallelism
- Embedding E is [H, v], v ~ 50257 (huge). Parallelize the embedding GEMM along the VOCAB dimension: E = [E1, E2] (column-wise in v). Input embedding: after lookup, an all-reduce (g) since each partition holds part of the table.
- Output embedding (tied weights): parallel GEMM [Y1,Y2]=[XE1, XE2] gives logits sharded over vocab. Naive: all-gather Y over vocab → communicates b*s*v elements (huge). Instead FUSE the parallel GEMM with cross-entropy: compute the loss locally and only communicate scalar losses → b*s elements. Big communication reduction.

## Duplicated (not communicated) ops
- Dropout, layer norm, residual connections are DUPLICATED on every GPU (cheap, elementwise) rather than split-and-broadcast. LayerNorm params kept as duplicate copies per GPU. Each worker optimizes its own params; no parameter communication needed since values are local or duplicated.

## (Also in paper, lighter) LayerNorm placement for BERT
- Scaling BERT naively degrades; rearranging layer norm + residual (pre-LN style) makes accuracy improve monotonically with size. (Mention as a design note; the core is the tensor parallelism.)

## Numbers (motivating/system, in scope as method capability not benchmark win)
- 1.2B on single V100 32GB at 39 TFLOPs (30% peak). 8.3B on 512 GPUs, 8-way model parallel → 15.1 PFLOPs, 76% scaling efficiency. ~1B params/GPU weak scaling.

## Scaffold ↔ final code correspondence
- f operator (copy_to_model_parallel_region) ← context stub: identity-fwd/allreduce-bwd primitive
- g operator (reduce_from_model_parallel_region) ← context stub: allreduce-fwd/identity-bwd primitive
- ColumnParallelLinear (split weight along output dim, X replicated) ← context stub: column-sharded linear
- RowParallelLinear (split weight along input dim, output all-reduced) ← context stub: row-sharded linear
- ParallelMLP (column then row + g) ← context stub: MLP block
- ParallelSelfAttention (QKV column by heads, output proj row + g) ← context stub: attention block

## OUT of scope (eval)
- WikiText103/LAMBADA/RACE SOTA numbers, the BERT-vs-GPT2 accuracy-vs-size experiments. The system scaling numbers describe the method's capability (mention lightly), but the downstream accuracy wins are out.
