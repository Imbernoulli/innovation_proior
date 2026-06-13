# Context: efficient training of fine-grained, sparse Mixture-of-Experts layers

## Research question

A Mixture-of-Experts (MoE) layer replaces the dense feed-forward (channel-mixer) block of a transformer with a router plus many smaller feed-forward sub-networks ("experts"). Each token is routed to only K of the E experts, so the layer holds the parameters of a very large model while spending the FLOPs of a small one. The practical pull is toward **fine-grained** MoEs — shrink each expert's intermediate width n (raise the granularity G = d/n, where d is the model embedding width) and proportionally raise the number of activated experts K so total FLOPs stay fixed — and toward **sparse** MoEs — raise the total expert count E while holding K fixed, lowering the activation ratio ρ = K/E. Both directions improve model quality per FLOP.

The problem: the theoretical FLOP saving does not turn into wall-clock training throughput. The goal is a way to train fine-grained, sparse MoE layers that (i) keeps activation memory from growing as experts get finer, (ii) keeps the matrix-multiply units near peak even though the layer is drifting into the memory-bound regime, and (iii) stops wasting compute on padding when each expert receives few tokens — all while leaving the MoE's mathematics (and therefore model quality) untouched, and without assuming a particular router.

## Background

**The MoE layer.** With router scores S ∈ R^{T×E} over T tokens and E experts, top-K token-choice routing sends each token to its K highest-scoring experts. Writing X_e ∈ R^{T_e×d} for the rows routed to expert e, a SwiGLU expert computes an up-projection H_e = X_e W1_e ∈ R^{T_e×2n}, an activation A_e = SwiGLU(H_e) ∈ R^{T_e×n} (SwiGLU splits H into a gate and a value half and returns SiLU(gate)⊙value; gated activations improve transformer MLP quality, Shazeer 2020), a down-projection Y_e = A_e W2_e ∈ R^{T_e×d}, and an aggregation O_t = Σ_e s_{t,e} Y_{e,t}. The forward and backward together cost F = 18·T·K·n·d FLOPs (6 forward, 12 backward).

**Grouped GEMM.** Because experts receive different token counts, the per-expert matmuls are batched into a *grouped GEMM*: a list of GEMMs sharing the weight dimensions (N,K) but with variable token dimension M ("varlen-M") in the forward and activation-gradient passes, and variable contraction dimension K ("varlen-K") in the weight-gradient passes. Inputs are either gathered from scattered token positions or already contiguously packed. On NVIDIA Hopper and Blackwell GPUs a GEMM runs asynchronously in a producer-consumer style: producer warps stream tiles from global memory (HBM/GMEM) to shared memory (SMEM); consumer warpgroups read those tiles, issue tiled matrix-multiply-accumulate (MMA) instructions on the tensor cores, accumulate over the contraction dimension (mainloop), then in the epilogue apply pointwise math and write results back to HBM. Tensor cores compute in fixed tiles (tileM, tileN, tileK) and pad any dimension not divisible by the tile.

**Roofline / arithmetic intensity.** Arithmetic intensity (FLOPs ÷ bytes transferred) decides whether a kernel is compute-bound or memory-bound. For one expert's forward, counting 2·T_e·2n·d + 2·T_e·n·d FLOPs against 4T_e·n + 6n·d + 4T_e·d bytes and substituting T_e = Tρ, G = d/n gives an intensity of 3 / ( (2 + 2G)/d + 3/(Tρ) ). For fixed d this *falls* as granularity G rises or as the activation ratio ρ falls — the IO cost per FLOP scales linearly with granularity. So the more fine-grained or sparse the MoE, the further it slides below the roofline ridge into the memory-bound regime, where the matmuls cannot reach peak unless the IO is hidden.

**Motivating measurements about existing systems.** (1) *Activation memory.* For fixed FLOPs F = 18TKnd and fixed d, T, the product nK is constant, so raising granularity lowers n and raises K. Any cached activation of size O(TKd) — the down-projection output Y, or the gathered inputs X_e and dO_e — therefore grows *linearly* with granularity; existing grouped-GEMM MoE kernels cache such tensors and run out of memory as experts get finer. (2) *Backward IO.* In the backward pass the weight-gradient kernels must gather X and dO and the activation-gradient kernel must gather dO; existing kernels fuse the gather in the forward pass but launch a *separate* gather kernel in the backward pass, paying an extra 2TKd bytes of traffic. (3) *Store stalls.* On Hopper, the only non-TMA instruction for a scatter-fused HBM store is the *synchronous* `st.global`, which blocks the next MMA tile; measured throughput drops ~20% on store-heavy epilogues, and warpgroup ping-pong cannot hide a synchronous store. (4) *Tile-quantization waste.* When ρ is small the expected tokens per expert T̄_e = Tρ shrinks (e.g. 16384 tokens but only 512 per expert at K=4, E=128), so each expert's last GEMM tile is mostly padding. The exact per-expert padding fraction on the token dimension is (⌈T_e/tileM⌉·tileM − T_e)/T_e; if the residue is roughly uniform, its expectation is about tileM/(2Tρ), hence O(tileM/(Tρ)). (5) *Router overhead.* The PyTorch top-K used by existing MoE kernels can be ~40% of router time.

## Baselines

**Sparsely-gated MoE / Switch Transformer (Shazeer et al. 2017; Fedus et al. 2022).** Introduced the router + experts with top-K token-choice routing. To obtain static shapes Switch used a fixed per-expert *capacity*: experts under capacity are padded (wasted FLOPs) and tokens over capacity are dropped (quality loss). Gap: capacity forces a padding-vs-dropping trade-off.

**MegaBlocks (Gale et al. 2023).** Casts MoE as a *block-sparse* matmul, eliminating both capacity padding and token dropping ("dropless"). The block-sparse kernels (via the STK library) are slower and more complex than dense grouped GEMM, and the gather + pad + scatter steps move on the order of 8TKd bytes. Gap: dense grouped GEMM is faster and simpler, and the IO is still unhidden.

**ScatterMoE (Tan et al. 2024).** A ~700-line Triton implementation built on a `ParallelLinear` primitive: a grouped GEMM with the input gather *fused* into the forward and the output scatter fused into the store, avoiding padding and input copies. It computes the score gradient as dS_{t,e} = ⟨dO_t, Y_{e,t}⟩, which requires the down-projection output Y to be cached (2TKd bytes) and reloaded. Gaps: (i) caching Y makes activation memory grow with granularity; (ii) gather is fused only in the forward, with a separate gather kernel in the backward; (iii) the scatter-fused output store needs the synchronous `st.global`; (iv) being on Triton it cannot express fine-grained asynchronous TMA loads/stores, warpgroup ping-pong, or heavy-epilogue overlap; (v) no gather fusion for varlen-K.

**MoMoE (Costin et al. 2025).** A memory-optimized Triton MoE; it fuses the dS computation into the up-projection activation-gradient kernel but still uses dS = ⟨dO, Y⟩ (caching Y), and it materializes the gathered dO_e and X_e in the backward pass (both scale with granularity). Same Triton-imposed lack of async TMA / ping-pong as ScatterMoE; its scatter-style aggregation is slow.

**DeepGEMM (DeepSeek, 2025).** A highly optimized grouped GEMM for *contiguously-packed, 128-padded* inputs, specialized for distributed expert parallelism with all-to-all. Its SM90 BF16 kernel has no ping-pong, no gather fusion, and no other epilogue fusion, so an MoE built on it must launch a separate gather-and-pad kernel and cannot overlap a heavy epilogue. It is the strongest grouped-GEMM building-block baseline.

**Megatron-LM GroupedMLP (Shoeybi et al. 2019).** A CUTLASS grouped GEMM with JIT epilogue fusion; assumes packed inputs (no gather fusion); a recent patch fuses the score weighting with SwiGLU so autograd follows the memory-light gradient path. Its expert aggregation (`torch.scatter_add`) is under-optimized, and its per-expert-stream variant suffers CUDA-stream bubbles.

**GPU-scheduling prior art.** FlashAttention-3 (Shah et al. 2024) established Hopper warp-specialization — asynchronous TMA producers feeding consumer-warpgroup WGMMA — and *ping-pong* scheduling, in which one warpgroup runs MMA while another runs the softmax/epilogue, then they swap, to keep tensor cores continuously fed. The CUTLASS ping-pong recipe (Wright & Hoque, 2024) uses one lightweight TMA producer warpgroup and two heavy consumer warpgroups on *separate* output tiles, handed off by async-pipeline barriers; the alternative *cooperative* schedule puts both consumers on the *same* tile and overlaps the epilogue less. Expert-choice routing (Zhou et al. 2022) lets experts pick tokens for perfect load balance but breaks autoregressive inference. Bitonic sort (Batcher 1968) and optimal low-latency sorting networks give data-oblivious, register-only parallel comparisons suited to a warp-level top-K.

## Evaluation settings

The natural yardstick is per-layer training of decoder MoE language models, sweeping configurations from ~1.4B up to ~120B total parameters and varying the two axes of interest: granularity (activated/total experts such as 2/32, 4/64, 8/128, 16/256, equivalently scaling n while holding nK fixed) and sparsity (raising E with K fixed). Hardware: NVIDIA Hopper (H100) and Blackwell (B300) GPUs; multi-GPU runs use FSDP-2. Metrics: per-layer activation-memory footprint (GB) versus granularity; achieved compute throughput (TFLOPS) and effective HBM bandwidth (TB/s) per kernel, against a dense batched-GEMM (cuBLAS BMM, perfect load balance) upper bound; end-to-end training throughput (tokens/day); and, for routing changes, downstream model quality at the ~1.4–1.8B scale together with kernel runtime in the sparse regime. Baselines for the kernel comparisons are ScatterMoE, MoMoE, MegaBlocks, Megatron GroupedMLP, and DeepGEMM-based MoE. Per-component micro-benchmarks cover grouped GEMM with and without gather fusion, the expert-aggregation kernel, and the top-K kernel (against PyTorch, Triton, Tilelang, RTop-K).

## Code framework

A standard MoE module needs a router, per-expert up/down projections, routing metadata, grouped GEMM, and token-wise aggregation. The empty bodies below are the computation slots to fill in.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def grouped_gemm(lhs, rhs, *, out=None, cu_seqlens_m=None, cu_seqlens_k=None,
                 A_idx=None, bias=None):
    raise NotImplementedError

def grouped_gemm_gated(x, w1, *, cu_seqlens_m, A_idx, preact_out, postact_out):
    raise NotImplementedError

def grouped_gemm_dgated(dout, w2, h, scores, *, cu_seqlens_m, A_idx,
                        dh_out, postact_scaled_out):
    raise NotImplementedError

def token_gather_sum(grouped_values, scores, reverse_scatter_idx, token_offsets, T, H):
    raise NotImplementedError

def topk_softmax(router_logits, K):
    top_logits, indices = router_logits.topk(K, dim=-1)
    scores = top_logits.softmax(dim=-1, dtype=torch.float32)
    return scores, indices

def routing_metadata(indices, E):
    raise NotImplementedError


class _UpProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, expert_offsets, x_gather_idx, reverse_scatter_idx, token_offsets):
        # TODO: grouped up-projection and activation.
        raise NotImplementedError

    @staticmethod
    def backward(ctx, unused_da, dh):
        # TODO: compute dX and dW1.
        raise NotImplementedError


class _DownProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, h, w2, scores, expert_offsets, x_gather_idx,
                scatter_idx, reverse_scatter_idx, token_offsets, T):
        # TODO: grouped down-projection and token aggregation.
        raise NotImplementedError

    @staticmethod
    def backward(ctx, dout):
        # TODO: compute dH, dW2, and dS.
        raise NotImplementedError


class MoE(nn.Module):
    def __init__(self, num_experts, top_k, hidden_size, intermediate_size):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(2 * intermediate_size, hidden_size, num_experts))
        self.w2 = nn.Parameter(torch.empty(hidden_size, intermediate_size, num_experts))

    def forward(self, x):
        x = x.view(-1, x.size(-1))
        logits = self.router(x)
        scores, indices = topk_softmax(logits, self.top_k)
        offsets, x_gather_idx, scatter_idx, reverse_scatter_idx, token_offsets = (
            routing_metadata(indices, self.router.out_features)
        )
        a, h = _UpProjection.apply(x, self.w1, offsets, x_gather_idx,
                                   reverse_scatter_idx, token_offsets)
        o = _DownProjection.apply(a, h, self.w2, scores, offsets, x_gather_idx,
                                  scatter_idx, reverse_scatter_idx, token_offsets, x.size(0))
        return o
```
