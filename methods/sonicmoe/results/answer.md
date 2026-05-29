# SonicMoE

SonicMoE accelerates fine-grained, sparse Mixture-of-Experts training by co-designing the MoE backward graph, grouped-GEMM kernels, and a tile-aware routing option. The MoE computation itself remains router-agnostic and mathematically unchanged; token rounding is an optional router that trades at most one boundary tile per expert for tile-divisible grouped-GEMM shapes.

## Problem

For fixed forward+backward FLOPs F = 18·T·K·n·d, with model width d and token count T fixed, iso-FLOPs means nK is constant. Raising granularity G = d/n lowers n and raises K, so any cached O(TKd) tensor grows linearly with G. The main offenders are Y, gathered X_e, and gathered dO_e.

One expert's forward arithmetic intensity is

3 / ((2 + 2G)/d + 3/(Tρ)),

from (2T_e·2n·d + 2T_e·n·d) FLOPs over (4T_e n + 6nd + 4T_e d) bytes with T_e = Tρ. Increasing G or decreasing activation ratio ρ = K/E lowers intensity. In sparse regimes, token-dimension padding adds another cost: the exact per-expert wasted fraction is (⌈T_e/tileM⌉·tileM − T_e)/T_e; under a roughly uniform residue this is about tileM/(2Tρ), hence O(tileM/(Tρ)).

## Key Ideas

1. **Memory-minimal backward.** Avoid Y and dY in the score-gradient path:

   dS_{t,e} = ⟨dO_t, Y_{e,t}⟩ = ⟨dO_t W2_e^T, A_{e,t}⟩ = ⟨dA'_{e,t}, A_{e,t}⟩.

   Here dA'_e = dO_e W2_e^T is already produced by the down-projection activation-gradient GEMM, and A_e is recomputed from cached H_e. Then dA_e = Broadcast(s_e)dA'_e, dH_e = dSwiGLU(dA_e, H_e), and dW2_e = (Broadcast(s_e)A_e)^T dO_e. In the code, w2 is stored transposed as (H, I, E), so the storage gradient is dO_e^T A'_e. The cached activations are X and H only: 2Td + 4TKn bytes.

2. **IO-aware kernels.** Eight kernels cover forward `{A, Y, O}` and backward `{dH+dS+A', dW2, dXtilde, dW1, dX}`. Gather is fused into grouped-GEMM prologues; SwiGLU/dSwiGLU/dS are fused into epilogues; heavy epilogues use Hopper ping-pong or Blackwell two-stage Tensor Memory overlap; weight-gradient kernels use cooperative scheduling/LPT ordering; contiguous TMA stores plus token-gather-and-sum avoid synchronous scatter stores.

3. **Token rounding.** For router probabilities S in [0,1], compute true top-K, frequencies f_e, and tile multiples ⌊f_e⌋, ⌈f_e⌉. Build S' = S − 1, then overwrite true top-K entries with their top-K scores. Real top-K entries are nonnegative and all padding candidates are negative, so sorting each expert column can only drop or add the boundary tile. Nearest rounding on frequency is the default; balanced rounding can bound total routed-token deviation by half a tile.

## Code

The core autograd structure is a PyTorch router feeding separate up- and down-projection autograd functions over CuTe-DSL grouped GEMMs. Bias handling is omitted here; with down-projection bias, dS adds the expected ⟨dO, b2_e⟩ term.

```python
import torch
import torch.nn.functional as F
from quack.gemm_interface import gemm, gemm_dgated, gemm_gated
from sonicmoe.functional import TC_Softmax_Topk_Router_Function
from sonicmoe.functional.backward import _token_broadcast_backward
from sonicmoe.functional.forward import _router_forward
from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton


class _UpProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, expert_offsets, x_gather_idx,
                reverse_scatter_idx, token_offsets):
        TK = int(x_gather_idx.numel())
        I = w1.size(0) // 2
        a = torch.empty(TK, I, dtype=x.dtype, device=x.device)
        h = torch.empty(TK, 2 * I, dtype=x.dtype, device=x.device)
        gemm_gated(x, w1.permute(2, 1, 0), activation="swiglu",
                   cu_seqlens_m=expert_offsets, A_idx=x_gather_idx,
                   preact_out=h, postact_out=a, store_preact=True)
        ctx.save_for_backward(x, w1, expert_offsets, x_gather_idx,
                              reverse_scatter_idx, token_offsets)
        return a, h

    @staticmethod
    def backward(ctx, unused_da, dh):
        x, w1, offsets, x_gather_idx, reverse_scatter_idx, token_offsets = ctx.saved_tensors
        dx_expanded = gemm(dh, w1.permute(2, 0, 1), cu_seqlens_m=offsets)
        dw1 = torch.empty_like(w1)
        gemm(x.T, dh, out=dw1.permute(2, 1, 0),
             cu_seqlens_k=offsets, A_idx=x_gather_idx)
        dx = torch.empty_like(x)
        varlen_k_max = reverse_scatter_idx.numel() // x.size(0)
        _token_broadcast_backward(dx, dx_expanded, reverse_scatter_idx,
                                  token_offsets, varlen_k_max, x.size(1),
                                  token_offsets is not None)
        return dx, dw1, None, None, None, None


class _DownProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, h, w2, scores, offsets, x_gather_idx,
                scatter_idx, reverse_scatter_idx, token_offsets, T):
        y = torch.empty(a.size(0), w2.size(0), dtype=a.dtype, device=a.device)
        gemm(a, w2.permute(2, 1, 0), out=y, cu_seqlens_m=offsets)
        o = torch.empty(T, w2.size(0), dtype=a.dtype, device=a.device)
        _router_forward(y, o, scores.view(-1), reverse_scatter_idx,
                        token_offsets, scores.size(-1), w2.size(0),
                        token_offsets is not None)
        ctx.save_for_backward(h, w2, scores.view(-1), offsets, x_gather_idx, scatter_idx)
        ctx.score_shape = scores.shape
        return o

    @staticmethod
    def backward(ctx, dout):
        h, w2, scores, offsets, x_gather_idx, scatter_idx = ctx.saved_tensors
        dh = torch.empty_like(h)
        a_prime = torch.empty(h.size(0), w2.size(1), dtype=h.dtype, device=h.device)
        s = scores[scatter_idx]
        _, _, ds_grouped = gemm_dgated(
            dout, w2.permute(2, 0, 1), PreAct=h, activation="swiglu",
            dx_out=dh, postact_out=a_prime, colvec_scale=s, colvec_reduce=True,
            cu_seqlens_m=offsets, A_idx=x_gather_idx)
        ds = torch.empty_like(scores)
        ds[scatter_idx] = ds_grouped
        dw2 = torch.empty_like(w2)
        gemm(dout.T, a_prime, out=dw2.permute(2, 0, 1),
             cu_seqlens_k=offsets, A_idx=x_gather_idx)
        return None, dh, dw2, ds.view(ctx.score_shape), None, None, None, None, None, None


def moe_tc_softmax_topk_layer(x, router_w, w1, w2, K):
    logits = F.linear(x, router_w)
    T, E, TK = x.size(0), router_w.size(0), x.size(0) * K
    scores, topk_idx = TC_Softmax_Topk_Router_Function.apply(logits, E, K, True, False)
    expert_freq = torch.empty(E, dtype=torch.int32, device=x.device)
    offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=x.device)
    scatter_idx = torch.empty(TK, dtype=torch.int32, device=x.device)
    reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=x.device)
    TC_topk_router_metadata_triton(topk_idx.to(torch.int32), E, expert_freq,
                                   offsets, x_gather_idx, scatter_idx,
                                   reverse_scatter_idx)
    a, h = _UpProjection.apply(x, w1, offsets, x_gather_idx,
                               reverse_scatter_idx, None)
    o = _DownProjection.apply(a, h, w2, scores, offsets, x_gather_idx,
                              scatter_idx, reverse_scatter_idx, None, T)
    return o, logits, expert_freq


def token_rounding(router_logits, K, E, tileM):
    scores = router_logits.softmax(dim=-1, dtype=torch.float32)
    topk_scores, topk_idx = scores.topk(K, dim=-1)
    freq = torch.bincount(topk_idx.flatten(), minlength=E).int()
    freq_up = ((freq + tileM - 1) // tileM) * tileM
    freq_down = (freq // tileM) * tileM
    rank_score = scores.scatter(1, topk_idx, topk_scores).detach() - 1.0
    rank_score.scatter_(1, topk_idx, topk_scores)
    routed = []
    for e in range(E):
        order = torch.argsort(rank_score[:, e], descending=True)
        keep = freq_up[e] if (freq_up[e] - freq[e]) < (freq[e] - freq_down[e]) else freq_down[e]
        routed.append(order[:keep])
    return routed
```
