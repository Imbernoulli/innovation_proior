I would present the method as SonicMoE, a training-time co-design for fine-grained, sparse Mixture-of-Experts layers. The starting observation is that the usual MoE scaling recipe—make each expert narrower while activating more experts to keep FLOPs fixed, and increase the total expert count while keeping the number of active experts fixed—does not translate into wall-clock speed or even into feasible memory usage. The FLOP count goes down, but the layer slides into the memory-bound regime and begins to cache tensors that grow with granularity. SonicMoE fixes this by rewriting the backward graph so that only the dense-equivalent activations need to be saved, fusing gather and activation math into grouped GEMM kernels, overlapping the remaining IO with compute, and optionally rounding per-expert token counts to the GEMM tile size.

The first thing to quantify is arithmetic intensity. For a single SwiGLU expert with intermediate width n, embedding width d, average tokens per expert T_e, forward plus backward FLOPs are 18 T_e n d, and the bytes moved are dominated by activations and weights. Substituting granularity G = d/n and activation ratio ρ = K/E so that T_e = Tρ, one expert's forward intensity becomes 3 / ((2 + 2G)/d + 3/(Tρ)). Because G appears in the denominator, raising granularity pushes the layer below the roofline ridge; lowering ρ does the same. So the very direction that improves quality per FLOP also makes the kernel memory-bound. That reframes the problem: the enemy is IO, not FLOPs.

The second wall is activation memory. At fixed FLOPs F = 18 T K n d, holding T and d fixed means nK is constant, so increasing granularity raises K in proportion as n shrinks. Any tensor of size O(T K d) therefore grows linearly with granularity. The worst offenders are the down-projection output Y, the gathered inputs X_e, and the gathered output gradients dO_e. Existing grouped-GEMM MoE kernels cache Y and materialize X_e and dO_e, which is exactly why training runs out of memory when experts get fine. The theoretical floor without recomputing a matmul is to cache only the layer input X, which costs 2 T d bytes, and the up-projection pre-activation H, which costs 4 T K n bytes. Because nK is constant, the H cache is flat in granularity. That is the target SonicMoE hits.

The key graph rewrite is in the score gradient. The standard aggregation is O_t = Σ_e s_{t,e} Y_{e,t}, and differentiating with respect to the router score gives dS_{t,e} = ⟨dO_t, Y_{e,t}⟩. Every prior kernel I know of computes it this way, which forces Y to be cached and later reloaded from HBM. But Y_{e,t} = A_{e,t} W2_e, so I can push the weight onto the other side of the inner product: dS_{t,e} = ⟨dO_t W2_e^T, A_{e,t}⟩. The term dO_t W2_e^T is exactly the grouped-GEMM output dA'_e that the down-projection activation-gradient kernel already produces, and A_{e,t} is recomputed from the cached H_{e,t} in the same kernel epilogue. Therefore dS can be computed as a reduction over the n-dimensional dot product ⟨dA'_{e,t}, A_{e,t}⟩ while both tensors still live in registers or shared memory. No Y, no dY, and no extra HMB traffic. The reduction is also cheaper because it runs over n instead of d, saving log2(G) rounds in the fine-grained regime where G is large.

Closing the rest of the backward without resurrecting Y is straightforward once dS is handled this way. Conceptually dY_{e,t} = s_{t,e} dO_t, but I never form it. Instead dA_e = Broadcast(s_e) dA'_e is produced fused in the epilogue, dH_e = dSwiGLU(dA_e, H_e) uses the cached H, and the down-projection weight gradient is dW2_e = (Broadcast(s_e) A_e)^T dO_e, where Broadcast(s_e) A_e is also produced in the same epilogue. The up-projection backward then computes dX̃_e = dH_e W1_e^T and dW1_e = X_e^T dH_e, with the gather of X fused into the weight-gradient kernel. The full activation footprint is 2 T d + 4 T K n bytes, which is the dense-equivalent minimum and does not grow with granularity. Y is still materialized transiently in the forward so it can be handed to the aggregation kernel, but that buffer is recycled per layer and is not part of the backward live set.

With the memory problem solved, the remaining issue is IO throughput. SonicMoE attacks it with fusion and overlap. The token gather is fused into the prologue of every grouped GEMM that needs scattered rows, both in the forward and in the backward, so X_e, dO_e, and similar tensors are never written to HBM. SwiGLU and dSwiGLU are fused into the up-projection and down-projection epilogues, and the heavy down-projection activation-gradient epilogue computes dH, dS, and A' together in one pass. On Hopper the heavy epilogues are overlapped with the matmul mainloop through ping-pong scheduling: while one consumer warpgroup finishes the epilogue, the other continues issuing MMA instructions, so tensor cores never wait for the epilogue. On Blackwell the same idea maps onto the two-stage Tensor Memory accumulator pipeline. Weight-gradient kernels, which are long-mainloop and light-epilogue, use a cooperative schedule with large tiles instead. The down-projection stores Y contiguously via asynchronous TMA, and a separate token-gather-and-sum kernel computes O_t = Σ_e s_{t,e} Y_{e,t}. This gather-summation looks like an extra kernel, but it avoids the synchronous scatter store that would otherwise block the next MMA tile and remove the benefit of ping-pong.

The router itself is also tuned. The default token-choice top-K is replaced by a register-level bitonic sort over the E scores per token, with column indices packed into the low mantissa bits so ties are deterministic and the argmax is recovered after sorting. This avoids the shared-memory scans and library overhead that can consume a large fraction of router time.

Finally, SonicMoE addresses the discrete padding waste that appears when the activation ratio ρ is small. The grouped GEMM tiles the token dimension in multiples of tileM, so when T_e = Tρ is only a few hundred tokens each expert's last tile is mostly padding. Token rounding first runs true token-choice top-K, counts per-expert frequencies f_e, and rounds each f_e to the nearest multiple of tileM. To touch only the boundary tile, it builds a preference-adjusted score S' = S − 1 and overwrites the genuine top-K entries with their original scores, so every real assignment is nonnegative and every padding candidate is negative. Sorting each expert's column then lets the kernel drop the lowest-scoring real tokens or admit the highest-scoring near-misses, but only within the last partial tile. The routing stays close to true top-K, and the grouped GEMM sees tile-divisible shapes.

The canonical name is SonicMoE. The method leaves the MoE mathematics unchanged and does not assume a particular router; token rounding is an optional drop-in that trades at most one boundary tile per expert for clean tile shapes. The gains come from treating the backward graph, the kernel epilogues, and the routing as one system rather than separate pieces.

```python
import torch
import torch.nn.functional as F


def swiglu(h):
    gate, value = h.chunk(2, dim=-1)
    return F.silu(gate) * value


def demo_sonicmoe_principles(d=1024, n=128, T=4096, K=8, E=128, tileM=128):
    # Arithmetic intensity for one expert at the given granularity and sparsity.
    G = d / n
    rho = K / E
    Te = T * rho
    flops = 2 * Te * 2 * n * d + 2 * Te * n * d
    bytes_moved = 4 * Te * n + 6 * n * d + 4 * Te * d
    intensity = flops / bytes_moved
    intensity_formula = 3 / ((2 + 2 * G) / d + 3 / (T * rho))

    # Activation-memory comparison: baseline caches Y; SonicMoE caches only X and H.
    mem_sonicmoe = 2 * T * d + 4 * T * K * n
    mem_baseline = mem_sonicmoe + 2 * T * K * d

    # Verify the score-gradient identity: <dO, Y> == <dO W2^T, A>.
    X = torch.randn(int(Te), d)
    W1 = torch.randn(2 * n, d)
    W2 = torch.randn(n, d)
    s = torch.rand(1)
    H = X @ W1.t()
    A = swiglu(H)
    Y = s * (A @ W2)
    dO = torch.randn(int(Te), d)
    dS_via_Y = (dO * Y).sum(dim=-1)
    dA_prime = dO @ W2.t()
    dS_via_A = s * (dA_prime * A).sum(dim=-1)
    score_grad_error = (dS_via_Y - dS_via_A).abs().max().item()

    # Token rounding to the nearest tileM multiple.
    tokens = int(Te)
    up = ((tokens + tileM - 1) // tileM) * tileM
    down = (tokens // tileM) * tileM
    rounded = up if (up - tokens) < (tokens - down) else down

    return {
        "G": G,
        "rho": rho,
        "Te": Te,
        "intensity_flops_per_byte": intensity,
        "intensity_formula": intensity_formula,
        "mem_sonicmoe_MB": mem_sonicmoe / 2**20,
        "mem_baseline_MB": mem_baseline / 2**20,
        "score_grad_max_error": score_grad_error,
        "tokens": tokens,
        "rounded_tokens": rounded,
    }


if __name__ == "__main__":
    for n, K in [(256, 4), (128, 8), (64, 16)]:
        print(f"n={n}, K={K}: {demo_sonicmoe_principles(d=1024, n=n, T=4096, K=K, E=128)}")
```
