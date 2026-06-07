# SonicMoE — synthesis notes (Phase 1.5)

Method short-name: **sonicmoe** (paper title "SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations"; official repo Dao-AILab/sonic-moe, CuTe-DSL kernels + PyTorch interface).

## The pain point / research question

MoE replaces a dense MLP channel-mixer with a router + many smaller "experts"; each token activates only K of E experts, so we get the parameter count of a big model at the FLOPs of a small one. Scaling laws for fine-grained MoE (Krajewski et al. 2024; Clark et al. 2022) say model quality-per-FLOP improves as you make experts **more granular** (smaller intermediate size n, hence larger granularity G=d/n, with K scaled up to keep FLOPs fixed) and **sparser** (E up, K fixed, activation ratio ρ=K/E down). Frontier models (DeepSeek-V3, Qwen3-MoE, gpt-oss, Kimi-K2) all moved this way.

But the theoretical FLOP win does not translate to wall-clock throughput, for three reasons that are all *facts about existing systems / hardware* (pre-method, motivating):

1. **Activation memory scales linearly with granularity.** For constant FLOPs F=18·T·K·n·d (6 fwd + 12 bwd), keeping d,T fixed means keeping n·K constant; raising G means lowering n and raising K. Any cached activation of size O(TKd) then grows linearly with G. ScatterMoE/MoMoE/MegaBlocks cache the down-proj output Y (TKd) and/or grouped X_e, dO_e (TKd each) → OOM as you go fine-grained.
2. **IO cost grows linearly with granularity / lower arithmetic intensity.** Derived in Eq. 1: per-expert fwd arithmetic intensity = 3 / ( (2+2G)/d + 3/(Tρ) ). Increasing G or decreasing ρ both lower it → MoE drifts into the memory-bound regime, so the GEMMs can't reach peak tensor-core throughput unless IO is hidden.
3. **Tile-quantization waste under sparsity.** GEMM tiles M into multiples of tileM (e.g. 128) and pads. Per-expert token count T_e = Tρ shrinks as ρ↓ (e.g. gpt-oss: 16384 tokens but only 512/expert). The last tile is mostly padding; wasted-FLOP fraction ≈ O(tileM/(Tρ)), linear in tileM and inversely linear in ρ.

A solution must: keep activation memory flat in G (ideally = a dense model with the same activated params, the minimum without GEMM recompute), hide/eliminate the IO, and remove tile-padding waste — all without changing the MoE math (so model quality is untouched) and agnostic to the router.

## The three contributions (and how they map to the three problems)

### (A) Memory-efficient backward — minimum activation, no extra FLOPs
Standard MoE-with-grouped-GEMM forward (per expert e, with SwiGLU):
- H_e = X_e W1_e  (up-proj, varlen-M grouped GEMM), X_e gathered from X by π
- A_e = SwiGLU(H_e)
- Y_e = A_e W2_e  (down-proj)
- O_t = Σ_e s_{t,e} Y_{e,t}  (expert aggregation; s = router score)

Standard autograd backward caches X_e (or X), H_e (or A_e), Y_e, and the gathered dO_e. The killers are the O(TKd) tensors: Y, gathered X_e, gathered dO_e.

SonicMoE's three moves:
1. **Fuse gather into the GMEM→SMEM load** (prologue), so X_e and dO_e are never materialized in HBM → save 2·(2TKd) cache + IO. (Backward needs gather for dW1 (gather X), dW2 & dH (gather dO); ScatterMoE/MoMoE fuse gather only in fwd and launch a *separate* gather kernel in bwd.)
2. **Never materialize/cache Y, and avoid dY.** dY = Broadcast(s)·dO never needs to be written.
3. **Compute dS via an alternative path that needs neither Y nor dY** (Appendix proof, see below). Cache only X (Td), H (2TKn=O(TKn), which is bounded since nK const) and routing metadata → total **2Td + 4TKn bytes/layer = same as a dense model with the same activated params** = minimum without GEMM recompute. (A transient Y is materialized but recycled across layers; with L≥K layers it's overshadowed. Removing it entirely would need a BF16 atomic-add → nondeterminism, numerical issues, incompat with EP.)

**The dS derivation (Appendix B/dZ_proof — load-bearing, must be lived inline):**
- O_t = Σ_e s_{t,e} Y_{e,t}  ⇒  dY_{e,t} = s_{t,e} dO_t, i.e. dY_e = Broadcast(s_e) dO_e.
- Define grouped-GEMM output dA'_e := dO_e W2_e^T ∈ R^{T_e×n}. Then dA_e = dY_e W2_e^T = Broadcast(s_e) dA'_e.
- **Score gradient**: dS_{t,e} = ⟨dO_t, Y_{e,t}⟩ = ⟨dO_t W2_e^T, A_{e,t}⟩ = ⟨dA'_{e,t}, A_{e,t}⟩.  (Used: Y_{e,t}=A_{e,t}W2_e, and ⟨u, M v_row⟩ identity / move W2^T onto dO.) So dS is a reduction over the n-dim of the *already-on-chip* dA' and A — no Y, no dY, no extra HBM load.
- **dH**: dH_e = dSwiGLU(dA_e, H_e), with A_e recomputed from cached H_e in the same epilogue.
- **dW2**: dW2_e = A_e^T dY_e = (Broadcast(s_e) A_e)^T dO_e = A'_e^T dO_e, where A'_e := Broadcast(s_e) A_e is produced in the dH-kernel epilogue and handed to the dW2 (varlen-K grouped GEMM) kernel.
- **dW1** = X_e^T dH_e (gather X, varlen-K). **d X̃** = dH_e W1_e^T (varlen-M). **dX_t** = Σ_e dX̃_{e,t} (expert aggregation).

**Why ⟨dA', A⟩ over ⟨dO, Y⟩** (three concrete reasons, Appendix B):
- Extra HBM traffic 0 vs 2TKd (dA' and A already on chip in the dH kernel; ⟨dO,Y⟩ needs to reload Y).
- Extra cached activation 0 vs 2TKd (⟨dO,Y⟩ forces caching Y — exactly why ScatterMoE/MoMoE/MegaBlocks fail to stay flat in G).
- Parallel reduction rounds log2(n) vs log2(d): dA',A reduce over n; dO,Y reduce over d; saves ≥ log2(d/n)=log2(G) rounds.
- The two are *mathematically identical* (it's the same dS, just realized from different intermediates) — depends only on whether you apply the score s before or after down-proj forward. ScatterMoE/MoMoE/MegaBlocks effectively apply s after (scatter Y) ⇒ ⟨dO,Y⟩; SonicMoE applies the gradient weighting via A' ⇒ ⟨dA',A⟩.

### (B) IO-aware kernel: reduce IO via fusion + hide IO via MMA/IO overlap
Built on an efficient **varlen-M** and **varlen-K** grouped GEMM. Two fusion classes + one overlap class:

- **Gather fusion into prologue** (GMEM→SMEM): fetch routed token indices, then `cp.async` the gathered rows directly into SMEM to feed the tensor core; no separate gather kernel, no materialized X_e/dO_e. Saves 2TKd bytes IO in backward. (DeepGEMM/Megatron/MegaBlocks assume contiguously-packed, padded-to-128 inputs and need a separate gather+pad kernel.) On **Blackwell** with 2-CTA clusters, `cp.async` only signals completion within its own CTA but the leader CTA's MMA must wait for both CTAs' gathers → a dedicated **relay warp** in CTA1 forwards the completion signal via cluster-scope mbarrier.
- **Epilogue fusion**: SwiGLU fused into up-proj A-kernel epilogue; dSwiGLU + dS + A' fused into the dH-kernel epilogue (heavy epilogue: load H via async TMA, compute dH, dS, A'). This makes one dH kernel replace ScatterMoE's separate {down-proj act, dS, dSwiGLU} kernels.
- **MMA ↔ async-IO overlap.**
  - *Hopper*: producer-consumer paradigm (TMA producers GMEM→SMEM; consumer warpgroups do WGMMA). With 2 consumer warpgroups, **Ping-Pong scheduling** alternates: while WG0 does the heavy epilogue, WG1 runs MMA, then they swap — keeps tensor cores continuously fed. Used for **Y kernel** (heavy HBM store epilogue, 2TKd) and **dH kernel** (heavy compute+load epilogue). Also: dedicated async-TMA load pipeline for H in dH epilogue. (FA3 established async overlap + ping-pong; applying it to fine-grained MoE's increasing IO is the new use.)
  - *Blackwell*: same spirit, different mechanism. Tensor Memory (TMEM): 256KB/SM, 128 rows × 512 cols of 32-bit cells. UMMA (single-threaded async) writes accumulators into TMEM; the 512 cols form a **two-stage accumulator pipeline** (2×256 cols): one stage does UMMA while the other does epilogue, then ownership of the TMEM stage is handed back. UMMA removes WGMMA's register pressure → better MMA/epilogue overlap.
- **Store choice (no scatter fusion).** In fwd down-proj Y and bwd up-proj dX̃, SonicMoE does *not* fuse scatter with the HBM store; it stores contiguously via **async TMA**, then a separate **expert-aggregation kernel** lets each *token gather-and-sum* its experts' outputs. Reason: scatter-fused store on Hopper requires the *synchronous* `st.global` (only PTX option without TMA-1D), which blocks the next MMA tile (~20% TFLOPS drop on heavy-store epilogue), and ping-pong can't hide a synchronous store. Token-gather and expert-scatter are mathematically equivalent; gather wins because it reuses the same gather index over the whole d-row (K index fetches/token) and uses asynchronous TMA store. ScatterMoE/MoMoE chose scatter-fused store → slow. (Atomic-add-fused store avoids the agg kernel but BF16 atomic-add → nondeterminism, numerical error, incompat with all2all/EP.)
- **LPT tile scheduling** for varlen-K (dW1, dW2): expert with most tokens first (Longest-Processing-Time) to cut tail effect; persistent tile scheduler (CLC on Blackwell).
- **Efficient top-K kernel**: torch.topk is ~40% of router time. SonicMoE: parallelize over T, **bitonic sort** per row over E values; pack column indices into the low log2(E) mantissa bits of the FP32 score so the sort is stable (unique cols ⇒ no ties); base cases (≤64) use **optimal low-latency sorting networks**; all compare-and-swap stays in registers / warp-shuffle (no SMEM scans). Supports E≤4096, K≤16, optional softmax fusion. Beats torch/triton/tilelang/RTop-K.

8 launched kernels total: fwd {A (up-proj), Y (down-proj), O (expert-agg)}; bwd {dH (down-proj act + dS + A'), dW2, dX̃ (up-proj act), dW1, dX (expert-agg)}.

### (C) Token rounding (TR) — eliminate tile-quantization waste, drop-in router
Tile quantization: ⌈T_e/tileM⌉·tileM − T_e padded rows/expert; waste fraction O(tileM/(Tρ)). TR rounds each expert's token count to a multiple of tileM, deviating by **at most one tile** from token-choice (TC).

Algorithm (2-step sort, drop-in, router-agnostic):
1. TC top-K: (S_topK, I_topK) = TopK(S, K). Expert freq f_e = Σ_t 1[e∈I_topK,t]. ⌈f_e⌉ = ⌈f_e/tileM⌉·tileM, ⌊f_e⌋ = ⌊f_e/tileM⌋·tileM.
2. Build a **TC-preferred** score S': S' = S − 1 (so all non-topK entries < 0), then overwrite S'[t, I_topK(t,k)] = S_topK,t,k. This guarantees TC tokens always outrank the EC "padding" candidates, so the rounding only ever touches the *last* tile per expert.
3. Per expert: sort by S'_e, then `round_and_sparsify` makes a **binary** decision — discard down to ⌊f_e⌋ (drop lowest-score TC tokens) or pad up to ⌈f_e⌉ (admit highest-score EC tokens). Default **NR-f (nearest rounding on frequency)**: pad if ⌈f_e⌉−f_e < f_e−⌊f_e⌋, else drop.
- The "preserve total tokens in expectation" / exact-preservation variant = **Balance-f** (Alg. in appendix): carry a running accumulator z of residual rounding error; choose up/down to minimize |r±+z|; guarantees max per-expert deviation ≤ tileM/2 and total deviation ≤ tileM/2. NR-f is the simple default; TR is robust to the subroutine.
- Why TC-preferred and ≤1-tile deviation: keeps TR's routing decisions almost identical to TC ⇒ preserves model quality (validated robust when T̄_e/tileM ≥ 2), while making every grouped-GEMM M-dim exactly tile-divisible ⇒ zero padding waste.

## Load-bearing ancestors (elaborate, don't name-drop)

- **Sparsely-gated MoE (Shazeer et al. 2017) / Switch Transformer (Fedus et al. 2022):** router + experts; top-K token-choice routing as default; Switch used fixed **expert capacity** with padding (wasted FLOPs) and **token dropping** (quality loss) to get static shapes. Pain it leaves: padding waste & dropped tokens; motivates dropless + tile-aware approaches.
- **MegaBlocks (Gale et al. 2023):** frames MoE as **block-sparse** matmul → dropless, no capacity padding/dropping. But block-sparse GEMM (STK) is slower than dense grouped GEMM, complex, and its gather+pad+scatter kernels cost ~8TKd bytes IO. Leaves: grouped GEMM is faster & simpler; IO still unhidden.
- **ScatterMoE (Tan et al. 2024):** the direct baseline. `ParallelLinear` = grouped GEMM with **fused gather (fwd) and fused scatter (output)**, avoiding pad/copies; ~700 lines of **Triton**. Limits SonicMoE attacks: (i) computes dS=⟨dO,Y⟩ ⇒ must cache Y (2TKd) ⇒ activation grows with G; (ii) fuses gather only in fwd, separate gather kernel in bwd; (iii) scatter-fused output store ⇒ synchronous st.global; (iv) Triton ⇒ no fine-grained async TMA / ping-pong / heavy-epilogue overlap; (v) no varlen-K gather fusion.
- **MoMoE (Costin et al. 2025, Tilde):** memory-optimized MoE; fuses dS into up-proj act-grad but still dS=⟨dO,Y⟩ (caches Y); materializes grouped dO_e, X_e in bwd (scales with G); Triton; scatter store via atomic/sum.
- **DeepGEMM (DeepSeek 2025):** very fast grouped GEMM for **contiguously-packed, 128-padded** inputs; specialized for EP + all2all; SM90 BF16 has no ping-pong, no gather fusion, no other epilogue fusion. "DeepGEMM++" = best MoE you can build on it without editing it; still pays separate gather+pad and lacks heavy-epilogue overlap → SonicMoE's baseline-to-beat.
- **Megatron-LM GroupedMLP / TEGroupedMLP (Shoeybi et al. 2019):** CUTLASS grouped GEMM with JIT epilogue; assumes packed inputs (no gather fusion); a recent patch makes autograd follow the ⟨dA',A⟩ path (like SonicMoE) by fusing s with SwiGLU; under-optimized expert agg (torch.scatter_add). TEGroupedMLP launches per-expert GEMMs on 4 streams → CUDA-stream bubbles.
- **FlashAttention-3 (Shah et al. 2024):** established the Hopper producer-consumer warp-specialization + async TMA/WGMMA + **ping-pong** overlap (interleave matmul of one warpgroup with softmax/epilogue of another) to reach near-peak. SonicMoE borrows the overlap machinery and applies it to MoE's growing IO; novelty is the application, not the mechanism.
- **CUTLASS Ping-Pong (Wright & Hoque / PyTorch blog 2024):** concrete recipe: 1 lightweight TMA producer WG + 2 heavy consumer WGs on *separate* output tiles; while WG0 does epilogue, WG1 does MMA, then swap; barriers via async pipeline; vs **cooperative** (both WGs on the *same* tile, less epilogue overlap). SonicMoE picks ping-pong for heavy-epilogue kernels (Y, dH), cooperative for long-mainloop kernels (dW1, dW2).
- **SwiGLU / GLU variants (Shazeer 2020):** expert = SwiGLU MLP: A = SiLU(gate)⊙value where [gate,value]=H=XW1; gating empirically improves transformer MLP quality. SonicMoE keeps it (act_func/dAct_func fused in epilogue) but is activation-agnostic (geglu/reglu/gelu/relu/relu² also supported).
- **Expert-choice routing (Zhou et al. 2022):** experts pick tokens ⇒ perfect load balance, but breaks autoregressive inference (train/infer mismatch, causality leak). TR borrows EC's *per-expert sort* idea only to choose which boundary tokens to drop/pad, while staying TC-equivalent at inference.
- **Bitonic sort (Batcher 1968) / optimal sorting networks:** data-oblivious, parallel, register-only comparisons → ideal for a warp-level top-K. Underpins SonicMoE's top-K kernel.
- **Roofline / arithmetic intensity (Williams et al.):** AI=FLOPs/bytes; ridge point separates memory- vs compute-bound. The lens that turns "fine-grained MoE is slow" into the precise Eq.1 statement that AI falls with G and 1/ρ.

## Design-decision → why table (with rejected alternatives)

| Decision | Why / what breaks otherwise | Rejected alt & its failure |
|---|---|---|
| Cache only X, H (+ metadata) = 2Td+4TKn | = dense-with-activated-params, the min without GEMM recompute; flat in G | Cache Y/X_e/dO_e (O(TKd)) → grows linearly with G → OOM (ScatterMoE/MoMoE) |
| dS = ⟨dA',A⟩ (reduce over n) | dA',A already on chip; 0 extra IO, 0 extra cache, log2(n) rounds | dS=⟨dO,Y⟩: reload+cache Y (2TKd each), log2(d) rounds (ScatterMoE/MoMoE/MegaBlocks) |
| dW2 = A'^T dO with A'=Broadcast(s)A | apply score weighting before down-proj in bwd graph ⇒ enables ⟨dA',A⟩ path | apply s after (scatter Y) ⇒ forces ⟨dO,Y⟩ ⇒ cache Y |
| Recompute A from cached H in dH epilogue | H is small (2TKn, bounded); cheap recompute beats caching A (TKn) and Y (TKd) | cache A and Y → more memory & IO |
| Fuse gather into prologue (fwd & bwd) | X_e/dO_e never materialized; −2TKd IO; feeds tensor core directly | separate gather kernel (ScatterMoE bwd, DeepGEMM/Megatron always): extra IO+launch |
| Async-TMA contiguous store + token-gather-sum agg kernel | TMA store is async, overlaps MMA; gather reuses 1 index per d-row | scatter-fused store: synchronous st.global blocks next MMA (~20% drop), ping-pong can't hide it (ScatterMoE/MoMoE) |
| No atomic-add-fused store | determinism, BF16 numerical accuracy, EP/all2all compat | BF16 atomic add: nondeterministic, inaccurate, breaks EP |
| Ping-Pong for Y & dH; cooperative for dW1/dW2 | heavy-epilogue kernels need epilogue↔MMA overlap; long-mainloop kernels want max tile | one scheme for all: stalls tensor cores on heavy epilogue, or small tiles on long mainloop |
| Async-TMA load of H in dH epilogue | overlaps the H reload with other epilogue ops | synchronous load: serializes epilogue |
| Blackwell relay warp for cp.async 2-CTA | cp.async signals only within its CTA; leader MMA needs both CTAs' gather done | no relay: leader MMA waits forever / wrong sync |
| Blackwell TMEM 2-stage accumulator (UMMA) | 512-col TMEM = 2×256; one stage MMA while other epilogue; UMMA removes WGMMA reg pressure | Hopper-style WGMMA on Blackwell: register pressure, worse overlap |
| LPT (longest-processing-time) scheduling for varlen-K | unequal per-expert token counts ⇒ tail; biggest-first cuts makespan | static expert-ID order: tail effect, idle SMs |
| Top-K: bitonic sort + mantissa-bit index packing + sorting networks, register-only | torch.topk ~40% router time; register comms + stable sort + min-step base cases | torch radix-select (SMEM scans), tilelang K-pass (bad for large K), RTop-K (iterative, SMEM) |
| TR: ≤1-tile deviation, TC-preferred S'=S−1 then overwrite topK | every M-dim exactly tile-divisible ⇒ 0 pad waste; near-TC ⇒ quality preserved | capacity+drop (Switch): quality loss + still-padded; ignore tiles (Rectify-Router): no FLOP saving |
| TR default NR-f (nearest on frequency) | simplest, robust; ablations show subroutine barely matters | Balance-f exact-total (more complex) only if total-token preservation needed |
| SwiGLU expert, act fused in epilogue | quality (GLU gating) + fused = no extra IO; but agnostic to act | unfused pointwise act: extra HBM read/write |

## Code grounding (official repo Dao-AILab/sonic-moe)
- `sonicmoe/moe.py`: `MoE` nn.Module — router Linear, two `Experts` (c_fc up-proj 2I, c_proj down-proj), `forward` calls `moe_TC_softmax_topk_layer`, switch-style aux loss.
- `sonicmoe/functional/__init__.py`: the heart. `TC_Softmax_Topk_Router_Function`, `_UpProjection` (fwd: gemm_gated stores preact H + postact A; bwd: dAct→dH, gemm dW1=X^T dH with gathered X, token_broadcast_backward dX), `_DownProjection` (fwd: gemm Y, `_router_forward` gather-sum O; bwd: `_down_projection_backward_act` computes dH, dS, A', then gemm dW2=dO^T A'), and the `moe_TC_softmax_topk_layer` driver wiring the 8 kernels. `gemm`/`gemm_gated`/`gemm_dgated` from QuACK (CuTe-DSL grouped GEMM).
- `sonicmoe/functional/forward.py`,`backward.py`: custom-op wrappers around CuTe-DSL kernels (topk, router gather-sum, db2_and_ds_kernel computes dS=⟨dout,Y via b2⟩ partial + handed dA' partial; dh; a_prime).
- The final reasoning/answer code mirrors this PyTorch autograd structure (kernels abstracted as grouped-GEMM + epilogue primitives), grounded in the repo.
