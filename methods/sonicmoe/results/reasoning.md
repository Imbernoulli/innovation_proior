OK, let me start from the thing that actually hurts. I have a Mixture-of-Experts layer: a router scores every token over E experts, each token goes to its top-K experts, each expert is a little SwiGLU MLP with intermediate width n, and the model embedding width is d. The whole appeal is that I get the parameter count of a giant model at the FLOP cost of a small one, because only K of E experts fire per token. And the scaling-law work on fine-grained MoE keeps telling me the same thing: for a fixed FLOP budget, I get better quality per FLOP if I make the experts *finer* — shrink n, and bump K up to compensate so the FLOPs stay put — and *sparser* — keep K fixed but blow E up, so the activation ratio ρ = K/E gets tiny. Every frontier MoE has been marching in that direction.

So in the FLOP model I should just keep shrinking n and growing E and the quality goes up for free. Except when I actually train it, the wall-clock throughput doesn't follow the FLOP count down at all, and at high granularity I run out of memory before I even get to measure throughput. The FLOPs are a lie about how long this takes. I need to understand *why*, precisely, because "it's slow" isn't actionable.

Let me write the cost of one expert's forward in real units and stare at it. Up-projection: X_e ∈ R^{T_e×d} times W1_e ∈ R^{d×2n} gives H_e ∈ R^{T_e×2n}, costing 2·T_e·2n·d FLOPs and moving roughly 2T_e d + 2·2n·d + 2T_e n bytes (load X, load the weight, write the result). Down-projection A_e W2_e: 2·T_e·n·d FLOPs and ~2T_e n + 2n d + 2T_e d bytes. Arithmetic intensity is FLOPs over bytes — the single number the roofline model uses to tell me whether I'm starved for compute or starved for bandwidth. Summing the two projections and dividing:

(2·T_e·2n·d + 2·T_e·n·d) / (4T_e n + 6n d + 4T_e d) = 3 / ( 2/d + 2/n + 3/T_e ).

Now substitute the MoE-specific quantities: T_e = Tρ on average (uniform routing), and the granularity G = d/n so that 1/n = G/d. Then 2/d + 2/n = (2 + 2G)/d, and

arithmetic intensity = 3 / ( (2 + 2G)/d + 3/(Tρ) ).

There it is, and it's worse than I feared in a *specific* way. For a fixed model width d, raising granularity G or lowering the activation ratio ρ both *increase* the denominator and push the intensity down. The IO cost per FLOP grows linearly with granularity. So every step I take toward "better quality per FLOP" slides the layer further below the roofline ridge into the memory-bound regime, where the tensor cores sit idle waiting on HBM and the beautiful FLOP reduction means nothing. The pursuit of fine-grained, sparse MoE is *the same thing* as the pursuit of a memory-bound kernel. That reframes the whole problem: the enemy is IO, not FLOPs.

But before I even get to throughput there's a harder wall: memory. Let me make the FLOP-budget constraint explicit. Forward plus backward of the whole layer is F = 18·T·K·n·d (6 for forward, 12 for backward). If I want to compare configurations at iso-FLOPs with d and T fixed, then n·K must be held constant. So "increase granularity" literally means "lower n, raise K, with nK fixed." Now ask: which cached activations have size proportional to TKd? The down-projection output Y is T·K·d (each of the TK token-expert pairs produces a d-vector). The gathered inputs X_e collected across experts are also TKd, and the gathered output-gradients dO_e likewise TKd. And nK constant means K rises as n falls — so anything of size O(TKd) grows *linearly with granularity* even though I'm holding FLOPs fixed. That's exactly the OOM I'm hitting. The existing grouped-GEMM MoE kernels cache Y (and materialize the grouped X_e, dO_e), so they cannot survive fine-grained training.

So I want two things from the algorithm side, before I even touch a kernel: (1) do not let any O(TKd) tensor be cached for the backward; (2) do not let any O(TKd) tensor be materialized in HBM during the backward either. What's the floor? A dense MLP with the same number of *activated* parameters caches its input and its hidden pre-activation. Here X has T·d BF16 entries, so 2Td bytes. The GLU pre-activation H has T·K·2n BF16 entries, so 4TKn bytes. If I can cache only X and H, the activation memory is 2Td + 4TKn bytes, and crucially, with nK fixed, 4TKn is *constant* in granularity. That would be the minimum achievable without recomputing a GEMM. That's the target.

Let me see what the standard backward forces me to keep. The forward is H_e = X_e W1_e, A_e = SwiGLU(H_e), Y_e = A_e W2_e, O_t = Σ_e s_{t,e} Y_{e,t}, where s = router score. The textbook reverse-mode pass needs, for the down-projection, both Y (to get the score gradient) and A (to get dW2 and dH). For the up-projection it needs X (for dW1) and the activation's input H. The O(TKd) offenders are Y and the gathered X_e, dO_e. Caching H is fine — it's only O(TKn), and I can recompute A from H cheaply in the epilogue, so I don't need to cache A separately. So the whole game reduces to: *can I do the backward without ever needing Y (or its gradient dY)?*

Where does Y actually get used in the backward? Two places. First, dW2 = A_e^T dY_e — but that's A and dY, not Y itself, so Y isn't strictly needed there if I can get dY. Second, and this is the one everyone trips on, the score gradient: from O_t = Σ_e s_{t,e} Y_{e,t}, differentiating with respect to the scalar s_{t,e} gives

dS_{t,e} = ⟨dO_t, Y_{e,t}⟩.

That's the natural expression — it's literally what you get if you implement aggregation as "scatter each expert's Y to its tokens and reduce," which is what ScatterMoE, MoMoE, and MegaBlocks all do. And it needs Y. So to compute dS the cheap-looking way, you cache Y, and you've lost — activation memory now scales with granularity. This is the trap.

Let me try to wriggle out. dS is a per-(token, expert) scalar; it shouldn't *intrinsically* need the d-dimensional Y. Let me expand Y_{e,t} = A_{e,t} W2_e and push the W2_e onto the other factor of the inner product:

dS_{t,e} = ⟨dO_t, Y_{e,t}⟩ = ⟨dO_t, A_{e,t} W2_e⟩ = ⟨dO_t W2_e^T, A_{e,t}⟩.

And dO_t W2_e^T is exactly the grouped-GEMM output I already have to compute on the way to the activation gradient — call it dA'_e := dO_e W2_e^T ∈ R^{T_e×n}. So

dS_{t,e} = ⟨dA'_{e,t}, A_{e,t}⟩.

dA' is the very first matmul of the down-projection backward, and A is what I recompute from cached H in the same kernel. Both live on chip already, in the down-projection activation-gradient kernel's epilogue. I never touch Y. I never write dY to HBM. dS is just a reduction over the n-dimension of two tensors I'm already holding. The two forms ⟨dA', A⟩ and ⟨dO, Y⟩ compute the *identical* number — it's the same dS — they only differ in *which intermediates* realize it, which in turn is a choice of whether you apply the score weighting before or after the down-projection in the graph. The scatter-Y crowd applies it after; if I apply the gradient weighting through A instead, I get the cheap form.

How much cheaper, exactly? Three ways. The extra HBM traffic to compute dS this way is zero, because dA' and A are already in registers/SMEM, versus 2TKd bytes to reload Y the other way. The extra activation memory I must cache is zero, versus 2TKd bytes to keep Y around — and *that* is the single thing that was making the other kernels' memory scale with granularity. And the reduction itself is over the n-dimension here rather than the d-dimension there, so the parallel reduction takes log2(n) rounds instead of log2(d), saving log2(d/n) = log2(G) rounds — which is a real saving precisely in the fine-grained regime where G is large. Everything points the same way.

Now let me close the loop on the rest of the backward so I'm sure I never resurrect Y or dY. I do still conceptually have dY_{e,t} = s_{t,e} dO_t — that's just chain rule through O_t = Σ_e s_{t,e} Y_{e,t} — i.e. dY_e = Broadcast(s_e)·dO_e. But I never need to *form* dY in memory. Watch dA: dA_e = dY_e W2_e^T = Broadcast(s_e)·(dO_e W2_e^T) = Broadcast(s_e)·dA'_e. So dA is dA' with the per-token score broadcast in — fused, not stored. Then dH_e = dSwiGLU(dA_e, H_e), computed from cached H. For the mathematical down-projection weight W2_e ∈ R^{n×d}, the gradient is dW2_e = A_e^T dY_e = A_e^T (Broadcast(s_e) dO_e) = (Broadcast(s_e) A_e)^T dO_e. Define A'_e := Broadcast(s_e) A_e, which I produce in the same epilogue where I'm already holding A, and hand it to the dW2 kernel as a contiguous input. In the actual PyTorch module the stored down-projection weight is transposed, with shape d×n per expert, so the kernel writes dO_e^T A'_e into that storage; it is the same gradient with the storage axes swapped. So one kernel — the down-projection activation-gradient kernel — loads H, gathers dO, computes dA', then in its epilogue computes dH, dS, and A', all at once, never seeing Y. That heavy-epilogue kernel is going to matter later.

The up-projection backward is the easy mirror: dH flows into d X̃_e = dH_e W1_e^T (a varlen-M grouped GEMM), dW1_e = X_e^T dH_e (gather X, varlen-K grouped GEMM), and dX_t = Σ_e d X̃_{e,t} (aggregate over experts per token). No O(TKd) cache appears. So the full picture is: cache only X (Td) and H (O(TKn)) plus the tiny routing metadata, total 2Td + 4TKn bytes, flat in granularity — the dense-equivalent minimum. One annoyance: I still transiently materialize Y in the forward to hand to the aggregation kernel. Could I kill it with a fused atomic-add to HBM so each expert adds straight into O? I could, but BF16 atomic-add is non-deterministic and numerically lossy, and it would tangle with all-to-all / expert-parallel communication later. Cleaner to materialize Y per layer and recycle the buffer; as long as there are more layers than K (always true at 7B+), that transient is overshadowed. Fine — keep the explicit Y, drop the atomic-add idea.

So the memory problem is solved by a graph rewrite that costs no extra FLOPs. Now back to the IO problem, because solving memory doesn't make the tensor cores busy. The intensity formula told me I'm memory-bound; to win I have to (1) move fewer bytes and (2) hide the bytes I do move behind the matmul.

First, fewer bytes — fusion. The grouped GEMM has a prologue that streams tiles from GMEM to SMEM and an epilogue that post-processes the result. The moment I have to gather scattered token rows for an expert, the lazy approach is a separate gather kernel that writes X_e back to HBM and then a GEMM that reads it — that's an extra round trip of TKd bytes per gathered tensor, and it materializes the very O(TKd) tensor I just worked to avoid. Instead I fuse the gather *into* the prologue: fetch the routed token indices, then issue the asynchronous copy that pulls those exact rows directly into SMEM to feed the tensor core. X_e is never written to HBM at all. The same fusion goes into the backward, where dW1 must gather X, and dW2 and the dH kernel must gather dO. The existing kernels fuse this gather in the forward but, oddly, fall back to a separate gather kernel in the backward — paying 2TKd bytes there. Fusing it everywhere saves that, and the backward of a fine-grained MoE is where most of the time goes.

(One hardware wrinkle on the newest GPUs: the asynchronous copy instruction only signals its own completion within its own thread block, but the matmul on Blackwell can be a two-block cooperative MMA where the leader block's MMA must wait for *both* blocks' gathers to land. So the follower block needs a dedicated relay warp that catches its own copy-completion signal and forwards it to the leader's MMA warp through a cluster-scope barrier. Bookkeeping, but it lets the gather fusion survive the two-block GEMM.)

Then the epilogue fusions. SwiGLU is trivially fuseable into the up-projection epilogue — compute it on the GEMM result in registers instead of writing H, reading it back, applying the activation, writing again. And the big one I already designed: the down-projection activation-gradient kernel's epilogue computes dH, dS, and A' together, so that one kernel replaces what would otherwise be three separate passes over HBM (down-proj activation gradient, then dS, then dSwiGLU). The catch is that this epilogue is now *heavy* — it loads H, runs the dSwiGLU math, does the dS reduction, broadcasts the scores — and a heavy epilogue, naively, stalls the tensor cores while it runs.

Which brings me to hiding the bytes — overlap. On Hopper the GEMM already runs producer-consumer: producer warps stream tiles in via the asynchronous tensor-memory-accelerator copies, consumer warpgroups do the warpgroup-matmul. The trick is that with two consumer warpgroups I can keep the tensor cores fed continuously: while warpgroup 0 grinds through that heavy epilogue, warpgroup 1 is issuing matmul on its tile; when 0 finishes the epilogue they swap roles. Ping-pong. The cores never wait for the epilogue because there's always the other warpgroup feeding them. This is exactly where it pays off: the down-projection forward kernel has a heavy *store* epilogue (it writes 2TKd bytes of Y), and the dH kernel has a heavy *compute+load* epilogue — both are textbook ping-pong candidates, and I also give the dH kernel a dedicated asynchronous copy pipeline to bring in H during the epilogue so even that load overlaps. For kernels that are all mainloop and little epilogue — the weight-gradient kernels dW1, dW2, which reduce over a long token dimension — I don't want ping-pong; I want the *largest* tile and a cooperative schedule where both warpgroups share the tile. So the schedule is chosen per kernel by where the weight sits: ping-pong for heavy-epilogue (Y, dH), cooperative for long-mainloop (dW1, dW2). The overlap machinery itself isn't new — warp-specialization and ping-pong came from attention kernels — but applying it to absorb the *growing IO of fine-grained MoE* is the point. (On Blackwell the same idea wears different clothes: accumulators live in a dedicated on-chip Tensor Memory whose 512 columns split into two 256-column stages, so one stage runs the matmul while the other runs the epilogue and then hands the stage back — a two-stage accumulator pipeline, ping-pong in spirit, and the single-threaded async matmul instruction frees the registers the old warpgroup-matmul ate.)

Now there's a store-side decision I almost got wrong. After the down-projection produces Y per expert, I need O_t = Σ_e s_{t,e} Y_{e,t}. The obvious move, and what the prior kernels do, is to *fuse the scatter into the GEMM store*: as each expert finishes a tile, scatter its rows straight to the tokens' output slots, then a tiny summation kernel reduces. Seems efficient — one fewer pass. But the scatter-fused store on Hopper has to use the synchronous global-store instruction (the asynchronous copy only works for loads, and the tensor-memory-accelerator store needs contiguous addressing, not scattered). A synchronous store *blocks the next matmul tile*, and worse, ping-pong can't rescue it — the epilogue warpgroup is stuck on the store and can't swap roles. On a store-heavy epilogue that's a ~20% throughput hit. So I flip it: the GEMM stores Y *contiguously* via the asynchronous tensor-memory copy (which overlaps the matmul fine), and a *separate* expert-aggregation kernel has each *token gather and sum* its experts' outputs. Each-token-gathers and each-expert-scatters compute the identical thing, but the gather version lets me reuse one gather index across the whole d-row (only K index fetches per token) and, decisively, lets me keep the asynchronous store. So the "unconventional" choice — don't fuse the scatter, do an extra gather-sum kernel — is the one that preserves the overlap I need, and it's the same gather-sum kernel I use for dX in the backward.

I almost forgot the router itself. The expert assignment is a top-K over the E scores per token, and the stock library top-K turns out to eat ~40% of the router's time — embarrassing once everything else is tuned. So a custom top-K: parallelize over tokens, and per row do a bitonic sort over the E scores, which is data-oblivious and maps to register-level and warp-shuffle compare-and-swap with no shared-memory scans. To keep ties deterministic and to recover the argmax indices, pack each score's column index into the low log2(E) mantissa bits of the FP32 value before sorting, with the sign-aware encoding needed for ordinary floating-point order; after sorting, mask those low bits off to recover the score and decode the index. Equal scores now have a fixed column tie-break, and unequal scores keep their order except for the deliberately sacrificed low bits. Then special-case the small base cases with optimal low-latency sorting networks that minimize the number of parallel steps. Everything stays in registers, which is the right shape for replacing the shared-memory-heavy library path.

That's the IO story end to end: fuse the gather into every prologue, fuse SwiGLU/dSwiGLU/dS into the epilogues, overlap the heavy epilogues with matmul via ping-pong (or the Blackwell two-stage pipeline), store contiguously and gather-sum instead of scatter-store, schedule per-kernel, and stop wasting time in top-K. Eight kernels in total: forward up-projection, forward down-projection, forward aggregation; backward dH (with dS and A' folded in), dW2, up-projection activation gradient, dW1, backward aggregation.

There's a third problem I haven't touched, and it only bites in the *sparse* regime. Look at the intensity formula again — lowering ρ also kills it, and there's a discrete reason on top of the continuous one. The grouped GEMM tiles the token dimension into multiples of some tileM (say 128) and pads the rest. When ρ is small, the expected tokens per expert T̄_e = Tρ is small — a 16k-token microbatch with K=4, E=128 leaves only 512 tokens per expert — so each expert's *last* tile is mostly padding. For one expert the exact padding fraction on that token dimension is (⌈T_e/tileM⌉·tileM − T_e)/T_e. If the residue T_e mod tileM is roughly uniform and small relative to T_e, the expected wasted rows are about tileM/2, so the expected fraction is about tileM/(2Tρ), i.e. O(tileM/(Tρ)): linear in the tile size, inverse in ρ. In the highly sparse regime that's a non-trivial slice of runtime, pure waste — multiply-accumulates on padding zeros.

The clean fix: make every expert's token count a multiple of tileM, so there's never a partial tile. But I have to do it without distorting the routing — if I just drop or duplicate tokens carelessly, model quality suffers. Switch Transformer's capacity does exactly the careless thing — it pads under-capacity experts (the waste I'm trying to remove) and drops over-capacity tokens (quality loss). I want something that touches the routing as little as possible: for each expert, change *at most one tile's worth* of tokens relative to the true top-K assignment.

So, token rounding. First run vanilla token-choice top-K and get each expert's true frequency f_e. Round it to the nearest tileM multiple: ⌊f_e⌋ = ⌊f_e/tileM⌋·tileM below, ⌈f_e⌉ = ⌈f_e/tileM⌉·tileM above. For each expert I'll make a single binary decision — round *down* by discarding the lowest-scoring of its assigned tokens, or round *up* by admitting the highest-scoring tokens that *just missed* the top-K cut (expert-choice-style padding) — and either way the change is confined to that last partial tile, so the deviation from true top-K is at most one tile.

The mechanism that guarantees "only the last tile is ever touched" is a little score trick. I want, per expert, a ranking where the genuine top-K tokens always sit strictly above the borderline padding candidates. This only works on router probabilities or other scores already in [0,1], not arbitrary logits. Build a preference-adjusted score S' = S − 1, then overwrite the true top-K entries with their original top-K scores S'[t, top-K(t)] = S_topK. Now for any expert, all its real top-K tokens have nonnegative scores and every other padding candidate sits below zero. Sort each expert's column of S' descending and the top-K tokens are all above the candidates. If I round down, I drop from the bottom — that's the lowest-scoring real top-K tokens, exactly the last partial tile. If I round up, I admit from just below the cut — the best near-miss tokens, again only filling the last tile. Either way the confident assignments are untouched and only the marginal tile moves. For the up/down decision, the default is nearest rounding on frequency: pad up if ⌈f_e⌉ − f_e < f_e − ⌊f_e⌋, else drop down. If I additionally want the *total* token count preserved regardless of E, I can carry a running residual accumulator z across experts and pick whichever rounding direction keeps the accumulated residual smallest — a balanced variant that bounds both the per-expert and total deviation by half a tile. The simple nearest rule is enough as a default because the rest of the construction has already limited the damage to one boundary tile per expert.

The payoff: the varlen-M kernels that dominate the forward and activation-gradient work now see tile-divisible token counts, so the padding tiles I was worried about disappear there; if the weight-gradient tileK divides tileM, the same rounding also removes the analogous token-reduction residue. Because I only perturb the borderline tile, the routing stays close to true top-K instead of becoming capacity dropping or full expert choice. It's a drop-in router that's agnostic to the rest of the kernel, exactly because the whole MoE computation I built is agnostic to the router.

Let me put the computation into code that mirrors how this is actually wired — a PyTorch module that routes, then an autograd Function whose forward is the three grouped-GEMM/epilogue-fused kernels and whose backward is the five, with the cached set held to X and H only. The grouped-GEMM calls below are the fused kernels (gather in the prologue, activation/store in the epilogue, MMA overlapping the IO); I name them as the primitives they are.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Names mirror the PyTorch/CuTe-DSL interface: weights are stored as
# w1: (2I, H, E) and w2: (H, I, E), so the stored dW2 is dO^T A'.
from quack.gemm_interface import gemm, gemm_dgated, gemm_gated
from sonicmoe.functional import TC_Softmax_Topk_Router_Function
from sonicmoe.functional.backward import _token_broadcast_backward
from sonicmoe.functional.forward import _router_forward
from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton


class _UpProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, expert_offsets, x_gather_idx, reverse_scatter_idx, token_offsets):
        # gather X rows in the prologue, up-proj, SwiGLU in the epilogue.
        # Cache H: TK x 2I BF16 = 4TKn bytes. A is not saved by autograd.
        TK = int(expert_offsets[-1])
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
    def backward(ctx, unused_da, dh):                  # dh comes from _DownProjection.backward
        x, w1, expert_offsets, x_gather_idx, reverse_scatter_idx, token_offsets = ctx.saved_tensors
        dx_expanded = gemm(dh, w1.permute(2, 0, 1),
                           cu_seqlens_m=expert_offsets)              # d Xtilde = dH W1^T
        dw1 = torch.empty_like(w1)
        gemm(x.T, dh, out=dw1.permute(2, 1, 0),
             cu_seqlens_k=expert_offsets, A_idx=x_gather_idx)        # stored as (2I,H,E)
        dx = torch.empty_like(x)
        K = reverse_scatter_idx.numel() // x.size(0)
        _token_broadcast_backward(dx, dx_expanded, reverse_scatter_idx,
                                  token_offsets, K, x.size(1),
                                  token_offsets is not None)
        return dx, dw1, None, None, None, None


class _DownProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, h, w2, scores, expert_offsets, x_gather_idx,
                scatter_idx, reverse_scatter_idx, token_offsets, T):
        y = torch.empty(a.size(0), w2.size(0), dtype=a.dtype, device=a.device)
        gemm(a, w2.permute(2, 1, 0), out=y, cu_seqlens_m=expert_offsets)  # Y = A W2
        o = torch.empty(T, w2.size(0), dtype=a.dtype, device=a.device)
        _router_forward(y, o, scores.view(-1), reverse_scatter_idx,
                        token_offsets, scores.size(-1), w2.size(0),
                        token_offsets is not None)
        ctx.save_for_backward(h, w2, scores.view(-1), expert_offsets, x_gather_idx, scatter_idx)
        ctx.score_shape = scores.shape
        return o

    @staticmethod
    def backward(ctx, dO):
        h, w2, scores, expert_offsets, x_gather_idx, scatter_idx = ctx.saved_tensors
        dh = torch.empty_like(h)
        a_prime = torch.empty(h.size(0), w2.size(1), dtype=h.dtype, device=h.device)
        # One heavy-epilogue kernel: gather dO, compute dA' = dO W2^T, then on chip
        #   dA = Broadcast(s) dA';  A = SwiGLU(H);  dH = dSwiGLU(dA, H);
        #   dS = <dA', A> (reduce over n);  A' = Broadcast(s) A.  Y is never touched.
        s = scores[scatter_idx]
        _, _, ds_scattered = gemm_dgated(
            dO, w2.permute(2, 0, 1), PreAct=h, activation="swiglu",
            dx_out=dh, postact_out=a_prime, colvec_scale=s, colvec_reduce=True,
            cu_seqlens_m=expert_offsets, A_idx=x_gather_idx)
        ds = torch.empty_like(scores)
        ds[scatter_idx] = ds_scattered
        dw2 = torch.empty_like(w2)
        gemm(dO.T, a_prime, out=dw2.permute(2, 0, 1),
             cu_seqlens_k=expert_offsets, A_idx=x_gather_idx)        # stored dW2 = dO^T A'
        return None, dh, dw2, ds.view(ctx.score_shape), None, None, None, None, None, None


def moe_layer(x, router_w, w1, w2, K):
    logits = F.linear(x, router_w)
    T, E, TK = x.size(0), router_w.size(0), x.size(0) * K
    scores, indices = TC_Softmax_Topk_Router_Function.apply(logits, E, K, True, False)
    expert_freq = torch.empty(E, dtype=torch.int32, device=x.device)
    offsets = torch.empty(E + 1, dtype=torch.int32, device=x.device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=x.device)
    scatter_idx = torch.empty(TK, dtype=torch.int32, device=x.device)
    reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=x.device)
    TC_topk_router_metadata_triton(indices.to(torch.int32), E, expert_freq,
                                   offsets, x_gather_idx, scatter_idx,
                                   reverse_scatter_idx)
    a, h = _UpProjection.apply(x, w1, offsets, x_gather_idx, reverse_scatter_idx, None)
    o = _DownProjection.apply(a, h, w2, scores, offsets, x_gather_idx,
                              scatter_idx, reverse_scatter_idx, None, T)
    return o


def token_rounding(router_logits, K, E, tileM):
    # (1) true token-choice top-K and per-expert frequency
    scores = router_logits.softmax(dim=-1, dtype=torch.float32)
    topk_scores, topk_idx = scores.topk(K, dim=-1)
    f = torch.bincount(topk_idx.flatten(), minlength=E)
    f_up   = ((f + tileM - 1) // tileM) * tileM
    f_down = (f // tileM) * tileM
    # (2) top-K-preferred score: non-top-K probabilities become negative.
    s_pref = scores.scatter(1, topk_idx, topk_scores).detach() - 1.0
    s_pref.scatter_(1, topk_idx, topk_scores)
    # (3) per expert: sort, then a binary discard/pad to a tileM multiple
    routed = []
    for e in range(E):
        order = torch.argsort(s_pref[:, e], descending=True)
        keep = f_up[e] if (f_up[e] - f[e]) < (f[e] - f_down[e]) else f_down[e]  # nearest
        routed.append(order[:keep])                      # ≤ 1 tile from true top-K
    return routed
```

The whole chain, in one breath: fine-grained, sparse MoE is unavoidably memory-bound — the arithmetic-intensity formula shows IO per FLOP rising with granularity and falling with sparsity — so I rewrote the backward to compute the score gradient as ⟨dA', A⟩ instead of ⟨dO, Y⟩, which never needs Y or dY and pins the cached activation at the dense-equivalent minimum 2Td + 4TKn, flat in granularity; I fused the token gather into every grouped-GEMM prologue and the activation/dS into the epilogues to move fewer bytes, overlapped those heavy epilogues with the matmul through ping-pong (and the Blackwell two-stage accumulator) to hide the bytes that remain, stored contiguously and gather-summed rather than scatter-stored so the store could stay asynchronous, and sped the router with a register-only bitonic top-K; and finally, to kill the discrete tile-padding waste that appears as ρ shrinks, I round each expert's token count to a tile multiple by perturbing only its borderline tile, leaving the routing close to true top-K.
