# Synthesis — Speculative Decoding (arXiv 2211.17192)

## Pain point / research question
Autoregressive decoding from a large Transformer M_p generates K tokens with K *serial* forward passes. Each pass is memory-bandwidth-bound at batch=1: the bottleneck is streaming the model weights (and the KV cache) from HBM, not the arithmetic. So the accelerator's FLOPs sit idle. Crucially: scoring K tokens *in parallel* (one forward pass over a length-K block, teacher-forced) costs about the same wall-time as scoring one token, because the same weights are streamed once. So there is slack: compute is free, latency is serial.

Goal: cut the number of serial M_p passes WITHOUT changing the output distribution and WITHOUT retraining / architecture change.

## Ancestors (load-bearing)
- Autoregressive Transformer decoding (Vaswani et al. 2017): the serial generation loop; one token per pass.
- Memory-bandwidth-bound inference: at batch 1, decode is dominated by weight + KV reads from HBM; arithmetic units underutilized. Scoring a block of K tokens costs ~one pass.
- Adaptive computation / early exit (Schuster et al., Graves, Elbayad et al., "Wisdom of Committees" Mamou et al.): "not all steps are equally hard" — but these change architecture/training and the output distribution.
- Speculative execution in processors (Burton 1985; Hennessy & Patterson): do a task in parallel with verifying whether it was needed; branch prediction. Payoff = concurrency when the guess is right. Here it's deterministic; the contribution is generalizing it to the *stochastic* setting.
- Rejection sampling (von Neumann): to sample p using proposal q, accept x~q w.p. p(x)/(M q(x)), M = max_x p/q; on reject, retry. The constant M can be huge → low accept rate, and retry discards the parallel work.
- Importance sampling: reweight q-samples by p/q — but that changes the *samples* (weighted), doesn't give exact p-distributed *draws*.
- Blockwise Parallel Decoding (Stern et al. 2018): predict several tokens in parallel, verify with the big model — but greedy-only, needs a custom-trained model, optimizes downstream quality not exact distribution.
- Shallow Aggressive Decoding (Sun et al.): parallel decode by copying input→output, only for input≈output tasks (GEC), greedy-only.

## The method (final)
1. Draft: run small M_q autoregressively to propose γ tokens x_1..x_γ (cost cγT, c≪1).
2. Verify: run M_p ONCE over prefix+[x_1..x_γ] (a length-(γ+1) block, teacher-forced) to get p_1=p(·|prefix), p_2=p(·|prefix,x_1), …, p_{γ+1}=p(·|prefix,x_1..x_γ) in parallel — one M_p pass.
3. Accept/reject (speculative sampling): for i=1..γ, draw r_i~U(0,1); accept x_i iff r_i ≤ p_i(x_i)/q_i(x_i) (always accept if p≥q). Let n = number accepted before first rejection.
4. Fix-up: if a rejection occurred at position n+1, sample the next token from the residual p'(x)=norm(max(0, p_{n+1}(x) − q_{n+1}(x))) and STOP. If all γ accepted, sample one free bonus token from p_{γ+1}.
5. Emit x_1..x_n plus the one fix-up/bonus token: between 1 and γ+1 tokens per M_p pass. Worst case 1 → never worse than standard decoding.

Standardization: argmax/top-k/nucleus/temperature all recast as standard sampling from an adjusted categorical, so the analysis only needs "sample from a distribution."

## Core theorem (exactness) — appendix:correctness
Single position. Sample x~q, accept w.p. min(1,p/q), else resample from p'=norm(max(0,p−q)).
- P(accept, x=x') = q(x')·min(1, p(x')/q(x')) = min(q(x'), p(x')).
- Normalizer of p': Σ_x max(0, p−q) = Σ_x (p − min(p,q)) = 1 − Σ_x min(p,q) = 1 − β.
- P(reject, x=x') = (1−β)·p'(x') = p(x') − min(q(x'), p(x')).
- Sum = min(p,q) + p − min(p,q) = p(x'). ∎  Holds for ANY q (no support/coverage condition, unlike rejection sampling).

## Acceptance rate β and α — sec analysis
- β = E_{x~q} min(1, p(x)/q(x)) = Σ_x min(p(x), q(x)).
- Divergence D_LK(p,q) = Σ|p−M| (M=(p+q)/2) = Σ|p−q|/2 = 1 − Σmin(p,q). Symmetric, in [0,1]; 0 iff p=q; 1 iff disjoint support. (total-variation distance)
- Theorem: β = 1 − D_LK(p,q). α := E(β) = E(min(p,q)) = 1 − E(D_LK).

## Expected tokens — eq:expected_num_tokens
i.i.d. β assumption, α=E(β). #accepted is geometric capped at γ; #tokens = #accepted + 1.
E[#tokens] = Σ_{i=0}^{γ} α^i = (1 − α^{γ+1})/(1 − α).

## Walltime — thm:total_walltime
c = (single M_q run)/(single M_p run). One step costs T(cγ+1) (γ draft runs + 1 parallel M_p block), yields (1−α^{γ+1})/(1−α) tokens.
Cost/token = T(cγ+1)(1−α)/(1−α^{γ+1}). Speedup vs T:
  (1 − α^{γ+1}) / ((1−α)(cγ+1)).
γ=1 ⇒ (1+α)/(1+c). Improvement exists iff α>c. Optimal γ found numerically.
Negligible-cost M_q (c≈0): speedup = (1−α^{γ+1})/(1−α) → 1/(1−α) as γ→∞.

## Arithmetic ops — thm:num_ops
ĉ = ops/token of M_q vs M_p. One step = T̂ĉγ + T̂(γ+1) ops. Increase factor = (1−α)(γĉ+γ+1)/(1−α^{γ+1}). Low α ⇒ more wasted compute. Memory accesses go DOWN by (1−α^{γ+1})/(1−α) (weights/KV read once per step).

## vs rejection sampling — sec:vs_rejection_sampling
Non-iterative rejection sampling accept prob = E_{x~q} p/(Mq) = (1/M)·1, M=max p/q ⇒ accept = Σ p · min_{x'} q/p ≤ Σ min(p,q) = α. So speculative sampling's accept rate ≥ rejection sampling's, and it reuses (not discards) the parallel work.

## Lenience (appendix)
Multiply q by l∈[0,1] before compare ⇒ α = Σ min(p/l, q); guarantees no token sampled > p/l. Trades exactness for speed.

## Beam search (appendix)
Draft beam width u≥w for γ steps; M_p checks all (w+uγ) candidates in parallel; accept while top_w(M_p) ⊆ top_u(M_q). Left as future work.

## Design choices → why
- Draft γ then verify in one block: exploits "K-token scoring ≈ 1-token cost" + memory-bound slack.
- min(1,p/q) accept (not rejection sampling's p/(Mq)): higher accept rate (α ≥ rejection's), and no global constant M.
- Residual norm(max(0,p−q)) on reject: exactly the mass p didn't share with q; makes total = p (the theorem). Resampling from raw p would double-count.
- Stop at first rejection: positions after a rejected token were conditioned on a token that won't be emitted — their M_p distributions are now invalid, so they must be discarded.
- Bonus token from p when all accepted: the (γ+1)-th M_p distribution p_{γ+1} is already computed for free in the same block; sampling it is exact and gains a token.
- M_q ~2 orders smaller: balances α (want big M_q) vs c (want small M_q).

## Canonical code
feifeibear/LLMSpeculativeSampling — speculative_sampling_v2 (DeepMind no-KV-cache variant) cleanly mirrors Algorithm 1. utils: norm_logits, sample (multinomial), max_fn = norm(max(x,0)). NOTE upstream residual indexing uses p[:,n,:]−q[:,n,:] at reject which equals position prefix_len+i-1 = n+... actually n is updated to prefix_len-1+accepted; at the rejecting i, the correct residual position is prefix_len+i-1. Will write final code with explicit correct indexing.
