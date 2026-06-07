# Switch Transformer synthesis

## Pain point
MoE (Shazeer 2017 "outrageously large NN") proved sparsely-gated experts give huge capacity at ~constant FLOPs/token, but adoption stalled on (1) complexity, (2) communication cost, (3) training instability. Goal: keep the capacity-without-compute win, but make routing dead simple, cheap to communicate, and stable enough to train in bf16 at trillion-parameter scale. Also: scale the *parameter count* as a separate axis from FLOPs (Kaplan 2020 power laws → params matter independent of compute).

## Ancestors (load-bearing)
- **Dense Transformer FFN** (Vaswani 2017): per-token FFN h=xW_in, y=ReLU(h)W_out, d_ff≈4·d_model. The slot the experts replace. Applied independently per token.
- **Sparsely-gated MoE** (Shazeer 2017): router p_i(x)=softmax(W_r x)_i; keep **top-k** (k>1) experts; y=Σ_{i∈T} p_i(x)E_i(x). Conjectured k≥2 NECESSARY for nontrivial router gradient ("can't learn to route without comparing ≥2 experts"). Used TWO aux losses (importance + load), noisy top-k gating. Collapse problem: router favors few experts, self-reinforcing.
- **Ramachandran & Le 2018** (diversity/depth of routing): found higher-k in lower layers helps when many routing layers. Reinforces "k>1 needed" prior.
- **GShard** (Lepikhin 2020): MoE Transformer for MT across 100 langs; top-2; trains in **float32 throughout** for stability; introduced fixed **expert capacity** + drop-to-residual on overflow; simplified to single aux load loss (dot of f and P). Mesh-TF.
- **Mesh-TensorFlow** (Shazeer 2018): SPMD lib; logical mesh of cores; tensors sharded by named dims; static shapes for TPU. Reintroduced MoE into Transformer FFN slot (no NLP results). The compute substrate.
- **Kaplan 2020 scaling laws**: power-law in params/data/compute; train big models on relatively little data is compute-optimal. Motivates "param count is its own axis."

## Method derived (the Switch layer)
- **Top-1 routing.** Replace top-k with k=1: route each token to argmax expert only. Output y = p_i(x)·E_i(x) where i=argmax. The gate value p_i still multiplies the output → router stays differentiable (gradient flows through p_i scaling the residual contribution). This REFUTES the Shazeer-2017 conjecture that k≥2 is needed.
  - Benefits: (1) router compute halved+ (only one selection); (2) expert capacity (per-expert batch) at least halved since each token goes to ONE expert; (3) simpler dispatch, less all-to-all communication.
- **Expert capacity + token dropping.** TPU needs static shapes → each expert has a FIXED buffer. expert_capacity = (tokens_per_batch / num_experts) × capacity_factor. CF>1 buffers imbalance. If an expert overflows, extra tokens are DROPPED → their representation passes through the residual unchanged (skip the FFN). CF too high = wasted pad compute/memory; CF too low = more drops. Switch works well at LOW CF (1.0–1.25) — that's its edge over top-2 MoE which needs more buffer.
- **Differentiable load-balancing loss.** Single aux loss (vs Shazeer's two):
    loss = α · N · Σ_{i=1}^N f_i · P_i
  f_i = fraction of tokens whose argmax = i = (1/T)Σ_x 1{argmax p(x)=i}  (NON-differentiable, hard count)
  P_i = mean router prob to expert i = (1/T)Σ_x p_i(x)  (differentiable)
  Minimized under uniform routing (f_i=P_i=1/N → Σ f_i P_i = N·(1/N²)=1/N → loss = α·N·(1/N)=α). The ×N keeps loss magnitude ≈ constant as N varies. Gradient flows through P_i only; f_i acts as a per-expert weight on P_i. Pushing P_i down for over-used experts (large f_i) and up for under-used ones → balances. α=1e-2 (swept 1e-1..1e-5).
  - Why this is the *right* surrogate: we want equal token COUNTS (f_i=1/N) but counts have no gradient. P_i is a smooth proxy for "how much this expert is wanted." Weighting P_i by the observed load f_i gives the loss its minimum exactly at uniform; minimizing the weighted-prob mass under high-load experts reduces their probabilities → fewer future assignments.
  - Appendix code: reduce_mean(density_proxy * density_1) * num_experts^2. density_1 = mean over tokens of one-hot argmax = f (length-N, sums to 1). density_proxy = mean over tokens of router_probs = P. reduce_mean over the N experts divides by N: (1/N)Σ f_i P_i · N² = N Σ f_i P_i. Matches main text (α applied at the model-loss level). Verified.
- **Selective precision.** bf16 softmax in router → instability (exp of bf16 logits). Cast router INPUT/logits to float32, do softmax in fp32, recast dispatch/combine tensors to bf16 BEFORE the all-to-all. So fp32 only LOCAL to the router on-device; no fp32 broadcast over the network. Gets fp32 stability at bf16 comm cost. (Table: bf16 diverges -3.78; selective-precision -1.716 ≈ fp32 -1.718, at bf16 speed.)
- **Smaller init scale.** Truncated normal, μ=0, σ=√(s/n), n=fan-in. Reduce default Transformer s=1.0 by 10× → s=0.1. Big improvement in mean quality + far lower variance across seeds (table: 0.1x-init -2.72±0.01 vs 1.0x-init -3.60±0.68). Hard-switching routing amplifies init noise → smaller init tames early instability.
- **Expert dropout.** Fine-tuning small downstream tasks overfits (Switch has many more params). Uniformly raising dropout hurts. Instead: low dropout (0.1) at non-expert layers, MUCH higher (0.4) inside experts → "expert dropout." Best across GLUE/CNNDM/SQuAD/SuperGLUE.
- **Router exploration noise.** Input jitter: multiply router input by uniform[1-eps,1+eps] during training. Beats argmax/sample-softmax/input-dropout (Table top1_noise: input jitter -1.468 best).

## Distributed mechanics (Section 6, appendix pseudo-code)
- Mesh-TF: cores N = n (data) × m (model). Switch-C uses pure expert-parallel (n=N=num_experts, m=1).
- Router produces dispatch_tensor (binary [cores, tokens_per_core, E, C]) and combine_tensor (float, same shape, holds the gate value at the chosen (expert,slot)).
- einsum(inputs, dispatch) → per-expert batches [E, cores, C, d_model]; all-to-all to reshard from cores→experts; FFN; all-to-all back; einsum(expert_outputs, combine) scales by gate value and scatters back.
- token_priority = cumsum of one-hot over tokens; keep where cumsum ≤ capacity → drops overflow.
- Comm cost of dispatch/combine all-to-all ∝ E·C·d_model in bf16. Top-1 halves C vs top-2 → halves this.

## No-Token-Left-Behind (appendix, NEGATIVE result)
Iteratively reroute overflow tokens to 2nd-highest expert to avoid drops. Guarantees ~0 drops. Hypothesized helps; found NO empirical benefit (suspect: once token↔expert assoc learned, rerouting to 2nd choice degrades). Mention as a tried-and-abandoned branch.

## Switch-for-Attention (appendix, future-work branch)
Replace Q/K/V weight matrices with Switch layers. Quality gains in fp32 but diverges in bf16 → left out of final model.

## Design-decision → why table
- top-1 not top-k → max sparsity, halve router compute + capacity + comm; differentiability preserved by the p_i output scaling (refutes need for k≥2).
- gate value multiplies expert output → keeps router in the gradient graph with k=1.
- capacity_factor>1 → static-shape buffer for imbalance; low CF favored (memory-scarce large regime) and Switch tolerates low CF.
- drop-to-residual on overflow → mathematically clean (skip = identity via residual), needed for static shapes.
- single f·P aux loss (not 2) → simpler; minimum at uniform; ×N keeps scale; α=1e-2 swept.
- f non-diff, P diff → gradient still trains router toward balance via P weighted by load.
- fp32 router only (selective precision) → stability of fp32 softmax without fp32 all-to-all cost.
- 0.1× init → tame hard-switch-amplified early variance.
- expert dropout 0.4 vs 0.1 elsewhere → regularize the high-capacity experts on small fine-tune sets without crippling shared layers.
- input jitter → exploration in the contextual-bandit-like routing; beats sampling/dropout.

## Code grounding
- HF SwitchTransformersTop1Router: fp32 cast → optional jitter → linear classifier → softmax(fp32)→ argmax (max) → one_hot → cumsum priority → capacity mask → return (gate prob, expert one-hot, logits).
- load_balancing_loss_func: one_hot argmax → mean over tokens (=f) ; mean router_probs over tokens (=P); mean(f*P)*N². Matches paper eq 4-6.
- SSExperts/SparseMLP: per-expert FFN (wi/wo, ReLU), index_add weighted by gate.
- _init_weights: experts wi/wo std=factor·d_model^-0.5; router classifier std=factor·1; factor=initializer_factor (the s reduced to ~0.1 for Switch). LayerNorm = RMSNorm (no mean, no bias) — T5 style.
- NOTE: HF also has router_z_loss (from 2202.08906, a LATER paper) — POSTERIOR to Switch, exclude from reasoning/answer.
- For answer/reasoning final code: present a self-contained PyTorch SwitchFFN (router top-1 + capacity drop + aux loss + fp32 router) grounded in HF, plus the Mesh-TF appendix pseudocode flavor for the distributed dispatch/combine.

## Equations to flag for Codex gate
1. aux loss = α·N·Σ f_i·P_i ; uniform value Σ f_i P_i = N·(1/N²)=1/N ; loss→α. (eq 4-6)
2. appendix code mean(f*P)*N² == N·Σ f_i P_i reconciliation.
3. expert_capacity = (tokens/experts)·CF.
4. f_i = (1/T)Σ 1{argmax=i} (non-diff), P_i=(1/T)Σ p_i(x) (diff).
5. top-1 differentiability via y=p_i(x)E_i(x) scaling.
