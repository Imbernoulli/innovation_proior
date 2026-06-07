# Synthesis — LoRA (Low-Rank Adaptation)

## The pain point (research question)
Pretrain-then-finetune is the dominant NLP paradigm. Full fine-tuning produces, for *each*
downstream task, a new parameter set ΔΦ with |ΔΦ| = |Φ₀|. For GPT-3 175B that is a 175B-param
checkpoint *per task*. Storage/serving cost: hosting N task-specialized models ≈ N × 350GB.
Switching tasks = swapping a whole model. Also training cost: Adam keeps optimizer state (m, v)
for every trainable param → ~3× the param memory; the gradient must be computed for all params.

What a solution must achieve simultaneously:
1. Tiny per-task footprint (store/ship megabytes, not hundreds of GB).
2. Match full-FT quality (prior efficient methods traded quality for size).
3. NO added inference latency (production, online, batch size 1).
4. NOT consume the context window (don't steal sequence length).
5. Cheap task switching at serve time.

## Load-bearing ancestors (verified against primary text + refs)

### Full fine-tuning (the baseline being reacted against)
Objective: max_Φ Σ_{(x,y)} Σ_t log P_Φ(y_t | x, y_<t). Init Φ=Φ₀, follow gradient. ΔΦ has
dimension = |Φ₀|. Gap: per-task |ΔΦ|=|Φ₀|; prohibitive storage/switching at GPT-3 scale; full
optimizer state.

### Adapter layers (Houlsby et al. 2019; Lin et al. 2020; Pfeiffer; Rebuffi 2017)
Insert small bottleneck MLP modules *between* existing layers. Houlsby: two adapters per
Transformer block, each is down-proj (d→r) + nonlinearity + up-proj (r→d) + residual, with
biases. Lin et al.: one adapter per block (after MLP) + a LayerNorm. Param count per adapter:
2·d_model·r + r + d_model (+ LayerNorm). Few params (<1% sometimes).
GAP — INFERENCE LATENCY: adapters add *depth*. They are computed *sequentially* (in addition to
the base block), can't be folded into existing weights, can't be parallelized away. Their FLOPs
are tiny but latency ≠ FLOPs: large nets rely on hardware parallelism; a thin extra layer still
forces an extra sequential GPU kernel + sync. Hurts most at batch size 1 / short sequence
(online inference). Measured: GPT-2 medium, bs=1, seqlen=128 → Adapter^H +30.3%, Adapter^L
+20.7% latency. Worse under model sharding (extra AllReduce/Broadcast).

### Prefix / prompt tuning (Li & Liang 2021 prefix-tuning; Lester 2021; Hambardzumyan 2020 WARP; Liu 2021 P-tuning)
Optimize *input activations* instead of weights. PreEmbed: prepend l_p (+infix l_i) trainable
"virtual token" embeddings; |Θ| = d_model·(l_p+l_i). PreLayer: also replace activations after
every layer; |Θ| = L·d_model·(l_p+l_i).
GAP 1 — eats sequence length: the prefix occupies positions in the context, reducing usable
length for the actual task. GAP 2 — hard to optimize: performance is non-monotonic in #params
(adding tokens can hurt); poor in low-data (GPT-3 PreEmbed on MNLI-100 ≈ chance, 37.6% vs 33.3%).

### Intrinsic dimensionality of fine-tuning (Li et al. 2018 "Measuring intrinsic dim"; Aghajanyan et al. 2020) — the MOTIVATION
Aghajanyan reparameterizes θ = θ₀ + P θ_d, where θ_d ∈ R^d (small), P: R^d → R^D a fixed random
(Fastfood) projection; only θ_d is trained. d90 = smallest d reaching 90% of full-FT perf.
Finding: d90 is astonishingly small — RoBERTa needs ~200 trainable params to hit 90% on MRPC.
Larger pretrained models have *lower* intrinsic dimension. So the *solution* of fine-tuning lives
on a very low-dimensional manifold even though the ambient param space is huge.
KEY LEAP this enables: Aghajanyan's projection is into a random subspace of the *full parameter
vector*. The natural next question: if the update lives in a low-dim subspace, can we pick a
*structured, per-matrix* low-dim subspace that (a) is learnable, (b) costs no inference latency,
(c) doesn't need a giant random projection matrix P? Answer → constrain ΔW per weight matrix to
be low *rank*.

## The method derivation (insight-before-method)

Hypothesis: the weight *update* ΔW during adaptation has low "intrinsic rank" (analogous to
low intrinsic dimension of the solution). So instead of learning a full d×k ΔW, write
ΔW = B A, with B ∈ R^{d×r}, A ∈ R^{r×k}, r ≪ min(d,k). Forward:
  h = W₀ x + ΔW x = W₀ x + B A x.
W₀ frozen; only A, B trained. Param count per matrix 2·d·r → e.g. d=12288, r=4 → ~10⁴× fewer
than d·k.

### Why this beats each ancestor, by construction
- vs adapters: BAx is a *parallel* branch on the *same* input as W₀x, not extra depth. At deploy,
  merge W = W₀ + BA (same d×k shape). Inference is one matmul through W → identical latency to the
  un-adapted / fully-fine-tuned model. Zero added latency BY CONSTRUCTION. Task switch: subtract
  BA, add B'A'.
- vs prefix: doesn't touch the input sequence at all → no lost context length, no
  non-monotonic optimization pathology.
- vs full FT: generalizes it — let r → rank(W₀) and apply to all matrices (+biases) and you
  roughly recover full-FT expressiveness. As you add params, LoRA → training the original model;
  adapters → an MLP; prefix → a model that can't take long inputs.

### Design choice: zero-init B, Gaussian-init A
Want ΔW = BA = 0 at step 0 so the adapted model = pretrained model exactly (no random
perturbation to the carefully-pretrained W₀; training starts from the known-good point and the
first gradient step is meaningful). With B=0, BA=0 regardless of A; A gets random Gaussian so it
isn't stuck (if both were 0, ∂L/∂B = (∂L/∂h)·(Ax)ᵀ would be 0 → B never moves; A's gradient is
Bᵀ·... = 0 too → dead). So exactly one of {A,B} is zero, the other random. (Canonical code puts
kaiming-uniform on A and zeros on B — same invariant BA=0, symmetric to the paper's "A Gaussian,
B zero". Either works; the invariant is what matters.)

### Design choice: scale ΔWx by α/r
Set ΔWx ← (α/r)·BAx. Why divide by r: as you raise r, BA sums over more rank-1 terms so its
typical magnitude grows with r; dividing by r keeps the *effective scale* of the update roughly
constant as r changes, so you don't have to re-tune the learning rate every time you change r
(connects to Yang et al. 2021 feature-learning / μP scaling). α is just a constant in r — set it
to the first r you try and leave it; with Adam, tuning α ≈ tuning the learning rate (up to init
scale). So α/r is a convenience knob that decouples r from the LR.

### Design choice: which matrices? only attention W_q, W_v (in main experiments)
Transformer has W_q,W_k,W_v,W_o (attention) + 2 MLP matrices. For simplicity + param efficiency,
adapt only attention weights, freeze MLP. Treat W_q etc. as single d_model×d_model matrices
(ignore head slicing). Diagnostic finding: given a fixed 18M budget on GPT-3, spreading rank
across {W_q,W_v} beats putting it all in one type at higher r; even r=1 or 2 across two matrices
works. (This is a motivating/diagnostic observation, in-frame ok as "what I'd want to check".)

## Empirical/diagnostic findings that motivate (allowed as context / "want to verify")
- Adapter latency table (above): the concrete pain that rules out adapters for online serving.
- Prefix non-monotonicity + low-data collapse: rules out prefix for robustness.
- Aghajanyan d90 ≈ 200 for RoBERTa/MRPC; larger ⇒ lower intrinsic dim.
NOTE: the proposed method's own win tables (GLUE/E2E/GPT-3 accuracy, the rank-deficiency SVD
study, amplification factor) are PROPOSED-METHOD RESULTS → excluded from context/reasoning,
mentioned only as "what I'd want to validate".

## Canonical implementation (microsoft/LoRA, loralib)
- `loralib/layers.py`: `LoRALayer` base (r, lora_alpha, lora_dropout, merge_weights, merged).
  `Linear(nn.Linear, LoRALayer)`: lora_A (r×in), lora_B (out×r), scaling=alpha/r, freeze
  self.weight. reset_parameters: kaiming_uniform A, zeros B. forward (unmerged):
  `F.linear(x, W, b) + (dropout(x) @ A.T @ B.T) * scaling`. train(mode): on eval, merge
  W += BA*scaling; on train, unmerge W -= BA*scaling (so you can keep training).
  fan_in_fan_out handles Conv1D-style stored weights (GPT-2).
  Also Embedding, MergedLinear (single qkv matrix, enable_lora mask + grouped conv1d to build
  ΔW), ConvLoRA.
- `loralib/utils.py`: mark_only_lora_as_trainable (set requires_grad False unless 'lora_' in
  name; optional bias='all'/'lora_only'); lora_state_dict (save only lora_ params + optional
  bias) → tiny checkpoints.
- Usage: replace nn.Linear with lora.Linear(..., r=8), mark_only_lora_as_trainable(model),
  train, torch.save(lora_state_dict(model)).

## Code-framework scaffold (pre-method) ↔ final code correspondence
Pre-method scaffold = a frozen base linear layer + ONE empty slot for "the parameter-efficient
adapter we will design" + a "mark trainable" stub + a "save only the small params" stub + a
"merge for deployment" stub. Final code fills: slot → lora_A/lora_B + alpha/r + forward;
mark-trainable → mark_only_lora_as_trainable; save-small → lora_state_dict; merge → train(mode)
weight merge. Piece-for-piece.
