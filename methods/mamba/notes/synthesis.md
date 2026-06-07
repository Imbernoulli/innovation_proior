# Synthesis — Selective State Space Models (Mamba)

## Pain point at the time
- Transformers/attention dominate sequence FMs. Attention routes information densely within a window but: (a) cannot model anything outside a finite window; (b) scales quadratically in sequence length for training; (c) autoregressive inference requires the full KV cache (linear-in-length per-step memory, slow). Efficient-attention variants trade away the quality that makes attention work and none have been empirically effective at scale across domains.
- Recurrent/SSM models are efficient (finite state → constant-time per-step inference, linear-time training) but had been *less effective on discrete, information-dense data like text*. They dominate continuous-signal benchmarks (Long Range Arena, audio) but lag attention on language.

## Core object: compression of context into a state
- Sequence modeling = compressing context into a (small) state. Attention does NOT compress (stores entire context → KV cache) → effective but inefficient. RNN/SSM compress into finite state → efficient but effectiveness bounded by how good the compression is.
- Efficiency↔effectiveness tradeoff is governed by state size. The goal: a model whose state can hold *all necessary* context, yet stays small and cheap. Resolution: make compression *content-aware* (selective) — keep what matters, discard the rest.

## Ancestors (load-bearing) and the gap each leaves
1. **Attention / Transformer (Vaswani 2017; Bahdanau 2015).** Dense pairwise routing. Gap: O(L²) train, KV cache O(L) at inference, finite window.
2. **Classical SSM / Kalman (1960).** Continuous linear system h'=Ah+Bx, y=Ch. Basis object.
3. **HiPPO (Gu 2020).** Theory of online function approximation: a specific A matrix optimally compresses history into coefficients of orthogonal polynomials. Gives SSMs principled long-range memory and the special A initialization. Gap: by itself just an init; LTI.
4. **S4 (Gu 2021/2022).** First *structured* SSM for deep learning. Discretize (ZOH) continuous (Δ,A,B) → (Ā,B̄). Two computation modes: linear recurrence (inference) and **global convolution** (training). Convolution kernel K̄ = (CB̄, CĀB̄, ..., CĀ^k B̄, ...), y = x ∗ K̄. To compute K̄ efficiently for the HiPPO A, S4 uses Normal-Plus-Low-Rank → Diagonal-Plus-Low-Rank, truncated generating function + Cauchy kernel + Woodbury identity + inverse FFT, giving O(N+L). Gap: **LTI** — (Δ,A,B,C) constant across time, which is exactly what enables the convolution; but constant dynamics cannot do content-based selection.
5. **S4D / DSS (Gupta 2022; Gu 2022).** Diagonal A suffices (drop low-rank). Simplifies S4 dramatically; A is just N numbers per channel. S4D-Real init A_n = -(n+1); S4D-Lin A_n = -1/2 + n·i. Gap: still LTI.
6. **S5 (Smith 2023).** Computes the diagonal SSM *recurrently with a parallel (associative) scan* instead of convolution; switches SISO→MIMO to keep the state small. First to use the scan in this line. Gap: MIMO lowers the effective per-channel state; still LTI/non-selective.
7. **Parallel associative scan (Blelloch 1990; Martin & Cundy 2018).** A linear recurrence h_t = a_t h_{t-1} + b_t is a prefix-scan under the associative operator (a,b)•(a',b') = (a'a, a'b + b'). Work-efficient O(L) work, O(log L) depth. This is what lets a recurrence be parallelized over time even when a_t,b_t vary — the key that survives dropping LTI.
8. **Linear attention (Katharopoulos 2020).** Softmax→kernel feature map makes attention a linear recurrence (a degenerate LTI SSM). Showed attention↔recurrence duality. Gap: LTI-ish, weak quality.
9. **H3 (Dao 2023).** SSM sandwiched by multiplicative gates + a shift-SSM (short conv). The de-facto SSM architecture block; interleaved with an MLP. Inspiration for the architecture.
10. **GAU (Hua 2022).** Merged attention + MLP into one block. Inspiration for collapsing H3-block + MLP into a single homogeneous Mamba block.
11. **Gated RNNs (LSTM/GRU); QRNN (Bradbury 2016), SRU (Lei 2017).** Input-dependent gates h_t=(1-g_t)h_{t-1}+g_t x_t. Powerful but: N=1 (no state expansion), heuristic gates, vanishing-gradient/efficiency issues historically. Connection: gating = discretization of a continuous SSM with input-dependent Δ (Tallec 2018, Funahashi 1993).
12. **FlashAttention (Dao 2022).** IO-aware kernel: most ops are memory-bandwidth bound; fuse + recompute to cut HBM traffic. The template for the hardware-aware scan.

## Diagnostic tasks (motivating, pre-method, allowed in context)
- **Copying** (Arjovsky 2016): constant spacing → solvable by LTI (a convolution kernel of the right length / a fixed-delay recurrence). Time-awareness only.
- **Selective Copying** (a.k.a. Denoising, Jing 2019): random spacing between data tokens, interspersed with noise tokens. Needs *content-aware* filtering: remember colored tokens, drop white ones. LTI fails — a static convolution kernel can't model variable spacing; constant recurrent dynamics can't choose what to keep.
- **Induction Heads** (Olsson 2022): associative recall ("Harry Potter"… then "Harry"→"Potter"). Needs context-aware retrieval at the right moment. Predictive of in-context learning.
- Diagnostic ablation finding (about SSMs generally, knowable pre-method): increasing recurrent state dimension N should help quality but costs compute/memory in the naive recurrence — the expressivity↔speed tension that motivates wanting state expansion *for free*.

## The method derivation (insight → form)
1. **Selection.** Let the SSM parameters be functions of the input. Specifically make Δ, B, C input-dependent (A stays static):
   - s_B(x)=Linear_N(x), s_C(x)=Linear_N(x), s_Δ(x)=Broadcast_D(Linear_1(x)), τ_Δ=softplus, Δ=softplus(param+s_Δ(x)).
   - Shapes change: B,C: (B,L,N); Δ: (B,L,D); Ā,B̄: (B,L,D,N). Model becomes **time-varying**.
   - Why Δ,B,C and not A: A only enters via Ā=exp(ΔA), so selective Δ already makes Ā,B̄ selective; selective A is redundant for selectivity → left out for simplicity. B controls what of x_t enters the state; C controls what of the state leaves to y_t; Δ is the master gate (focus current input vs persist state).
2. **Wall: time-varying kills convolution.** The convolution form requires a single fixed kernel K̄ = (CB̄, CĀB̄, …) reused at every position — only valid under LTI. Once Ā_t,B̄_t,C_t vary per position there is no single kernel; the whole O(N+L) FFT machinery of S4 is gone. Must fall back to the recurrence.
3. **Wall: naive recurrence materializes (B,L,D,N) state → memory blowup + sequential.** State is N× bigger than input. Two sub-problems: sequential dependency, and HBM memory.
4. **Patch A — parallel scan.** Even though dynamics vary, h_t = Ā_t h_{t-1} + B̄_t x_t is still a *first-order linear* recurrence → associative scan with operator (a,b)•(a',b') = (a'a, a'b+b'). Work-efficient, parallel over L. Recurrent FLOPs O(BLDN) (linear in L) vs convolution O(BLD·logL); recurrence has the smaller constant, so for long L and modest N it's actually fewer FLOPs.
5. **Patch B — kernel fusion (IO-aware), à la FlashAttention.** Don't materialize Ā,B̄ (size BLDN) in HBM. Load (Δ,A,B,C) from HBM→SRAM, discretize in SRAM, scan in SRAM, multiply by C and sum, write only y (size BLD) back. Cuts HBM IO by ~N.
6. **Patch C — recomputation.** Don't store the (B,L,D,N) intermediate scan states for backward; recompute them in the backward pass from the (small) inputs reloaded HBM→SRAM. Same activation memory as FlashAttention.
7. **Discretization (ZOH), worked.** Ā=exp(ΔA); B̄=(ΔA)^{-1}(exp(ΔA)-I)·ΔB. Diagonal A → elementwise. Often simplified to Euler-ish B̄≈ΔB.
8. **Gating theorem.** N=1, A=-1, B=1, s_Δ=Linear(x), τ_Δ=softplus ⇒ continuous system is the leaky integrator h'=-h+x. Δ_t=softplus(Linear(x_t)). ZOH: Ā=exp(ΔA)=1/(1+e^{Linear})=σ(-Linear)=1-σ(Linear); B̄=-(exp(ΔA)-I)=1-Ā=σ(Linear). So with g_t=σ(Linear(x_t)): h_t=(1-g_t)h_{t-1}+g_t x_t — exactly the classic RNN gate. ⇒ "discretization is the principled foundation of heuristic gating", and motivates choosing softplus for Δ.
9. **Architecture.** Collapse H3-block (SSM with two gates) + MLP block into one homogeneous block (à la GAU): in_proj expands D by E=2 into two branches (x and z); x → short causal conv1d → SiLU → selective SSM; multiply by SiLU(z) gate; out_proj back to D. Stack homogeneously with norm + residual. Params ≈ 3ED² per block (2ED² in, ED² out); two blocks ≈ Transformer's 12D² (MHA+MLP). Real-valued default; S4D-Real init A_n=-(n+1); Δ bias init so softplus(bias)∈[0.001,0.1].

## Design-decision → why table
- **Selective Δ,B,C, static A** — A acts only through exp(ΔA), so selective Δ already gives selective Ā; selective A redundant → simplicity. Ablation: Δ most important (gating link), but Δ+B+C synergize.
- **softplus for Δ** — falls out of the gating equivalence (gives the σ gate via ZOH); keeps Δ>0 (a positive timestep).
- **Project Δ through dim-1 then broadcast** — if a token must be ignored, all D channels must ignore it together → a single scalar gate per token, broadcast. Generalizable to rank-R low-rank projection.
- **Diagonal A** — N numbers/channel; elementwise discretization; the only structure that keeps the scan cheap.
- **Real-valued default** — complex helps continuous modalities (audio) but not discrete (text/DNA); real is simpler and hardware-friendly.
- **State expansion N (≈16)** — bigger N = bigger effective recurrent memory = better compression; only affordable because the scan never materializes the NDB state in HBM. Helps strongly only when B,C selective.
- **Parallel scan not convolution** — convolution needs LTI (single kernel); selectivity forbids it; scan parallelizes the time-varying recurrence.
- **Kernel fusion + recomputation** — recurrence's enemy is memory IO/blowup; FlashAttention-style fusion fixes it; recompute states in backward to match attention's memory.
- **Short causal conv before SSM** — local context mixing (the H3 shift-SSM role); cheap.
- **z gate (SiLU) / single homogeneous block** — merges H3+MLP (GAU idea); SiLU makes the gated MLP a SwiGLU-like unit; replaces H3's first multiplicative gate with an activation.
- **E=2, two blocks** — match Transformer's 12D² params (MHA+MLP) with two ~6D² Mamba blocks.
- **Δ bias init softplus∈[0.001,0.1]** — follows prior SSM init so initial timesteps span a reasonable range of memory horizons.
- **dt_rank ≈ d_model/16** — small low-rank Δ projection; negligible params, captures most of the benefit.

## Canonical code (grounded)
- `mamba_ssm/modules/mamba_simple.py`: the `Mamba` block (in_proj→conv1d→SiLU→x_proj for Δ,B,C→dt_proj→selective_scan→×SiLU(z)→out_proj); A_log param (A=-exp(A_log)); D skip; dt bias init.
- `mamba_ssm/ops/selective_scan_interface.py`: `selective_scan_ref` (pure-PyTorch reference loop = the scan, exact math) and the fused CUDA `selective_scan_fn`.
- `mamba_ssm/models/mixer_seq_simple.py`: MixerModel = embedding → stack of (norm+residual) Mamba blocks → final norm → lm_head.
- The reference `selective_scan_ref` math: deltaA=exp(einsum(delta,A)); deltaB_u=einsum(delta,B,u); loop x = deltaA·x + deltaB_u; y = einsum(x,C); out = y + D·u; out *= SiLU(z).
