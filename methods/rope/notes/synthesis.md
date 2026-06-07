# RoPE — Synthesis Notes (Phase 1.5)

## The pain point (research question)
Transformer self-attention is permutation-equivariant: with no position signal, `softmax(q_m^T k_n / sqrt d)` is identical for any reordering of tokens. So position MUST be injected. The whole field is asking: *what is the right way to inject position into attention?* Two families exist (below), both with concrete defects. The precise object we care about is the attention **logit** `q_m^T k_n` — that is the only place position needs to enter for it to affect which tokens attend to which. The dream: make `q_m^T k_n` depend on `x_m`, `x_n`, and the **relative offset `m-n`** only — never on absolute `m`, `n` separately — while injecting position **multiplicatively per-token** (so it is compatible with linear/kernelized attention, which never forms the N×N logit matrix).

## Load-bearing ancestors (verified against primary text §2)

### Sinusoidal absolute PE (Vaswani et al. 2017)
- `p_{i,2t}=sin(i/10000^{2t/d})`, `p_{i,2t+1}=cos(i/10000^{2t/d})`. Add to embedding: `f_t(x_i,i)=W_t(x_i+p_i)`.
- Idea: fixed (non-learned) so generalizes to unseen lengths; geometric progression of wavelengths from 2π to ~10000·2π gives a multi-resolution "clock."
- Key fact they noticed but did NOT exploit: `p_{m+k}` is a linear function of `p_m` (rotation in each 2-d sin/cos pair) — i.e. sinusoids already have a relative structure latent in them. RoPE will turn that latent rotation into the actual mechanism.
- Limitation: it is **absolute** and **additive**. Expanding `(x_m+p_m)^T W_q^T W_k (x_n+p_n)` produces cross terms `x_m^T W^T W p_n`, etc., that depend on absolute `m`,`n`, not on `m-n`. No clean relative dependence.

### Learned absolute PE (BERT, GPT, ALBERT, ELECTRA)
- `p_i` is a trainable vector per position up to max length L.
- Limitation: hard length cap at L; nothing learned for positions > L; still absolute/additive with the same cross-term problem.

### Shaw et al. 2018 (relative PE, the seminal one)
- `f_q(x_m)=W_q x_m`, `f_k(x_n,n)=W_k(x_n + p̃^k_r)`, `f_v(x_n,n)=W_v(x_n + p̃^v_r)`, where `r=clip(m-n, r_min, r_max)`.
- Adds a learned **relative** embedding (clipped beyond a window) into key and value. First to make attention explicitly relative.
- Limitations: (a) clipping throws away long-range distinctions; (b) adds a learned bias table — extra params, and not a clean closed form; (c) injects into the **values** too, complicating things; (d) crucially, it modifies the **expanded dot product**, so it cannot factor as a per-token transform — incompatible with linear attention.

### Transformer-XL (Dai et al. 2019) / decomposition family
- Expand `q_m^T k_n = x_m^T W_q^T W_k x_n + x_m^T W_q^T W_k p_n + p_m^T W_q^T W_k x_n + p_m^T W_q^T W_k p_n` (eq rela-posi1).
- Replace absolute `p_n` by sinusoid relative `p̃_{m-n}`; replace `p_m` in terms 3,4 by trainable global vectors `u`,`v`; split `W_k` into content/position versions.
- T5 (Raffel 2020): collapse all of it to `x_m^T W_q^T W_k x_n + b_{m,n}` — a single learned scalar **relative bias** per bucket. Simplest, very effective.
- DeBERTa (He 2020): keep the two middle "content×position" cross terms with relative embeddings.
- The whole family's shared move: **start from the additive-absolute decomposition and surgically edit terms** to make them relative. RoPE's contrast: don't start from "add p then expand"; instead **demand the relative property as a constraint and solve for `f`**.
- Shared limitation of ALL of these: position enters as an **additive term inside the N×N logit matrix**. Linear attention replaces `exp(q^T k)` with `φ(q)^T φ(k)` and uses associativity `(φ(Q)^T)(φ(K) V)` to avoid the N×N matrix — but an additive relative bias `b_{m,n}` lives exactly in that matrix you are trying to avoid. So none of them port to linear attention.

## The desire, stated as an equation
Want `f_q, f_k` such that `<f_q(x_m,m), f_k(x_n,n)> = g(x_m, x_n, m-n)`. Boundary: `f_q(x,0)=W_q x`, `f_k(x,0)=W_k x` (no position ⇒ ordinary projection). This is the formulation (eq fn:formulation). Everything else is solving this functional equation.

## The 2D derivation (appendix appendix:rope-deriv) — FULL
Take d=2, identify R^2 with C. Inner product `<a,b> = Re[a b*]`.
Write polar: `f_q(x_q,m)=R_q(x_q,m) e^{iΘ_q(x_q,m)}`, same for k, and `g=R_g e^{iΘ_g}`.
Plug `f_q f_k^* = g` (works because `Re[f_q f_k^*]` is the inner product and the magnitude/phase must match the target g written the same way):
- magnitudes: `R_q(x_q,m) R_k(x_k,n) = R_g(x_q,x_k,n-m)`
- phases: `Θ_k(x_k,n) - Θ_q(x_q,m) = Θ_g(x_q,x_k,n-m)`
Boundary at 0: `q = ||q|| e^{iθ_q} = f_q(x_q,0)`, etc.

**Set m=n** (so relative offset = 0):
- `R_q(x_q,m) R_k(x_k,m) = R_g(...,0) = R_q(x_q,0)R_k(x_k,0) = ||q|| ||k||`. A clean solution: `R_q(x_q,m)=||q||` independent of m (and same for k, g). ⇒ **magnitude carries no position info**; rotation only.
- `Θ_k(x_k,m) - Θ_q(x_q,m) = Θ_g(...,0) = θ_k - θ_q`. Rearrange: `Θ_q(x_q,m) - θ_q = Θ_k(x_k,m) - θ_k`. LHS depends only on (x_q,m), RHS only on (x_k,m), yet they're equal ⇒ both equal a function of m alone, independent of the embedding. Call it `φ(m)`. So `Θ_f(x,m) = φ(m) + θ_x` where `Θ_f := Θ_q = Θ_k`.

**Set n=m+1** in the phase relation and use `Θ_f=φ+θ`:
`φ(m+1) - φ(m) = Θ_g(x_q,x_k,1) + θ_q - θ_k`. RHS is a constant in m ⇒ `φ` is **arithmetic**: `φ(m) = mθ + γ`. Choose γ=0 (free, folds into boundary).
⇒ `f_q(x_m,m) = (W_q x_m) e^{imθ}`, `f_k(x_n,n)=(W_k x_n) e^{inθ}`. The position is a **pure rotation by mθ**.
Check: `<f_q,f_k> = Re[(W_q x_m)(W_k x_n)^* e^{i(m-n)θ}]` — depends only on m-n. 

Matrix form (d=2): `f(x_m,m) = [[cos mθ, -sin mθ],[sin mθ, cos mθ]] W x_m`.

## General d (even) — block diagonal
Split d-dim space into d/2 independent 2-d planes; by linearity of inner product, rotate each plane by its own frequency. `f_{q,k}(x_m,m) = R^d_{Θ,m} W_{q,k} x_m`, `R^d_{Θ,m}` = block-diagonal of 2×2 rotations with angles `mθ_1,...,mθ_{d/2}`.
Then `q_m^T k_n = (R^d_{Θ,m}W_q x_m)^T (R^d_{Θ,n} W_k x_n) = x_m^T W_q^T R^d_{Θ,n-m} W_k x_n`, using `R^T_m R_n = R_{n-m}` (rotations compose; orthogonal, norm-preserving ⇒ stable). [Primary text has an `R^d_{Θ,n-m}=(R_m)^T R_n` identity; note paper's eq has a typo writing it as one thing then transpose — the correct relation is `(R_m)^T R_n = R_{n-m}`.]
Frequencies: `θ_i = 10000^{-2(i-1)/d}` (i=1..d/2), exactly the sinusoidal geometric progression — so RoPE *is* "the relative version of sinusoidal." This is the design-choice tie-back: reusing 10000-base geometric spacing gives a spectrum of rotation speeds from fast (local) to near-static (global).

## Efficient realization (appendix appendix:rope-efficient)
Don't build the sparse matrix. Elementwise:
`R^d_{Θ,m} x = x ⊙ [cos mθ_1, cos mθ_1, cos mθ_2, cos mθ_2, ...] + rotate(x) ⊙ [sin mθ_1, sin mθ_1, ...]`
where `rotate(x) = [-x_2, x_1, -x_4, x_3, ...]` (interleaved pairs). O(d) not O(d²).

## Long-term decay (appendix appendix:long-term-decay) — FULL
Group q,k into d/2 complex pairs: `q_m^T k_n = Re[ Σ_{i=0}^{d/2-1} q_{[2i:2i+1]} k_{[2i:2i+1]}^* e^{i(m-n)θ_i} ]`.
Let `h_i = q_{[2i:2i+1]} k_{[2i:2i+1]}^*` (content, position-free), `S_j = Σ_{i=0}^{j-1} e^{i(m-n)θ_i}` (partial sums of the phase factors), with `h_{d/2}=0`, `S_0=0`.
**Abel summation (summation by parts):** `Σ h_i (S_{i+1}-S_i) = -Σ S_{i+1}(h_{i+1}-h_i)`.
So `|Σ_i h_i e^{i(m-n)θ_i}| = |Σ S_{i+1}(h_{i+1}-h_i)| ≤ (max_i |h_{i+1}-h_i|) Σ_i |S_{i+1}|`.
The bound factors content (`max|Δh|`) from a purely positional envelope `Σ|S_{i+1}|`. Because the `θ_i` are a geometric sweep, as `|m-n|` grows the `e^{i(m-n)θ_i}` phases spread / dephase across frequencies and `(1/(d/2))Σ|S_i|` **decays** with `|m-n|`. ⇒ far-apart tokens have a bounded-and-shrinking positional contribution to the logit — exactly the inductive bias we want. (Not monotone; it's a decaying envelope.)

## RoPE with linear attention
Linear attn: `Attn_m = Σ_n φ(q_m)^T ψ(k_n) v_n / Σ_n φ(q_m)^T ψ(k_n)`, φ,ψ ≥ 0 (e.g. elu+1), computed via associativity → O(N).
RoPE is a rotation = **norm-preserving**, so we can apply it AFTER the non-negative feature map without breaking non-negativity of the scalar weights' intent: `Attn_m = Σ_n (R_m φ(q_m))^T (R_n ψ(k_n)) v_n / Σ_n φ(q_m)^T ψ(k_n)`. The rotations sit on the per-token features → still factorizes → O(N) with relative position. Additive biases `b_{m,n}` could NEVER do this (they live in the N×N matrix). This is the property no prior relative scheme had. Denominator kept un-rotated to avoid divide-by-zero / negative normalizer.

## Design-decision → why table
- **Constrain the dot product (vs add then expand):** solving the functional equation yields a *clean closed form* with exact relative dependence and no extra params; the edit-the-expansion family only ever approximates relative-ness with learned tables/biases.
- **Multiplicative rotation (vs additive p):** rotation is per-token & norm-preserving ⇒ composes to `R_{n-m}` automatically (relative emerges for free) AND ports to linear attention. Additive does neither.
- **Magnitude position-independent / pure rotation:** falls out of the m=n branch of the functional equation; also keeps norms stable (orthogonal R).
- **θ_i = 10000^{-2(i-1)/d}:** inherits sinusoidal's multi-resolution spectrum; gives the long-term-decay envelope; learning θ barely moves them, so fix them (no params).
- **Even d, pairwise 2-planes:** the 2-d rotation is the only solution to the functional equation; tile it by linearity of inner product.
- **Apply to q,k only (not v by default):** position belongs in the attention logit; rotating v is optional (HF exposes `rotary_value`, off by default).
- **Interleaved pairs (paper/HF RoFormer) vs split-half rotate_half (GPT-NeoX/LLaMA):** mathematically equivalent up to a fixed permutation of dimensions; split-half is a contiguous-slice trick that's cheaper/cleaner in PyTorch. Both implement the same block rotation.
- **Keep denominator unrotated in linear attn:** numerator can have negative terms post-rotation; un-rotated denominator avoids zero-division and keeps a sane normalizer.

## Canonical code (grounded)
- HF `RoFormerSelfAttention.apply_rotary_position_embeddings` (interleaved): sin/cos from sinusoidal table, `sin_pos`/`cos_pos` repeat-interleaved, `rotate_half = stack([-x[...,1::2], x[...,::2]])`, then `x*cos + rotate_half*sin`. Exactly the appendix's elementwise formula. (file code/hf_modeling_roformer.py:220)
- HF LLaMA `rotate_half` (split-half) + `apply_rotary_pos_emb`: `inv_freq=1/base^{arange(0,d,2)/d}`, `freqs=pos⊗inv_freq`, `emb=cat(freqs,freqs)`, `q*cos + rotate_half(q)*sin` with `rotate_half(x)=cat(-x2,x1)`. The de-facto modern convention. (file code/hf_modeling_llama.py:138)
- Final code in answer.md will mirror the LLaMA/standard convention (most widely used) AND note the interleaved RoFormer form.

## In-frame reminders
Never name "RoPE/RoFormer" as a paper; never "this paper"; cite ancestors (Vaswani 2017, Shaw 2018, Dai 2019, Raffel 2020, He 2020, Katharopoulos 2020) freely. The method name "Rotary Position Embedding / RoPE" may be used in answer.md as the thing being built. Scaffold = bare attention harness with empty position slot, no rotary names.
