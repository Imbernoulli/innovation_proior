# Muon — Synthesis (Phase 1.5)

## One-line
Muon optimizes 2D hidden weight matrices by taking SGD-momentum, then orthogonalizing the
momentum matrix (replace it with the nearest semi-orthogonal matrix, = its polar factor U V^T =
msign(M)), computed cheaply by a Newton–Schulz quintic iteration in bf16. AdamW is kept for
embeddings, the LM head, and all 1D params. Scaling to large LLMs needs two extra pieces:
decoupled weight decay, and a per-shape learning-rate scale √(max(A,B)) (set so RMS ≈ 0.2 to match AdamW).

## The pain point / research question
- Adam/AdamW treat every weight entry as an independent scalar (sign descent / per-entry
  normalization). They ignore that a layer's weight is a **matrix** acting as a linear operator.
- Empirically, the SGD-momentum / Adam update matrices for transformer 2D params are nearly
  low-rank (very high condition number): a few singular directions dominate, "rare directions"
  that matter for learning get a tiny share of the update. What is the right notion of a "unit
  step" for a matrix parameter?

## Load-bearing ancestors → contribution / limitation
- **SGD with momentum** (Polyak/Nesterov; Sutskever et al. 2013): EMA of gradients M = μM + g.
  Muon's first stage is exactly this. Limitation: raw update inherits the gradient's anisotropy.
- **Adam (Kingma & Ba 2015) / AdamW (Loshchilov & Hutter 2019)**: per-coordinate adaptive scaling
  m̂/√v̂; with EMA off, Adam = sign(g) = steepest descent under the ∞-norm / "max-of-max" ℓ1→ℓ∞
  operator norm (Bernstein & Newhouse). Per-matrix normalization but **per-entry**, blind to the
  spectral/operator structure. AdamW = Adam + decoupled weight decay (decay applied to W directly,
  not through the adaptive denominator). Update RMS ≈ 0.2–0.4 in practice.
- **Shampoo (Gupta et al. 2018) / full preconditioners**: W ← W − η L^{-1/4} G R^{-1/4} with
  L=ΣGGᵀ, R=ΣGᵀG. Full Kronecker preconditioner; with accumulation OFF it reduces to
  (GGᵀ)^{-1/4} G (GᵀG)^{-1/4} = U Vᵀ — exactly orthogonalization. But maintaining/inverting the
  d×d preconditioners is O(d³) and memory-heavy → too expensive at LLM scale. Muon keeps the
  orthogonalized-update *idea* but drops the accumulators and the explicit inverse roots.
- **Matrix sign function / polar decomposition** (Higham): for G = UΣVᵀ, msign(G) = U Vᵀ is the
  orthogonal polar factor; = sign applied to each singular value (push σ→1). The dualization of
  the gradient under the spectral norm.
- **Newton–Schulz iteration** (Higham Eq 5.22; Kovarik 1970, Björck 1971): iteration of an odd
  polynomial that drives the singular values of a normalized matrix toward 1 without computing an
  SVD or any inverse; stable, matmul-only, bf16-friendly.
- **Steepest-descent-under-a-norm view** (Bernstein & Newhouse 2024 "Old Optimizer, New Norm";
  Bernstein et al. 2024 "Modular Duality"): optimizers = steepest descent under a chosen norm;
  the spectral norm is the natural operator norm for a weight matrix.

## The central derivation (must be lived inline in reasoning.md)

### 1. Steepest descent under a norm (the template)
Local quadratic model: minimize over ΔW
   ⟨G, ΔW⟩ + (λ/2)‖ΔW‖²  (G = gradient/momentum, λ = sharpness).
Split ΔW = c·T, c≥0, ‖T‖=1:
   min_{c≥0}[ c·min_{‖T‖=1}⟨G,T⟩ + (λ/2)c² ]
            = min_{c≥0}[ −c·‖G‖† + (λ/2)c² ]
where ‖G‖† = max_{‖T‖=1}⟨G,T⟩ is the **dual norm**. So:
   direction T* = argmax_{‖T‖=1}⟨G,T⟩,  magnitude c* = ‖G‖†/λ,
   ΔW = −(‖G‖†/λ)·argmax_{‖T‖=1}⟨G,T⟩.
The whole question is: **which norm?**

### 2. Adam falls out under the ∞ / max-of-max norm
‖·‖∞ on flattened weights → argmax_{‖T‖∞=1}⟨g,t⟩ = sign(g), dual = ‖g‖₁. So sign descent. And
‖w‖∞ = max_l ‖W_l‖_{ℓ1→ℓ∞} (a max of a max is a max), so Adam-without-EMA is per-matrix sign
descent under the max ℓ1→ℓ∞ operator norm. Per-entry; the operator structure of W is thrown away.

### 3. Spectral norm → orthogonalized update (the key)
A weight matrix is an operator on a (locally Euclidean) hidden space, so its natural norm is the
**ℓ2→ℓ2 induced (spectral) norm** ‖·‖₂. Solve direction:
   T* = argmax_{‖T‖₂=1} ⟨G, T⟩.
With G = Σᵢσᵢuᵢvᵢᵀ: ⟨G,T⟩ = Σᵢσᵢ uᵢᵀ T vᵢ ≤ Σᵢσᵢ since ‖T‖₂=1 ⇒ uᵀTv ≤ 1. Equality at
T* = Σᵢ uᵢvᵢᵀ = U Vᵀ. So:
   **direction = U Vᵀ = msign(G)**,  dual norm ‖G‖₂† = Σᵢσᵢ = trace Σ (the nuclear norm).
   ΔW = −η · U Vᵀ.
Equivalently UVᵀ = (GGᵀ)^{−1/2}G = argmin over semi-orthogonal A of ‖A−G‖_F (closest
semi-orthogonal matrix): ‖A−G‖_F² = ‖A‖_F² −2⟨A,G⟩ + ‖G‖_F² and ‖A‖_F²=min(m,n) fixed, so
minimizing distance = maximizing ⟨A,G⟩ = Σσᵢ at A=UVᵀ.
**Why desirable**: it keeps the singular *vectors* (directions) of the momentum but flattens all
singular *values* to 1 — every direction gets an equal-size step, instead of a few dominant σ's
eating the whole update; the rare-but-important low-σ directions are no longer starved.

### 4. Why spectral and not Frobenius/something else — the loss bound
Linear model y=Wx, square loss, ‖xᵢ‖=√d_in:
   ℓ(W+ΔW) ≤ ℓ(W) + ⟨∇ℓ,ΔW⟩ + ½·(d_in/d_out)·‖ΔW‖₂².
The natural majorizer is quadratic in the **spectral** norm. Minimizing this majorizer IS steepest
descent under the spectral norm (majorization-minimization). The d_in/d_out (fan_in/fan_out)
factor is exactly the per-shape scale that will reappear in the learning-rate adjustment.

### 5. Computing U Vᵀ without an SVD — Newton–Schulz
SVD is slow/unstable on GPU. Want a matmul-only iteration that sends every singular value to 1.
Build it from an odd polynomial applied to singular values. Start from the degree-3 sign iteration
(Higham 5.22): normalize X₀ = G/‖G‖ (Frobenius or spectral) so all σ ≤ 1 (more precisely <√3),
then
   X_{k+1} = 1.5 X_k − 0.5 X_k X_kᵀ X_k.
Because X = UΣVᵀ ⇒ X_kX_kᵀX_k = UΣ³Vᵀ, the iteration acts as the scalar map
   f(σ) = 1.5σ − 0.5σ³
on each singular value, leaving U,V untouched. f has stable fixed point at σ=1: for 0<σ<√3, f
pushes σ→1. So X_k → U Vᵀ.
**Why normalize first**: f only contracts toward 1 inside (0,√3); if any σ>√3 the cubic diverges.
Frobenius normalization guarantees σ_max ≤ ‖X‖₂ ≤ ‖X‖_F = 1, safely inside the basin.
**Quintic + tuned coefficients**: generalize to
   X_{k+1} = a X_k + b (X_kX_kᵀ)X_k + c (X_kX_kᵀ)²X_k, scalar map g(σ)=aσ+bσ³+cσ⁵.
Want g≈sign(σ) on (0,1]. Convergence rate near 0 is g'(0)=a, so make **a large** to lift small
singular values fast (the near-low-rank update has many tiny σ). Trade-off: pushing a up makes g
overshoot — it no longer converges exactly to 1 but to roughly Uniform(0.5,1.5). That's fine:
exact orthogonality isn't needed, only roughly-equalized σ. Tuned: **a=3.4445, b=−4.7750,
c=2.0315**. With these, **5 iterations** in **bf16** suffice (the iteration is numerically stable
in low precision because it's all matmuls and bounded; the coupled-inverse Newton iteration would
need fp32). N=10 gives a more accurate orthogonalization but no better loss → N=5 for efficiency.
Transpose so the short side is rows (work with the smaller GGᵀ) for speed.

### 6. Which params get Muon
Only **2D hidden weight matrices** (the linear operators). Embedding and final LM head are 2D but
behave like vocab-indexed lookups / class scorers, not hidden operators — empirically AdamW is
better there. Scalars/vectors (norm gains, biases) have no matrix structure. So: Muon for hidden
2D, AdamW for everything else. Conv weights → flatten last 3 dims to 2D. (One momentum buffer for
Muon vs two for Adam → half the optimizer memory on the Muon params.)

## Scaling Muon to large LLMs (the two crucial techniques)

### A. Decoupled weight decay
Vanilla Muon has no weight decay. At scale, weight RMS and layer-output RMS grow unboundedly,
eventually exceeding bf16's high-precision range → hurts the over-trained regime. Add AdamW-style
**decoupled** decay:
   W_t = W_{t-1} − η_t (O_t + λ W_{t-1}).
Vanilla Muon converges faster early but is overtaken; with decay Muon beats both vanilla Muon and
AdamW in the over-train regime. (Applying decay to the RMSNorm γ too is important for stability.)

### B. Consistent update RMS + match to AdamW (the √max(A,B) factor)
Lemma: for a full-rank [A,B] matrix, the orthogonalized update O=U_{:,:r}V_{:r,:} (r=min(A,B)) has
   RMS(O) = √(r/(AB)) = √(1/max(A,B)) for full rank.
   Proof: RMS² = (1/AB)ΣᵢΣⱼ(Σ_k U_{ik}V_{kj})² = (1/AB)Σ_k(Σ_i U_{ik}²)(Σ_j V_{kj}²) =
   (1/AB)Σ_{k=1}^r 1 = r/(AB).  (cross terms vanish by orthonormal columns of U,V.)
So Muon's update RMS depends on the **shape**: huge for tiny matrices, tiny for big MLP matrices.
- max(A,B) too large (dense MLP) → updates too small → underfit/under-trained capacity.
- max(A,B) too small (per-head GQA/MLA slices) → updates too large → instability.
Fix: scale each matrix's update by **√max(A,B)** to cancel the shape dependence (Keller's
original √max(1,A/B) = √(fan_out/fan_in) is equivalent up to a global scale when the second dim is
shared). Then to share one (η, λ) with the AdamW-managed params, match AdamW's typical update
RMS ≈ 0.2: use
   W_t = W_{t-1} − η_t (0.2 · O_t · √max(A,B) + λ W_{t-1}).
RMS sweep [0.05,0.1,0.2,0.4,0.8] → 0.2 and 0.4 best; pick 0.2. This lets Muon **reuse** the
learning rate and weight decay tuned for AdamW out of the box. (Keller's repo uses the equivalent
update *= max(1, fan_out/fan_in)^0.5 with lr in "spectral norm per update" units.)

### C. Other hyperparameters
N=5 NS steps; momentum μ=0.95 (no consistent gain from tuning).

### D. Distributed Muon (ZeRO-1 style)
ZeRO-1 partitions optimizer state across DP. AdamW updates are elementwise so the partition is
trivial; **Muon needs the full gradient matrix** to orthogonalize. Distributed Muon: reduce-scatter
G → apply momentum on local shard → **DP-gather** the sharded momentum into the full matrix →
Newton–Schulz on the full matrix → discard all but the local partition → apply update →
all-gather updated params. One momentum buffer (½ Adam's optimizer memory). NS only in bf16
(comm ½ of fp32). Total comm in (1, 1.25]× of distributed AdamW; latency negligible (~1–3% of
fwd/bwd) since N≈5.

## Design-decision → why (the table)
| Decision | Why this, not the alternative |
|---|---|
| Orthogonalize the momentum (replace with UVᵀ) | = steepest-descent direction under the spectral norm, the natural operator norm of a weight matrix; equalizes the step across singular directions so rare directions aren't starved. Alt (raw SGDm/Adam) lets a few dominant σ eat the update. |
| Polar factor UVᵀ specifically (drop Σ) | UVᵀ = argmax⟨G,T⟩ over ‖T‖₂=1 AND argmin‖A−G‖_F over semi-orthogonal A: it's both the spectral-norm dual direction and the nearest semi-orthogonal matrix. Keeping Σ would just give back the anisotropic raw update. |
| Newton–Schulz (not SVD, not inverse-root Newton) | SVD is slow/awkward on GPU; inverse-root Newton needs fp32 and breaks on low-rank G. NS is matmul-only, bf16-stable, works on all singular values incl. tiny ones. |
| Frobenius-normalize X₀ | NS scalar map only contracts σ→1 inside (0,√3); ‖X‖_F=1 ⇒ σ_max≤1, safely in the basin. Without it the cubic/quintic diverges. |
| Quintic with a=3.4445,b=−4.775,c=2.0315 | g'(0)=a controls convergence of small σ; make a large to lift the many tiny σ of a near-low-rank update fast. Quintic gives enough freedom to push a high while staying bounded; the cost is σ converges to ~U(0.5,1.5) not exactly 1, which doesn't hurt loss. |
| N=5 steps, bf16 | N=10 is more accurate but no better loss; N=5 is cheaper. bf16 stable because NS is bounded matmuls; comm halved. |
| Muon only for 2D hidden weights; AdamW for embed/head/scalars | Only hidden 2D params are operators with meaningful spectral geometry; embeddings/head behave like lookups/scorers (AdamW empirically better); scalars/vectors have no matrix structure. |
| Add decoupled weight decay | Without it weight/output RMS grow past bf16's precise range and the over-trained regime suffers; decoupled (on W, not through a denominator) keeps Muon's geometry intact. |
| Scale update by √max(A,B) | Lemma: bare update RMS = √(1/max(A,B)) is shape-dependent → big matrices under-step, tiny ones over-step/instability. √max(A,B) cancels it for uniform RMS. |
| 0.2 prefactor (RMS≈0.2) | Matches AdamW's empirical update RMS (0.2–0.4) so a single (η,λ) tuned for AdamW transfers to Muon; sweep picked 0.2. |
| μ=0.95 | No consistent gain from tuning momentum. |
| Distributed: DP-gather + NS on full matrix | ZeRO-1 shards state, but orthogonalization is non-elementwise and needs the whole matrix; gather only the local-param momentum, NS, keep local slice. One buffer = ½ Adam memory. |

## Canonical code (grounded)
- Keller Jordan `muon.py`: zeropower_via_newtonschulz5 (a,b,c=3.4445,−4.775,2.0315; bf16; transpose
  so rows≤cols; X/=‖X‖+1e-7; 5 steps of A=XXᵀ, B=bA+cAA, X=aX+BX); muon_update (Nesterov lerp,
  flatten conv, NS, update*=max(1,fan_out/fan_in)^0.5); Muon optimizer (p*=1−lr·wd; p+=−lr·update);
  MuonWithAuxAdam.
- Moonlight: same NS; adjust_lr_for_muon: adjusted = 0.2·√max(A,B); p.mul_(1−lr·wd);
  p.add_(u, alpha=−adjusted_lr); Muon for ndim==2 except embed/head, AdamW otherwise.

## In-frame discipline notes
- Never name the target report / "Moonlight" / arXiv id. The method is "Muon" (nameable in answer).
- Ancestor citations (Bernstein & Newhouse 2024, Shampoo Gupta 2018, Adam Kingma 2015, AdamW
  Loshchilov 2019, Higham, Sutskever 2013) stay and get elaborated.
- reasoning.md: start from "Adam treats each weight as a scalar; what's a unit step for a MATRIX?"
  → steepest descent template → try Frobenius (gives back raw grad, no good) → spectral norm →
  argmax gives UVᵀ → "that's dropping the singular values" → why good (equalize directions) →
  closest-semi-orthogonal equivalence → Shampoo-without-accumulation = same thing but O(d³) →
  need cheap UVᵀ → Newton–Schulz cubic from sign function → why normalize → quintic + big a →
  5 steps bf16 → which params → at scale: RMS lemma + √max(A,B) + 0.2 match → weight decay →
  distributed. Land on code.
</content>
</invoke>
