# HiPPO synthesis notes

## Pain point / research question
- Sequence models need to maintain a bounded-size summary of an unbounded, online-arriving history f(x) for x ≤ t, updated incrementally.
- RNNs (the established stateful approach) suffer vanishing/exploding gradients → limited memory horizon.
- Heuristic fixes: LSTM/GRU gates, orthogonal/unitary RNNs (control spectrum), Fourier Recurrent Unit, Legendre Memory Unit. All either heuristic, require a timescale prior (window length / step size), or lack gradient guarantees.
- Three goals: (i) unified view of memory mechanisms, (ii) handle any timescale without priors, (iii) rigorous theory (gradient bounds, approx error).

## The core reframing
Memory = online function approximation. At each t, compress history f_{≤t} by its optimal projection onto an N-dim subspace of polynomials, w.r.t. a measure μ^(t) on (−∞,t] weighting "what to remember." Store the N coefficients c(t).

- Hilbert space: ⟨f,g⟩_μ = ∫ f g dμ. OPs {g_n} orthonormal for μ^(t). Optimal coefficients c_n(t) = ⟨f_{≤t}, g_n^(t)⟩ (closed form, no optimization).
- Key trick: differentiate c_n(t) in t. Two terms (product rule on integrand × measure):
  dc_n/dt = ∫ f (∂_t g_n) ω dx + ∫ f g_n (∂_t ω) dx
  (with normalized/no-tilt case ζ=1, χ=1).
- If ∂_t P_n is a poly of degree n−1 → linear combo of c_0..c_{n−1}; and ∂_t(ω/χ) expressible in ω/χ → second term also linear in c's and f. Result: linear ODE dc/dt = A(t)c + B(t)f.
- Leibniz rule via Dirac deltas: ∂_t 𝟙_{[α(t),β(t)]} = β'(t)δ_{β} − α'(t)δ_{α}.

## Legendre facts (sec:legendre-properties)
- P_n orthogonal on [−1,1] w.r.t. uniform; (2n+1)/2 ∫_{-1}^1 P_n P_m = δ. P_n(1)=1, P_n(−1)=(−1)^n.
- Normalized basis on [0,t] uniform: (2n+1)^{1/2} P_n(2x/t − 1).
- Recurrences: (2n+1)P_n = P_{n+1}' − P_{n−1}'; P_{n+1}' = (n+1)P_n + xP_n'.
  ⇒ P_n' = (2n−1)P_{n−1} + (2n−5)P_{n−3} + …   (eq legendre-d, used for LegT)
  ⇒ (x+1)P_n'(x) = nP_n + (2n−1)P_{n−1} + (2n−3)P_{n−2} + …  (eq legendre-xd, used for LegS)

## LegT (translated Legendre, sliding window [t−θ,t]) — recovers LMU
- ω = (1/θ)𝟙_{[t−θ,t]}; g_n = λ_n (2n+1)^{1/2} P_n(2(x−t)/θ + 1).
- ∂_t ω = (1/θ)(δ_t − δ_{t−θ}).
- ∂_t g_n uses P_n' = (2n−1)P_{n−1}+… → linear combo of g_{n−1}, g_{n−3},…
- Endpoints: g_n(t,t)=λ_n(2n+1)^{1/2}; g_n(t,t−θ)=λ_n(−1)^n(2n+1)^{1/2}.
- WALL: need f(t−θ), no longer available → approximate it by current reconstruction f(t−θ) ≈ Σ_k λ_k^{−1} c_k (2k+1)^{1/2}(−1)^k.
- With λ_n=1 (orthonormal):
  dc/dt = −(1/θ)A c + (1/θ)B f,
  A_{nk} = (2n+1)^{1/2}(2k+1)^{1/2} × {1 if k≤n; (−1)^{n−k} if k≥n}, B_n=(2n+1)^{1/2}.
- With λ_n=(2n+1)^{1/2}(−1)^n: A_{nk}=(2n+1)×{(−1)^{n−k} k≤n; 1 k≥n}, B_n=(2n+1)(−1)^n = exactly the LMU.

## LegS (scaled Legendre, whole history [0,t]) — the novel time-invariant-after-scaling op
- ω=(1/t)𝟙_{[0,t]}; g_n=(2n+1)^{1/2}P_n(2x/t−1); χ=1,ζ=1,λ=1.
- ∂_t ω = t^{−1}(−ω + δ_t).
- ∂_t g_n = −(2n+1)^{1/2} t^{−1}(z+1)P_n'(z), z=2x/t−1, then eq legendre-xd:
  = −t^{−1}(2n+1)^{1/2}[ n(2n+1)^{−1/2} g_n + (2n−1)^{1/2}g_{n−1} + (2n−3)^{1/2}g_{n−2}+…].
- Plug in; diagonal: from first term n(2n+1)^{−1/2}·(2n+1)^{1/2}=n times c_n; plus second-term −t^{−1}c_n (from −ω) → total −(n+1)c_n.
- RESULT (VERIFIED): dc/dt = −(1/t)A c + (1/t)B f,
  A_{nk} = (2n+1)^{1/2}(2k+1)^{1/2} (n>k);  n+1 (n=k);  0 (n<k);  B_n=(2n+1)^{1/2}.
- Factorization A = D(M + lower)... actually A = T M T^{-1} where T=diag((2n+1)^{1/2}),
  M_{nk}=2k+1 (k<n), k+1 (k=n), 0 (k>n). (code: M=-(where(row>=col,r,0)-diag(q)), A=T@M@inv(T), with overall sign folded so ODE uses +A_code/t.)

## Discretization
- Euler: c(t+Δt)=(I+ΔtA)c+ΔtBf. Bilinear/GBT/ZOH in sec:discretization-full.
- LegS discrete (Euler): c_{k+1} = (1 − A/k)c_k + (1/k)B f_k. **Δt-invariant** under GBT (the Δt cancels because both A and B scale as 1/t).
- N=1 LagT (A=B=1): c(t+Δt)=(1−Δt)c+Δt f. With input-dependent f and adaptive Δt = gated RNN (GRU); shows gates = order-1 HiPPO.

## LegS theory (proofs)
- Timescale equivariance: h(t)=f(αt) ⇒ hippo(h)(t)=hippo(f)(αt). Proof: change of var x↦x/α in coefficient integral; the 1/t and the P_n(2x/t−1) both rescale consistently → no timescale hyperparameter. Discrete recurrence Δt-invariant.
- Efficiency O(N): A = D_1(L+D_0)D_2, L all-ones lower-triangular = cumsum; inverse (L+D)^{-1} solved by a scalar recurrence s_k = d_k/(1+d_k) s_{k−1} + y_k/(1+d_k), computed via cumsum/cumprod. O(N) per step vs O(N^2).
- Gradient bound: ∂c_{ℓ+1}/∂f_k = (I−A/ℓ)...(I−A/(k+1)) B/k. A triangular with eigenvalues 1..N. Largest mode ρ = ∏_{i=k+1}^ℓ (1−1/i) · 1/k. Telescopes: ∏(1−1/i)=k/ℓ, so ρ=Θ(1/ℓ). So ‖∂c(t_1)/∂f(t_0)‖=Θ(1/t_1) — polynomial decay, no exponential vanishing.
- Approx error: Parseval ‖f−g‖² = Σ_{n≥N} c_n². Integration by parts using P_n=(1/(2n+1))(P_{n+1}−P_{n-1})', boundary terms vanish (P_{n±1}(±1) equal). c_n = O(tL/n) ⇒ Σ ~ t²L²/N ⇒ error O(tL/√N). Order-k derivatives ⇒ O(t^k N^{−k+1/2}).

## Lineage / baselines (rw.tex)
- LSTM (Hochreiter & Schmidhuber 1997), GRU (Cho 2014): gates smooth optimization, heuristic memory.
- Orthogonal/unitary RNNs (Arjovsky 2016): control recurrent spectrum, less robust across tasks (Henaff 2016).
- Tallec & Ollivier 2018: gates ≈ time dilation / learnable timescale.
- Fourier Recurrent Unit (Zhang 2018): Fourier basis per random frequency; bounded gradient IF timescale chosen ~1/T. Special case of framework (Fourier = OPs z^n on unit circle).
- LMU (Voelker 2018/2019): derived from spiking-neuron LTI + Padé approx in frequency domain; observes Legendre interpretation but not optimality; no self-contained proof. = LegT special case.
- Neural ODE (Chen 2018): general nonlinear ODE, expensive solvers; HiPPO ODE is linear, fast classical discretization. Adaptive Δt from timestamps handles irregular sampling/missing data.
- Sliding transforms in signal processing (sliding DFT/DCT/Hadamard/...): discrete, fixed-window. HiPPO differs: continuous-time, time-varying & scaled measures, general OP families.

## Code grounding
- Original repo HazyResearch/hippo-code: model/op.py `transition(measure,N)` builds A,B; model/hippo.py HiPPO_LegT / HiPPO_LegS modules with forward() recurrence + reconstruct() via eval_legendre.
- LegS: A,B from transition('legs'), then per-t discretize A/t,B/t (bilinear via solve_triangular), unroll c_k=A_k c_{k-1}+B_k f_k. reconstruct via eval_matrix = B ⊙ eval_legendre(n, 2v−1).
- modern state-spaces/.../hippo.py confirms same `legs` branch.
```python
elif measure == 'legs':
    q = np.arange(N); col,row = np.meshgrid(q,q); r = 2*q+1
    M = -(np.where(row>=col, r, 0) - np.diag(q))   # = -(paper M)
    T = np.sqrt(np.diag(2*q+1))
    A = T @ M @ np.linalg.inv(T); B = np.diag(T)[:,None]
```
