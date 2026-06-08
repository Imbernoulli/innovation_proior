# LSVI-UCB (Least-Squares Value Iteration with UCB)

## Problem

Reinforcement learning in an episodic MDP(S, A, H, P, r) where the state space S is huge or infinite, so function approximation is mandatory. No simulator (cannot query arbitrary (x,a)), no resets, adversarial initial states, rewards in [0,1]. Goal: a provably efficient algorithm whose regret is **independent of S and A**, and whose implementation avoids any enumeration over S while using finite-action maximization over A. The tabular minimax lower bound Ω(√(H²SAT)) makes the √S unavoidable without structure, so structure must be assumed.

## Key idea

Assume a **linear MDP**: the transition kernel and reward are linear in a known feature map φ(x,a) ∈ ℝ^d,

  P_h(·|x,a) = ⟨φ(x,a), μ_h(·)⟩,  r_h(x,a) = ⟨φ(x,a), θ_h⟩,

with μ_h an *unknown signed measure* over S (so the model has infinite degrees of freedom despite being "linear") and θ_h ∈ ℝ^d unknown; normalization ‖φ‖≤1, ‖θ_h‖,‖μ_h(S)‖ ≤ √d. (Tabular MDPs are the special case φ = e_{(x,a)}, d = SA.)

Two consequences make RL tractable:
1. **Value-linearity (all policies).** By Bellman, Q^π_h = ⟨φ, w^π_h⟩ with w^π_h = θ_h + ∫V^π_{h+1}(x')dμ_h(x'), and ‖w^π_h‖ ≤ 2H√d. So it suffices to maintain linear action-value functions.
2. **Bellman backup = ridge regression.** The target r_h + P_h V_{h+1} is linear in φ, so one step of value iteration is one least-squares fit (LSVI).

Then **lift optimism from linear bandits (OFUL) into RL**: add the self-normalized/elliptical confidence width β·√(φ⊤Λ⁻¹φ) as a UCB bonus to the ridge estimate. This is the first provably efficient RL with function approximation, with regret Õ(√(d³H³T)) and **no dependence on the size of the state space**.

## Algorithm

```
LSVI-UCB.  Input: feature map φ, λ = 1, β = c·dH√(log(2dT/p)).
for episode k = 1, …, K:
    observe x_1^k
    for h = H, …, 1:                                  # backward sweep: build Q_h
        Λ_h ← Σ_{τ<k} φ(x_h^τ,a_h^τ) φ(x_h^τ,a_h^τ)⊤ + λI
        w_h ← Λ_h^{-1} Σ_{τ<k} φ(x_h^τ,a_h^τ) [ r_h(x_h^τ,a_h^τ) + max_a Q_{h+1}(x_{h+1}^τ, a) ]
        Q_h(·,·) ← min{ w_h⊤φ(·,·) + β·(φ(·,·)⊤Λ_h^{-1}φ(·,·))^{1/2},  H }
    for h = 1, …, H:                                  # forward sweep: act greedily
        a_h^k ← argmax_a Q_h(x_h^k, a);  observe x_{h+1}^k
```
w_h is the closed form of the ridge problem argmin_w Σ_τ [r_h^τ + max_a Q_{h+1}(x_{h+1}^τ,a) − w⊤φ_h^τ]² + λ‖w‖². With Q_{H+1} ≡ 0. Using Sherman-Morrison for Λ_h^{-1}: runtime O(d²AKT), space O(d²H + dAT) — both independent of S.

## Main theorem

**Theorem (regret).** Under the linear-MDP assumption, with λ = 1 and β = c·dH√ι (ι = log(2dT/p)), with probability 1−p,

  Regret(K) = O(√(d³ H³ T · ι²)) = Õ(√(d³ H³ T)),  T = KH,

independent of S and A. With a fixed initial state, this converts to Õ(d³H⁴/ε²) samples for an ε-optimal policy with constant success probability.

**Theorem (misspecification).** If the MDP is only ζ-approximately linear (‖P_h − ⟨φ,μ_h⟩‖_TV ≤ ζ, |r_h − ⟨φ,θ_h⟩| ≤ ζ, ζ ≤ 1), then with β_k = c·(d√ι + ζ√(kd))H, with probability 1−p,

  Regret(K) = Õ(√(d³ H³ T) + ζ·dHT).

The extra ζdHT is linear in T — the unavoidable bias of a wrong linear model (O(ζ) per step).

## Proof sketch

1. **Ridge-weight decomposition.** For any fixed π, w_h^k − w^π_h splits into q₁ (regularization bias, −λΛ⁻¹w^π_h), q₂ (stochastic noise, Λ⁻¹Σφ[V−P_hV]), q₃ (recursion, Λ⁻¹Σφ·P_h(V^k−V^π)). After ⟨φ,·⟩: q₁ and q₃ are each ≤ O(H√d)·‖φ‖_{Λ⁻¹} (Cauchy-Schwarz in the Λ⁻¹ norm; q₃ reduces to P_h(V^k−V^π) plus bias via P_h = ⟨φ,μ_h⟩).

2. **Self-normalized concentration with covering (the bonus).** q₂'s numerator Σ_τ φ_h^τ[V_{h+1}(x_{h+1}^τ) − P_hV_{h+1}] would be a martingale-difference sum *if* V_{h+1} were fixed — but it is computed by LSVI from the same data, hence data-dependent. Resolve by uniform concentration over the value class 𝒱 = {min(max_a φ⊤w + β√(φ⊤Λ⁻¹φ), H)} (bounded by ridge). Its covering number: reparametrize A = β²Λ⁻¹; since min/max are contractions, dist(V₁,V₂) ≤ ‖w₁−w₂‖ + √‖A₁−A₂‖_F, giving log N_ε ≤ d log(1+4L/ε) + d²log(1+8√d B²/(λε²)) — dominated by **d²** (the bonus matrix A ∈ ℝ^{d×d}). Plugging into the self-normalized tail bound (Abbasi-Yadkori et al. 2011): ‖q₂-numerator‖_{Λ⁻¹} ≤ Õ(dH). Hence ⟨φ,q₂⟩ ≤ O(dH√ι)·‖φ‖_{Λ⁻¹}.

3. **Key relation.** Combining, |⟨φ,w_h^k⟩ − Q^π_h − P_h(V^k_{h+1}−V^π_{h+1})| ≤ c'·dH√ι·‖φ‖_{Λ⁻¹}. Set β = c·dH√ι ≥ this (a constant fixed-point c'√(ι+log(c+1)) ≤ c√ι).

4. **Optimism (UCB lemma).** By backward induction Q_h^k ≥ Q*_h for all (x,a,h,k): base h=H from the key relation; step uses P_h(V^k_{h+1}−V*_{h+1}) ≥ 0 (inductive hypothesis) so the recursion term only helps. Capping at H is harmless since Q*_h ≤ H.

5. **Regret decomposition + elliptical potential.** With δ_h^k = V_h^k(x_h^k) − V^{π_k}_h(x_h^k) and ζ_{h+1}^k the martingale difference E[δ_{h+1}^k|x,a] − δ_{h+1}^k:
   δ_h^k ≤ δ_{h+1}^k + ζ_{h+1}^k + 2β‖φ_h^k‖_{Λ⁻¹}.
   By optimism, Regret(K) ≤ Σ_k δ₁^k ≤ Σ_{k,h} ζ_h^k + 2β Σ_{k,h} ‖φ_h^k‖_{Λ⁻¹}. First term ≤ 2H√(Tι) by Azuma-Hoeffding. Second: the **elliptical potential lemma** gives Σ_k (φ_h^k)⊤(Λ_h^k)⁻¹φ_h^k ≤ 2 log[det(Λ_h^{K+1})/det(Λ_h^1)] ≤ 2d log((λ+K)/λ) ≤ 2dι; Cauchy-Schwarz over k then h gives Σ_{k,h} ‖φ‖_{Λ⁻¹} ≤ H√(2dKι).

6. **Powers.** Regret ≤ 2H√(Tι) + 2β·H√(2dKι); with β = c·dH√ι, the bonus term = O(dH·H√(dK)·ι) = O(d^{3/2}H²√K·ι) = O(√(d³H⁴K·ι²)) = O(√(d³H³T·ι²)). The d^{3/2} comes from d in β (the d² covering) times √d from the potential lemma; the H²√K becomes √(H³T) because T = KH. ∎

## Why each piece

- **Linear kernel (not linear policy):** makes Q linear for *all* π and the backup a regression; a linear-policy assumption gives neither.
- **μ_h an unknown measure (infinite DOF):** keeps the model genuinely large; the win is that one never needs to learn it in TV — only P̂_h V ≈ P_h V on the small class 𝒱 — so LSVI-UCB is effectively model-free.
- **Ridge (λI), not OLS:** invertibility when k<d, ‖w_h^k‖ ≤ 2H√(dk/λ), and bounded covering capacity.
- **Elliptical bonus β√(φ⊤Λ⁻¹φ):** the exact uncertainty notion OFUL's confidence ellipsoid induces; (φ⊤Λ⁻¹φ)⁻¹ ≈ effective samples along φ.
- **Clip at H, uniform concentration over 𝒱:** rewards in [0,1] ⇒ V≤H bounds the class; the clip + bonus push V outside the linear class, and V's data-dependence forces concentration over a class, not a fixed function.
- **Recursion-aware optimism:** P_h(V^k−V*) ≥ 0 in the induction propagates optimism *down the horizon*, avoiding the exponential-in-H blowup of a naive per-step bandit reduction; H-dependence stays polynomial (H^{3/2}).
