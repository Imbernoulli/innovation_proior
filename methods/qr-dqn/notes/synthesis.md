# QR-DQN synthesis (arXiv 1710.10044, AAAI 2018; Dabney, Rowland, Bellemare, Munos)

## Pain point (the theory-practice gap left by C51)
- C51 proved: distributional Bellman operator T^π is a γ-contraction in the MAXIMAL Wasserstein metric d̄_p (Lemma 3 of C51). Wasserstein is the "right" metric — robust to disjoint supports (no overlap pathology that KL/TV suffer).
- BUT: Wasserstein-as-a-loss cannot be minimized by SGD from samples — biased gradients (Bellemare et al. 2017 Cramer, Theorem 1). The minimizer of E_samples[W_p(empirical, B_μ)] ≠ minimizer of W_p(B, B_μ).
- So C51 did NOT minimize Wasserstein. It used a workaround: fixed support comb z_1≤...≤z_N (N atoms, fixed locations, predetermined [V_min,V_max]), learnable probabilities q_i (logits/softmax), a heuristic projection Φ of the Bellman target onto the support, then KL minimization. Theory says contract in Wasserstein; algorithm minimizes KL after projection. Disconnect.
- Open question: is there an algorithm that operates end-to-end on Wasserstein in a stochastic-approximation (online, SGD) setting? Answer here: yes, via quantile regression.

## The transpose idea
C51: N fixed LOCATIONS z_i, adjust PROBABILITIES q_i.
QR-DQN: N fixed PROBABILITIES (uniform q_i = 1/N), adjust LOCATIONS θ_i.
=> "quantile distribution" Z_θ(x,a) = (1/N) Σ_i δ_{θ_i(x,a)}. Each θ_i estimates a quantile of the return.

## Benefits of the transpose
1. No prespecified support bounds / uniform resolution. Locations adapt to actual return range per state. Better when return range varies across states. No domain knowledge of [V_min,V_max] needed.
2. No projection step Φ (no disjoint-support issue — locations move freely).
3. Can minimize Wasserstein loss without biased gradients — via quantile regression.

## Cumulative probabilities and midpoints
- τ_i = i/N for i=1..N, τ_0 = 0.
- Quantile midpoints: τ̂_i = (τ_{i-1}+τ_i)/2 = (2i-1)/(2N) for i=1..N.

## W1 of a distribution vs N uniform Diracs
For Y with bounded first moment and U = (1/N)Σδ_{θ_i}:
  W_1(Y,U) = Σ_{i=1}^N ∫_{τ_{i-1}}^{τ_i} |F_Y^{-1}(ω) - θ_i| dω.
Because U's CDF is a step function; on the ω-interval (τ_{i-1},τ_i] the inverse-CDF of U equals θ_i; W1 = ∫_0^1 |F_Y^{-1}(ω)-F_U^{-1}(ω)| dω.

## Lemma 2 (quantile midpoint minimizer) — KEY
For τ<τ', the set of θ minimizing ∫_τ^{τ'} |F^{-1}(ω) - θ| dω is {θ : F(θ) = (τ+τ')/2}.
Proof: θ↦|F^{-1}(ω)-θ| convex, subgradient +1 if θ<F^{-1}(ω), -1 if θ>, [-1,1] at equality.
Integral's subgradient = ∫_τ^{F(θ)}(-1)dω + ∫_{F(θ)}^{τ'}(+1)dω = -(F(θ)-τ) + (τ'-F(θ)) = (τ+τ') - 2F(θ).
Set =0 → F(θ) = (τ+τ')/2. So θ = F^{-1}((τ+τ')/2). Wait sign: for the cost ∫|F^{-1}-θ|, the subgradient in θ of |F^{-1}(ω)-θ| is +1 when θ<F^{-1}(ω) (increasing θ toward F^{-1} decreases cost? No: d/dθ|c-θ| = -sign(c-θ) = +1 if θ>c). Careful: paper states subgrad of θ↦|F^{-1}(ω)-θ| is 1 if θ<F^{-1}(ω). That is d/dθ |F^{-1}-θ|; for θ<F^{-1}, |F^{-1}-θ|=F^{-1}-θ, derivative = -1. Hmm. The paper's convention: they write subgradient as +1 if θ<F^{-1}(ω). Let me recompute: f(θ)=|F^{-1}(ω)-θ|. For θ < F^{-1}(ω): f=F^{-1}-θ, f'=-1. For θ>F^{-1}: f=θ-F^{-1}, f'=+1. So paper's table has signs FLIPPED from naive — they must mean derivative of the *quantile-regression-style* arrangement, OR there's a sign convention. Net effect for the integral: ∫_τ^{τ'} f'(θ)dω. Using correct f': for ω with F^{-1}(ω)>θ (i.e. ω>F(θ)) f'=-1; for ω<F(θ), f'=+1. So integral subgrad = ∫_τ^{F(θ)}(+1)dω + ∫_{F(θ)}^{τ'}(-1)dω = (F(θ)-τ) - (τ'-F(θ)) = 2F(θ)-(τ+τ'). Set=0 → F(θ)=(τ+τ')/2. SAME ANSWER. Good — minimizer θ_i = F_Y^{-1}(τ̂_i). (The paper's printed signs are its own convention; the conclusion F(θ)=(τ+τ')/2 is what matters and is correct.)
=> minimizing W1: θ_i = F_Y^{-1}(τ̂_i). The optimal locations are the midpoint quantiles.

## Quantile regression loss
QR loss for quantile τ: ρ_τ(u) = u(τ - 1_{u<0}), u = ẑ - θ (ẑ sample, θ prediction).
- u>0 (underestimate, θ<ẑ): weight τ.   u<0 (overestimate): weight (1-τ) since u(τ-1)= -u(1-τ) = |u|(1-τ).
- Minimizer of E_{ẑ~Z}[ρ_τ(ẑ-θ)] is F_Z^{-1}(τ). (Standard quantile regression: setting derivative E[τ - 1_{ẑ<θ}]=0 → P(ẑ<θ)=τ.)
- Gives UNBIASED sample gradients (gradient is τ - 1_{u<0}, depends only on sign, no projection). This is the whole point.
- Overall location objective: Σ_i E_{ẑ~Z}[ρ_{τ̂_i}(ẑ - θ_i)] minimizes W1(Z,Z_θ) (by Lemma 2).

## Biased-gradient caveat (Prop 2)
Quantile param does NOT by itself fix biased W_p gradients (there exists Z s.t. argmin E[W_p(Ẑ_m, Z_θ)] ≠ argmin W_p(Z,Z_θ)) — the FIX is using the QR loss, not W_p directly. (Z = (1/N)Σδ_i; gradient on θ_1 at θ_1=1 is <0 because with nonzero prob the empirical has no atom at 1.)

## Quantile Huber loss (smoothing)
QR loss not smooth at 0; gradient constant as u→0+ (magnitude τ or 1-τ), discontinuous derivative. Hypothesized to hurt nonlinear function approximation. Replace |u| part by Huber:
  L_κ(u) = ½u² if |u|≤κ;  κ(|u|-½κ) otherwise.
  ρ^κ_τ(u) = |τ - 1_{u<0}| · L_κ(u).
  ρ^0_τ = ρ_τ (standard QR loss).
- For κ=1, |u|≤1: ½u²; |u|>1: |u|-½. (DQN's gradient-clipped squared error ≡ Huber κ=1.)

## QRTD (tabular policy evaluation update)
θ_i(x) ← θ_i(x) + α(τ̂_i - 1_{r+γz' < θ_i(x)}),  z'~Z_θ(x').
- This is the SGD on ρ_{τ̂_i}: ∂ρ_{τ̂_i}(u)/∂θ = -(τ̂_i - 1_{u<0}); u = (r+γz') - θ_i(x). So update = -α·∂/∂θ = α(τ̂_i - 1_{u<0}) = α(τ̂_i - 1_{r+γz'<θ_i}). Correct.
- In practice: draw many z'~Z(x'), or compute update for ALL pairs (θ_i(x), θ_j(x')).

## QR-DQN control
- Distributional Bellman OPTIMALITY operator: TZ(x,a) =_D R(x,a)+γZ(x',a'), a' = argmax_{a'} E_{z~Z(x',a')}[z] (greedy on MEAN of next-state distribution).
- Three minimal changes to DQN:
  1. Output layer size |A|×N (N quantile locations per action).
  2. Replace Huber loss with quantile Huber loss.
  3. Replace RMSProp with Adam.
- Algorithm (per transition x,a,r,x'):
  Q(x',a') = Σ_j q_j θ_j(x',a') = (1/N)Σ_j θ_j(x',a').
  a* = argmax_{a'} Q(x',a').  [NOTE: paper alg line writes argmax over Q(x,a'); but greedy should be on x'. The text says greedy on next-state mean. Implementations use x'. Flag: the printed algorithm line "a* ← argmax_{a'} Q(x, a')" appears to be a typo for x'.]
  Tθ_j ← r + γθ_j(x', a*)  ∀j  (target quantile locations; γ=0 if terminal).
  Loss = Σ_{i=1}^N E_j[ ρ^κ_{τ̂_i}(Tθ_j - θ_i(x,a)) ] = (1/N) Σ_i Σ_j ρ^κ_{τ̂_i}(Tθ_j - θ_i(x,a)).
- N=200 (Atari), κ=1 (QR-DQN-1) or κ=0 (QR-DQN-0). Adam α=5e-5, ε_ADAM=0.01/32.

## Contraction result (Prop 3, the main theory)
Π_{W1} T^π is a γ-contraction in d̄_∞ (∞-Wasserstein = max gap between CDFs):
  d̄_∞(Π_{W1}T^π Z_1, Π_{W1}T^π Z_2) ≤ γ d̄_∞(Z_1,Z_2).
=> unique fixed point Ẑ^π; repeated application (and its stochastic approx QRTD) converges. Since d̄_p ≤ d̄_∞, convergence for all p∈[1,∞].
- CAVEAT: contraction does NOT hold for p<∞ (Lemma: Π_{W1}T^π not a non-expansion in d̄_p, p∈[1,∞)) — counterexample with N=2.
- Proof structure: reduce to deterministic reward, γ=1 (Wasserstein translation-invariant, T^π already γ-contraction so suffices γ=1). Reduce N-Dirac value dists to single-Dirac via a transformed MDP (split each transition into N branches each prob p_i/N). Lemma WinftyIsMaxQuantileDiff: d_∞(Π_{W1}ν_1,Π_{W1}ν_2)=max_i|F_{ν1}^{-1}(τ̂_i)-F_{ν2}^{-1}(τ̂_i)| (optimal coupling pairs equal cumulative-prob quantiles). Lemma 1DiracCase: for single Diracs, Π_τ T^π is non-expansion in d̄_∞ — proof by contradiction on quantile ordering (if |θ_u-ψ_v|>|θ_i-ψ_i| ∀i then I_{≤θ_u}⊆I_{<ψ_v} forces τth quantile of T^πY < ψ_v, contradiction).

## Counterexample d_p non-contraction (concrete)
N=2, γ=1, x→x_1 (p=2/3), x→x_2 (p=1/3), rewards 0.
Z(x_1)=½δ_0+½δ_2, Y(x_1)=½δ_1+½δ_2; Z(x_2)=½δ_3+½δ_5, Y(x_2)=½δ_4+½δ_5.
d_p(Z(x_1),Y(x_1)) = (½·1)^{1/p} = 2^{-1/p}; same for x_2; d̄_p(Z,Y)=2^{-1/p}.
T^πZ(x)=⅓δ_0+⅓δ_2+⅙δ_3+⅙δ_5; T^πY(x)=⅓δ_1+⅓δ_2+⅙δ_4+⅙δ_5.
Project onto 2 Diracs = 25% and 75% quantiles:
Π Z = ½δ_0+½δ_3; Π Y = ½δ_1+½δ_4.
d̄_p(ΠT^πZ,ΠT^πY) = (½(1^p+1^p))^{1/p} = 1 > 2^{-1/p}. Expands. So p<∞ fails; need d_∞.

## Canonical implementation (SB3-contrib quantile_huber_loss; Kaixhin uses C51 not QR)
def quantile_huber_loss(current_quantiles, target_quantiles, cum_prob=None, sum_over_quantiles=True):
  # current: (batch, N), target: (batch, N')
  if cum_prob is None: cum_prob = (arange(N)+0.5)/N  -> τ̂_i = (i+0.5)/N  (= (2i+1)/(2N), i=0..N-1; matches midpoints)
     reshape to (1, N, 1) to broadcast.
  pairwise_delta = target.unsqueeze(-2) - current.unsqueeze(-1)   # (batch, N, N'): u_ij = Tθ_j - θ_i
  abs_delta = pairwise_delta.abs()
  huber = where(abs_delta>1, abs_delta-0.5, 0.5*pairwise_delta**2)   # κ=1
  loss = abs(cum_prob - (pairwise_delta.detach()<0).float()) * huber  # |τ_i - 1_{u<0}| L_κ
  if sum_over_quantiles: loss = loss.sum(dim=-2).mean()   # sum over current-quantile dim i, mean over batch & target j
  else: loss = loss.mean()
- network: output (batch, |A|, N); greedy a* = (quantiles.mean(dim=2)).argmax(1).

## Design-decision table
| choice | why | rejected alt |
|---|---|---|
| transpose: fixed prob, variable locations | enables Wasserstein-consistent SGD via QR; no support bounds; no projection | C51's fixed atoms + learnable probs (needs projection, KL, [Vmin,Vmax]) |
| quantile regression loss ρ_τ | unbiased sample gradient for quantile; minimizer = F^{-1}(τ) | direct W_p loss (biased gradients, Thm 1) |
| midpoint quantiles τ̂_i=(2i-1)/(2N) | exact W1-minimizing locations (Lemma 2) | endpoints τ_i=i/N (not the W1 minimizer of each cell) |
| quantile Huber, κ=1 | smooth at 0; stable gradients for nonlinear FA; matches DQN clipping | κ=0 hard QR loss (constant gradient near 0) |
| greedy on mean of next dist | acts to max expected return (same objective as DQN) | greedy on some quantile / risk measure (changes objective) |
| Adam instead of RMSProp | empirically better with the new loss | RMSProp (DQN default) |
| contraction stated in d̄_∞ | Π_{W1}T^π only contracts in ∞-Wasserstein, not p<∞ | claim d̄_p contraction (false, counterexample) |
| N=200 quantiles | resolution of distribution; N→ from DQN-like to fine quantiles | small N (coarser) |
