# C51 synthesis (pre-Phase-2 understanding)

## Pain point / research question
Value-based RL (Q-learning, SARSA, DQN) learns only the SCALAR EXPECTED return
Q(x,a)=E[Z(x,a)]. Bellman: Q(x,a)=E R(x,a)+γ E Q(X',A'). Two contraction operators
(T^π policy eval, T optimality) in L∞; Banach → unique fixed point Q^π / Q*.
But the return Z is a RANDOM VARIABLE — multimodal, skewed, heavy-tailed. Collapsing
to the mean throws away that structure. Question: can we learn the WHOLE distribution
of returns, does a Bellman recursion still hold for it, does it still contract, and how
do we represent + update a distribution with a neural net?

## Distributional Bellman equation (equality in distribution)
Z^π(x,a) =_D R(x,a) + γ Z^π(X',A'),  X'~P(·|x,a), A'~π(·|X').
Three INDEPENDENT sources of randomness: reward R, transition→(X',A'), next return Z(X',A').
Operators:
  P^π Z(x,a) :=_D Z(X',A')           (transition operator on distributions)
  T^π Z(x,a) :=_D R(x,a) + γ P^π Z(x,a)
T (optimality) := any T^π with π greedy w.r.t. E Z (π ∈ G_Z, maximizes the MEAN).

## Wasserstein metric (the right metric)
d_p(F,G) = inf over couplings ||U−V||_p = (∫_0^1 |F^{-1}(u)−G^{-1}(u)|^p du)^{1/p}.
Properties (A indep of U,V; a scalar):
  P1: d_p(aU,aV) ≤ |a| d_p(U,V)
  P2: d_p(A+U,A+V) ≤ d_p(U,V)
  P3: d_p(AU,AV) ≤ ||A||_p d_p(U,V)
Maximal/sup metric over value distributions:  d̄_p(Z1,Z2) := sup_{x,a} d_p(Z1(x,a),Z2(x,a)).

## Contraction of T^π in d̄_p  (Lemma)
d_p(T^π Z1(x,a), T^π Z2(x,a))
 = d_p(R+γP^πZ1, R+γP^πZ2)
 ≤ (P2, same R shifts both) d_p(γP^πZ1, γP^πZ2)
 ≤ (P1) γ d_p(P^πZ1(x,a), P^πZ2(x,a))
 ≤ γ sup_{x',a'} d_p(Z1(x',a'),Z2(x',a'))   [def of P^π: P^πZ(x,a)=_D Z(X',A')]
Take sup over (x,a):  d̄_p(T^πZ1,T^πZ2) ≤ γ d̄_p(Z1,Z2).  → γ-contraction.
Banach (Z has bounded moments) → unique fixed point = Z^π; all moments converge geom.

## METRIC SUBTLETY (the easy place to err)
T^π is a γ-contraction in the MAXIMAL WASSERSTEIN d̄_p. It is NOT a contraction in:
  - total variation (Chung & Sobel 1987 give the counterexample)
  - KL divergence
  - Kolmogorov (sup-CDF) distance.
Reason intuitively: γ-scaling SHRINKS the support toward 0 (γ<1). Wasserstein measures
horizontal/transport distance, so scaling by γ scales the distance by γ — contraction.
TV/KL/Kolmogorov are "vertical"/overlap-based: two scaled distributions with disjoint
support still have TV=1 no matter how small γ is, so no contraction. This is why d_p is
THE tool. (Property P1 is exactly the horizontal-scaling fact; TV has no P1 analogue.)

## Centered moments (appendix, Sobel 1982)
d_2^2(U,V) ≤ E[(U−V)^2] = Var(C)+(E C)^2, C=U−V coupling → can't bound variance diff
directly via d_2. BUT by independence of R and P^πZ:
  Var(T^π Z(x,a)) = Var R(x,a) + γ^2 Var(P^π Z(x,a)).
⇒ ||Var(T^πZ1)−Var(T^πZ2)||_∞ ≤ γ^2 ||Var Z1 − Var Z2||_∞. Contraction in variance with
modulus γ^2 (not γ). Not a contraction in pth centered moment p>2 in general, but moments
still converge geometrically.

## Control setting (the instability — negative results)
Optimality op T Z = T^π Z for some greedy π∈G_Z (greedy w.r.t. E Z).
Lemma: ||E T Z1 − E T Z2||_∞ ≤ γ ||E Z1 − E Z2||_∞ (mean behaves: E T = T_E E, linearity).
So E Z_k → Q* geometrically — the MEAN is fine.
BUT:
  Prop: T is NOT a contraction in ANY metric over distributions.
    2-state example: x1→x2; at x2, a1 gives 0, a2 gives 1+ε / −1+ε w.p.½ each; both terminal.
    Unique optimal policy (a2 at x2), unique fixed point Z*. Take Z that equals Z* except at
    (x2,a2) where it is −ε±1. Then d̄_1(Z,Z*)=d_1(Z(x2,a2),Z*(x2,a2))=2ε.
    Apply T: greedy now picks a1 at x2 (because E of −ε±1 = −ε < 0 = E of a1)... wait the
    paper: T Z(x1)=Z(x2,a1)=0; mean of a1 (0) vs a2 (−ε) → a1 greedy. So TZ(x1)=δ_0.
    d_1(TZ,TZ*)=d_1(TZ(x1),Z*(x1)) where Z*(x1)=ε±1. = ½|1−ε|+½|1+ε| ≈ 1 > 2ε for small ε.
    So d̄_1(TZ,TZ*) > d̄_1(Z,Z*): EXPANSION. With γ<1 same proof works (not a contraction).
  Prop: some optimality ops have NO fixed point (tie-breaking that oscillates a1/a2 forever).
  Prop: even with a fixed point, {Z_k} need not converge to Z* (nonstationary example:
    a1 reward ½ deterministic, a2 reward 0/1 each w.p.½, γ=½, both optimal. p=0 return =1;
    p=1 return uniform on [0,2] (binary expansion .R1R2...); nonstationary a1-then-a2 gives
    uniform on [1/2,3/2], a distribution achievable by NO stationary p).
Best we can prove: convergence (pointwise / uniform if X finite) to the set of NONSTATIONARY
optimal value distributions Z**, NOT to Z*. Theorem (control convergence) proof:
  ε_k=γ^k B, B=2 sup||Z||. X_k = {x: gap Q*(x,π*(x))−max_{a≠π*}Q*(x,a) > 2ε_k}.
  Lemma exp-conv-mean ⇒ |Q_k−Q*|≤ε_k ⇒ on X_k greedy π_k = π*. X_k ↑ X.
  Recursively X_{k,i} = states in X_k whose successor under π* lands in X_{k-1,i-1} w.p. ≥1−δ.
  Partition lemma splits transition into "solved" S and "unsolved" S̄.
  d_p(W_{k+1}(x),W*(x)) ≤ γ d_p(P^{π_k}W_k,P^{π*}W*) (P1,P2)
   ≤ γ d_p(S W_k, S W*) + γ d_p(S̄ W_k, S̄ W*) (partition lemma)
   second term ≤ γ ||S̄(X')||_p sup d_p(W_k,W*) ≤ γ δ B (P3).
  Induct on i: d_p(W_{k+i}(x),W*(x)) ≤ γ^i B + δB/(1−γ) → 0 as δ→0,i→∞,k→∞.
Practical upshot: model the full distribution → averages out chattering (like conservative
policy iteration), more stable than greedy mean updates.

## Partition lemma (appendix, used in control proof + sample-Wasserstein)
A_i partition of Ω (indicator, exactly one =1). Then d_p(U,V) ≤ Σ_i d_p(A_i U, A_i V).
Proof: d_p^p(A_iU,A_iV)=inf E[|Y_i−Z_i|^p]; |A_iU−A_iV|=0 when A_i=0 so set those couplings=0.
The inf over (U,V) couplings is ≤ inf over per-component couplings (extra constraint:
must preserve conditional CDFs — can only reorder WITHIN each partition element, not across).
Components independently minimized ⇒ Σ_i d_p(A_iU,A_iV).

## Why NOT minimize Wasserstein loss with SGD (appendix — crucial design block)
Even though d_p is the right metric for the operator, you CANNOT minimize it from sampled
transitions with SGD. Sample-Wasserstein lemma: P = mixture P_I (I random index indep),
Q indep of I. Then d_p(P,Q) ≤ E_{i~I} d_p(P_i,Q), strict in general, and
  ∇_Q d_p(P_I,Q) ≠ E_i ∇_Q d_p(P_i,Q).
So a single sampled transition gives a BIASED gradient of the true Wasserstein loss.
Counterexample: P=½δ_0+½δ_1 (I∈{1,2},P_1=0,P_2=1). Q=p·δ_0+(1−p)δ_1. d_1(P,Q)=|p−½|,
which is <½ for p∈(0,1); but E_I d_1(P_i,Q)=½p+½(1−p)=½. So sampled loss expectation ½
> true ½... err: true d_1(P,Q)=|p−½|≤½, sampled-mean =½, biased upward, gradient wrong.
CliffWalk experiment confirms: Wasserstein-SGD converges to wrong fixed point (local minima),
categorical-projection+cross-entropy actually minimizes the Wasserstein metric in practice.
This is the dead end that forces the categorical/projection route.

## Categorical representation (the parametric choice)
Fixed support: N atoms z_i = Vmin + iΔz, i=0..N−1, Δz=(Vmax−Vmin)/(N−1).
Distribution Z_θ(x,a) = Σ_i p_i(x,a) δ_{z_i}, p_i = softmax(θ_i(x,a)).
N=51, Vmax=−Vmin=10 chosen from preliminary Atari sweeps (over {3,10,100}). Hence "C51".
Discrete categorical = expressive + computationally friendly (cf. pixelCNN softmax-over-bins).
Gaussian was tried before (Morimura, Tamar) — limited to unimodal; categorical keeps modes.

## The support-mismatch problem → categorical projection Φ
Apply T̂ to atoms: T̂ z_j = r + γ z_j. These shifted atoms generally do NOT land on {z_i}
(γ-scaling + r-shift). So T̂Z_θ and Z_θ have (almost always) disjoint support — can't take
KL directly. Solution: PROJECT the shifted distribution back onto the fixed grid by splitting
each shifted atom's probability onto its two nearest grid neighbours (linear interpolation),
with clamp to [Vmin,Vmax].
Projection formula (eqn cat_proj):
  (Φ T̂ Z_θ(x,a))_i = Σ_{j=0}^{N-1} [ 1 − |[T̂ z_j]_{Vmin}^{Vmax} − z_i| / Δz ]_0^1  p_j(x',π(x'))
where [·]_{Vmin}^{Vmax} clamps the value, [·]_0^1 clamps the triangular weight to [0,1].
The weight [1−|·|/Δz]_0^1 is a triangular kernel of half-width Δz centered at the shifted
atom — nonzero only for the ≤2 grid atoms within Δz. For shifted value falling at fractional
position b_j=(T̂z_j−Vmin)/Δz, with l=⌊b_j⌋, u=⌈b_j⌉:
  atom l gets weight (u − b_j),  atom u gets weight (b_j − l),  times p_j.
Check: at z_l, |T̂z_j−z_l|/Δz = (b_j−l), weight 1−(b_j−l)=(u−b_j) when u=l+1. ✓
       at z_u, |T̂z_j−z_u|/Δz = (u−b_j), weight 1−(u−b_j)=(b_j−l). ✓ Sum = 1 (mass preserved).
Algorithm 1 = this in O(N) per atom: m_l += p_j(u−b_j); m_u += p_j(b_j−l). Terminal: γ_t=0.
Integer-b_j edge case (l=u): naive (u−b_j)=0 would drop mass; impl adds (l==u) so atom l
gets full p_j. This is the categorical projection's only subtlety in code.

## Loss
Target distribution m = Φ T̂ Z_{θ̃}(x,a) from TARGET net θ̃ (greedy a* = argmax_a Σ_i z_i p_i(x',a)).
Predicted = Z_θ(x,a). Loss = cross-entropy term of KL(Φ T̂ Z_{θ̃} || Z_θ):
  L_{x,a}(θ) = − Σ_i m_i log p_i(x,a).
Bellman update reduces to MULTICLASS CLASSIFICATION over the N atoms. Minimized by SGD/Adam.
KL between categoricals is easy to optimize; KL insensitive to atom VALUES (only matches mass)
— a Wasserstein-aware loss should do even better but isn't SGD-able from samples.

## Categorical DQN / C51 architecture (grounded in CleanRL + DQN)
DQN nature CNN: conv(4,32,8,s4)-relu-conv(32,64,4,s2)-relu-conv(64,64,3,s1)-relu-fc(3136,512)
-relu-fc(512, n_actions*N). Reshape to (n_actions,N), softmax over N → p_i(x,a).
Q(x,a)=Σ_i z_i p_i(x,a) for ε-greedy action selection. Target net for θ̃. Adam (eps=0.01/L,
L=32 batch), lr 2.5e-4 (DQN's 0.00025). γ=0.99. Reward clipping [-1,1], frame stack 4, etc.
Bernoulli (N=2) one-param alternative: Φ T̂ Z := [(E[T̂Z]−Vmin)/Δz]_0^1. Beats DQN in 3/5 games.

## Design decisions → why
- Distribution not mean: preserves multimodality, averages chattering, propagates rare events,
  more stable target under approximation + nonstationary policy.
- Wasserstein metric for analysis: only metric where γ-scaling gives contraction (P1); TV/KL/
  Kolmogorov fail because they're overlap/vertical, blind to γ-shrink of support.
- Categorical (fixed-support discrete) not Gaussian: multimodal, expressive, cheap softmax.
- N=51, [Vmin,Vmax]=[−10,10]: empirical sweep; more atoms always helped (capacity for finer
  resolution); bounded support also acts as inductive bias (clips extremal returns as equal).
- Projection Φ + cross-entropy NOT Wasserstein-SGD: Wasserstein gradient from a single sample
  is biased (sample-Wasserstein lemma); cross-entropy on projected target is unbiased and easy.
- Target network for θ̃: same as DQN, stabilizes the bootstrap target.
- Greedy on mean (Σ z_i p_i) not on distribution: keeps it a drop-in DQN replacement; richer
  distribution-based action selection left open.
