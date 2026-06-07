# CQL synthesis notes (pre-Phase-2)

## Pain point
Offline RL: fixed dataset D ~ d^β(s)π_β(a|s), no environment interaction. Naive off-policy
actor-critic / Q-learning fails. Diagnosis: policy-eval bootstraps target via
E_{a'~π}[Q(s',a')] with π trained to maximize Q. π drifts to OOD actions (a' s.t.
π_β(a'|s')≈0). Q at OOD actions is arbitrary (never trained there); any erroneous high value
gets bootstrapped into target → overestimation compounds across Bellman backups. Online RL
self-corrects (try action, observe true value); offline can't. Result: divergent Q, garbage policy.
Note: NO state distribution shift in Q training — backup only queries Q at s∈D and a'~π at s'∈D;
the issue is purely action distribution shift.

## Prior baselines and gaps
- BCQ (Fujimoto 2018): generative model of π_β, only allow actions close to data; constrains
  candidate actions. Gap: needs accurate behavior model; conservative; couples to generative model quality.
- BEAR (Kumar 2019): support-constraint via MMD between π and π_β; uses ensemble + min for
  uncertainty. Gap: still needs behavior model; uncertainty sets in offline RL must be very
  high-fidelity, loose sets fail. Empirically Δ^k (gap between OOD max-Q and in-dist Q) stays
  POSITIVE & grows → unlearning, even with constraint, because function-approx coupling makes
  OOD Q high anyway and the constraint doesn't regularize Q itself.
- KL / Wasserstein policy-constraint (BRAC, AWR, ABM): penalize divergence to π_β. Gap: same
  reliance on behavior estimate; too conservative or ineffective.
- Uncertainty / UCB methods (from exploration, optimism): in offline want pessimism = pointwise
  lower bound. Gap: calibration too loose for offline demands.
- SPIBB (Laroche 2017): bootstrap with behavior policy on unseen actions; safe-PI theorems.
- Robust MDP: overly conservative.

Key shared gap: all need to ESTIMATE π_β explicitly, and all constrain the POLICY but leave the
Q-FUNCTION unregularized, so function-approx error at OOD actions is uncorrected.

## The idea (discovery order)
Instead of constraining the policy away from OOD actions, regularize Q directly to be
PESSIMISTIC about OOD actions, provably lower-bounding the value. Then optimizing against a
lower bound is safe.

### Eq.1 (basic): min_Q  α E_{s~D,a~μ}[Q] + ½ E_{s,a,s'~D}[(Q − B̂^π Q̂^k)²]
- push DOWN Q under chosen μ(a|s) (state marginal = data's), keep Bellman fit.
- derivative=0 → Q̂^{k+1}(s,a) = B̂^π Q̂^k(s,a) − α μ(a|s)/π_β(a|s). Pointwise underestimate of
  the (empirical) Bellman target. Fixed point: Q̂^π ≤ Q^π − α(I−γP^π)^{-1}[μ/π_β] + sampling term.
  → POINTWISE lower bound for large enough α. (Theorem 1)
- This over-penalizes: pushes down EVERY action incl. in-data ones.

### Eq.2 (tighter): add push-UP term under data dist π_β
min_Q α( E_{s~D,a~μ}[Q] − E_{s,a~D}[Q] ) + ½ Bellman²
- only the policy's VALUE E_π[Q] needs to be a lower bound (that's all policy eval/improvement uses),
  not pointwise Q. So we can give back value on in-data actions.
- derivative=0 → Q̂^{k+1} = B̂^π Q̂^k − α(μ/π_β − 1). NOT pointwise lower bound (where μ<π_β, adds positive).
- but with μ=π: E_π[Q̂^{k+1}] = B^π V̂^k − α E_π[π/π_β − 1], and
  D_CQL(s)=Σ_a π(π/π_β −1) = Σ_a (π−π_β)²/π_β ≥ 0 (cross terms cancel since Σπ=Σπ_β=1).
  → V̂^π(s) ≤ V^π(s), TIGHTER (subtract 1). (Theorem 2)
- Necessity of π_β for push-up (App): solve max_ν min_π Σπ(π−ν)/π_β → optimal ν=π_β, value 0;
  any ν≠π_β admits a π making penalty negative → no guaranteed bound. So push-up MUST be under π_β.

### CQL(R) family: make μ adaptive instead of fixed
min_Q max_μ α(E_μ[Q] − E_{π_β}[Q]) + ½ Bellman² + R(μ).
- R(μ)=H(μ): solve max_μ E_μ[Q]+H(μ) s.t. Σμ=1 → μ*∝exp(Q), and the term becomes
  log Σ_a exp Q(s,a). → CQL(H):
  min_Q α E_{s~D}[ log Σ_a exp Q(s,a) − E_{a~π_β}[Q] ] + ½ Bellman².
  Soft-maximum push-down. Self-adjusting: hammers whichever action Q currently inflates.
- R(μ)=−D_KL(μ‖ρ): μ*∝ρ·exp(Q). ρ=Unif → CQL(H). ρ=π̂^{k-1} → exp-weighted avg of prev policy
  Q (CQL(ρ)); more stable in high-dim action spaces where logsumexp sampling is high-variance.
- R=var (DRO): penalize variance of Q across actions.

### Full algorithm properties
- CQL learns lower-bounded values across iterations (Theorem 3) under slow policy updates
  (D_TV(π^{k+1}, π_{Q̂^k}) ≤ ε); condition: E_{π_Q}[π_Q/π_β −1] ≥ max_a(π_Q/π_β)·ε. Hence the
  small policy LR (3e-5 vs 3e-4 for Q) — keep ε tiny.
- Gap-expanding (Theorem 4): CQL backup increases (E_{π_β}[Q] − E_{μ}[Q]) beyond the true gap,
  for large enough α. Push-up under π_β adds 0 expected shift (β^T(μ−β)/β = Σ(μ−β)=0); push-down
  under μ adds α Δ̂^k>0. So in-dist actions favored; implicitly prevents OOD without explicit constraint.
- Well-defined objective (Theorem 5): CQL fixed point = solving an MDP with reward
  r(s,a) − α(π/π_β − 1); equals max_π J(π, M̂) − α/(1−γ) E[D_CQL(π,π_β)]. Penalized empirical-MDP return.
- ζ-safe policy improvement (Theorem 6): J(π*,M) ≥ J(π_β,M) − ζ, ζ from sampling error
  ~ √(D_CQL+1)·√|A|/√|D(s)| minus the empirical-MDP gain. Smaller α suffices as data grows.

### Sampling error handling
Empirical Bellman B̂^π vs true B^π: concentration assumptions on r and T give
|B̂^π Q − B^π Q| ≤ C_{r,T,δ}R_max/((1−γ)√|D(s,a)|). Choose α big enough to cancel; threshold
DECREASES as |D| grows (limit: any α>0).

## Implementation grounding (CORL / young-geng single-file SAC+CQL)
- Critic loss = TD(twin Q, target min, optional SAC entropy backup) + α_cql·(logsumexp_term − Q(s,a_data)).
- logsumexp over actions in continuous control via importance sampling: cat [random_actions − log Unif,
  current-policy actions − logπ(·|s), next-policy actions − logπ(·|s')], logsumexp over the N samples,
  N=10. (Eq A: ½ uniform + ½ policy importance estimate.)
- diff = logsumexp_ood − q_data; min_q_loss = α_cql · diff. Optional clamp.
- Lagrange (auto-α): min_Q max_{α≥0} α(diff − τ); α_prime optimizer ascends, threshold τ=10 (mujoco)/5.
- Actor: SAC, max E[Q − logπ]. Q lr 3e-4, policy lr 3e-5.
- Discrete (QR-DQN): logsumexp exact over actions, B* backup.

## Design-decision → why
- Why push-down under μ with data state-marginal? Q only ever queried at in-data states; no state
  shift; restrict μ(s,a)=d^β(s)μ(a|s).
- Why add push-up term? Tightens bound; only value needs bounding; recovers in-data value lost by Eq.1.
- Why push-up under π_β specifically (not μ or uniform)? Proven necessity — only π_β guarantees a
  bound for worst-case π.
- Why logsumexp / μ∝exp Q? Closed-form max of E_μ[Q]+H(μ); soft-max self-targets the inflated action.
- Why CQL(ρ) with ρ=prev policy? logsumexp via sampling is high variance in big action spaces.
- Why min over twin Q + entropy backup? inherited from SAC; extra pessimism in target.
- Why importance sampling for logsumexp in continuous? can't enumerate actions; mix uniform+policy.
- Why policy lr ≪ Q lr (3e-5)? Theorem 3 needs slow policy change (small ε) for the per-iteration
  lower-bound to hold.
- Why Lagrange α? Hard to pick fixed α across datasets; dual descent targets a fixed gap τ.
- Why no behavior-policy estimator? CQL regularizes Q directly; gap-expanding implicitly constrains
  the policy → no explicit π_β model needed (unlike BCQ/BEAR/BRAC).
