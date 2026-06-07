# Synthesis — Reservoir SDP (Stedinger, Sule & Loucks 1984)

## Three sources
1. PRIMARY: Stedinger, Sule & Loucks 1984, "Stochastic Dynamic Programming Models for Reservoir
   Operation Optimization", Water Resources Research 20(11):1499–1505. Full text PAYWALLED (Wiley
   402; ResearchGate blocks curl). Obtained: abstract + the precise contribution (best forecast of
   current-period inflow as the hydrologic state variable, vs the previous-period inflow used by
   prior SDP; quadratic penalties for failing flood/irrigation/power targets; stationary periodic
   policies). GAP FLAGGED. The exact equations are recovered from Loucks's own textbook (#2), which
   reproduces this method.
2. BACKGROUND/SECONDARY (reproduces the equations): Loucks & van Beek 2017, *Water Resource Systems
   Planning and Management*, Springer open access (CC BY-NC). Ch.4 (deterministic reservoir DP,
   mass balance Eq.4.107/4.113, steady-state policy, backward recursion Eq.4.115) and Ch.7 Modeling
   Uncertainty (Markov transition probabilities §7.5 Eqs.7.19–7.23; SDP recursion §7.6 Eqs.7.37/7.38/
   7.51; joint-probability steady-state Eqs.7.42–7.46; 2-D rule curve + hedging Fig.7.12; the caveat
   that current inflow isn't known at decision time). By the same author Loucks → faithful.
   Plus Yakowitz 1982 survey "Dynamic programming applications in water resources" WRR 18(4):673
   (curse of dimensionality, Howard policy iteration) for the field state.
3. THIRD-PARTY EXPLAINER + CODE: swd-turner `reservoir` R package (CRAN, cran/reservoir mirror):
   sdp_hydro / sdp_supply functions + rdrr.io docs. Canonical implementation: storage discretized
   into S_disc states, inflow into Q classes by quantiles, periodic first-order Markov transition
   matrices estimated from history, backward value iteration to a stationary periodic policy,
   simulate. Markov=TRUE adds current-period inflow class as a state.

## The problem (in-frame, pre-method)
Operate a single reservoir over the seasons of a year, indefinitely, to maximize expected benefit
(hydropower energy ∝ release × head; or water-supply delivery) minus penalties for shortage / flood.
Inflows are RANDOM and serially correlated. Need a state-dependent release POLICY, not a single
deterministic release schedule, because next year's inflows aren't known.

## Mass balance (continuity)
S_{t+1} = S_t + Q_t − R_t − L_t(S_t,S_{t+1}) − Spill_t, with 0 ≤ S_{t+1} ≤ K (capacity).
L = evaporation/seepage (area-dependent). Spill = forced overflow when S would exceed K.
Release bounded: 0 ≤ R_t ≤ S_t + Q_t − L (can't release more than available, can't go below dead pool).

## Why a single deterministic schedule fails / why mean-inflow DP is insufficient
Deterministic DP on mean inflows gives one release per (storage, season) but ignores that a wet/dry
year changes the right release. Stochasticity has a "deep impact on performance" (curse: averaging
inflows ≠ averaging the nonlinear penalty). Need the expectation INSIDE the recursion.

## Inflow as a Markov chain
Serial correlation is real (low flows follow low flows). Discretize each period's flow into classes i;
estimate transition probabilities Ptij = Pr{Q_{t+1} in class j | Q_t in class i} by counting historical
transitions, one stochastic matrix per period t (periodic chain). Discrete representative flow per class
chosen to preserve mean & variance. Steady-state marginal flow probs from PQj = Σ_i PQi Pij, ΣPQi=1.

## Bellman recursion (the heart) — value iteration, backward, to stationary periodic policy
State = (storage class k, inflow class i, season t). Decision = release (equiv. final storage class l).
With n periods remaining:
  F_t^n(k,i) = opt over feasible l { benefit/−cost(k,i,l) + Σ_j Ptij · F_{t+1}^{n−1}(l,j) }
For supply (minimize): F_t^n(S_k,q_i)=min_l { ((target−R)/target)^p + Σ_j Ptij F_{t+1}^{n-1}(S_l,q_j) }
For hydropower (maximize): F_t^n = max_R { R·H(S) + Σ_j Ptij F_{t+1}^{n-1}(S_{t+1},q_j) }, H=head.
Subject to S_{l,t+1}=S_k+q_i−R−L (mass balance), 0≤S_{l,t+1}≤K, R≤S_k+q_i−L.
The Σ_j Ptij(·) is the EXPECTATION over next-period inflow, conditioned on the current inflow class —
that's exactly where the inflow Markov chain enters the value function.
Run backward season-by-season; periods cycle 1..frq..1..; F^0=0; iterate years until the argmax/argmin
policy l*(k,i,t) repeats from one year to the next → STATIONARY (steady-state) periodic policy. This is
value iteration on a periodic (seasonal) MDP; convergence of the policy is the stopping rule.

## The Stedinger refinement (primary contribution)
Most prior SDP used PREVIOUS-period inflow Q_{t-1} as the hydrologic state. But the current decision
benefits from a FORECAST of the current/coming inflow. Using the best forecast of the current period's
inflow as the hydrologic state variable (and the transition probabilities of that forecast) to define
the policy and compute expected future benefits improves simulated operations. Caveat (textbook §7.6):
if you condition on the *actual* current inflow you can't follow the policy in real time (inflow not
yet observed when you decide) → either condition on the previous inflow / forecast, or report the
expected release per storage (Fig.7.12) as the implementable 1-D rule with hedging.

## Curse of dimensionality (context fact, Yakowitz 1982)
State space = (storage discretization)^(#reservoirs) × (#inflow classes)^(#hydrologic states).
Each added reservoir multiplies the state count exponentially; computation/storage explode. SDP for a
single reservoir with one inflow state is tractable; multi-reservoir is not, directly.

## Hedging (emerges from the convex penalty)
Quadratic/convex shortage penalty ⇒ optimal to accept a small deficit now to avoid a large one later
⇒ release < demand when storage is low (carry water over). Marginal value of current release =
marginal value of carryover storage. This falls out of the recursion; the rule curve bends below the
demand line at low storage.

## Code structure (cran/reservoir sdp_hydro/sdp_supply) — final code grounds here
- frequency frq (12 or 4); reshape Q into year×period matrix.
- S_states = seq(0,K,K/S_disc); R_disc_x = seq(0,qmax,qmax/R_disc) (or final-storage grid).
- Q classes via quantiles Q_disc (default 5 classes); class median preserves location.
- Markov: per-period transition matrix Q_trans_probs[i,j,t] from counting Q_class transitions.
- Backward loop t=frq..1: build R/mass-balance arrays, implied next-storage state index, immediate
  reward/cost array, expectation = Σ_j Ptij · (to-go at implied next storage, class j), add, take
  max/min over release → R_policy[,,t]; carry Cost/Rev_to_go.
- repeat full-year sweeps until fraction of unchanged policy entries > tol → stationary.
- Simulate: walk the historical inflow series, look up policy by (storage state, [inflow class], period),
  apply mass balance with spill, accumulate energy/deficit.

## Design choices → why
- State = storage (+inflow class): storage is the only controllable carryover; inflow class captures
  the serially-correlated hydrology the next inflow depends on. Without inflow state the expectation
  uses the unconditional flow distribution (loses persistence info) — Markov=TRUE conditions on it.
- Discretize storage finely (default 1000), inflow coarsely (5 classes): value function is smooth in
  storage (need resolution for the policy) but the inflow only enters via an expectation (a few classes
  preserve mean/variance; more classes blow up the chain & data needed per cell).
- Quantile-based inflow classes: equal-probability-ish bins keep enough transition counts per class.
- Quadratic (loss_exp=2) penalty: convex ⇒ hedging; "many small deficits better than few large".
- Backward value iteration + stationarity stop: periodic infinite-horizon MDP; the periodic policy is
  the fixed point; policy-repeat is the practical convergence test (cheaper than comparing values).
- Representative flow preserves mean & variance: discretization shouldn't bias the expected inflow.
