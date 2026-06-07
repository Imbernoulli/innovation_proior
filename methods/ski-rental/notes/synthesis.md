# Synthesis — ski-rental / rent-or-buy randomized e/(e-1)

## Sources actually read this run (refs/)
- `karlin1994_nonuniform.{pdf,txt}` — PRIMARY. Karlin, Manasse, McGeoch, Owicki, "Competitive Randomized Algorithms for Non-Uniform Problems" (SODA prelim / Algorithmica 1994). Full text, 9 pp.
- `sleator_tarjan_1985.{pdf,txt}` — antecedent. Sleator & Tarjan, "Amortized Efficiency of List Update and Paging Rules," CACM 28(2):202-208, 1985. Origin of comparing on-line cost to optimum off-line cost (C_A(s) ≤ 2·C_OPT(s)); competitive analysis.
- `cs369_online_notes.{pdf,txt}` — analysis. Stanford CS369 online-algorithms notes; deterministic ski-rental break-even, 2−1/k, oblivious adversary def.
- `gatech_ski_rental.{pdf,txt}` — analysis. Harik & Chung, Georgia Tech, "Generalizations of Ski-Rental" (2005). Continuous rent/buy: deterministic factor 2, randomized density p(z)=e^z/(e−1) on [0,1], adversary q(u), primal-dual / minimax → e/(e−1).
- `bonn_yao_principle.{pdf,txt}` — antecedent + analysis. Kesselheim, "Algorithms and Uncertainty" L5, Yao's Principle (statement + proof) and the discrete ski-rental randomized lower bound with adversary Pr[X≥t]=(1−1/B)^{t−1} → e/(e−1).

## The problem (rent-or-buy / ski rental)
- Each day you ski you pay rent = 1. At any time you can buy for B (=p in the cache paper, =k in CS369, =q/p in gatech). Bought once, never rent again. Horizon (#days) unknown and adversarial.
- OPT (knows horizon x): cost = min(x, B). Rent all if x<B; buy day 1 if x≥B.
- Snoopy-caching mapping (primary): a "write run of length k" = active cache writes k times; each broadcast update costs 1 (=rent/day); invalidating the block in the other cache costs p (=buy B). Algorithm A_i = "broadcast updates until the run exceeds i, then invalidate" = rent i days then buy. Spin-block (§3): continuous version — spin at cost ∝ time (rent) vs block at cost C (buy). All three are the SAME object.

## Deterministic result (folklore, but grounded)
- Family A_x = rent x days then buy on day x+1. Cost(A_x, x_days=d) = d if d≤x else x+B. Worst case d=x+1: cost = x+B, OPT = min(x+1,B)=B (when x=B−1). Ratio (B−1+B)/B = (2B−1)/B = 2 − 1/B.
- Best deterministic: x = B−1 (break-even: rent until cumulative rent equals buy price). Ratio 2−1/B, optimal among deterministic (gatech c(A_x)=(px+q)/min(px,q) ≥ 2; CS369: 2−1/k both upper & lower bound). Sleator-Tarjan break-even reflex: pay-to-rent until you've spent what buying costs.
- WALL: deterministic is stuck at 2. A strong (adaptive) adversary simulates the deterministic algorithm and sets horizon to x+1 every time. Primary, §1: "if the on-line algorithm is deterministic then a weak adversary can simulate a strong one." Theorem in [9] (Karlin et al 1988): no on-line snoopy alg beats 2 vs strong adversary.

## The leap: randomize the buy-day (oblivious/weak adversary)
- Against a WEAK (oblivious) adversary — fixes the sequence without seeing coin tosses — the adversary can no longer aim at the realized buy day. So randomize over A_i.
- Discrete distribution (PRIMARY, Theorem 1 proof). π_i = prob of choosing A_i (drop after i updates), i=1..p. Set up so expected cost ≤ (1+α)·OPT for every horizon k:
  - For a run of length k ≤ p (OPT = k): E[C_A] = Σ_{i<k} π_i (p + i) + (1 − Σ_{i<k} π_i)·k. [if buy-day i<k you paid i rents + p buy; if i≥k you paid k rents only.]
  - Require E[C_A(σ_k)] ≤ (1+α)k for k≤p and ≤ (1+α)p for k>p.
  - Setting the inequalities to equalities and solving the difference equation gives a geometric tail:
    π_i = α·(1−α)^{i−1}, i=1..p  (the (1−1/B)^{B−i}-type geometric weight on the buy day).
  - Normalize Σ_{i=1}^{p} π_i = 1 ⇒ α·(1−(1−α)^p)/α = 1−(1−α)^p = 1 needs the residual mass; primary solves α by Σπ_i=1 to get
    α = 1 / ((1+1/p)^p − 1) = 1/(e_p − 1),  where e_p := (1+1/p)^p.
  - Competitive factor = 1 + α = 1 + 1/(e_p−1) = e_p/(e_p − 1) → e/(e−1) ≈ 1.58 as p→∞.
- Continuous version (gatech / spin-block, primary §3). Buy-time z drawn on [0,1] (scale B=1) with density p(z) = e^z/(e−1). Then for every horizon the ratio is exactly e/(e−1). Derivation: want E[cost(d)] ≤ (1+α)·OPT(d) for all d; differentiate the equality constraint twice ⇒ ODE p′=p ⇒ p(z)=Ce^z; normalize ∫_0^1 p=1 and the boundary condition give C=1/(e−1) and 1+α = e/(e−1).
  - WALL inside the derivation: density must be supported on [0,1] only (buying after the break-even point B never helps — pure waste). Boundary at z=1 carries an atom-free cutoff; this is what forces the e/(e−1) and not something smaller.

## Lower bound: this is optimal vs weak adversary (Yao + primary Theorem 2)
- PRIMARY Theorem 2: for any on-line B with prob a_i of dropping after i writes, take the smallest k with a_k ≤ Σ_{i<k} π_i (the algorithm's own CDF must cross the optimal CDF). Because π was chosen to make all the constraints (*) tight, E[C_B(σ_k)] ≥ E[C_A(σ_k)], so no algorithm beats e_p/(e_p−1). Repeating the run makes the additive constant irrelevant ⇒ ratio ≥ e_p/(e_p−1) on arbitrarily expensive sequences.
- Yao's principle (bonn, antecedent). max_x E[c(A,x)] ≥ min_a E[c(a,X)] for any distribution X over instances. Choose the adversary horizon distribution X with Pr[X≥t] = (1−1/B)^{t−1}. Then:
  - E[c(OPT,X)] = Σ_{t=1}^B Pr[X≥t] = B(1−(1−1/B)^B).
  - Every deterministic a has E[c(a,X)] = Σ_{t=1}^a Pr[X≥t] + B·Pr[X>a] = B (all equalized — the defining property of the bad distribution).
  - So min_a E[c(a,X)] / E[c(OPT,X)] = 1/(1−(1−1/B)^B) → 1/(1−1/e) = e/(e−1). Matching lower bound.
  - This is the SAME geometric distribution as the algorithm's π, viewed from the adversary side — primal/dual.

## Design decisions → why
- A_i family (rent i days then buy): the only sensible deterministic strategies; buying earlier than the realized need or later both dominated — choice of cutoff is the only degree of freedom. (gatech, primary.)
- Break-even cutoff B−1 deterministically: equalizes the two regrets (you over-rent vs you over-buy); minimizes the max ratio at 2−1/B. (Sleator-Tarjan break-even intuition, CS369.)
- Randomize over the cutoff: defeats the adaptive adversary by hiding the realized buy day from an oblivious adversary. (primary §1 weak-vs-strong; CS369 oblivious adversary.)
- Geometric / exponential weighting (1−α)^{i−1} resp. e^z: forced, not chosen — it is the unique solution making the per-horizon competitive ratio constant across ALL horizons (difference eqn / ODE). Any flatter and short horizons are bad; any steeper and long horizons are bad. (primary Theorem 1 proof; gatech ODE.)
- α = 1/(e_p−1): forced by normalization Σπ=1. (primary.)
- Support only up to the buy point: buying after break-even is pure waste; truncation at p (resp. z=1) is what yields the finite e/(e−1). (gatech, primary.)
- Optimality: Theorem 2 / Yao — the construction is tight because π and the adversary's X are the same geometric object (primal-dual). (primary, bonn.)

## Code
- Real, runnable simulation: implement A_i, OPT, deterministic break-even, the geometric randomized policy with π_i = α(1−α)^{i−1}, α=1/(e_p−1); Monte-Carlo over adversarial horizons; verify empirical competitive ratio → e/(e−1). Plus the continuous density p(z)=e^z/(e−1) sampler. Grounded in the primary's distribution; no external canonical repo for ski-rental, so the code is a faithful transcription of the paper's algorithm.

## In-frame notes
- Method name to use in answer.md: "the randomized rent-or-buy (ski-rental) algorithm." Do not cite the target paper as artifact. Sleator-Tarjan 1985, Yao 1977 are prior-art ancestors and may be cited.
