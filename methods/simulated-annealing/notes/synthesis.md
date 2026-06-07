# Synthesis — Simulated Annealing

## Three sources (all retrieved & read this run)
1. PRIMARY: Kirkpatrick, Gelatt & Vecchi 1983, "Optimization by Simulated Annealing", Science 220(4598):671–680. PDF in refs/kirkpatrick1983.pdf. Read pages 3–12 via pdf_extract.
2. BACKGROUND: Metropolis, Rosenbluth, Rosenbluth, Teller & Teller 1953, "Equation of State Calculations by Fast Computing Machines", J. Chem. Phys. 21(6):1087–1092. refs/metropolis1953.pdf. Read the general-method + detailed-balance proof (pp.1087–1089). Also the Boltzmann distribution / partition function / free energy as read directly out of Kirkpatrick's "Statistical Mechanics" section.
3. EXPLAINER: scipy `dual_annealing` docs (generalized SA, GSA visiting/acceptance/temperature formulas) + Wikipedia SA page (Metropolis acceptance, quenching vs slow cooling, hill-climbing contrast). Code: perrygeo/simanneal (canonical classic SA) cloned into code/simanneal; scipy `_dual_annealing.py` curled into code/.

## Pain point (pre-method)
- Combinatorial optimization: minimize a cost f over a huge discrete configuration space. TSP is the canonical example; NP-complete; exact methods scale exponentially (≤ a few hundred cities feasible). So heuristics.
- Two heuristic strategies (Kirkpatrick names them): divide-and-conquer, and iterative improvement (= local search / hill-climbing). Iterative improvement: from a config, apply rearrangements, accept only downhill ones, stop when no neighbor improves. Gets stuck in a LOCAL but not global optimum; customary patch = many random restarts, keep best. Greedy TSP tour (nearest-neighbor) has step length ~1.12 on avg.
- The defect: accepting ONLY downhill moves = "extremely rapid quenching from high T to T=0", which freezes the system into a metastable (poor) state. This is the load-bearing pre-method observation.

## The physics borrowed (background facts)
- A system at temperature T in thermal equilibrium: each configuration {r_i} weighted by its Boltzmann factor exp(−E({r_i})/k_B T), E = energy, k_B = Boltzmann's constant. (Kirkpatrick p.4)
- Low-T limit: the Boltzmann distribution collapses onto the lowest-energy (ground) state(s). Ground states are exponentially rare among all configs yet dominate at low T because of the Boltzmann weighting.
- Real materials reach ground states (single crystal) NOT by sudden cooling but by careful ANNEALING: melt, then lower T slowly, spending a long time near the freezing point. Sudden quench → defects/glass/metastable.
- Partition function Z = Tr exp(−E/k_B T) (sum over all configs). F(T) = −k_B T ln Z = free energy; <E(T)>, entropy S(T) = ln(number of contributing configs) derivable from F. Specific heat C(T) = d<E>/dT; a peak in C signals a change of order (phase transition / freezing) — useful diagnostic for where to cool slowly.
- Spin glass / frustration: when interactions can't all be satisfied (e.g. a_ij random ferromagnetic + long-range antiferromagnetic A imbalance term), many near-degenerate ground states, no symmetry. Partition objective f = sum a_ij(μ_i−μ_j)^2/... is exactly a random-magnet Hamiltonian with μ_i = ±1.

## The Metropolis algorithm (1953, the engine)
- Goal there: compute canonical-ensemble averages F = ∫ F exp(−E/kT) / ∫ exp(−E/kT) over 2N-dim config space. Naive MC (sample uniformly, weight by exp(−E/kT)) wastes nearly all samples on low-weight configs.
- Modified MC ("importance sampling"): instead of sampling uniformly and weighting, sample configs WITH probability ∝ exp(−E/kT) and weight evenly.
- Mechanism: from current config, propose a small random displacement x → x + aξ (ξ uniform in [−1,1], a = max displacement). Compute ΔE.
  - If ΔE ≤ 0: accept.
  - If ΔE > 0: accept with probability exp(−ΔE/kT) (draw r~U(0,1), accept iff r < exp(−ΔE/kT)); else KEEP THE OLD CONFIG (and count it again in averages).
- Detailed-balance / stationary-distribution proof (Metropolis pp.1088–1089): proposal symmetric, P_rs = P_sr (equally likely to move anywhere in the 2a square). Take E_r > E_s. Number moving r→s = ν_r P_rs (all downhill accepted). Number moving s→r = ν_s P_sr exp(−(E_r−E_s)/kT) (uphill weighted). Net flux s→r = P_rs(ν_s exp(−(E_r−E_s)/kT) − ν_r). So if ν_r/ν_s > exp(−E_r/kT)/exp(−E_s/kT) the net flow is from r to s; the only fixed point is ν_r ∝ exp(−E_r/kT). Ergodicity (any state reachable) + this drift ⇒ the chain converges to the canonical (Boltzmann) distribution. Note: after a forbidden move you must re-count the initial config, else you wrongly deplete state s.
- Tuning note (Metropolis): a too large ⇒ most moves rejected; a too small ⇒ config barely changes; both slow equilibration.

## The leap (Kirkpatrick's contribution)
- Identify cost function f ↔ energy E; configurations ↔ system states; introduce a control parameter T (an "effective temperature" for optimization, with k_B ≡ 1, units of the cost). Then run Metropolis at temperature T: from a config, propose a random rearrangement, ΔE = Δcost; accept downhill always, uphill with prob exp(−ΔE/T).
- Iterative improvement = Metropolis at T=0 (only downhill accepted) = quench. Metropolis at T>0 generalizes it: controlled UPHILL moves let the walk escape local minima.
- Annealing schedule: start at high T (≈ all proposals accepted → explore freely, sample broadly), then SLOWLY lower T toward 0. As T falls the Boltzmann distribution concentrates on low-cost configs; at each T spend enough steps to reach equilibrium, then cool. The schedule = sequence of temperatures + #steps per temperature.
- Why slow: too-fast cooling = quench = freezes into a poor local minimum (the same metastable-glass failure mode as accept-only-downhill). Slow cooling lets the system stay near equilibrium and track the shifting Boltzmann distribution down to the ground state. "Adaptive divide-and-conquer": gross/large-scale features settle at high T (large-ΔE rearrangements still happen), fine details at low T.
- Specific-heat diagnostic: C(T) = d<f>/dT computed by watching variance of f at each T; a peak indicates a phase-transition-like reordering, telling you which T ranges need slow cooling.

## Cooling schedule specifics (grounded)
- Kirkpatrick partitioning run: start T0=10 (essentially all flips accepted), cool exponentially T_{n+1} = (T1/T0)^n T0 ... they write the geometric rule with ratio T1/T0 = 0.9 per stage; "frozen" when no further improvement. TSP: schedule found empirically; "melt" the system then cool in slow stages.
- simanneal library (canonical classic-SA code): EXPONENTIAL (geometric) schedule, T(step) = Tmax * exp(Tfactor * step/steps), Tfactor = −ln(Tmax/Tmin). At step=0 → Tmax; at step=steps → Tmax*exp(−ln(Tmax/Tmin)) = Tmin. Defaults Tmax=25000, Tmin=2.5, steps=50000. Accept rule: if dE>0 and exp(−dE/T) < random() → reject (restore prev). i.e. accept iff dE<=0 OR random() <= exp(−dE/T). Track best-ever separately. `auto()` calibrates Tmax to ~98% acceptance and Tmin to ~0% improvement.
- scipy dual_annealing = Generalized SA (Tsallis): distorted Cauchy-Lorentz visiting distribution (param q_v), acceptance p = min{1,[1−(1−q_a)βΔE]^(1/(1−q_a))}, temperature T_qv(t)=T_qv(1)(2^{q_v−1}−1)/((1+t)^{q_v−1}−1), plus local-search polish. Defaults init temp 5230, visit 2.62, accept −5.0. This is the explainer/generalization; the classic algorithm is the simanneal one.

## Design decisions → why (table)
- Cost as energy / Boltzmann weight exp(−f/T): so the low-cost configs become the high-probability ones; minimizing f ↔ finding the ground state. Alternative (uniform sampling, weight after) wastes samples — Metropolis importance-sampling fixes that.
- Metropolis acceptance exp(−ΔE/T) for uphill (not, say, a fixed accept prob, not a softmax over neighbors): it's the UNIQUE rule (with symmetric proposals) whose stationary distribution is Boltzmann (detailed-balance proof). Anything else doesn't sample exp(−E/T).
- Accept downhill always: special case ΔE≤0 ⇒ exp(−ΔE/T) ≥ 1, so "accept with prob min(1,exp(−ΔE/T))" = always. Falls out of the same formula.
- Temperature T as a knob: at high T, exp(−ΔE/T)→1, almost everything accepted ⇒ exploration / random walk; at low T, only tiny uphill moves accepted ⇒ exploitation / hill-climb. T interpolates between random search and greedy.
- SLOW cooling (annealing) rather than jump to T=0: T=0 = greedy = quench = metastable local min. Slow cooling keeps the chain near equilibrium at each T so it tracks the Boltzmann ground state. The metallurgy analogy is literal: anneal vs quench.
- Geometric/exponential schedule T_{k+1}=αT_k (α≈0.9–0.95): simple, spends more "stages" at low T where it matters; logarithmic schedule T_k ∝ c/ln(k) has an asymptotic global-convergence guarantee but is impractically slow, so geometric is the practical choice. (Explainer.)
- Enough steps per temperature (inner loop): need to (re)equilibrate at each T before cooling, else you're effectively quenching even with a slow outer schedule.
- Keep best-ever state separately: the final config at T_min is a sample, not necessarily the minimum visited; record the best seen.
- Tune move size / Tmax for ~high (but not ~100%) acceptance at the start, Tmin for ~0 improvement at the end: mirrors Metropolis's "a not too large/too small" and simanneal's auto().

## Canonical implementation to mirror
- perrygeo/simanneal Annealer.anneal(): exponential schedule, Metropolis accept/restore, best-state tracking. Final code mirrors this structure; TSP move/energy from examples/salesman.py.
