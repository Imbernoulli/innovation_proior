# L-SHADE, distilled

L-SHADE (Success-History based Adaptive Differential Evolution with Linear Population Size
Reduction) is a self-tuning differential evolution algorithm for real-parameter black-box
optimization. It removes DE's per-problem hand-tuning of the scaling factor `F` and crossover
rate `CR` by learning them online from a historical memory of recently successful values, and it
removes the fixed-population-size compromise by shrinking the population linearly over the
evaluation budget. It builds on JADE's current-to-pbest/1 mutation and external archive, and on
SHADE's history-based parameter adaptation.

## Problem it solves

Continuous black-box optimization `min_x f(x)`, `x in [lo, hi]^D`, with only function evaluations
available (no gradients), under a fixed evaluation budget, across diverse landscapes (unimodal,
ill-conditioned, multimodal, composite). Classical DE works but its three parameters `N`, `F`,
`CR` are problem-dependent and must be hand-tuned; L-SHADE makes `F`, `CR` self-adapting and `N`
scheduled.

## Key ideas

1. **Self-adapting `F` and `CR` from a success-history memory.** Keep `H` memory slots
   `M_F`, `M_CR` (all initialized to `0.5`). Each individual `i` picks a slot `r` uniformly and
   samples
   - `CR_i = randn(M_{CR,r}, 0.1)`, clamped to `[0,1]` (Normal: `CR` should settle stably);
   - `F_i = randc(M_{F,r}, 0.1)`, truncated to `1` if above, **re-sampled** if `<= 0` (Cauchy's
     heavy tail keeps `F` diverse, fighting premature convergence; `F <= 0` is degenerate).
2. **current-to-pbest/1 mutation with an external archive (from JADE).**
   `v_i = x_i + F_i*(x_pbest - x_i) + F_i*(x_{r1} - x_{r2})`, where `x_pbest` is a *random* member
   of the top `N*p` (not the single best — avoids funnelling the whole population into one basin),
   `r1` is from the population `P`, and `r2` is from `P ∪ A`, the population union the archive of
   recently *losing* parents (the archive injects abandoned directions for diversity). `|A|` is
   capped at `arc_rate * N` with random overflow deletion.
3. **History updated round-robin by the improvement-weighted Lehmer mean.** Each generation,
   winners (trials with `f(u) < f(x)`) contribute `(F_i, CR_i)` weighted by improvement
   `Δf = |f(x) - f(u)|`. One memory slot `k` (cycling `1..H`) is overwritten with
   `mean_WL(S) = (Σ_k w_k S_k^2) / (Σ_k w_k S_k)`, `w_k = Δf_k / Σ Δf`. For positive `S`, this
   weighted Lehmer mean is at least the weighted arithmetic mean because
   `Σ wS^2 - (Σ wS)^2 >= 0`; it therefore upweights larger successful values, especially for
   `F`, preventing the mutation magnitude from drifting down. Updating *one* slot per generation
   means a single unlucky generation can corrupt at most `1/H` of the adaptation — the robustness
   fix over JADE's single `(mu_F, mu_CR)`. If the write slot is already terminal or the slot's
   successful `CR` are all `0` (`max(S_CR)=0`), the slot is set to a **terminal** marker that
   locks its `CR` to `0` thereafter (a "change-one-coordinate-at-a-time" policy, effective on some
   multimodal functions). No success in a generation ⇒ no memory update.
4. **Linear Population Size Reduction (LPSR) — the L-SHADE contribution.** Reduce `N` linearly
   from `N_init` to `N_min` as a function of evaluations spent:
   `N_{G+1} = round[ ((N_min - N_init)/MAX_NFE) * NFE + N_init ]`. `N_min = 4` (the minimum
   `current-to-pbest/1` needs: `x_i`, `x_pbest`, `x_{r1}`, `x_{r2}` distinct). When `N` drops,
   delete the **worst** individuals (elitist) and rescale the archive. Shrinking converts budget
   that a fixed large `N` would waste on redundant late exploration into many small refinement
   generations; it is deterministic, so it adds only one new parameter (`N_init`). It is the
   linear special case of the SVPS framework.

## Concrete parameters

- `N_init = 18 * D`; `N_min = 4`.
- `H = 5` (memory slots); `p = 0.11` with `pNP = max(round(p*N), 2)`.
- archive cap `= r_arc * N`, `r_arc = 1.4`.
- sampling spread `0.1` for both Normal (`CR`) and Cauchy (`F`).

## Final algorithm

```
init population P of N_init random vectors in [lo,hi]^D; evaluate
M_F = M_CR = [0.5]*H;  k = 0;  archive A = {}
while NFE < MAX_NFE:
    sort P by fitness
    S_F = S_CR = Δf = []
    for each individual i:
        r  = uniform index in [0, H)
        CR_i = 0 if slot r terminal else clip(randn(M_CR[r], 0.1), 0, 1)
        F_i  = resample randc(M_F[r], 0.1) until >0;  F_i = min(F_i, 1)
        pNP = max(round(p*N), 2);  x_pbest = random of top pNP
        r1 in P (≠ i);  r2 in P∪A (≠ i, ≠ r1)
        v_i = x_i + F_i*(x_pbest - x_i) + F_i*(x_r1 - x_r2);  repair to bounds
        u_i = binomial_crossover(x_i, v_i, CR_i)   # ≥1 coord from v via j_rand
        evaluate u_i
        if f(u_i) < f(x_i):
            record F_i,CR_i in S_F,S_CR; Δf += |f(x_i)-f(u_i)|; A += x_i
            x_i <- u_i
    remove duplicate archive entries; if |A| exceeds cap, delete random entries
    if S_F nonempty:
        w = Δf / sum(Δf)
        M_F[k]  = Σ w F^2 / Σ w F                          # weighted Lehmer
        M_CR[k] = -1 if M_CR[k] terminal or max(S_CR)==0 else Σ w CR^2 / Σ w CR
        k = (k+1) mod H
    N_next = max(N_min, round(((N_min - N_init)/MAX_NFE)*NFE + N_init))
    if N_next < N:  delete worst (N - N_next); rescale A to r_arc*N_next
return best individual, per-generation best-fitness history
```

## Working code

One concrete NumPy implementation fills the reproduction-and-population-management slot of the
population-based optimization harness.

```python
import numpy as np


def update_archive(archive, new_points, cap, rng):
    if new_points.size == 0 or cap == 0:
        return archive[:0] if cap == 0 else archive
    archive = np.vstack([archive, new_points])
    _, first_idx = np.unique(archive, axis=0, return_index=True)
    archive = archive[np.sort(first_idx)]
    if len(archive) > cap:
        archive = archive[rng.permutation(len(archive))[:cap]]
    return archive


def run_evolution(evaluate, dim, lo, hi, pop_size, max_nfes, seed):
    """L-SHADE: success-history adaptive DE (current-to-pbest/1 + archive,
    Cauchy F / Normal CR from an H-slot memory) with linear population reduction."""
    rng = np.random.default_rng(seed)

    H = 5                       # memory slots
    p_best_rate = 0.11          # fixed pbest greediness
    arc_rate = 1.4              # archive cap = arc_rate * current N
    max_pop_size = pop_size     # N_init
    min_pop_size = 4            # N_min (current-to-pbest/1 needs 4 distinct individuals)

    pop = lo + rng.random((pop_size, dim)) * (hi - lo)
    fitness = np.array([evaluate(ind) for ind in pop])
    nfes = pop_size

    memory_sf = np.full(H, 0.5)         # success-history of F centers
    memory_cr = np.full(H, 0.5)         # success-history of CR centers
    memory_pos = 0                      # round-robin write index

    archive = np.zeros((0, dim))        # external archive of losing parents
    archive_cap = int(round(arc_rate * pop_size))
    fitness_history = []

    while nfes < max_nfes:
        N = pop_size
        sorted_idx = np.argsort(fitness)            # best first

        # sample CR ~ Normal(M_CR[r], .1), F ~ Cauchy(M_F[r], .1)
        ridx = rng.integers(0, H, N)
        mu_cr, mu_sf = memory_cr[ridx], memory_sf[ridx]

        cr = rng.normal(mu_cr, 0.1)
        cr[mu_cr == -1] = 0.0                        # terminal slot -> CR = 0
        cr = np.clip(cr, 0.0, 1.0)

        sf = mu_sf + 0.1 * np.tan(np.pi * (rng.random(N) - 0.5))
        bad = sf <= 0
        while np.any(bad):                           # F <= 0 degenerate -> resample
            sf[bad] = mu_sf[bad] + 0.1 * np.tan(np.pi * (rng.random(bad.sum()) - 0.5))
            bad = sf <= 0
        sf = np.minimum(sf, 1.0)                     # F > 1 -> 1

        # current-to-pbest/1 with r2 drawn from P u A
        pop_all = np.vstack([pop, archive])
        donors = np.empty((N, dim))
        pnp = max(2, int(round(p_best_rate * N)))
        for i in range(N):
            pbest = pop[sorted_idx[rng.integers(0, pnp)]]
            r1 = rng.integers(0, N)
            while r1 == i:
                r1 = rng.integers(0, N)
            r2 = rng.integers(0, len(pop_all))
            while r2 == i or r2 == r1:
                r2 = rng.integers(0, len(pop_all))
            donors[i] = pop[i] + sf[i] * (pbest - pop[i] + pop[r1] - pop_all[r2])

        # bound repair: violated coord -> midpoint of bound and parent
        donors = np.where(donors < lo, (lo + pop) / 2.0, donors)
        donors = np.where(donors > hi, (hi + pop) / 2.0, donors)

        # binomial crossover with guaranteed donor coord
        cross = rng.random((N, dim)) < cr[:, None]
        cross[np.arange(N), rng.integers(0, dim, N)] = True
        trials = np.where(cross, donors, pop)

        trial_fit = np.array([evaluate(t) for t in trials])
        nfes += N

        improved = trial_fit < fitness
        dif = np.abs(fitness - trial_fit)
        good_cr, good_sf, good_df = cr[improved], sf[improved], dif[improved]
        if improved.any():
            archive = update_archive(archive, pop[improved], archive_cap, rng)

        pop[improved] = trials[improved]
        fitness[improved] = trial_fit[improved]

        # round-robin memory update with improvement-weighted Lehmer mean
        if good_df.size > 0:
            w = good_df / good_df.sum()
            memory_sf[memory_pos] = np.sum(w * good_sf ** 2) / np.sum(w * good_sf)
            if good_cr.max() == 0 or memory_cr[memory_pos] == -1:
                memory_cr[memory_pos] = -1
            else:
                memory_cr[memory_pos] = np.sum(w * good_cr ** 2) / np.sum(w * good_cr)
            memory_pos = (memory_pos + 1) % H

        # linear population size reduction (LPSR), elitist deletion
        plan = max(min_pop_size,
                   round(((min_pop_size - max_pop_size) / max_nfes) * nfes + max_pop_size))
        if plan < pop_size:
            keep = np.argsort(fitness)[:plan]
            pop, fitness = pop[keep], fitness[keep]
            pop_size = plan
            archive_cap = int(round(arc_rate * pop_size))
            if len(archive) > archive_cap:
                archive = archive[rng.permutation(len(archive))[:archive_cap]]

        fitness_history.append(float(fitness.min()))

    return pop[int(np.argmin(fitness))], fitness_history
```

## Relation to prior methods

- **Classical DE (Storn & Price 1997)** = fixed `F`, `CR`, `N`; L-SHADE makes `F`, `CR` adapt
  and `N` shrink.
- **JADE (Zhang & Sanderson 2009)** = current-to-pbest/1 + archive + adaptive `(mu_F, mu_CR)`
  with Cauchy `F`/Normal `CR`, arithmetic updating for `CR`, and a Lehmer mean for `F`.
- **SHADE / SHADE 1.1 (Tanabe & Fukunaga 2013)** = replace JADE's single center with the `H`-slot
  success-history memory. The SHADE 1.1 variant used by L-SHADE updates one slot round-robin,
  uses improvement-weighted Lehmer means in the concrete loop, and includes the terminal-`CR`
  rule.
- **L-SHADE** = SHADE 1.1 + LPSR; removing the LPSR option recovers SHADE 1.1.
