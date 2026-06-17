**Problem.** L-SHADE already self-tunes `F`/`CR` and shrinks the population, but it starts its memory
neutral (wasting early budget rediscovering that high `CR` helps), it overwrites memory slots with one
noisy generation, and — most importantly — it applies a single `F` to *both* the elite-pull term and
the random-difference term of current-to-pbest/1, forcing exploration and exploitation strength to
move in lockstep even though the right balance changes over the run.

**Key idea (jSO).** Keep all of L-SHADE, add: (1) memory biased high (`M_CR=0.8`, `M_F=0.3`) with one
sampled period fixed at `0.9,0.9` as a permanent aggressive reservoir; (2) phase rails — floor `CR` and cap
`F` early, relaxing over the budget; (3) blended memory update `M[k] <- (mean_WL + M[k]_old)/2`;
(4) a *decreasing* pbest fraction `p: 0.25 -> 0.125`; and the heart of it, (5) the **weighted
mutation** `v = x_i + Fw*(x_pbest - x_i) + F*(x_r1 - x_r2)` where `Fw = 0.7F / 0.8F / 1.2F` at budget
fractions `<0.2 / <0.4 / else` — decoupling the elite pull from the random perturbation so the balance
slides from exploration-heavy early to exploitation-heavy late.

**Why it beats L-SHADE.** The weighted factor lets the elite pull be weak while exploring and strong
while refining without coupling it to the perturbation magnitude; the high/frozen memory and phase
rails save early budget and prevent the search going quiet; the blended update damps adaptation noise.

**Hyperparameters.** `H=5` (last sampled period uses 0.9,0.9); `M_CR` init 0.8, `M_F` init 0.3;
`p_max=0.25`, `p_min=0.125`; `Fw` schedule (0.7,0.8,1.2); `CR>=0.7` (nfes<0.25), `>=0.6` (nfes<0.5);
`F<=0.7` (nfes<0.6); `N_init=round(25 log(D) sqrt(D))`, `N_min=4`, archive `|A|=N`; Cauchy/Normal
spread 0.1.

```python
import numpy as np
import math


def run_evolution(evaluate, dim, lo, hi, pop_size, max_nfes, seed):
    """jSO: weighted current-to-pbest-w/1 success-history adaptive DE with linear
    population reduction (Brest, Maucec & Boskovic, CEC 2017). Fills the
    reproduction-and-population policy slot."""
    rng = np.random.default_rng(seed)

    H = 5
    p_max, p_min = 0.25, 0.125
    arc_rate = 1.0
    n_min = 4
    n_init = max(n_min, int(round(25.0 * math.log(dim) * math.sqrt(dim))))

    pop = lo + rng.random((n_init, dim)) * (hi - lo)
    fitness = np.array([evaluate(ind) for ind in pop])
    nfes = n_init
    N = n_init

    # H memory periods; drawing the last period uses fixed centers (0.9, 0.9).
    memory_f = np.full(H, 0.3)
    memory_cr = np.full(H, 0.8)
    mem_pos = 0

    archive = np.zeros((0, dim))
    archive_cap = int(round(arc_rate * N))

    fitness_history = []

    while nfes < max_nfes:
        order = np.argsort(fitness)

        ridx = rng.integers(0, H, N)
        mu_cr = memory_cr[ridx].copy()
        mu_f = memory_f[ridx].copy()
        fixed = ridx == H - 1
        mu_cr[fixed] = 0.9
        mu_f[fixed] = 0.9

        # CR ~ Normal(mu_cr, 0.1); negative memory => terminal CR=0 before phase rails
        cr = rng.normal(mu_cr, 0.1)
        cr = np.where(mu_cr < 0, 0.0, cr)
        cr = np.clip(cr, 0.0, 1.0)
        # phase rails on CR (early): floor high
        if nfes < 0.25 * max_nfes:
            cr = np.maximum(cr, 0.7)
        elif nfes < 0.5 * max_nfes:
            cr = np.maximum(cr, 0.6)

        # F ~ Cauchy(mu_f, 0.1), resample if <=0, truncate to 1
        sf = mu_f + 0.1 * np.tan(np.pi * (rng.random(N) - 0.5))
        bad = sf <= 0
        while np.any(bad):
            sf[bad] = mu_f[bad] + 0.1 * np.tan(np.pi * (rng.random(bad.sum()) - 0.5))
            bad = sf <= 0
        sf = np.minimum(sf, 1.0)
        # phase cap on F (early)
        if nfes < 0.6 * max_nfes:
            sf = np.minimum(sf, 0.7)

        # weighted mutation factor Fw by budget phase
        if nfes < 0.2 * max_nfes:
            fw = 0.7 * sf
        elif nfes < 0.4 * max_nfes:
            fw = 0.8 * sf
        else:
            fw = 1.2 * sf

        # decreasing pbest fraction: 0.25 -> 0.125
        p = p_max - (p_max - p_min) * min(1.0, nfes / max_nfes)
        pnp = max(2, int(round(p * N)))

        pop_all = np.vstack([pop, archive])
        donors = np.empty((N, dim))
        for i in range(N):
            pbest = pop[order[rng.integers(0, pnp)]]
            r1 = rng.integers(0, N)
            while r1 == i:
                r1 = rng.integers(0, N)
            r2 = rng.integers(0, len(pop_all))
            while r2 == i or r2 == r1:
                r2 = rng.integers(0, len(pop_all))
            # v = x_i + Fw(x_pbest - x_i) + F(x_r1 - x_r2)
            donors[i] = pop[i] + fw[i] * (pbest - pop[i]) + sf[i] * (pop[r1] - pop_all[r2])

        # bound repair: midpoint of violated bound and parent
        below, above = donors < lo, donors > hi
        donors = np.where(below, (lo + pop) / 2.0, donors)
        donors = np.where(above, (hi + pop) / 2.0, donors)

        # binomial crossover with guaranteed donor coord j_rand
        cross = rng.random((N, dim)) < cr[:, None]
        jrand = rng.integers(0, dim, N)
        cross[np.arange(N), jrand] = True
        trials = np.where(cross, donors, pop)

        trial_fit = np.array([evaluate(t) for t in trials])
        nfes += N

        selected = trial_fit <= fitness
        improved = trial_fit < fitness
        dif = np.abs(fitness - trial_fit)
        good_cr = cr[improved]
        good_sf = sf[improved]
        good_df = dif[improved]
        if improved.any():
            archive = update_archive(archive, pop[improved], archive_cap, rng)

        pop[selected] = trials[selected]
        fitness[selected] = trial_fit[selected]

        # blended memory update (new + old)/2 with weighted Lehmer means
        if good_df.size > 0:
            w = good_df / good_df.sum()
            new_f = np.sum(w * good_sf ** 2) / np.sum(w * good_sf)
            memory_f[mem_pos] = 0.5 * (new_f + memory_f[mem_pos])
            if memory_cr[mem_pos] < 0 or good_cr.max() == 0:
                new_cr = -1.0
            else:
                new_cr = np.sum(w * good_cr ** 2) / np.sum(w * good_cr)
            memory_cr[mem_pos] = 0.5 * (new_cr + memory_cr[mem_pos])
            mem_pos = (mem_pos + 1) % H

        # linear population size reduction
        plan = round((n_min - n_init) / max_nfes * nfes + n_init)
        plan = max(n_min, plan)
        if plan < N:
            keep = np.argsort(fitness)[:plan]
            pop, fitness = pop[keep], fitness[keep]
            N = plan
            archive_cap = int(round(arc_rate * N))
            if len(archive) > archive_cap:
                archive = archive[rng.permutation(len(archive))[:archive_cap]]

        fitness_history.append(float(fitness.min()))

    best = pop[int(np.argmin(fitness))]
    return best, fitness_history


def update_archive(archive, new_points, cap, rng):
    if new_points.size == 0 or cap == 0:
        return archive[:0] if cap == 0 else archive
    for point in new_points:
        if len(archive) < cap:
            archive = np.vstack([archive, point[None, :]])
        else:
            archive[int(rng.integers(0, cap))] = point
    return archive
```
