I have a continuous objective $f: \mathbb{R}^D \to \mathbb{R}$ that I can only probe by evaluation — no gradient, no structure, just "give me a point, I'll tell you its cost." The landscapes span the whole zoo: smooth unimodal bowls, ill-conditioned valleys, highly multimodal egg-cartons, hybrid and composite functions, with dimensionality from 10 to 100 and a fixed budget of function evaluations, so every wasted evaluation is gone for good. The strongest off-the-shelf engine for this regime is the success-history adaptive DE family — SHADE and its linear-population-reduction extension L-SHADE. L-SHADE already self-tunes $F$ and $CR$ from a memory of what has been working, mutates with current-to-pbest/1 plus an external archive of losing parents, and shrinks the population linearly over the budget. The adaptation is genuinely good and robust to a single bad generation. So the question is not how to make DE self-adaptive — that is solved — but where this family still spends evaluations awkwardly. Three pressure points stand out. First, the memory starts neutral at $0.5$ for both $F$ and $CR$, so the early generations are spent rediscovering, on every new problem, that a high $CR$ (most donor coordinates taken, large coordinated moves) helps while the population is still spread across the box. Second, a slot is overwritten with the weighted Lehmer summary of one generation's winners, throwing away accumulated history and letting a single noisy generation swing the adaptation. Third, and most importantly, current-to-pbest/1 applies one scale factor $F_i$ to two structurally different vector differences, forcing exploration and exploitation strength to move in lockstep even though the right balance between them changes over the run.

I propose jSO: L-SHADE's reproduction and population schedule, kept intact, plus five changes whose heart is a weighted mutation factor that decouples the elite pull from the random perturbation. The starting point is the donor formula of current-to-pbest/1, $v_i = x_i + F_i(x_{\text{pbest}} - x_i) + F_i(x_{r1} - x_{r2})$. Look hard at the two difference terms. The first, $x_{\text{pbest}} - x_i$, is a directed pull toward the elite — pure exploitation, dragging the individual toward one of the best solutions found. The second, $x_{r1} - x_{r2}$, is the self-scaling random difference — pure exploration, the perturbation that carries the population's own scatter. Using a single $F_i$ for both means the strength of the elite-pull and the strength of the random perturbation cannot be set independently, yet the right ratio between them is broad and diversifying early (do not commit to the current elite while still exploring) and focused and elite-following late (follow the good solutions once the basin is found). So I give the elite-pull term its own factor: replace $F_i(x_{\text{pbest}} - x_i)$ with $F_w(x_{\text{pbest}} - x_i)$, where $F_w$ is $F_i$ scaled by a phase-dependent multiplier, while leaving $F_i$ on the random-difference term. The donor becomes
$$v_i = x_i + F_w\,(x_{\text{pbest}} - x_i) + F_i\,(x_{r1} - x_{r2}),$$
with a three-step schedule on the budget fraction $\text{nfes}/\text{max\_nfes}$: $F_w = 0.7\,F_i$ for the first $0.2$ of the budget, $F_w = 0.8\,F_i$ for the first $0.4$, and $F_w = 1.2\,F_i$ thereafter. Early the individual is perturbed more than it is pulled toward the elite, preserving diversity; late it is pulled more than it is perturbed, sharpening convergence. This is the single most consequential change over the unweighted operator — it turns the exploration/exploitation balance from one coupled knob into one that follows the run's phase by construction.

The same phase argument reshapes the greediness $p$ of the pbest pool, the top $N\,p$ individuals from which $x_{\text{pbest}}$ is drawn. A larger $p$ early makes the elite pool broad, so the pull is toward a diverse set of good solutions rather than a single incumbent — the diversity I want early — and a smaller $p$ late narrows the pool toward the very best, sharpening convergence. So $p$ should decrease over the run, linearly from a larger value to a smaller one,
$$p = p_{\max} - (p_{\max} - p_{\min})\,\frac{\text{nfes}}{\text{max\_nfes}},$$
with $p_{\max} = 0.25$ and $p_{\min} = p_{\max}/2 = 0.125$: early draw the elite guide from the top quarter, late from the top eighth. The three remaining changes all concern parameter propagation. I initialize the memory with the bias I already believe in rather than at neutral: all $CR$ slots start high at $0.8$ and $F$ slots conservatively at $0.3$, a free head start that the adaptation will still drive down if a near-separable landscape turns out to prefer low $CR$. I reserve one sampled period — the last memory index — to use fixed centers $M_F = M_{CR} = 0.9$ no matter what the adaptive arrays hold; because each individual picks its memory index uniformly, roughly $1/H$ of the population always samples around an aggressive $(0.9, 0.9)$, a permanent reservoir of aggression the adaptation can never extinguish, cheap insurance against the whole population going quiet on a hard multimodal problem. I add phase rails as hard guardrails (not adapted quantities): because the heavy-tailed Cauchy on $F$ will occasionally throw a near-1 $F$ that produces a destabilizing donor on a spread population, and the Normal on $CR$ will occasionally propose a premature near-0 $CR$, I floor $CR \ge 0.7$ for the first quarter of the budget and $CR \ge 0.6$ for the first half, and cap $F \le 0.7$ for the first $0.6$ of the budget, all relaxing as the run proceeds so the adaptation takes over for refinement. Finally I blend the memory update instead of overwriting: rather than writing a slot with this generation's weighted Lehmer summary outright, I write
$$M[k] \leftarrow \tfrac{1}{2}\big(\text{mean}_{WL}(S) + M[k]_{\text{old}}\big),$$
so a single generation can move a slot at most halfway toward its summary, damping noise while retaining what worked before. The weighted Lehmer mean itself is $\text{mean}_{WL}(S) = \big(\sum_k w_k S_k^2\big)/\big(\sum_k w_k S_k\big)$, with weights $w_k$ proportional to each winner's fitness improvement $|\Delta f|$; it sits at or above the weighted arithmetic mean and is pulled up by larger successful values.

Two edge cases pin the adaptation down so it never divides by zero. If a generation produces no winners, no slot is updated — no evidence, no change. And if a slot's successful $CR$ values are all zero, the weighted Lehmer mean is $0/0$; I read this as "this slot means $CR = 0$," the one-coordinate-at-a-time policy that is slow but thorough on rugged functions, mark the slot terminal by storing a negative center, and whenever an individual draws from a terminal slot it sets $CR_i = 0$ (with the early floors still applied on top, so the literal zero only takes effect once the rails relax). The remaining constants follow standard adaptive-DE practice: memory size $H = 5$ (enough that one bad generation cannot dominate, small enough that stale entries do not steer the search, and with the last period frozen at $0.9$ four ordinary periods carry learned centers); initial population $N_{\text{init}} = \text{round}(25\log(D)\sqrt{D})$, large enough early for exploration and good adaptation statistics, whittled by linear reduction to $N_{\min} = 4$, the minimum current-to-pbest/1 can run on; archive capped at $|A| = N$ and rescaled as $N$ shrinks with random deletion on overflow; Cauchy and Normal spread $0.1$, tight enough that samples stay near the learned centers yet loose enough (with Cauchy's tail) to keep exploring parameter space; and bound violations repaired by moving the offending coordinate to the midpoint of the violated bound and the parent's coordinate. Assembled, that is the entire method — L-SHADE's reproduction and schedule, with the high and frozen memory initialization, the phase rails, the blended update, the decreasing pbest fraction, and at its heart the weighted mutation factor that decouples the elite pull from the random perturbation and slides their balance from exploration-heavy early to exploitation-heavy late.

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
