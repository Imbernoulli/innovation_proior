I have a continuous objective $f:\mathbb{R}^D \to \mathbb{R}$ that I can only probe by evaluation — feed it a vector, get back a number, no gradients and no structure — and I want one optimizer that holds up across the whole zoo of landscapes: smooth bowls, ill-conditioned Rosenbrock valleys, multimodal Rastrigin egg-cartons, hybrid and composite functions, in dimensions from a handful to a hundred, all under a fixed evaluation budget where every wasted evaluation is gone for good. Differential evolution is the natural engine here, because its one genuinely clever idea is that it never invents a step size: the mutation builds a donor by adding a scaled population difference to a base vector, $v = x_{r1} + F\,(x_{r2} - x_{r3})$, and that difference carries the population's own scatter inside it — large and exploratory when the population is splattered across the box, shrinking on its own toward refinement as everyone crowds into a basin. The trouble is that DE's behavior is governed by three control parameters — population size $N$, scaling factor $F$, crossover rate $CR$ — and the single most-cited fact about DE is that their optimal settings are problem-dependent and coupled. A high $F$ and large $N$ resist premature convergence on rugged landscapes but crawl; a low $F$, small $N$, and the right $CR$ converge fast on smooth or separable ones but collapse into the nearest local pit on rough ones. So I end up re-tuning the same algorithm by hand for every new problem, and that hand-tuning is exactly the cost I want to remove.

The prior adaptive variants each take a bite out of this and none finishes the meal. jDE attaches an $F_i, CR_i$ to each individual and lets good values survive selection, but the "what worked" signal is diffuse and entangled with selection, with no explicit record of which parameter values caused success. SaDE tracks a single central tendency per strategy and re-samples around it, representing the good region of parameter space by one point and adapting only $CR$. EPSDE and CoDE restrict parameters to a coarse, hand-built discrete menu rather than learning continuous values. JADE is the closest ancestor and contributes the pieces worth keeping — a less-greedy mutation, an external archive, and Cauchy/Normal sampling of $F$/$CR$ — but its adaptation carries a *single* pair $(\mu_F, \mu_{CR})$ steering every individual, so on a hard problem where selection is noisy, one unlucky generation whose winners happen to carry mediocre values slides that one center toward them and degrades the entire population's sampling next generation. And across all of these, $N$ is held fixed for the whole run, so the broad-early/fine-late compromise it forces is never resolved.

I propose L-SHADE — Success-History based Adaptive Differential Evolution with Linear Population Size Reduction — which makes $F$ and $CR$ self-adapting from a robust history memory and makes $N$ a deterministic linear schedule, on top of an archived current-to-pbest/1 DE. The starting observation is that the signal for tuning $F$ and $CR$ is already sitting in the selection step: every generation some trials beat their parents and some don't, and the winners were produced by particular $(F_i, CR_i)$ values, so those values are evidence — "these worked on this landscape this generation" — and next generation's sampling should be biased toward them. So I do not fix $F$ and $CR$; I sample them per individual from distributions centered on recently successful values, and slide the centers toward the latest winners.

The mutation I build on is current-to-pbest/1 rather than the greedy current-to-best/1. Pulling every individual toward the single incumbent best, $v_i = x_i + F_i(x_{best} - x_i) + F_i(x_{r1} - x_{r2})$, funnels the whole population into whatever basin that best sits in, and on a multimodal function that is premature convergence baked right in. The fix is to pull toward a *random one of the top few* instead of the one best:
$$v_i = x_i + F_i\,(x_{pbest} - x_i) + F_i\,(x_{r1} - x_{r2}),$$
where $x_{pbest}$ is drawn uniformly from the top $\lceil N p\rceil$ individuals. Here $p$ is a greediness dial — $p \to 0$ recovers current-to-best, a moderate $p$ spreads the attraction over several good basins. I keep $p = 0.11$ fixed and small but enforce a pool of at least two, $pNP = \max(\mathrm{round}(p\,N), 2)$, so the pool never collapses to a single guide and undoes the whole point. The second difference $x_{r1} - x_{r2}$ shrinks toward zero as the population converges, losing diversification exactly when I might be stuck, so I keep an external archive $A$ of parents that just *lost* selection — they encode where the search recently was and chose to leave — and draw $x_{r2}$ from $P \cup A$ while $r_1$ comes from $P$ only. This reaches the difference vector back into a recently-abandoned region without enlarging the live population (which would cost evaluations). The archive is capped at $r_{arc}\,N$ with $r_{arc} = 1.4$, and on overflow I drop random members so it stays a reservoir of recent history rather than a museum that biases the search.

The sampling distributions are deliberately asymmetric between $CR$ and $F$, and the asymmetry is load-bearing. $CR$ is a probability whose job is to settle near a stable good value, so I draw $CR_i = \mathrm{randn}(\mu_{CR}, 0.1)$ from a tight Normal and clamp out-of-range draws back to the nearest of $0$ or $1$ — clamping is fine because both endpoints are meaningful. $F$ controls mutation magnitude, and the failure mode I most fear is $F$ collapsing to small values too early and the search going quiet, so I want a distribution that keeps proposing large $F$ even when the center has drifted down. That is Cauchy: $F_i = \mathrm{randc}(\mu_F, 0.1)$, same center as a Normal but heavy tails. The truncation rules respect what $F$ is: $F > 1$ is truncated to $1$ (huge $F$ is unstable and $1$ is plenty), but $F \le 0$ is degenerate — a non-positive scaling factor inverts or kills the mutation — so I *re-sample* until positive rather than clamping to an arbitrary floor.

The same downward-drift worry dictates how I summarize the winners into a new center. Arithmetic averaging of the successful $F$ values pulls toward the bulk of the successes, which on a converging population skew small, so $\mu_F$ drifts down and down and I manufacture the very premature convergence I was avoiding. I need a summary that resists that pull by upweighting the larger successful values, which is exactly the Lehmer mean, $\mathrm{mean}_L(S) = (\sum_k S_k^2)/(\sum_k S_k)$: each term contributes its square in the numerator but only its first power in the denominator. Not every success is equally informative either — a trial that improved fitness a lot is stronger evidence than one that barely squeaked past its parent — so I weight each winner $k$ by its improvement $\Delta f_k = |f(x_k) - f(u_k)|$, normalized to $w_k = \Delta f_k / \sum_l \Delta f_l$, and use the weighted Lehmer mean
$$\mathrm{mean}_{WL}(S) = \frac{\sum_k w_k\,S_k^2}{\sum_k w_k\,S_k}.$$
For positive $S$ this sits at or above the weighted arithmetic mean, because
$$\frac{\sum_k w_k S_k^2}{\sum_k w_k S_k} - \sum_k w_k S_k = \frac{\sum_k w_k S_k^2 - (\sum_k w_k S_k)^2}{\sum_k w_k S_k},$$
whose numerator is the weighted variance times the total weight and hence nonnegative, with equality only when all successful values are identical. So Lehmer-averaging keeps $\mu_F$ biased toward the larger successes and stops the mutation magnitude from quietly dying.

The piece that distinguishes this from JADE is what I do with the adaptation *state*, because a single $(\mu_F, \mu_{CR})$ is fragile: one unlucky generation whose winners carry poor values slides the one center toward them, and next generation the entire population samples from the contaminated center. The cure is redundancy. Instead of one center I keep $H = 5$ memory slots, $M_F = [M_{F,1}, \dots, M_{F,H}]$ and $M_{CR}$ likewise, all initialized to $0.5$. When an individual needs parameters it picks an index $r$ uniformly from $[1,H]$ and samples around that slot, and each generation I write the winners' weighted-Lehmer summary into *one* slot $k$, cycling $k$ round-robin through $1..H$. Now a contaminated generation lands in exactly one slot; next generation only about $1/H$ of the population samples from it while the other $H-1$ slots — snapshots of what was working at earlier, healthy points in the run — keep the sampling sane, and the bad slot is overwritten within a few generations. The round-robin is essential: overwriting every slot each generation would collapse back to one effective center, whereas cycling one slot per generation keeps the $H$ slots diverse. Two edge cases in the update are load-bearing. If a generation produces no winners, the memory holds — no evidence, no change. And for $CR$, some multimodal problems do best with $CR$ driven to $0$, i.e. mutating exactly one coordinate at a time (since $j_{rand}$ already forces one donor coordinate); when a slot's successful $CR$ are all zero, $\mathrm{max}(S_{CR}) = 0$ and the Lehmer mean would be $0/0$, so I recognize this as "this slot has found that $CR = 0$ works here," mark it terminal with a sentinel $-1$, and thereafter set $CR_i = 0$ directly for individuals drawing from it; once terminal it stays terminal.

Self-adapting $F$ and $CR$ removes two of the three knobs; the third, $N$, is still a fixed compromise the budget cannot satisfy in both directions. Early I want a large population — broad coverage so I don't miss a basin, and a large pool of successes per generation so the memory has good statistics to learn from — but late, once the population has localized, a large $N$ is pure waste, spending hundreds of evaluations per generation on one tiny refinement step when I want *many* generations of small precise moves. With fixed $N$ the budget buys exactly $\mathrm{MAX\_NFE}/N$ generations and forces a single compromise. So I let $N$ start large and shrink: early generations cost a lot of evaluations each but buy exploration and adaptation statistics, and as $N$ shrinks the same remaining budget buys more generations — shrinking literally manufactures the late refinement generations a fixed large population would have burned on redundant exploration, and it respects DE's self-scaling since a smaller converged population still has small differences. I reject reactive resizing because it would reintroduce a fistful of meta-parameters governing when and how much to resize, after I just spent the whole derivation eliminating hand-tuned ones; a deterministic monotone schedule adds essentially no tuning surface. The simplest monotone decrease is linear, needing only the initial size:
$$N_{G+1} = \mathrm{round}\!\left[\frac{N_{min} - N_{init}}{\mathrm{MAX\_NFE}}\cdot \mathrm{NFE} + N_{init}\right],$$
which is $N_{init}$ at $\mathrm{NFE}=0$, $N_{min}$ at the budget's end, and linear in between. The final size $N_{min} = 4$ is pinned by the operator, not chosen: current-to-pbest/1 needs four distinct individuals ($x_i$, $x_{pbest}$, $x_{r1}$, $x_{r2}$). When $N$ must drop I delete the *worst* individuals by fitness — elitist shrinking concentrates the population on its good members and accelerates late-stage focusing — and rescale the archive cap to $r_{arc}\,N$, dropping random archive members on overflow so the loser reservoir stays scaled to the shrinking search. I set $N_{init} = 18D$, following standard DE practice of scaling the population with problem difficulty, knowing the schedule whittles it to $4$ by the end, and use spread $0.1$ for both the Normal and the Cauchy.

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
