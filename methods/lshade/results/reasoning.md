Let me start from what actually hurts when I run differential evolution on a new problem. I have a continuous objective I can only poke at — feed it a vector, get back a number — and DE is the tool I reach for, because its one genuinely clever idea is that it never has to invent a step size. The mutation builds a donor by adding a scaled difference of two population members to a third, `v = x_r1 + F·(x_r2 - x_r3)`, and that difference vector carries the population's own scatter inside it. Early in the run the population is splattered all over the box, so the differences are huge and I explore; as everyone crowds toward a basin, the differences shrink on their own and I refine. Nobody scheduled that; it falls out of reusing the population as its own ruler. I cross the donor with the parent coordinate-wise with probability `CR` (forcing at least one donor coordinate via a random index `j_rand`, so the trial is never just a copy of the parent), and I keep whichever of parent and trial is better. Clean, greedy, simple.

So what's the pain? Every time I move to a new function I have to re-pick `F`, `CR`, and the population size `N`, and the right values genuinely change with the landscape. On a smooth bowl I want a smallish `N`, a moderate `F`, a high `CR` — converge fast and stop. On a Rastrigin egg-carton that same setting collapses straight into the nearest local pit; there I want a bigger `N` and a more diversifying `F` to keep from committing too early. And `N` itself is a two-edged knife: shrink it and I converge quickly but get trapped, grow it and I explore widely but spend my whole evaluation budget inching forward one slow generation at a time. The three knobs are coupled and all problem-dependent. I am hand-tuning the same algorithm over and over, and that is the thing I want to kill: the parameters should learn themselves from what is working on *this* problem, right now, as the run proceeds.

How would I even learn `F` and `CR` online? The signal is sitting right there in the selection step. Every generation, some trials beat their parents and some don't. The trials that *won* were produced by particular `(F_i, CR_i)` values; the trials that *lost* by others. So the winners' parameter values are evidence — "these worked on this landscape this generation" — and I should bias next generation's sampling toward them. That's the direction I want to push: don't fix `F` and `CR`, sample them per individual from distributions centered on values that have recently been succeeding, and slide those centers toward the latest winners.

Let me try the most direct version. Keep two running centers `mu_F` and `mu_CR`, both starting at `0.5` since I have no prior. Each individual draws `CR_i ~ randn(mu_CR, 0.1)` and `F_i` from something centered at `mu_F`. After the generation, collect the winners' values into sets `S_F`, `S_CR`, and nudge the centers toward their means with a small learning rate `c`, say `mu <- (1-c)·mu + c·mean(S)`. That's the skeleton. But each piece of it hides a decision that, if I get it wrong, quietly breaks the whole thing, so let me work through them one at a time and pressure-test each before I commit.

First, the mutation strategy. The plain `DE/rand/1` picks a random base vector, which is fine but slow — it doesn't lean on the good solutions I've already found. The obvious greedy fix is `current-to-best/1`, `v = x_i + F·(x_best - x_i) + F·(x_r1 - x_r2)`: pull each individual partway toward the single best, plus a random difference. That converges fast on a unimodal bowl, but stare at it on a multimodal function — *everyone* is being dragged toward the *one* incumbent best, so the whole population funnels into whatever basin that best happens to sit in, and if it's the wrong basin I've lost. Premature convergence, baked right into "best." The fix is to not commit to *the* best but to a *random one of the top few*. Replace `x_best` with `x_pbest`, drawn uniformly from the top `N·p` individuals for some fraction `p`:

  v_i = x_i + F_i·(x_pbest - x_i) + F_i·(x_r1 - x_r2),  x_pbest ~ uniform over the best ⌈N·p⌉.

Now `p` is a greediness dial: `p → 0` recovers the aggressive current-to-best behavior, while a moderate `p` spreads the attraction over several good basins so the population doesn't all rush the same hole. Let me sanity-check that the `p → 0` limit really does collapse to current-to-best. With `N = 540` and `p = 0.11`, the pool size is `⌈0.11·540⌉ = 60`, so `x_pbest` is one of 60 good guides; push `p` to `1/540` and the pool is `⌈1⌉ = 1`, which is `x_best` itself — current-to-best exactly. So this operator contains current-to-best as a special case and only relaxes it. That's the mutation I'll build on.

Second, the difference `x_r1 - x_r2`. If `r1` and `r2` both come from the current population, then as the population converges this difference shrinks toward zero along with everything else, and I lose diversification exactly when I might be stuck. I want a way to inject directions that aren't in the current cloud anymore. Here's the move: the parents that just *lost* selection — I'm about to throw them away — actually encode where the search recently *was* and chose to leave. Keep them in an external archive `A`, and draw `x_r2` not just from the population `P` but from `P ∪ A`. Now the difference `x_r1 - x_r2` can reach back to a recently-abandoned region, adding a progress-direction flavor and diversity without me having to enlarge the live population (which would cost evaluations). I cap the archive at roughly the population size and, when it overflows, drop random members — I don't want it to become a museum that biases the search, just a reservoir of recent history. So: `r1` from `P` only, `r2` from `P ∪ A`, both distinct from each other and from `i`.

Third — and this is where the sampling distributions earn their keep — what do I sample `F` and `CR` *from*, and how do I summarize the winners? Take `CR` first. It's a probability in `[0,1]`, I want it to settle near a stable good value, and Normal is the natural choice: `CR_i = randn(mu_CR, 0.1)`, clamp anything outside `[0,1]` back to the nearest of `0` or `1`. A tight Normal keeps `CR` near its learned center without wild excursions, which is what I want for a parameter whose job is steady.

`F` is different. `F` controls mutation magnitude, and the failure mode I'm most afraid of — premature convergence — is precisely `F` collapsing to small values too early and the search going quiet. So for `F` I want a distribution that keeps *proposing* large values even when the center has drifted down, a heavy tail. That points at Cauchy: `F_i = randc(mu_F, 0.1)`. Same center as a Normal but fat tails. Before I lean on "fat tails," let me actually measure whether the difference matters at this spread. I draw 200,000 samples of each at `mu = 0.5, spread = 0.1` and count how often each proposes a genuinely large `F`, say `F ≥ 0.9` — four spreads above center, the kind of aggressive mutation that re-energizes a stalling search:

  Cauchy:  P(F ≥ 0.9) ≈ 0.077,   P(F ≤ 0) ≈ 0.063
  Normal:  P(F ≥ 0.9) ≈ 0.000,   P(F ≤ 0) ≈ 0.000

So the Normal essentially *never* proposes a large `F` at this spread — four sigma out is one-in-thirty-thousand — while Cauchy does it about 8% of the time. That is exactly the property I wanted: even after the center drifts down, Cauchy keeps lobbing in occasional big mutations. The same measurement tells me the truncation rules I'll need: Cauchy lands `F ≤ 0` about 6% of the time, which I cannot allow — a non-positive scaling factor inverts or kills the mutation — so I *re-sample* those until positive rather than clamping to some arbitrary floor. And `F > 1` I truncate to `1` (huge `F` is unstable, and `1` is plenty). Clamping `CR` is fine because both its endpoints are meaningful; resampling `F` is right because only the lower end is degenerate.

And how do I summarize the successful `F` values into the new center? My first instinct is the arithmetic mean. But think about what arithmetic averaging does to `F` over many generations: it pulls toward the bulk of the successes, which on a converging population tend to be smallish, so `mu_F` drifts down and down, mutation gets weaker and weaker, and I've manufactured the exact premature convergence I was trying to avoid. I need a summary that *resists* that downward pull — that gives extra weight to the larger successful `F` values, because those are the ones keeping the search alive. The Lehmer mean is a candidate:

  mean_L(S) = (Σ_k S_k²) / (Σ_k S_k).

Each term contributes its *square* in the numerator but only its first power in the denominator, so larger elements should pull the ratio up. Let me not just assert that — let me put a concrete success set through both means. Say three winners carried `S = (0.3, 0.5, 0.9)`. The arithmetic mean is `(0.3+0.5+0.9)/3 = 0.567`. The Lehmer mean is `(0.09+0.25+0.81)/(0.3+0.5+0.9) = 1.15/1.7 = 0.676`. So Lehmer sits `0.11` higher — pulled up toward the `0.9` winner, precisely the direction that keeps `mu_F` from sagging. Good, the effect is real and the right sign.

Is it *always* at least the arithmetic mean, or did I get lucky with this set? Let me prove it for the weighted version I'll actually use (normalized weights `w_k`, positive values). The gap is

  (Σ_k w_k S_k²)/(Σ_k w_k S_k) - Σ_k w_k S_k
  = (Σ_k w_k S_k² - (Σ_k w_k S_k)²)/(Σ_k w_k S_k).

The numerator `Σ w S² - (Σ w S)²` is exactly the weighted variance of `S`, which is nonnegative, zero only when all the `S_k` are equal. I'll confirm I haven't bungled the algebra numerically: with `S = (0.3, 0.5, 0.9)` and improvement weights proportional to `(10, 1, 0.5)` — a big first win, two small ones — I get weighted arithmetic `0.343`, weighted Lehmer `0.394`, and the numerator `Σ wS² − (Σ wS)² = 0.01724`, which matches the weighted variance `Σ w(S − 0.343)² = 0.01724` to the digit. So the identity holds and the Lehmer mean is genuinely `≥` the arithmetic mean, with equality only when the winners agree. That asymmetry — Normal/arithmetic-flavored for `CR`, Cauchy/Lehmer-flavored for `F` — isn't decoration; it's "let `CR` settle, but never let `F` go gentle into that good night."

There's one more refinement I can fold in here, because not every success is equally informative. A trial that improved the fitness by a lot is stronger evidence for its parameters than one that barely squeaked past its parent. So weight each winner by how much it improved: let `Δf_k = |f(u_k) - f(x_k)|` be the improvement, normalize `w_k = Δf_k / Σ_l Δf_l`, and use the *weighted* Lehmer mean

  mean_WL(S) = (Σ_k w_k S_k²) / (Σ_k w_k S_k).

Now a big-improvement winner pulls the center harder than a marginal one. For `CR`, the older single-center recipe can use an arithmetic-style update. If I use the weighted Lehmer form for `CR` too, I get the same improvement weighting and the same upward-bias property when the successful values are positive; when all successful `CR` values are zero, the denominator is zero and I need a separate terminal rule instead of pretending there is a mean.

So I have a coherent self-tuning DE: current-to-pbest/1, the archive, stochastic `F` and `CR`, and winners as parameter evidence, with a single pair `(mu_F, mu_CR)` updated toward the winners each generation. But the more I stare at that single pair, the worse one thing looks. There is *one* pair `(mu_F, mu_CR)` steering every individual's sampling. Selection is stochastic; some generation, by sheer luck, the trials that happen to win will carry mediocre `F`/`CR` values — a couple of bad draws that succeeded for unrelated reasons. The update then slides my one and only center toward those bad values, and *next* generation the *entire* population samples from the contaminated center. One unlucky generation poisons the whole adaptation, because the adaptation has exactly one piece of state and no redundancy. On a hard multimodal problem where success is noisy and rare, this is not a corner case — it's the normal weather. That's a real failure mode and I need to deal with it before going further.

How do I make the adaptation robust to a single bad generation? The disease is "all my eggs in one center." The cure has to be redundancy: don't keep one `(mu_F, mu_CR)`, keep a *set* of them — a historical memory of `H` centers, `M_F = [M_{F,1}, ..., M_{F,H}]` and `M_{CR}` likewise, all initialized to `0.5`. Now when an individual needs parameters, it picks an index `r` uniformly from `[1,H]` and samples around *that* slot: `CR_i = randn(M_{CR,r}, 0.1)`, `F_i = randc(M_{F,r}, 0.1)`. And here's the key change to the update: each generation I write the winners' summary into *one* memory slot, cycling through the slots with a counter `k` that wraps around `1..H`. So a generation's `S_F`, `S_CR` overwrite only slot `k`, not everything.

Does this actually fix the fragility, and by how much? Suppose one bad generation produces a contaminated summary. It lands in exactly one of the `H` slots. Next generation, each individual still picks a slot uniformly, so the fraction of the population exposed to the poisoned slot should be about `1/H`. Let me put a number on it with `H = 5` and `N = 540`: drawing 540 uniform slot indices and counting, the five slots get population shares `0.202, 0.202, 0.207, 0.200, 0.189` — right on `1/H = 0.2`, and no slot exceeds `~0.21`. So a single poisoned slot can misdirect at most about a fifth of the population for one generation, while the other four slots — holding summaries from earlier, healthy generations — keep the rest sampling sanely; and the bad slot gets overwritten within `H` generations as the counter cycles. With the single center, that same bad generation drove `100%` of the next population. So the memory cuts the worst-case contamination from all of it to roughly `1/H`, and it comes from turning one number into `H` numbers and updating them round-robin rather than all-at-once. The round-robin matters too: if I overwrote *every* slot each generation I'd be back to a single effective center with extra steps; cycling one slot per generation means the `H` slots stay diverse, each a snapshot of what was working at a different time.

I should pin down two edge cases in the memory update, because they're load-bearing. First: if a generation produces *no* winners at all (`S_F` and `S_CR` empty), I update nothing — the memory holds. No success, no evidence, no change. Second, a subtle one for `CR`. On some multimodal problems the search does best with `CR` driven all the way to `0`, which corresponds to a "change one coordinate at a time" policy (recall `j_rand` forces exactly one donor coordinate, so `CR = 0` means *only* that one coordinate mutates per individual). If a slot's successful `CR` values are all `0`, then `max(S_CR) = 0`, and I want to recognize this as "this slot has discovered that `CR = 0` is what works here" and *lock* it: assign the slot a terminal marker (I'll use the value `-1` in code, sentinel for "this slot means `CR = 0`"), and whenever an individual draws from a terminal slot, set its `CR_i = 0` directly. Once a slot goes terminal it stays terminal — it has converged on the one-coordinate-at-a-time regime, which is slow but thorough and exactly right on certain rugged functions. Without this, the Lehmer mean of a bag of zeros is `0/0` and I'd be guessing.

One last knob from the mutation: the greediness `p`. I do not want a `pbest` set of size one, because that collapses back to current-to-best and the whole point was to avoid pulling everyone toward a single incumbent. So I keep `p` small but fixed, and when I build the `pbest` pool I enforce at least two candidates: `pNP = max(round(p·N), 2)`. Then each individual draws `x_pbest` uniformly from those `pNP` best members. The randomness is in the selected good guide, not in a new per-individual `p` value, and the fixed `p` remains the greediness control.

Now I have a genuinely self-tuning DE: current-to-pbest/1 mutation, the archive, Cauchy sampling for `F`, Normal sampling for `CR`, and winner-based adaptation, but with the single center replaced by an `H`-slot historical memory updated round-robin and with the terminal `CR = 0` edge case handled explicitly. It removes the hand-tuning of `F` and `CR`. But it has done nothing about the third knob, `N`, which is still fixed for the whole run — and `N` is the one I argued earliest cuts both ways. Let me think about `N` properly now, because there's free performance hiding in it.

Here's the tension, sharply. Early in the run I genuinely want a *large* population: broad coverage of the box so I don't miss a basin, and — this matters specifically for *my* method — a large, diverse pool of successes per generation so the memory adaptation has good statistics to learn `F` and `CR` from. But late in the run, once the population has localized to the promising region, a large `N` is pure waste: I'm spending, say, a few hundred evaluations per generation to make one tiny refinement step, when what I actually want near the end is *many generations* of small, precise moves to polish the best basin. With a fixed `N`, the budget `MAX_NFE` buys me `MAX_NFE / N` generations, full stop — I cannot have both the broad early population and the many late generations. The budget forces a single compromise `N` that is too small to explore well or too large to refine well.

Unless `N` isn't fixed. What if I *start* large and *shrink* over the run? Early generations cost a lot of evaluations each but buy exploration and good adaptation statistics; as I shrink, each generation costs fewer evaluations, so the *same remaining budget* buys *more* generations — exactly the many-small-steps regime I want at the end. Shrinking literally manufactures late generations out of the budget that a fixed large `N` would have burned on redundant exploration. And it respects DE's self-scaling: a smaller, converged population still has small differences, so the refinement stays fine-grained.

So I want to reduce `N` over the run, and now the only question is the schedule. I considered reactive resizing — grow/shrink `N` based on how the search is going — but the literature is clear and my own taste agrees: reactive population sizing replaces the one parameter `N` with several meta-parameters governing *when* and *how much* to resize, and those are themselves miserable to tune. I just spent this whole derivation eliminating hand-tuned parameters; I will not reintroduce a fistful of them to fix the last one. A *deterministic* monotone schedule is the disciplined choice: it changes `N` on a fixed rule that doesn't react to the search state, so it adds essentially no new tuning surface. Among deterministic schedules, halving at intervals (DPSR) needs a frequency parameter tuned to the dimensionality, and the general SVPS framework has a curve-shape parameter pair. The simplest possible monotone decrease is *linear*, and linear needs only one genuinely new number: the initial size `N_init` (the final size is forced, as I'll show). So: reduce `N` linearly from `N_init` down to `N_min` as a function of the evaluations spent. After the generation that has consumed `NFE` evaluations out of `MAX_NFE`, the target size is

  N_{G+1} = round[ ((N_min - N_init) / MAX_NFE) · NFE + N_init ].

At `NFE = 0` this is `N_init`; at `NFE = MAX_NFE` it's `N_min`; in between it interpolates linearly. (This is the special case of SVPS with its shape parameters set to the linear setting, which is why it needs only `N_init`.) Let me actually run the formula for a concrete budget to see the shape of the decay — `D = 30`, so `N_init = 18·30 = 540`, `N_min = 4`, `MAX_NFE = 30·10000 = 300000`:

  NFE =      0 (  0%): N = 540
  NFE =  75000 ( 25%): N = 406
  NFE = 150000 ( 50%): N = 272
  NFE = 225000 ( 75%): N = 138
  NFE = 300000 (100%): N = 4

Endpoints land where I wanted — `540` at the start, `4` at exhaustion — and the middle marches down in equal `~134`-individual steps, a clean straight line. The payoff is in counting generations: a fixed `N = 540` would buy `300000/540 ≈ 555` generations, all at full cost. Under the schedule the early generations cost `540` evaluations each, but the late ones cost a few dozen and then a handful, so the same `300000`-evaluation budget buys *far* more total generations, and the extra ones all land late — exactly the many-small-refinement-steps regime I want at the end. The shrink literally converts early-exploration budget into late-refinement generations. What is `N_min`? It's pinned by the mutation, not chosen: `current-to-pbest/1` needs the current `x_i`, a distinct `x_pbest`, a distinct `x_r1`, and a distinct `x_r2` — four distinct individuals — so the smallest population the operator can run on is `4`. `N_min = 4`.

And *which* individuals do I delete when `N` must drop? When `N_{G+1} < N_G`, I remove the `N_G - N_{G+1}` *worst* individuals by fitness. This is the obvious right call: shrinking the population should concentrate it on its best members, so deletion is elitist — I keep the good ones and discard the laggards, which both saves the elites and accelerates the late-stage focusing. I also resize the archive proportionally to the new population (capping `|A|` at a multiple of `N`), and when the archive overflows I drop random members — keeping the loser-reservoir scaled to the shrinking search rather than letting a stale, oversized archive dominate the difference vectors.

That linear population-size reduction is the last piece. The remaining knobs have to stay few and concrete. The memory size `H`: large enough to hold a diverse history so one bad generation can't dominate, but not so large that stale entries from long ago keep steering the search — a handful, `H = 5`, is enough for the concrete loop. The greediness parameter `p = 0.11` keeps the `pbest` pool small and exploitative while the "at least two" rule prevents collapse to the single best. The Cauchy/Normal spread `0.1` is tight enough that samples stay near the learned centers, loose enough (especially with Cauchy's tail) to keep exploring parameter space. The initial size `N_init`: following standard DE practice of scaling the population with problem difficulty, set it proportional to dimensionality, on the order of `N_init = 18·D` — big enough early for exploration and good adaptation statistics, knowing the linear schedule will whittle it down to `4` by the end. The archive cap multiple `r_arc = 1.4` keeps the loser reservoir comparable to the live population, enough recent history to diversify without swamping it.

Let me write the algorithm out as a loop so I can see the whole machine. Initialize a population of `N_init` random vectors and evaluate them; set all `H` memory slots `M_F = M_CR = 0.5`; empty archive; memory counter `k = 1`. Then each generation: sort the population by fitness and define the `pbest` pool as the top `max(round(p·N), 2)` members; for each individual `i`, pick a memory index `r`, draw `CR_i` from `randn(M_{CR,r}, 0.1)` (or `0` if that slot is terminal) clamped to `[0,1]`, draw `F_i` from `randc(M_{F,r}, 0.1)` (resample if `≤ 0`, truncate to `1`), choose `x_pbest` uniformly from that pool, pick distinct `r1` from `P` and `r2` from `P ∪ A`, form the donor `v_i = x_i + F_i·(x_pbest - x_i) + F_i·(x_{r1} - x_{r2})`, repair any out-of-bounds coordinate (set it to the midpoint between the violated bound and the parent's value, a standard DE bound-repair that keeps the trial inside the box without snapping it hard to the wall), binomial-crossover with the parent to get the trial `u_i`, and evaluate it. A strict improvement is the only evidence strong enough for the archive and the memory, so only if `f(u_i) < f(x_i)` do I store the losing parent, record `(F_i, CR_i, |f(x_i)-f(u_i)|)` in the success set, and replace the parent. After the generation, if there were any successes, write the improvement-weighted Lehmer means of `S_F` and `S_CR` into slot `k` (applying the terminal-`CR` rule), advance `k` round-robin; then compute the linear target size and, if it's smaller than the current `N`, delete the worst individuals and resize the archive.

Now let me turn that into real code, filling the one empty slot in the harness — the reproduction-and-population-management policy. I write it as a generation-level loop: two memory vectors (`memory_sf`, `memory_cr`), an archive with duplicate removal and random overflow deletion, the fixed `pbest` pool, the linear schedule, and the round-robin memory write.

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
    """L-SHADE: success-history adaptive DE (current-to-pbest/1 + external
    archive, Cauchy F / Normal CR sampled from an H-slot memory) with linear
    population size reduction. Fills the reproduction-and-population policy slot."""
    rng = np.random.default_rng(seed)

    # --- concrete constants for this loop ---
    H = 5                       # memory slots: diverse history, one bad gen can't dominate
    p_best_rate = 0.11          # fixed current-to-pbest greediness
    arc_rate = 1.4              # archive cap = arc_rate * current N (loser reservoir)
    max_pop_size = pop_size     # N_init (large early: exploration + adaptation stats)
    min_pop_size = 4            # N_min: current-to-pbest/1 needs 4 distinct individuals

    # population of real vectors + fitnesses
    pop = lo + rng.random((pop_size, dim)) * (hi - lo)
    fitness = np.array([evaluate(ind) for ind in pop])
    nfes = pop_size

    # H-slot historical memory of successful F and CR centers, all init to 0.5
    memory_sf = np.full(H, 0.5)
    memory_cr = np.full(H, 0.5)
    memory_pos = 0              # round-robin write index k

    archive = np.zeros((0, dim))   # external archive of losing parents
    archive_cap = int(round(arc_rate * pop_size))

    fitness_history = []

    while nfes < max_nfes:
        N = pop_size
        sorted_idx = np.argsort(fitness)          # ascending: best first (for pbest + deletion)

        # --- sample CR from Normal(M_CR[r], 0.1), F from Cauchy(M_F[r], 0.1) ---
        ridx = rng.integers(0, H, N)              # each individual picks a memory slot r
        mu_cr = memory_cr[ridx]
        mu_sf = memory_sf[ridx]

        cr = rng.normal(mu_cr, 0.1)
        cr[mu_cr == -1] = 0.0                     # terminal slot => CR locked to 0
        cr = np.clip(cr, 0.0, 1.0)                # CR bounded => clamp is fine

        sf = mu_sf + 0.1 * np.tan(np.pi * (rng.random(N) - 0.5))   # Cauchy(mu, 0.1)
        bad = sf <= 0
        while np.any(bad):                        # F<=0 is degenerate => resample, don't clamp
            sf[bad] = mu_sf[bad] + 0.1 * np.tan(np.pi * (rng.random(bad.sum()) - 0.5))
            bad = sf <= 0
        sf = np.minimum(sf, 1.0)                  # F>1 truncated to 1

        # --- current-to-pbest/1 donor with archive in the second difference ---
        pop_all = np.vstack([pop, archive])       # r2 drawn from P u A
        donors = np.empty((N, dim))
        pnp = max(2, int(round(p_best_rate * N)))
        for i in range(N):
            pbest = pop[sorted_idx[rng.integers(0, pnp)]]   # random of top p% (not THE best)

            r1 = rng.integers(0, N)                # r1 from P, r1 != i
            while r1 == i:
                r1 = rng.integers(0, N)
            r2 = rng.integers(0, len(pop_all))     # r2 from P u A, r2 != i, r2 != r1
            while r2 == i or r2 == r1:
                r2 = rng.integers(0, len(pop_all))

            # v = x_i + F(x_pbest - x_i) + F(x_r1 - x_r2)
            donors[i] = pop[i] + sf[i] * (pbest - pop[i] + pop[r1] - pop_all[r2])

        # bound repair: violated coord -> midpoint of bound and parent (standard DE)
        below, above = donors < lo, donors > hi
        donors = np.where(below, (lo + pop) / 2.0, donors)
        donors = np.where(above, (hi + pop) / 2.0, donors)

        # --- binomial crossover, with a guaranteed donor coord j_rand ---
        cross = rng.random((N, dim)) < cr[:, None]
        jrand = rng.integers(0, dim, N)
        cross[np.arange(N), jrand] = True
        trials = np.where(cross, donors, pop)

        trial_fit = np.array([evaluate(t) for t in trials])
        nfes += N

        # --- greedy selection; collect successes for the memory ---
        improved = trial_fit < fitness            # strict improvement => a success
        dif = np.abs(fitness - trial_fit)

        good_cr = cr[improved]
        good_sf = sf[improved]
        good_df = dif[improved]
        if improved.any():                        # losing parents -> archive
            archive = update_archive(archive, pop[improved], archive_cap, rng)

        pop[improved] = trials[improved]
        fitness[improved] = trial_fit[improved]

        # --- write one memory slot, round-robin, weighted-Lehmer of the winners ---
        if good_df.size > 0:
            w = good_df / good_df.sum()
            memory_sf[memory_pos] = np.sum(w * good_sf ** 2) / np.sum(w * good_sf)   # Lehmer: keeps F large
            if good_cr.max() == 0 or memory_cr[memory_pos] == -1:
                memory_cr[memory_pos] = -1                                           # lock CR=0 (terminal)
            else:
                memory_cr[memory_pos] = np.sum(w * good_cr ** 2) / np.sum(w * good_cr)
            memory_pos = (memory_pos + 1) % H

        # --- linear population size reduction: N_init -> N_min over the budget ---
        plan = round(((min_pop_size - max_pop_size) / max_nfes) * nfes + max_pop_size)
        plan = max(min_pop_size, plan)
        if plan < pop_size:
            order = np.argsort(fitness)            # delete the worst (elitist shrink)
            keep = order[:plan]
            pop, fitness = pop[keep], fitness[keep]
            pop_size = plan
            archive_cap = int(round(arc_rate * pop_size))
            if len(archive) > archive_cap:         # rescale loser reservoir to the smaller search
                ridx = rng.permutation(len(archive))[:archive_cap]
                archive = archive[ridx]

        fitness_history.append(float(fitness.min()))

    best = pop[int(np.argmin(fitness))]
    return best, fitness_history
```
