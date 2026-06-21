Random search told me exactly what its structure promised, and it told me in the per-seed columns. The means are respectable — 93.38 on CIFAR-10, 70.65 on CIFAR-100, 44.57 on ImageNet16-120 — the volume argument coming true: thirty independent shots at a space with a large good region usually catch something decent. But the spread is the story. CIFAR-10 is tight (±0.33), as the compressed top of its distribution predicted; CIFAR-100 wider (±0.65), and ImageNet16-120 falls in a clean monotone 45.47, 44.87, 44.60, 44.10, 43.80 — that is luck-of-the-draw, not signal. The seed-3 CIFAR-100 of 69.92 and the seed-4 ImageNet of 43.80 are the tell: runs where thirty memoryless gambles never drew into the top region, and random search has no way to *recover* because it never looks at what it already found. The variance is on the same order as any gap I could hope to open, so the lever is narrow and obvious: stop throwing every draw away — when a query reveals a strong architecture, spend the next queries *near* it.

I propose REA, a low-budget variant of Regularized Evolution (Real et al., 2019). It is the barest possible exploit-the-good loop, chosen precisely because it adds no model and no schedule — at 30 queries the controller machinery of RL-based search would never pay itself back. I keep a small population of evaluated architectures and repeat a steady-state cycle: pick a good member as parent, mutate it by one small edit, evaluate the child, insert it, and let the population turn over. No backprop, nothing coupling the budget to a learner — just selection and mutation, which *is* the "spend later queries near the good draws" lever the random-search numbers asked for. The whole design problem is what the knobs and the turnover rule must be at $K = 30$, because the standard recipes were tuned for thousands of evaluations.

The first knob is selective pressure, and I make it a tournament: sample $S$ members at random, take the highest-accuracy one as parent, mutate, evaluate, insert. $S$ is the greed knob, and the takeover-time result fixes its scale — for an $S$-ary tournament the best individual floods a population of size $P$ in roughly

$$t^\* \approx \frac{1}{\ln S}\,\big[\ln P + \ln\ln P\big],$$

so the dependence on $S$ is through $1/\ln S$ and bumping $S$ from 2 to 4 roughly halves the takeover time. With only $\sim$20 cycles of evolution after seeding I cannot afford the leisurely takeover $S = 2$ gives, but I also cannot be so greedy that the search collapses onto the first decent cell. The budget arithmetic then sets both sizes. Thirty queries split into seeding and evolution: I seed the population by drawing $P = 10$ random architectures and evaluating each — the first 10 queries, which is itself a depth-10 random search, so the population starts already holding a reasonable best — leaving $\sim$20 queries to evolve. $P = 10$ is small enough that those 10 seeds are a real fraction of the population's churn yet large enough to hold genuine diversity. $S = 3$ of 10 is moderate greed: the parent is the best of a 3-sample, usually one of the stronger members but not deterministically the global champion, so a little exploration stays alive. A larger $S$ (5 or 7 of 10) would make almost every parent the current best and turn the 20 evolution steps into a hill-climb on one lineage — fatal if the seeds were unlucky, which is exactly the random-search failure I am fixing. The takeover formula confirms $S = 3$ floods $P = 10$ in only a handful of cycles: fast enough to exploit a good seed within 20 steps, loose enough not to freeze on a bad one.

The eviction rule is where evolution either filters noise or accumulates it, and here I deliberately do *not* take the textbook choice. The greedy thing is to remove the *worst* member each cycle so the population's average ratchets up. On NAS-Bench-201 the tabular accuracy is deterministic — a query is a table lookup, not a noisy training run — so the classic argument that kill-the-worst homesteads a *lucky* evaluation does not bite. But there is a different reason to kill the *oldest*, and it is about exploration: kill-the-worst keeps every high-scoring cell in the population permanently, so once a strong cell is found it sits there forever, gets re-selected as parent over and over, and the population concentrates around it — the same premature-convergence collapse, now driven by genuine score instead of luck. With only 20 evolution steps, concentrating early means I stop exploring far too soon, right back to the random-search problem of being unable to leave a mediocre region. Kill-the-oldest gives every architecture a fixed bounded lifespan of about $P$ cycles regardless of its score, so a strong cell can only persist by being *re-discovered* — a parent carrying it must keep producing children that also score well. That age-based turnover *is* the regularizer: it constrains the survivors to architectures the search keeps re-finding, and it spreads the 20 precious steps across the space instead of piling them onto one lineage. Encoding age is free — append the child on the right, pop index 0 — so it costs no extra knob; $P$ and $S$ remain the only two.

The mutation matches the edit surface exactly. An architecture is a list of six op-indices in $[0,4]$; a single-edge mutation picks one edge and changes its operation to a different one of the five — precisely the helper `mutate_architecture(parent)`. One edit is the right granularity: it is the smallest move, so a child stays close to a parent the tournament already judged good (local exploitation), and chaining single edits can reach any architecture from any other, so the space stays connected and the search is not trapped in a neighborhood. I use no crossover — there is no meaningful way to splice two six-edge cells that chained single mutations cannot already reach, so it would add machinery for no gain. The one guard is validity: a mutation could yield an all-`none` degenerate cell, so I re-mutate until `is_valid_arch` passes before spending a query — costing only cheap helper calls, no budget. And I track the best-seen architecture *separately* from the population (an `_update_best` on every evaluation, seeding and evolution alike), so that even if age evicts a strong cell I still return it: the metric rewards the best architecture *ever found*, not the best currently alive.

The delta from random search is therefore concrete and small. Where random search drew all 30 architectures independently and kept the best, REA draws only the first 10 to seed a population, then spends the remaining $\sim$20 queries on children of tournament-selected parents, mutating one edge at a time and evicting the oldest each cycle, while tracking the best-ever for return. The exploitation random search structurally lacked is now the whole point. I expect the win where random search *failed to recover* — the wide-spread, low-saturation setting — so ImageNet16-120 should rise above 44.57; a modest lift or wash on CIFAR-100; and a near-wash or slight slip on near-saturated CIFAR-10, where the narrow-band ceiling leaves almost no room above 93.38. The clean signature I am after is REA beating random search on ImageNet16-120 while staying within noise on CIFAR-10. If instead REA *loses* on ImageNet16-120, the diagnosis flips: single-edge hill-climbing from random seeds cannot reach the good region in 20 steps, and the next rung must *model* the accuracy surface to extrapolate beyond the seeds rather than only mutate them.

```python
# EDITABLE region of naslib/custom_nas_search.py (lines 163-234) — step 2: REA
class NASOptimizer:
    """REA — Regularized Evolution Algorithm for NAS (low-budget variant).

    Population size 10 and tournament size 3, tuned for K=30 queries
    following NAS-Bench-Suite (White et al., 2022) low-budget recipes.
    """

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed

        self.population_size = 10
        self.tournament_size = 3
        self.population = []  # list of (arch, val_acc)
        self.best_arch = None
        self.best_val_acc = -1.0

    def _update_best(self, arch, val_acc):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = list(arch)

    def search_step(self, epoch):
        if epoch < self.population_size:
            # Seed initial population with random architectures
            arch = random_architecture()
            val_acc = self.api.query_val_accuracy(arch)
            self.population.append((arch, val_acc))
        else:
            # Tournament selection
            k = min(self.tournament_size, len(self.population))
            sample_indices = random.sample(range(len(self.population)), k)
            parent_idx = max(sample_indices, key=lambda i: self.population[i][1])
            parent_arch = self.population[parent_idx][0]

            # Mutation
            child_arch = mutate_architecture(parent_arch)
            while not is_valid_arch(child_arch):
                child_arch = mutate_architecture(parent_arch)
            child_val_acc = self.api.query_val_accuracy(child_arch)

            # Add child and remove oldest (regularization)
            self.population.append((child_arch, child_val_acc))
            self.population.pop(0)
            arch, val_acc = child_arch, child_val_acc

        self._update_best(arch, val_acc)

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "population_size": len(self.population),
        }

    def get_best_architecture(self):
        return self.best_arch
```
