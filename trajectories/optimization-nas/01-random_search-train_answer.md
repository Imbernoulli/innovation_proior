The strategy is the entire object of study here, but it has to bolt onto a fixed budget and a fixed space, so before I reach for anything clever I want the honest floor that every adaptive method will have to beat. I have 15,625 architectures in the NAS-Bench-201 cell and exactly 30 validation queries to spend; the objective $f(a)$ — validation accuracy on the tabular benchmark — is a gradient-free black box over a discrete cell graph, and one query costs one of my thirty units. The question is narrow: out of 30 evaluations, which architectures do I look at, and which one do I return for the unbudgeted final test? The temptation in any budgeted black-box search is to compound information — model the $(a, f(a))$ pairs seen so far and let them steer the next draw — but every such mechanism also carries an assumption (the landscape is locally smooth, the surrogate generalizes) that 30 queries may not justify. The baseline must make *no* such assumption, so that adaptivity has something to prove its worth against.

I propose plain uniform Random Search: draw each architecture independently and uniformly from the valid space, query its validation accuracy, keep the best-by-validation seen, and return that one. Its defining property is that it is *deliberately non-adaptive* — the history is never consulted to decide the next draw — so there is no wrong prior about where good architectures live that could mislead it. That same property is exactly what makes it the right yardstick rather than the last word.

What makes this a serious floor and not a strawman is a volume argument. Idealize "find a good architecture" as hitting a target region that occupies a fraction $p$ of the 15,625 architectures. Each draw lands in the target with probability $p$ and misses with $1-p$; the draws are independent, so

$$P(\text{at least one hit in 30}) = 1 - (1-p)^{30}.$$

Notice what is *absent*: the ambient size 15,625 never appears — only the *relative* volume $p$ of the good region. Whether 30 draws suffice has nothing to do with how big the space is and everything to do with how big a fraction of it is good. On NAS-Bench-201 that fraction is mild: the benchmark is small and many edge configurations are near-equivalent (swapping a `nor_conv_1x1` for a `nor_conv_3x3` on a non-critical edge barely moves the accuracy), so a large fraction of cells sit within a couple of accuracy points of the best. If the top $\sim$5% count as "good," then $1 - 0.95^{30} \approx 0.79$ — four runs in five land a good architecture in 30 draws, purely by volume; if "good" is the top $\sim$10%, it is $1 - 0.90^{30} \approx 0.96$. That is precisely why random search is hard to beat on this benchmark: the good region's relative volume is large enough that thirty independent shots usually catch it, so the ceiling a smarter method can reclaim above this floor is *narrow* — and that narrow band is exactly the sample-efficiency regime ($K \le 50$) the question targets.

It is worth being precise about *why* I draw uniformly rather than on a structured sweep of the six edges. A grid over the cell fixes a few operation values per edge and takes their Cartesian product, but with six edges that product is exponential, and worse, an aligned grid probes each individual edge at only the sixth root of the budget — thirty grid points on the 6-dimensional cell resolve each edge at barely two settings and project down onto any one edge as two stacks of coincident points. Uniform independent draws collapse under no such projection: each draw chooses all six edges independently, so the thirty draws probe every edge at (almost) thirty distinct settings at once, giving whichever edges actually matter for accuracy the full budget's worth of resolution *without my having to know in advance which edges those are*. This is the same per-axis-resolution argument that makes random search beat grid in hyperparameter optimization (Bergstra and Bengio, 2012), and it is why the helper `random_architecture()` draws each of the six op-indices independently rather than enumerating a lattice.

The one thing I must get right is the bookkeeping, because the budget is hard. The loop calls `search_step(epoch)` up to 30 times and counts every `query_val_accuracy` call; a 31st aborts the run. So each step queries exactly once: I draw one valid architecture from `random_architecture()` (which already rejects degenerate all-`none` cells), query its validation accuracy, and store it if it beats the running best. There is no state worth keeping beyond the running best and its score — the method is non-adaptive by construction.

There is one subtlety I name now because it is what the next rung will react to: the architecture I *return* is chosen by **validation** accuracy, but the metric is **test** accuracy, and the two are not perfectly monotone. Among several near-tied validation scores, which one I crown is itself a little noisy, and with only 30 samples the running best is often decided by one or two near-ties — so the reported test accuracy inherits that selection noise as seed-to-seed variance, and uniform sampling does nothing to suppress it. I therefore expect the variance across the five seeds, not the mean, to be the dominant feature of this baseline. CIFAR-10 should be tightest, since its accuracy distribution is compressed near the top (many cells reach $\sim$93–94%) so even unlucky draws land close; CIFAR-100 and especially ImageNet16-120 have a wider spread of quality, so a run that fails to draw into the top region pays more and the spread is larger. The structural failure is plain: thirty independent gambles with no exploitation of what they reveal — when a seed draws strong early it cannot *build* on it, and when it draws poorly it cannot *recover*. That is exactly the lever the next rung will pull: stop throwing every draw away, and spend later queries near the best-seen.

```python
# EDITABLE region of naslib/custom_nas_search.py (lines 163-234) — step 1: random search
class NASOptimizer:
    """Random Search — uniformly sample architectures and track the best."""

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed
        self.best_arch = None
        self.best_val_acc = -1.0

    def search_step(self, epoch):
        arch = random_architecture()
        val_acc = self.api.query_val_accuracy(arch)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = arch

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "current_val_acc": val_acc,
        }

    def get_best_architecture(self):
        return self.best_arch
```
