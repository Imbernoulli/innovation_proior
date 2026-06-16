**Problem.** Sample-efficient NAS on NAS-Bench-201: 15,625 architectures, a hard budget of 30
validation queries per dataset per seed, return the architecture to test on the unbudgeted split.
The objective is a gradient-free black box over a discrete cell graph; the question is purely which 30
architectures to query and which one to return.

**Key idea (the floor).** Draw each architecture independently and uniformly from the valid space,
query its validation accuracy, keep the best-by-validation seen. Deliberately **non-adaptive** — the
history is never consulted to choose the next draw — so it makes no assumption about where good
architectures live and cannot be misled by a wrong prior. It is the yardstick every adaptive rung must
beat.

**Why it is hard to beat here.** "Find a good architecture" is hitting a target of relative volume `p`;
`P(at least one hit in 30) = 1 - (1-p)^30`, with the ambient size absent. On this small benchmark the
top few percent of cells are near-equivalent, so `p` is large (top-5% gives ≈ 0.79, top-10% ≈ 0.96):
thirty independent shots usually catch the good region, leaving only a narrow band for adaptivity to
reclaim. Each draw also probes all six edges independently, so the budget's resolution lands on whichever
edges matter without a grid's exponential blowup.

**What to watch.** The mean should be respectable-but-unremarkable on each dataset; the **seed-to-seed
variance** is the dominant feature, because the returned architecture is picked by validation among near
ties and the metric is test. Expect the tightest spread on CIFAR-10 (compressed top of the accuracy
distribution) and wider spread on CIFAR-100 / ImageNet16-120. The structural failure — thirty memoryless
gambles with no exploitation — is what forces step 2 to keep and build on the best-seen.

**Hyperparameters.** None beyond the seed; one query per step; 30 steps.

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
