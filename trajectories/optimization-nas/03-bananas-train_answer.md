REA's numbers landed right where my falsifiable prediction said the risk was, and the variance is the diagnosis written in a number. REA did *not* clearly beat random search: CIFAR-10 went 93.38 → 93.40 (a wash — the compressed top is near-saturated, no room), CIFAR-100 actually *slipped* 70.65 → 70.43, and ImageNet16-120 rose 44.57 → 45.25 but its spread blew up to ±1.44, more than double random search's ±0.58, with seed 2 cratering to 42.70 while seed 0 hit 46.80. That is exactly the premature-convergence failure I worried about: on the good seeds the 20 evolution steps hill-climbed a strong seed into a genuinely better cell (46.8, 46.4), but on seed 2 the tournament locked onto a mediocre seeding region and single-edge mutation could not climb out in 20 steps. REA is high-variance because it is a *local* search — the mechanism that helped, mutating the best-seen, is the same one that hurt, because mutation only ever moves one edit from something already evaluated. It cannot *extrapolate*; it can only refine what luck handed it. So the lever this time is different from last time: REA has exploitation but no way to use the 30 queries to reason about architectures it has *not* yet evaluated.

I propose BANANAS: predictor-guided search that compounds every query into a *model* of the accuracy surface and uses it to reach across the whole 15,625-cell space each step, not just a 1-edit neighborhood. This is Bayesian optimization in spirit — model what you have seen, pick the next point to maximize expected progress, evaluate, update — but with three deliberate substitutions that make it fit a 30-query budget inside this scaffold: a neural surrogate in place of a Gaussian process, the path encoding in place of adjacency, and a uniformly-random global candidate pool in place of REA's mutate-the-best.

The surrogate comes first, because the textbook BO choice fails here. A GP *is* its kernel, and the inputs are labeled cell graphs — there is no off-the-shelf kernel on DAGs, building a bespoke distance over architectures is a hard modeling problem of its own, and GP inference is cubic in the number of observations. I need none of that. All I actually require is: consume the architectures seen, predict the validation accuracy of an unseen one, and let me pick the most promising candidate. A neural network does exactly that and *learns its own notion of similarity* from a feature vector, so the "invent a kernel on DAGs" problem evaporates — I hand the net a featurization and let it fit.

The encoding is the next decision, and the scaffold already made the right one available. The default move is an adjacency encoding — a bit per possible edge plus the op at each node — but that is order-dependent and its features are violently inter-dependent (an edge bit means nothing without a path to the output), which is what a small net struggles to fit from a handful of examples. Instead I use `path_encoding`: a binary indicator over input→output operation-paths, one feature per "the tensor can flow through this sequence of operations," length $5 + 5^2 + 5^3 = 155$ (the encoding of White et al., 2020). Each feature is a self-contained statement about what the cell *computes* rather than a fragment of its wiring, so the features are far less entangled and a single architecture maps to a single encoding — no isomorphism ambiguity. With 155 dimensions and a 30-cell budget the full encoding is small enough to use as-is; there is no need to truncate it to the most-frequent short paths the way one would on a large cell where the encoding length explodes — here it does not, so I take the whole vector.

The predictor itself has to respect what this scaffold can run, so I build it in numpy: a two-layer MLP, $\mathrm{Linear}(155, 64) \to \mathrm{ReLU} \to \mathrm{Linear}(64, 1)$, trained with Adam on plain mean-squared error against the validation accuracies. Why MSE and not a loss that up-weights the high-accuracy cells? At this budget — at most 29 training points by the final step — the predictor's job is modest: rank a pool well enough to pick a good one, not to nail the error of near-optimal cells to the decimal. Plain MSE on the 155-dim path features is the honest, low-variance choice: it fits fast (200 Adam steps, $\mathrm{lr} = 10^{-2}$), has no extra hyperparameter to tune against a budget that cannot afford tuning, and predicts accuracy *directly* so selection is a plain argmax over candidates. Width 64 and a $1/\sqrt{\text{fan-in}}$ initialization keep it from overfitting the tiny training set.

Every acquisition rule wants more than a point prediction — it wants an uncertainty, so the search can explore where the model is ignorant. A single net gives only a point estimate; the cheap, well-calibrated way to get an uncertainty is an ensemble. I train $M = 5$ independent copies from different random initializations and read their disagreement as the uncertainty — small enough to retrain from scratch every step (the MLP's cost is a rounding error against the budgeted query, which is just a table lookup), large enough that the ensemble mean is a stable estimate. Each member gets its own seed (`self.seed + i + 1`) so the members genuinely disagree.

Now the acquisition, where I deliberately stay simple. With the ensemble I have, for any candidate, the mean prediction across the 5 members, and the rule is the plainest exploitation: **score every candidate by the ensemble mean and query the single highest-scoring one.** I considered the uncertainty-aware alternatives — upper-confidence-bound ($\text{mean} + \beta\cdot\text{std}$) or Thompson-style sampling from each candidate's predictive spread — and they are the principled way to balance exploration against exploitation. But here the exploration is already supplied by *where the candidates come from*, and adding an explicit uncertainty bonus on top of a 29-point predictor risks chasing the model's high-variance phantoms into regions it has no business trusting. So I let the candidate pool carry the exploration and let the acquisition be pure greedy exploitation of the mean — which is also robust to a poorly-calibrated ensemble std at tiny sample sizes.

The candidate pool is the last piece and the one that most distinguishes this rung from REA. REA's fatal limitation was that it only ever looked one edit away from an evaluated architecture, so it could not escape a bad seeding region — that is the 42.70 seed. To fix exactly that, I draw the candidate pool **uniformly at random over the whole space**: each step I sample a large pool (500) of *unseen* random architectures and score all of them with the ensemble. This is the deliberate opposite of mutate-the-best: instead of refining the current neighborhood, the predictor reaches across the entire 15,625-cell space and pulls in whichever architecture it predicts is best, even if it is many edits away from anything seen. The model is what generalizes from the seen points to the unseen pool, so a single good predictor lets me *extrapolate* where REA could only *interpolate* locally. The pool is large (500) so the argmax sweeps a broad slice of the space, and it is cheap — scoring 500 candidates with five tiny MLPs is nothing against the budgeted query.

Assembling the loop: warm-start with $N_0 = 10$ random architectures, evaluated and recorded, just so the ensemble has enough points to fit something meaningful (with fewer than two points the predictor is hopeless, so I keep sampling randomly until then). Then each remaining step fits the 5-MLP ensemble on the path-encoded $(\text{arch}, \text{val\_acc})$ pairs seen so far, draws 500 unseen random candidates, encodes them, scores each by the ensemble mean, queries the single argmax, records it, and refits. Track the best-seen and return it. The strongest prediction is on **variance**: a poor seeding draw no longer traps the search, because the predictor is free to pull in a strong architecture from anywhere, so I expect the ImageNet16-120 spread to come *down* from REA's ±1.44 with the mean holding at or above 45.25; CIFAR-100 to recover above random search's 70.65; CIFAR-10 a near-wash around 93.4–93.5 at the saturated top. The signature I am looking for is BANANAS matching or slightly beating REA's mean on every setting while clearly *tightening* the ImageNet16-120 variance — predictor-guided global selection buying robustness, not just average accuracy. If instead its mean is no better and its variance no tighter, the diagnosis would be that 29 path-encoded points are too few to train a predictor that generalizes, and the next move would be a richer encoding or a zero-cost proxy to give the surrogate more signal per query.

```python
# EDITABLE region of naslib/custom_nas_search.py (lines 163-234) — step 3: BANANAS
class _TinyMLP:
    """2-layer numpy MLP regressor trained with Adam + MSE."""

    def __init__(self, in_dim, hidden=64, seed=0):
        rs = np.random.RandomState(seed)
        self.W1 = rs.randn(in_dim, hidden).astype(np.float32) * (1.0 / np.sqrt(in_dim))
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rs.randn(hidden, 1).astype(np.float32) * (1.0 / np.sqrt(hidden))
        self.b2 = np.zeros(1, dtype=np.float32)

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0.0)

    def forward(self, X):
        self._X = X
        self._z1 = X @ self.W1 + self.b1
        self._a1 = self._relu(self._z1)
        return (self._a1 @ self.W2 + self.b2).squeeze(-1)

    def fit(self, X, y, epochs=200, lr=1e-2):
        y = y.astype(np.float32).reshape(-1)
        m = {k: np.zeros_like(v) for k, v in self._params().items()}
        v = {k: np.zeros_like(p) for k, p in self._params().items()}
        b1_, b2_, eps, t = 0.9, 0.999, 1e-8, 0
        for _ in range(epochs):
            t += 1
            pred = self.forward(X)
            err = (pred - y) / max(1, len(X))
            dW2 = self._a1.T @ err.reshape(-1, 1)
            db2 = err.sum(keepdims=True)
            dA1 = err.reshape(-1, 1) @ self.W2.T
            dZ1 = dA1 * (self._z1 > 0)
            dW1 = X.T @ dZ1
            db1 = dZ1.sum(axis=0)
            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
            for k, g in grads.items():
                m[k] = b1_ * m[k] + (1 - b1_) * g
                v[k] = b2_ * v[k] + (1 - b2_) * (g * g)
                mhat = m[k] / (1 - b1_ ** t)
                vhat = v[k] / (1 - b2_ ** t)
                setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}


class NASOptimizer:
    """BANANAS — predictor-guided sample-efficient NAS.

    Strategy:
    1. Warm start with N0=10 random architectures.
    2. Fit an ensemble of M=5 small MLPs on path-encoded (arch, val_acc) pairs.
    3. Each remaining step: draw a large random pool of candidates, score
       them with ensemble-mean predictions, pick the top unseen candidate,
       query its val accuracy, refit the ensemble.
    """

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed

        self.warm_start = min(10, num_epochs)
        self.ensemble_size = 5
        self.candidate_pool = 500

        self.seen = {}           # arch_tuple -> val_acc
        self.best_arch = None
        self.best_val_acc = -1.0

    def _record(self, arch, val_acc):
        self.seen[tuple(arch)] = val_acc
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = list(arch)

    def _fit_ensemble(self):
        X = np.stack([path_encoding(list(a)) for a in self.seen])
        y = np.array([self.seen[a] for a in self.seen], dtype=np.float32)
        ensemble = []
        for i in range(self.ensemble_size):
            mlp = _TinyMLP(X.shape[1], hidden=64, seed=self.seed + i + 1)
            mlp.fit(X, y, epochs=200, lr=1e-2)
            ensemble.append(mlp)
        return ensemble

    def _propose_next(self):
        ensemble = self._fit_ensemble()
        # Large random candidate pool
        cands = []
        while len(cands) < self.candidate_pool:
            a = random_architecture()
            t = tuple(a)
            if t not in self.seen:
                cands.append(a)
        Xc = np.stack([path_encoding(a) for a in cands])
        preds = np.mean([m.forward(Xc) for m in ensemble], axis=0)
        idx = int(np.argmax(preds))
        return cands[idx]

    def search_step(self, epoch):
        if epoch < self.warm_start or len(self.seen) < 2:
            arch = random_architecture()
            while tuple(arch) in self.seen:
                arch = random_architecture()
        else:
            arch = self._propose_next()

        val_acc = self.api.query_val_accuracy(arch)
        self._record(arch, val_acc)

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "current_val_acc": val_acc,
        }

    def get_best_architecture(self):
        return self.best_arch
```
