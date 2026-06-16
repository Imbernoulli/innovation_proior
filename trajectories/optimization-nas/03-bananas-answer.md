**Problem.** REA exploited the best-seen but only one edit at a time, so it could not extrapolate: it
hill-climbed strong seeds well (ImageNet16-120 seed 0 → 46.8) but cratered when the seeds were poor
(seed 2 → 42.7), blowing the spread to ±1.44 and even slipping below random search on CIFAR-100. The
lever is to **compound the queries into a model** of the accuracy surface and use it to reach across the
whole space, not just a 1-edit neighborhood.

**Key idea (BANANAS, as this scaffold runs it).** Predictor-guided search. Warm-start with 10 random
architectures, then each step fit an **ensemble of 5 tiny numpy MLPs** (Linear(155,64)→ReLU→Linear(64,1),
Adam, plain MSE) on the **path-encoded** `(arch, val_acc)` pairs seen so far, draw a **pool of 500 unseen
random** architectures, score each by the **ensemble mean**, query the single argmax, refit. The neural
surrogate removes the GP's need for a kernel on DAGs; the 155-dim path encoding gives near-independent,
order-free features (used in full — no truncation needed at this size); the ensemble supplies a cheap
uncertainty; the **uniformly-random pool** gives the global reach REA lacked. Return the best-seen.

**Why these choices.** MSE (not a high-accuracy-weighted loss) and greedy ensemble-mean acquisition
(not UCB / Thompson) are the low-variance choices for ≤ 29 training points — exploration is supplied by
the random candidate pool, so an explicit uncertainty bonus would only chase the predictor's phantoms.
M = 5 retrains cheaply each step (the query, a table lookup, is the budgeted resource, not the MLP).
Predicting accuracy directly lets selection be a plain argmax.

**What to watch.** Strongest prediction is on **variance**: a bad seeding draw no longer traps the
search, so ImageNet16-120 spread should drop from REA's ±1.44 with the mean holding ≥ 45.25; CIFAR-100
should recover above random search's 70.65; CIFAR-10 a near-wash at the saturated top (~93.4–93.5). The
signature is matching/beating REA's mean everywhere while **tightening** the ImageNet16-120 variance.

**Hyperparameters.** `warm_start = min(10, num_epochs)`, `ensemble_size = 5`, `candidate_pool = 500`,
MLP `hidden = 64`, Adam `lr = 1e-2`, `epochs = 200`; one query per step; 30 steps.

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
