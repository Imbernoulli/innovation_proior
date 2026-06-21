The problem is sample-efficient neural architecture search. We have a discrete space of cell-based architectures represented as labeled DAGs, and an expensive black-box objective: the validation error obtained after training an architecture. Each query consumes a large fraction of the budget, so the search must extract as much information as possible from every evaluation. The standard Bayesian optimization framework is a natural fit, because it maintains a surrogate model of the objective and uses it to select informative queries. However, the classic Gaussian process surrogate does not apply cleanly here: there is no off-the-shelf kernel on the space of DAGs, and GP inference scales cubically with the number of observations, which becomes painful as the budget grows. Random search and regularized evolution avoid the modeling problem, but they do not build a global model of the objective and therefore cannot extrapolate beyond local neighborhoods.

The right fix is to replace the GP with a neural surrogate and design an architecture encoding that a neural network can actually read. A neural predictor consumes a fixed-length feature vector and learns its own similarity metric, so it removes the need to hand-engineer a kernel on DAGs. The remaining question is how to turn a labeled DAG into such a vector. The adjacency-matrix encoding is awkward because node orderings are arbitrary, which makes the representation non-unique, and individual edge features are not meaningful in isolation. A better representation is the path encoding: list every input-to-output operation path present in the cell and represent the architecture as a binary indicator vector over those paths. Each feature is a complete computational route rather than a wiring fragment, so features are nearly independent and the encoding is invariant to node ordering.

The method is BANANAS, which stands for Bayesian Optimization with Neural Architectures for Neural Architecture Search. It is a predictor-guided NAS loop built around three ideas: a path encoding of architectures, an ensemble of small neural predictors, and acquisition by the ensemble mean. At each step BANANAS encodes all architectures seen so far, trains an ensemble of tiny MLPs on the encoded architecture-to-validation-error pairs, scores a candidate pool with the ensemble mean, and evaluates the most promising candidate. The ensemble provides cheap predictive uncertainty through disagreement, the path encoding gives the surrogate meaningful and stable inputs, and the predictor lets the search reach across the whole space instead of hill-climbing one mutation at a time.

In the canonical small-cell setting the path encoding has one binary coordinate per possible input-to-output operation path. For a four-node cell with six edges and five operations per edge, there are five length-one paths, twenty-five length-two paths, and 125 length-three paths, giving a 155-dimensional vector. The full vector is used directly because it is already compact. The predictor is a two-layer MLP trained with Adam and mean-squared error. An ensemble of five such networks is trained from different random seeds; their average prediction is used to score candidates. The candidate pool is generated uniformly at random from the search space, filtered to avoid re-evaluating architectures, and the candidate with the highest predicted validation accuracy (or lowest predicted validation error) is queried next. Warm-starting with ten random architectures gives the ensemble enough data to be meaningful before model-guided selection begins.

The choice of a simple MSE loss and greedy ensemble-mean acquisition is deliberate at this scale. With only a few dozen training points, the predictor is noisy, and an explicit uncertainty bonus such as UCB or Thompson sampling can chase spurious highs and lows in the surrogate. The random candidate pool already supplies exploration, so the ensemble mean is the low-variance selector. Because evaluating an architecture is the budgeted operation and training the tiny MLPs is cheap, retraining the ensemble from scratch every step is affordable. The result is a search that compounds its queries into a global model, avoids the kernel-design and cubic-cost problems of GP-based BO, and remains simple to implement.

```python
import numpy as np


NUM_OPS = 5


def path_encoding(arch):
    """Encode a NAS-Bench-201 architecture as a 155-dim path feature vector.

    arch: list of 6 operation indices in [0, 4]. The cell contains five
    length-1, twenty-five length-2, and 125 length-3 input-to-output paths.
    """
    o = arch
    v = np.zeros(155, dtype=np.float32)
    # length-1 paths
    v[o[3]] = 1.0
    # length-2 paths
    off = NUM_OPS
    v[off + o[0] * NUM_OPS + o[4]] = 1.0
    v[off + o[1] * NUM_OPS + o[5]] = 1.0
    # length-3 paths
    off = NUM_OPS + NUM_OPS ** 2
    v[off + o[0] * NUM_OPS ** 2 + o[2] * NUM_OPS + o[5]] = 1.0
    return v


class _TinyMLP:
    """2-layer MLP regressor trained with Adam and MSE."""

    def __init__(self, in_dim, hidden=64, seed=0):
        rs = np.random.RandomState(seed)
        self.W1 = rs.randn(in_dim, hidden).astype(np.float32) / np.sqrt(in_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rs.randn(hidden, 1).astype(np.float32) / np.sqrt(hidden)
        self.b2 = np.zeros(1, dtype=np.float32)

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X):
        self._X = X
        self._z1 = X @ self.W1 + self.b1
        self._a1 = np.maximum(self._z1, 0.0)
        return (self._a1 @ self.W2 + self.b2).ravel()

    def fit(self, X, y, epochs=200, lr=1e-2):
        y = y.astype(np.float32).ravel()
        m = {k: np.zeros_like(v) for k, v in self._params().items()}
        v = {k: np.zeros_like(p) for k, p in self._params().items()}
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        for _ in range(epochs):
            t += 1
            pred = self.forward(X)
            err = (pred - y) / max(1, len(X))
            dW2 = self._a1.T @ err.reshape(-1, 1)
            db2 = err.sum(keepdims=True)
            dA1 = err.reshape(-1, 1) @ self.W2.T
            dZ1 = dA1 * (self._z1 > 0)
            dW1 = self._X.T @ dZ1
            db1 = dZ1.sum(axis=0)
            grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
            for k, g in grads.items():
                m[k] = b1 * m[k] + (1 - b1) * g
                v[k] = b2 * v[k] + (1 - b2) * (g * g)
                mhat = m[k] / (1 - b1 ** t)
                vhat = v[k] / (1 - b2 ** t)
                p = getattr(self, k)
                p -= lr * mhat / (np.sqrt(vhat) + eps)
                setattr(self, k, p)


class NASOptimizer:
    """BANANAS: predictor-guided sample-efficient NAS."""

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed
        self.warm_start = min(10, num_epochs)
        self.ensemble_size = 5
        self.candidate_pool = 500
        self.seen = {}
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
        return [
            _TinyMLP(X.shape[1], hidden=64, seed=self.seed + i + 1).fit(X, y)
            for i in range(self.ensemble_size)
        ]

    def _propose_next(self):
        ensemble = self._fit_ensemble()
        cands = []
        while len(cands) < self.candidate_pool:
            a = random_architecture()
            if tuple(a) not in self.seen:
                cands.append(a)
        Xc = np.stack([path_encoding(a) for a in cands])
        preds = np.mean([m.forward(Xc) for m in ensemble], axis=0)
        return cands[int(np.argmax(preds))]

    def search_step(self, epoch):
        if epoch < self.warm_start or len(self.seen) < 2:
            arch = random_architecture()
            while tuple(arch) in self.seen:
                arch = random_architecture()
        else:
            arch = self._propose_next()
        val_acc = self.api.query_val_accuracy(arch)
        self._record(arch, val_acc)
        return {"best_val_acc": self.best_val_acc, "queries": self.api.query_count}

    def get_best_architecture(self):
        return self.best_arch
```
