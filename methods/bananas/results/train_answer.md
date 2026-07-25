The problem is sample-efficient neural architecture search. We have a discrete space of cell-based architectures represented as labeled DAGs, and an expensive black-box objective: the validation error obtained after training an architecture. Each query consumes a large fraction of the budget, so the search must extract as much information as possible from every evaluation. The standard Bayesian optimization framework is a natural fit, because it maintains a surrogate model of the objective and uses it to select informative queries. However, the classic Gaussian process surrogate does not apply cleanly here: there is no off-the-shelf kernel on the space of DAGs, and GP inference scales cubically with the number of observations, which becomes painful as the budget grows. Random search and regularized evolution avoid the modeling problem, but they do not build a global model of the objective and therefore cannot extrapolate beyond local neighborhoods.

The right fix is to replace the GP with a neural surrogate and design an architecture encoding that a neural network can actually read. A neural predictor consumes a fixed-length feature vector and learns its own similarity metric, so it removes the need to hand-engineer a kernel on DAGs. The remaining question is how to turn a labeled DAG into such a vector. The adjacency-matrix encoding is awkward because node orderings are arbitrary, which makes the representation non-unique, and individual edge features are not meaningful in isolation. A better representation is the path encoding: list every input-to-output operation path present in the cell and represent the architecture as a binary indicator vector over those paths. Each feature is a complete computational route rather than a wiring fragment, so features are nearly independent and the encoding is invariant to node ordering.

The method is BANANAS, which stands for Bayesian Optimization with Neural Architectures for Neural Architecture Search. It is a predictor-guided NAS loop built around four ideas: a truncated path encoding of architectures, an ensemble of feedforward neural predictors trained with a loss that weights good architectures more heavily, an acquisition rule that samples independently per candidate, and a candidate pool built by mutating the best architectures found so far rather than drawn from nowhere. At each step BANANAS encodes every architecture seen so far, trains the ensemble on the encoded architecture-to-validation-error pairs, generates candidates by editing the current best architectures, scores each candidate with an independent Thompson-sampling draw from its ensemble mean and standard deviation, and evaluates the candidate with the lowest sampled error. The ensemble supplies cheap predictive uncertainty through disagreement, the path encoding gives the surrogate meaningful and stable inputs, and the predictor lets the search reach across the whole space instead of hill-climbing one mutation at a time on the raw objective.

In the canonical small-cell setting the full path encoding has one binary coordinate per possible input-to-output operation path: for a four-node cell with six edges and five operations per edge there are five length-one paths, twenty-five length-two paths, and 125 length-three paths, a 155-dimensional vector. Because a long path needs many specific edges to co-occur, long paths are exponentially rare under sparse random sampling, so the vector is truncated to a cutoff of the thirty shortest, most-frequent paths rather than used in full. The predictor is a ten-layer, width-twenty feedforward network trained with Adam and a loss that measures each prediction's error relative to a fixed lower bound on the best achievable validation error, so a given absolute miss is penalized far more heavily on a near-optimal architecture than on a bad one. An ensemble of five such networks is trained from different random seeds; their mean and standard deviation over the candidates are read off as the predictive uncertainty. The candidate pool is generated not by uniform random draws but by mutating the architectures with the lowest validation error found so far with single-edit changes, which keeps candidates close to where the predictor has actually seen data and clustered near the current optimum; each candidate is then scored by an independent normal draw from its own ensemble mean and standard deviation, and the candidate with the lowest sampled value is queried next. Warm-starting with ten random architectures gives the ensemble enough data to be meaningful before model-guided selection begins.

The choice of a relative-error loss, mutation-based candidates, and independent Thompson sampling is deliberate at this scale. Plain mean-absolute error would fit a bad architecture's error as carefully as a near-optimal one's, spending the predictor's capacity where the search does not need it; measuring error against a floor on the best achievable validation error redirects that capacity toward the good architectures the search actually has to discriminate between. Uniformly random candidates would almost surely sit far from every architecture the ensemble has trained on, so the predictor would be asked to extrapolate with unwarranted confidence; mutating the current best architectures keeps candidates edit-close to the training data and near the region that matters. Ordinary Thompson sampling with an ensemble draws one member and applies it to every candidate, correlating all the scores through a single coincidence; sampling independently per candidate keeps the exploration decorrelated across the pool, and being stochastic it slots directly into batch evaluation by taking the several best draws instead of only one. Because evaluating an architecture is the budgeted operation and training the small predictors is cheap, retraining the ensemble from scratch every step is affordable. The result is a search that compounds its queries into a global model, avoids the kernel-design and cubic-cost problems of GP-based BO, and remains simple to implement.

```python
import numpy as np
from tensorflow import keras
import tensorflow as tf

NUM_OPS = 5
LONGEST_PATH = 3


# ---- truncated path encoding ----
def encode_paths(arch):
    """One binary feature per input->output operation-path; full length
    NUM_OPS + NUM_OPS**2 + NUM_OPS**3."""
    o = arch
    L = sum(NUM_OPS ** i for i in range(1, LONGEST_PATH + 1))
    v = np.zeros(L, dtype=np.float32)
    v[o[3]] = 1.0                                                 # length-1 path
    off = NUM_OPS
    v[off + o[0] * NUM_OPS + o[4]] = 1.0                          # length-2 path
    v[off + o[1] * NUM_OPS + o[5]] = 1.0                          # length-2 path
    off = NUM_OPS + NUM_OPS ** 2
    v[off + o[0] * NUM_OPS ** 2 + o[2] * NUM_OPS + o[5]] = 1.0    # length-3 path
    return v


def path_encoding(arch, cutoff=30):
    """Keep the cutoff shortest/most-frequent paths."""
    full = encode_paths(arch)
    return full[:cutoff] if cutoff else full


# ---- feedforward predictor with MAPE loss ----
def mape_loss(y_true, y_pred):
    y_lb = 4.5                                  # lower bound on best val error
    return tf.abs((y_pred - y_lb) / (y_true - y_lb) - 1.0)


class Predictor:
    def fit(self, X, y, num_layers=10, width=20, epochs=150, lr=0.01):
        net = keras.models.Sequential(
            [keras.layers.Dense(width, activation='relu') for _ in range(num_layers)]
            + [keras.layers.Dense(1)])
        net.compile(optimizer=keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.99),
                    loss=mape_loss)
        net.fit(X, y, batch_size=32, epochs=epochs, verbose=0)
        self.net = net
        return self

    def predict(self, X):
        return np.squeeze(self.net.predict(X))


# ---- independent Thompson sampling ----
def acq_its(ensemble_preds):
    """ensemble_preds: (M, num_candidates). One independent N(fhat, sigmahat^2)
    draw per candidate; lower (predicted error) is better."""
    preds = np.array(ensemble_preds)
    fhat = preds.mean(axis=0)
    sigmahat = preds.std(axis=0, ddof=1)
    return np.random.normal(fhat, sigmahat)


class NASOptimizer:
    """BANANAS predictor-guided NAS."""

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed
        self.warm_start = min(10, num_epochs)
        self.ensemble_size = 5
        self.num_candidates = 100
        self.num_arches_to_mutate = 1
        self.patience_factor = 5
        self.seen = {}
        self.best_arch, self.best_val_loss = None, np.inf

    def _record(self, arch, val_loss):
        self.seen[tuple(arch)] = val_loss
        if val_loss < self.best_val_loss:
            self.best_val_loss, self.best_arch = val_loss, list(arch)

    def _fit_ensemble(self):
        X = np.stack([path_encoding(list(a)) for a in self.seen])
        y = np.array([self.seen[a] for a in self.seen], dtype=np.float32)
        return [Predictor().fit(X, y) for _ in range(self.ensemble_size)]

    def _propose_next(self):
        ensemble = self._fit_ensemble()
        best = sorted(self.seen, key=lambda a: self.seen[a])[
            : self.num_arches_to_mutate * self.patience_factor
        ]
        cands = []
        while len(cands) < self.num_candidates:
            parent = list(best[np.random.randint(len(best))])
            child = mutate_architecture(parent)               # single-edit mutation
            if tuple(child) not in self.seen:
                cands.append(child)
        Xc = np.stack([path_encoding(a) for a in cands])
        preds = [p.predict(Xc) for p in ensemble]
        return cands[int(np.argmin(acq_its(preds)))]          # min predicted error

    def search_step(self, epoch):
        if epoch < self.warm_start or len(self.seen) < 2:
            arch = random_architecture()
            while tuple(arch) in self.seen:
                arch = random_architecture()
        else:
            arch = self._propose_next()
        val_loss = self.api.query_val_loss(arch)
        self._record(arch, val_loss)
        return {"best_val_loss": self.best_val_loss, "queries": self.api.query_count}

    def get_best_architecture(self):
        return self.best_arch
```
