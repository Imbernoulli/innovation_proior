## Research question

Learn a hidden DNF concept — a function `f(x) = T_1(x) ∨ … ∨ T_s(x)`, each term `T_l` a conjunction of
literals over `n` Boolean variables — from *uniformly drawn* labelled examples `(x, f(x))`, and predict
`f` on a fresh uniform test sample. The one thing being designed is the **learner**: the model class,
the training set it is fit on, and the fitting routine. Everything else — how the target DNF is
sampled, how the held-out test set is built, how accuracy is scored — is frozen. The task is a
neural-learner ML-science question, *not* a SAT solver or a membership-query algorithm: the target DNF
is a hidden label oracle for uniform examples only, never to be inspected or reconstructed directly.

Three target families are evaluated independently and combined by geometric mean: `random-n30-s10-w4`
(mixed-polarity DNF, n=30, s=10, w=4), `monotone-n40-s20-w4` (positive literals only, n=40, s=20, w=4),
and `sparse-n60-s10-w4` (a 12-variable junta inside n=60, s=10, w=4). Each family stresses a different
axis: width and polarity in the random case, term count and monotonicity in the monotone case, and
variable irrelevance in the sparse case.

## Prior art before the first rung

DNF learnability under the uniform distribution is one of the most studied problems in computational
learning theory; the lineage the first rung reacts to is the gap between what is provably learnable and
what a gradient-trained network actually delivers from random examples alone.

- **Quasi-polynomial bound (Verbeurgt 1990).** Any `s`-term DNF is learnable in `n^{O(log s / ε)}` time
  under the uniform distribution from random examples — but the dependence on `s` is super-polynomial,
  so this is an existence result, not a practical recipe. Gap: not polynomial; gives no concrete model.
- **Harmonic Sieve (Jackson 1997).** Polynomial-time DNF learning — but it *requires membership
  queries* (it actively probes `f` at chosen points), so it is outside the "learner from random
  samples" setting this task fixes. Gap: needs queries the task forbids.
- **Random-DNF average case (Jackson & Servedio 2005).** Polynomial-time algorithms exist for *random*
  monotone DNF with `s = O(n^{2−γ})` terms and random non-monotone DNF with `s = O(n^{3/2−γ})` terms —
  evidence that the random regimes here are tractable in principle, by Fourier/correlation arguments,
  but again not a drop-in model. Gap: algorithm-specific, not a differentiable learner.
- **Tree ensembles on Boolean tabular data (Breiman 2001; Friedman 2001).** A decision tree splits on
  one variable at a time, so a depth-`w` root-to-leaf path is exactly a width-`w` conjunction — a single
  term. Bagging (random forests) and boosting (GBDT) are the canonical strong practical baselines on
  Boolean tabular data, and they encode conjunctive structure for free. Gap: not "neural," and it is an
  open empirical question whether a differentiable learner can match them here.
- **Differentiable DNF architectures (Payani & Fekri 2019; Mat_DNF 2023).** Encode the DNF inductive
  bias directly into the network — soft conjunctions with learnable literal memberships, OR'd by a
  noisy-OR — so the hypothesis class *is* relaxed DNF. The promise: the right inductive bias should beat
  a generic MLP. The question this task poses: does that promise hold from random examples only, against
  the tree ensembles?

## The fixed substrate

A frozen driver samples one hidden DNF per top-level seed, draws uniform Boolean examples in `{0,1}^n`,
fits the agent-defined learner on one fresh training set, and scores 0/1 test accuracy on a 10k held-out
uniform sample. The driver owns DNF sampling (`sample_dnf`), labeling (`evaluate_dnf`), uniform-example
generation (`make_uniform_examples`), dataset/output validation, and the per-family configuration
(`TaskConfig`: `n_features`, `num_terms`, `term_width`, `monotone`, `sparse_subset`, `train_size=20000`,
`test_size=10000`). The learner may read these config fields and may call the two exposed helpers
`evaluate_dnf` and `make_uniform_examples`, but may not peek at the test set or the term list.

## The editable interface

Exactly one region of `pytorch-examples/synth_dnf/custom_strategy.py` is editable (lines 217–304): the
three functions `build_model`, `make_dataset`, `fit_and_predict`. The contract:

1. `build_model(config, seed) -> (model, info)` returns an opaque model object consumed only by your own
   `fit_and_predict`, plus an `info` dict.
2. `make_dataset(dnf, config, seed) -> (x, y)` constructs the training set from `make_uniform_examples`
   and `evaluate_dnf` only, with inputs in `{0,1}^n`, labels in `{0,1}`, and at most `4 * train_size`
   examples.
3. `fit_and_predict(model, info, train_x, train_y, test_x, config, seed) -> preds` trains and returns an
   integer 0/1 vector of length `test_x.shape[0]`.

The starting point is the scaffold default: a 2-hidden-layer ReLU MLP trained with AdamW + BCE. Each
method on the ladder replaces exactly these three definitions and nothing else.

```python
# EDITABLE region of custom_strategy.py — default fill (MLP + AdamW + BCE)
def build_model(config: TaskConfig, seed: int):
    """2-hidden-layer MLP (256, 256) with ReLU (the scaffold default)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    hidden = 256
    model = nn.Sequential(
        nn.Linear(config.n_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    return model, {"kind": "mlp", "hidden": hidden}


def make_dataset(
    dnf: tuple[tuple[tuple[int, int], ...], ...],
    config: TaskConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform Boolean training set."""
    x = make_uniform_examples(config, config.train_size, seed)
    y = evaluate_dnf(x, dnf)
    return x, y


def fit_and_predict(
    model,
    info: dict,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: TaskConfig,
    seed: int,
) -> np.ndarray:
    """AdamW + BCE training, 20 epochs."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_x_t = torch.from_numpy(train_x).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().to(device)
    test_x_t = torch.from_numpy(test_x).float().to(device)

    epochs = 20
    batch_size = 256
    n = train_x_t.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            logits = model(train_x_t[idx]).view(-1)
            loss = criterion(logits, train_y_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(test_x_t)).view(-1) >= 0.5).long().cpu().numpy()
    return preds
```

## Evaluation settings

Three families (`random-n30-s10-w4`, `monotone-n40-s20-w4`, `sparse-n60-s10-w4`), one top-level seed
(42), `train_size = 20000`, `test_size = 10000`. Each family reports `test_accuracy` (emitted also as
`score`); the overall result is the geometric mean across the three families. Higher is better; the
geometric mean punishes any single family that lags. A soft per-run wall-clock budget applies.
