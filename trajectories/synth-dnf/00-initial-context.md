## Research question

Learn a hidden DNF concept `f(x) = T_1(x) ∨ … ∨ T_s(x)`, where each term is a width-`w` conjunction of literals over `n` Boolean variables, from *uniformly drawn* labelled examples `(x, f(x))`, and predict `f` on a fresh uniform test sample. The only design freedom is the **learner**: its model class, training data, and fitting routine. How the target DNF is sampled, how the held-out test set is built, and how accuracy is scored are all frozen. The task is a neural-learner ML-science question, not a SAT solver or membership-query algorithm: the hidden DNF is a label oracle for uniform examples only and is never inspected directly.

Three target families are evaluated independently and combined by geometric mean:

- `random-n30-s10-w4`: mixed-polarity DNF, `n=30`, `s=10`, `w=4`.
- `monotone-n40-s20-w4`: positive literals only, `n=40`, `s=20`, `w=4`.
- `sparse-n60-s10-w4`: a 12-variable junta inside `n=60`, `s=10`, `w=4`.

Each family stresses a different axis: width/polarity, term count/monotonicity, and variable irrelevance.

## Prior art / Background / Baselines

DNF learnability under the uniform distribution has a large literature. The relevant baselines for this random-sample setting are:

- **Quasi-polynomial bound (Verbeurgt 1990).** Learns any `s`-term DNF in `n^{O(log s / ε)}` time from uniform random examples by collecting all low-width conjunctions consistent with the sample and ORing them. **Gap:** the dependence on `s` is super-polynomial, so it is not a practical learner for moderate `s`.
- **Harmonic Sieve (Jackson 1997).** A polynomial-time DNF learner that uses Fourier-based boosting. **Gap:** it requires membership queries, which this task forbids.
- **Random-DNF average case (Jackson & Servedio 2005).** Polynomial-time algorithms exist for random monotone DNF with `s = O(n^{2−γ})` terms and random non-monotone DNF with `s = O(n^{3/2−γ})` terms, via Fourier/correlation arguments. **Gap:** the analysis is specific to planted random families and does not yield a general trainable model.
- **Tree ensembles on Boolean tabular data (Breiman 2001; Friedman 2001).** A decision tree’s depth-`w` root-to-leaf path is exactly a width-`w` conjunction; bagging and boosting are the canonical strong practical baselines on Boolean tabular data. **Gap:** they are not differentiable, and it is an open empirical question whether a neural learner can match them here from random samples alone.
- **Differentiable DNF architectures (Payani & Fekri 2019; Mat_DNF 2023).** Encode soft DNF structure directly into the network: learnable literal memberships, soft conjunctions, and an OR layer. **Gap:** their performance against tree ensembles under a random-sample-only, fixed-family protocol is not established; whether the DNF inductive bias pays off in this exact setting is open.

## Fixed substrate / Code framework

A frozen driver samples one hidden DNF per top-level seed, draws uniform Boolean examples in `{0,1}^n`, fits the agent-defined learner on one fresh training set, and scores 0/1 test accuracy on a 10k held-out uniform sample. The driver owns DNF sampling (`sample_dnf`), labelling (`evaluate_dnf`), uniform-example generation (`make_uniform_examples`), dataset/output validation, and the per-family configuration (`TaskConfig`: `n_features`, `num_terms`, `term_width`, `monotone`, `sparse_subset`, `train_size=20000`, `test_size=10000`). The learner may read these config fields and may call `evaluate_dnf` and `make_uniform_examples`, but it may not peek at the test set or the term list.

## Editable interface

Exactly one region of `pytorch-examples/synth_dnf/custom_strategy.py` is editable (lines 217–304): the three functions `build_model`, `make_dataset`, and `fit_and_predict`. The contract:

1. `build_model(config, seed) -> (model, info)` returns an opaque model object consumed only by your own `fit_and_predict`, plus an `info` dict.
2. `make_dataset(dnf, config, seed) -> (x, y)` constructs the training set from `make_uniform_examples` and `evaluate_dnf` only, with inputs in `{0,1}^n`, labels in `{0,1}`, and at most `4 * train_size` examples.
3. `fit_and_predict(model, info, train_x, train_y, test_x, config, seed) -> preds` trains and returns an integer 0/1 vector of length `test_x.shape[0]`.

The starting point is the scaffold default: a 2-hidden-layer ReLU MLP trained with AdamW + BCE. Each method replaces exactly these three definitions and nothing else.

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

Three families (`random-n30-s10-w4`, `monotone-n40-s20-w4`, `sparse-n60-s10-w4`), one top-level seed (42), `train_size = 20000`, `test_size = 10000`. Each family reports `test_accuracy` (emitted also as `score`); the overall result is the geometric mean across the three families. Higher is better; the geometric mean punishes any single family that lags. A soft per-run wall-clock budget applies.
