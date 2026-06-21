# Context: predicting LLM loss when unique data is finite (circa 2022-2023)

## Research question

Compute-optimal scaling has, by this point, a clean prescription: pick a FLOP budget `C`, split it
between model parameters `N` and training tokens `D` to minimize held-out loss, and the answer is to
grow `N` and `D` in roughly equal proportion as `C` grows. That prescription is fit on single-epoch
runs, where each token is seen exactly once, and it equates "training tokens `D`" with "fresh tokens."

Extrapolating the equal-scaling rule, a 530-billion-parameter model would call for on the order of 11
trillion tokens — tens of terabytes of text. Estimates of the total stock of high-quality English text
on the internet put exhaustion of that supply around the mid-2020s, and for almost every language other
than English the available corpus is several orders of magnitude short of what the rule demands. The
regime now in view: a fixed, finite pool of unique tokens `U`, and a compute budget large enough that
the only way to spend it is to pass over that pool more than once, or to make the model bigger than the
pool can justify.

The setting to model: a loss law `L` as a function of parameters `N`, total tokens processed `D`, and
the size of the unique-data pool `U`, where `D` may exceed `U` because tokens are repeated. The
*Allocation* question in this regime — given a fixed unique-data pool, how should new compute be split
between more passes over the data and more parameters — and the *Return* question — how fast the value
of additional compute decays once you are forced to repeat. The deliverable is a single compact symbolic
law, with coefficients fit per data family, that gives sensible asymptotics as repetition and
over-parameterization grow.

## Background

**Scaling laws as the planning instrument.** Two questions govern how to spend a training budget:
*Allocation* (what is the optimal split of resources?) and *Return* (what is the value of additional
resources?). For LLMs the resource is compute in FLOPs, the metric is cross-entropy loss on held-out
data, and the standard cost model (Kaplan et al. 2020) is `FLOPs(N, D) ≈ 6ND`. *Return* is empirically
a power law — loss falls as a power of compute — and *Allocation*, on the established single-epoch
evidence, is balanced: parameters and tokens scale together. These laws were established by training
many models across a grid of sizes and token counts and carefully extrapolating.

**The risk-decomposition view of a loss law.** A productive way to write a parametric loss is as a sum
of three nonnegative pieces, each with a clear meaning: an irreducible floor `E` (the entropy of the
text itself — no model beats it); a term that vanishes as the model grows (the gap left because a
finite-capacity transformer cannot match the ideal generative process); and a term that vanishes as
data grows (the gap left because the model is trained on a finite sample for a finite number of steps).
Writing each shrinking gap as a power law in its resource gives the single-epoch form `L = E + A/N^α +
B/D^β`, with five learnable constants. Because loss is multiplicative/power-law in nature, such forms
are fit in *log* space, and to keep `log(E + A/N^α + B/D^β)` numerically stable the three additive terms
are combined with a log-sum-exp; a Huber penalty on the residual (with a small `δ`) keeps the fit robust
to the noisiest, lowest-compute runs.

**The empirical picture of repetition.** Three pre-existing observations about *existing* systems frame
the problem. First, repeating data is ordinary practice in machine learning broadly, yet the large-LM
literature had largely trained for a single epoch, and some work explicitly argued against reusing
tokens. Second, a model trained for several epochs (e.g. a science LLM trained ~4 epochs) showed
continually decreasing validation loss across repeats. Third, work on the "deep bootstrap" established
that good online learners are good offline generalizers, supporting the intuition that for a *small*
number of passes, a repeated token behaves much like a fresh one.

**The symmetric story for parameters.** The single-epoch law gives every parameter the same marginal
value regardless of how much data exists: under `A/N^α`, going from 1B to 10B parameters lowers loss by
the same absolute amount whether the dataset is one token or a billion tokens. The same question that
arises for repeated *data* — how its value behaves under saturation — arises for *excess* parameters
relative to the data they are trained on.

## Baselines

**Kaplan et al. (2020), neural-LM scaling laws.** The first systematic study fitting test loss to scale.
Single-factor power laws `L(N) = (N_c/N)^{α_N}`, `L(D) = (D_c/D)^{α_D}`, `L(C) = (C_c/C)^{α_C}`, each
holding when the other factors are abundant, fit by log-log regression. Their compute-optimal analysis
concluded that as compute grows, almost all of it should go into a *bigger model* (exponent `a ≈ 0.73`
for `N ∝ C^a`), with data growing slowly. The runs used early stopping and a learning-rate schedule that
did not match each run's token horizon, and the data axis is unique tokens.

**Hoffmann et al. (2022), compute-optimal scaling.** Re-did the allocation question with three
independent estimators — (i) the lower envelope of fixed-`N` training curves, (ii) parabola vertices of
fixed-`C` IsoFLOP profiles, (iii) a parametric fit of the loss surface — all matching the cosine
learning-rate cycle to each run's token horizon. The parametric form is

```
L(N, D) = E + A / N^α + B / D^β
```

with `{A, B, E, α, β}` fit by minimizing a Huber loss (δ = 1e-3) between `log L` and `log L_hat`
(computed via log-sum-exp), using L-BFGS from a grid of initializations. The closed-form efficient
frontier follows by minimizing `L` under `6ND = C`:

```
N_opt(C) = G (C/6)^a,   D_opt(C) = G^{-1} (C/6)^b,
G = (αA / βB)^{1/(α+β)},   a = β/(α+β),   b = α/(α+β).
```

All three estimators agree that `α ≈ β`, hence `a ≈ b ≈ 0.5` — parameters and tokens should scale
together, sharply against the earlier `a ≈ 0.73`. The consequence cited at the time: a 70B model trained
on ~1.4T tokens beats a 280B model at similar compute. The parametric fit is built and validated on
single-epoch runs, with `D` being unique tokens. For the C4 corpus specifically, this line reports only
that `a = b = 0.5`, i.e. `α = β`, and does not publish the coefficients `A, B, E` needed to evaluate the
loss surface on C4 — so using it as a single-epoch baseline on C4 requires re-fitting its five constants
from the C4 runs.

**Single-epoch dogma / no-reuse practice.** The de facto baseline for "what to do with more compute on a
fixed corpus" was: don't repeat — collect more unique data, or accept that the corpus caps you. Some
work argued reuse actively hurts. This is a policy rather than a model.

## Evaluation settings

The natural yardsticks for a data-constrained loss law are symbolic-regression tasks built from large
collections of real LLM training runs, where the model must fit a single functional form per data family
and extrapolate to held-out, harder (denser / larger) configurations rather than memorize.

- **Inputs.** Per training run: numeric descriptors and a categorical *group*. For the data-constrained
  family the numeric descriptors are the unique-token pool `U` (`unique_tokens`), the parameter count
  `N` (`params`), and the total tokens processed `D` (`tokens`), where `D` may exceed `U` because data
  is repeated. The target is the held-out cross-entropy `loss`.
- **Underlying experimental grid (pre-existing runs the law is fit/tested on).** Transformer LMs with a
  GPT-2 architecture and tokenizer, up to ~9B parameters and up to ~900B total tokens, trained on
  subsets of a web corpus (C4) with carefully nested data constraints so that a smaller-unique-data run
  always uses a subset of a larger one; cosine LR schedules decaying 10× over each run's horizon
  (matching the cycle to the horizon, since mismatched schedules were known to distort intermediate-
  horizon loss estimates); held-out test loss rather than training loss, because repetition can cause
  extreme overfitting. Unique-data budgets in the hundreds of millions to low billions of tokens;
  repetition ranging from a single epoch up to hundreds of epochs.
- **Metrics.** Primary: held-out test `R²` per data family (higher is better) — does the symbolic law
  predict loss on configurations it was not fit on. Secondary: `MAE`, `RMSE`, `NMAE` (lower is better).
- **Protocol.** Fit coefficients per group rather than pooling all groups; preserve sensible asymptotics
  on larger/denser test points (extrapolation, not memorization). Fits use nonlinear least squares /
  quasi-Newton optimization with multi-start initialization over a grid.

## Code framework

The law plugs into a fixed regression harness. The harness already provides: per-group splitting of the
runs, a log-space objective for additive power-law loss terms, a multi-start quasi-Newton optimizer for
fitting coefficients, and a predict path that dispatches per group. What is *not* settled is the
functional form relating `(U, N, D)` to loss and which coefficients are free during fitting — that form
is what is to be designed. So the substrate is the generic per-group curve-fitting machinery plus an
empty slot for the loss law.

```python
import numpy as np
import torch


def group_labels(X_cat):
    """Map categorical metadata rows to a group key (the family the run belongs to)."""
    return np.asarray([row[0] if len(row) else "default" for row in X_cat])


def _log_terms(X_num, params):
    # TODO: map numeric descriptors and coefficients to the three additive
    #       log terms whose log-sum-exp is the predicted log loss.
    raise NotImplementedError


def _after_backward(params):
    # TODO: optionally zero gradients for coefficients held fixed by the form.
    return None


def _initial_grid(X_num, y):
    # TODO: starting points for the coefficients of the law.
    raise NotImplementedError


def _fit_one_form(X_num, y, init_grid, steps=50):
    """Multi-start log-space fit for one symbolic loss form."""
    X = torch.tensor(np.asarray(X_num, float), dtype=torch.float32)
    target = torch.tensor(np.asarray(y, float), dtype=torch.float32)
    best_params, best_loss = None, float("inf")

    for init in init_grid:
        params = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
        opt = torch.optim.LBFGS([params], lr=1e-1, history_size=10, max_iter=20,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            pred = torch.logsumexp(_log_terms(X, params), dim=0)
            loss = torch.nn.functional.huber_loss(
                pred, torch.log(target), delta=1e-3, reduction="none"
            ).sum()
            loss.backward()
            _after_backward(params)
            return loss

        for _ in range(steps):
            opt.step(closure)

        with torch.no_grad():
            loss = float(torch.nn.functional.huber_loss(
                torch.logsumexp(_log_terms(X, params), dim=0),
                torch.log(target),
                delta=1e-3,
                reduction="none",
            ).sum())
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_params = params.detach().cpu().numpy()
    return best_params


def _predict_form(X_num, params):
    # TODO: numpy prediction path for the same loss law used in _log_terms.
    raise NotImplementedError


class ScalingLawModel:
    """Fits one symbolic loss law (shared form, per-group coefficients)."""

    def __init__(self, benchmark_name, numeric_names=None, categorical_names=None):
        self.benchmark_name = benchmark_name
        self.numeric_names = list(numeric_names or [])
        self.categorical_names = list(categorical_names or [])
        self.group_params_ = {}
        self.default_params_ = None

    def fit(self, X_num, X_cat, y):
        X_num, y = np.asarray(X_num, float), np.asarray(y, float)
        labels = group_labels(X_cat)
        fitted = []
        for g in sorted(set(labels.tolist())):
            m = labels == g
            p = _fit_one_form(X_num[m], y[m], _initial_grid(X_num[m], y[m]))
            self.group_params_[g] = p
            fitted.append(p)
        self.default_params_ = np.median(np.stack(fitted, 0), 0)
        return self

    def predict(self, X_num, X_cat):
        X_num = np.asarray(X_num, float)
        labels = group_labels(X_cat)
        out = np.zeros(len(labels))
        for g in sorted(set(labels.tolist())):
            m = labels == g
            out[m] = _predict_form(X_num[m],
                                   self.group_params_.get(g, self.default_params_))
        return out
```

The empty slots are `_log_terms`, `_after_backward`, `_initial_grid`, and `_predict_form`: together they
specify the loss law, which coefficients move during fitting, how the search starts, and how prediction
is evaluated. The per-group dispatch, log-sum-exp objective, Huber penalty, and L-BFGS optimizer already
exist.
