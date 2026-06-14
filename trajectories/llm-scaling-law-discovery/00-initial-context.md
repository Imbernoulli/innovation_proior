## Research question

Given a few thousand real LLM training runs from the scaling-law literature, recovered as a table of
numeric descriptors plus a categorical *group* and a measured loss, fit **one compact symbolic law per
benchmark family** that *extrapolates* — fit on the cheaper, smaller-scale runs and asked to predict the
held-out larger/denser configurations. The single thing being designed is the body of the
`ScalingLawModel` class: the functional form relating the descriptors to loss (and how its coefficients
are fit, per group). Everything else — data loading, the train/test split, the metrics — is fixed.

There are three families, each harder (less saturated) than the original SLDBench `parallel`/`moe`/`sft`
trio, and each may use a different shared form:

- **`sld-vocab`** — vocabulary scaling: the *unigram-normalised* loss (which **can be negative** — do not
  clip) as a function of non-vocabulary parameters `N`, vocabulary size `V`, and training characters `D`.
- **`sld-lrbsz`** — learning-rate & batch-size scaling: LM loss as a joint function of learning rate `l`,
  batch size `b`, training tokens `D`, and non-embedding parameters `N`. This one has an *interior
  optimum* in `(l, b)` and is the hard family — held-out `R²` is routinely negative here.
- **`sld-dataconstrained`** — data-constrained scaling: loss as a function of unique tokens `U`,
  parameters `N`, and total tokens `D`, where `D` can exceed `U` because data is repeated.

The intended contribution is a *law*, not generic tabular regression: a shared symbolic expression per
family with group-specific coefficients, preserving sensible asymptotics on the extrapolation region.

## Prior art before the first rung (the loss-law lineage)

The forms each rung reacts to are the established power-law loss models the field already converged on;
the first rung is a black-box regressor that ignores them, and the ladder climbs back toward them.

- **Kaplan et al. (2020), neural-LM scaling laws.** Single-factor power laws `L(N)=(N_c/N)^{α_N}`,
  `L(D)=(D_c/D)^{α_D}` fit by log-log regression, each valid when the other factor is abundant.
  Established that loss falls as a power of scale. Gap: one factor at a time, no joint surface, and the
  data axis is *fresh* tokens — no notion of repetition or of an interior optimum in `(l, b)`.
- **Hoffmann et al. (2022), compute-optimal (Chinchilla) law.** The joint additive form
  `L(N, D) = E + A/N^α + B/D^β` — an irreducible floor plus a capacity-gap power term plus a
  data-gap power term — fit in log space (log-sum-exp of the three terms, Huber residual, L-BFGS from a
  grid of starts). This is the backbone every symbolic rung here reuses. Gap: additive and decoupled —
  it has no cross-axis interaction (so it cannot bend a basin in `lr`/`bsz`), treats a re-read token as
  worth a fresh one (so it cannot model `D > U`), and was validated only on single-epoch runs.
- **Tao et al. (2024), vocabulary scaling.** Adds a vocabulary axis: loss as a function of `N`, `V`, `D`
  via per-axis power terms on the unigram-normalised loss. Motivates the `sld-vocab` form. Gap: the
  established version is additive across axes; whether `V` and `D` interact is left open.
- **Muennighoff et al. (2023), data-constrained scaling.** Models repetition by replacing the raw token
  count with an *effective* count that saturates as data is repeated, plus a symmetric story for excess
  parameters. Motivates `sld-dataconstrained`. Gap: the saturation form has free constants that must be
  re-fit per family, and the published coefficients do not transfer to these subsets.
- **SLDBench / Step Law lineage (Li et al. 2025; SLDBench, Liu et al. 2025).** For learning
  rate & batch size, prior practice fits only the *optima* `lr* = c·N^a·D^b`, `bsz* = d·D^g` from a
  handful of best configurations, rather than the whole loss surface. Motivates `sld-lrbsz`. Gap:
  modelling only the optima discards most of the runs and gives no loss prediction off the optimum.

## The fixed substrate

A pure NumPy/SciPy benchmark loop is frozen and must not be touched. It loads the official SLDBench
train/test `jsonl` splits for the requested family, builds `X_num` (the family's numeric descriptors in a
fixed column order), `X_cat` (the `group`), and `y` (the target), then constructs **one**
`ScalingLawModel(benchmark_name, numeric_names, categorical_names)`, calls `.fit(X_num, X_cat, y)` on the
train split, and `.predict` on both splits. It computes the metrics itself — held-out `R²` (primary,
higher is better, can be negative — *not* clipped), and `MAE`, `RMSE`, `NMAE` (lower is better) — and
writes them out. The loop also provides, in the same module, the helper `group_labels(X_cat)` (maps the
categorical rows to a clean group key, with `"__MISSING__"`/`"__all__"` fallbacks), the constant `EPS`,
and (already imported at module top) `numpy as np` and `scipy.optimize.least_squares`. The observed
training trials are mirrored read-only into `observed_trials/sld_*_train.jsonl` for direct inspection.

The numeric columns per family are fixed: `sld-vocab` → `[non_vocab_parameters, vocab_size,
num_characters]`; `sld-lrbsz` → `[lr, bsz, data_size, non_embedding_param_size]`; `sld-dataconstrained`
→ `[unique_tokens, params, tokens]`. The target for vocab can be negative.

## The editable interface

Exactly one region is editable — the `ScalingLawModel` class body in `custom_scaling_law.py` (the loop
fills in everything around it). The contract is three methods: `__init__(benchmark_name, numeric_names,
categorical_names)`; `fit(X_num, X_cat, y) -> self`; `predict(X_num, X_cat) -> y_pred`. The same class is
instantiated for all three families, dispatching on `benchmark_name`, so one fill may carry a different
symbolic form for vocab, lrbsz, and dataconstrained while keeping a single shared expression *per*
family and fitting its coefficients per `group`. Large pretrained LMs are not allowed; the model is fit
from the observed trials alone.

The starting point is the scaffold default: **predict the train-target mean** for every test point — no
law, no per-group structure, no use of the descriptors at all. Each rung replaces exactly this class body
and nothing else.

```python
# EDITABLE region of custom_scaling_law.py — default fill (constant-mean predictor)
class ScalingLawModel:
    """Editable benchmark-specific symbolic law scaffold.

    You may implement different symbolic forms for:
    - sld-vocab
    - sld-lrbsz
    - sld-dataconstrained

    The raw observed training trials are mirrored in:
    - observed_trials/sld_vocab_train.jsonl
    - observed_trials/sld_lrbsz_train.jsonl
    - observed_trials/sld_dataconstrained_train.jsonl
    """

    def __init__(self, benchmark_name: str, numeric_names=None, categorical_names=None):
        self.benchmark_name = benchmark_name
        self.numeric_names = list(numeric_names or [])
        self.categorical_names = list(categorical_names or [])

    def fit(self, X_num, X_cat, y):
        self.mean_ = float(np.mean(y))            # ignores every descriptor and the group
        return self

    def predict(self, X_num, X_cat):
        return np.full(len(X_num), self.mean_)    # constant -> R^2 = 0 on the train split by construction
```

## Evaluation settings

Three families — `sld-vocab`, `sld-lrbsz` (hidden), `sld-dataconstrained` — each run at the single seed
`{42}` (the fit is deterministic given the split). Four metrics per family: `R²` (primary, weighted 2×,
higher is better, can be negative), and `MAE`, `RMSE`, `NMAE` (lower is better, weighted 1× each). The
per-family score is their weighted mean and the task score is the **geometric mean across the three
families**, so a single family with a badly negative `R²` drags the whole score down. On `sld-lrbsz`,
where `R²` is routinely negative on the held-out split, `MAE`/`RMSE` are the practical discriminators.
Strong solutions fit coefficients per `group` rather than pooling, and keep sensible asymptotics on the
larger/denser test points (extrapolation, not memorization).
