## Research question

Given a few thousand real LLM training runs, recover a table of numeric descriptors, a categorical *group*, and a measured loss. Fit **one compact symbolic law per benchmark family** that extrapolates: train on cheaper, smaller-scale runs and predict held-out larger/denser configurations. The design target is the body of the `ScalingLawModel` class — the functional form relating descriptors to loss and how its coefficients are fit per group. Data loading, the train/test split, and metrics are fixed.

There are three families:

- **`sld-vocab`** — vocabulary scaling: the *unigram-normalised* loss (which **can be negative** — do not clip) as a function of non-vocabulary parameters `N`, vocabulary size `V`, and training characters `D`.
- **`sld-lrbsz`** — learning-rate & batch-size scaling: LM loss as a joint function of learning rate `l`, batch size `b`, training tokens `D`, and non-embedding parameters `N`. This family has an *interior optimum* in `(l, b)` and is the hardest; held-out `R²` is routinely negative.
- **`sld-dataconstrained`** — data-constrained scaling: loss as a function of unique tokens `U`, parameters `N`, and total tokens `D`, where `D` can exceed `U` because data is repeated.

The intended contribution is a *law*, not a generic tabular regressor: a shared symbolic expression per family with group-specific coefficients and sensible asymptotics on the extrapolation region.

## Prior art / Background / Baselines

- **Kaplan et al. (2020)** — Neural language-model scaling is described by single-factor power laws, `L(N) ∝ N^{-α_N}` and `L(D) ∝ D^{-α_D}`, fitted in log-log space. Gap: each law holds only when the other factor is abundant, so there is no joint surface and no treatment of repeated data or of an interior optimum in `(l, b)`.
- **Hoffmann et al. (2022)** — Loss follows an irreducible floor plus independent capacity-gap and data-gap power terms, `L(N, D) = E + A/N^α + B/D^β`, fitted in log space. Gap: the additive, decoupled form has no cross-axis interaction, treats a re-read token as equivalent to a fresh one, and has been validated only on single-epoch runs.
- **Tao et al. (2024)** — Vocabulary scaling extends the power-law story to a third axis, modeling unigram-normalised loss as a function of `N`, `V`, and `D`. Gap: the established form is additive across axes; whether `V` and `D` interact is left open.
- **Muennighoff et al. (2023)** — Data-constrained scaling replaces raw token counts with effective counts that saturate under repetition, paired with a symmetric excess-parameter term. Gap: the saturation constants must be re-fit per family, and the published coefficients do not transfer to these subsets.
- **SLDBench / Step-law lineage (Li et al. 2025; SLDBench, Liu et al. 2025)** — For learning rate and batch size, current practice fits only the optimal `lr*` and `bsz*` as power laws of `N` and `D` from a handful of best runs. Gap: modeling only the optima discards most of the data and gives no loss prediction off the optimum.

## Fixed substrate / Code framework

A pure NumPy/SciPy benchmark loop is frozen. It loads the official SLDBench train/test `jsonl` splits for the requested family, builds `X_num` (numeric descriptors in a fixed column order), `X_cat` (the `group`), and `y` (target), then constructs one `ScalingLawModel(benchmark_name, numeric_names, categorical_names)`, calls `.fit(X_num, X_cat, y)` on the train split, and `.predict` on both splits. Metrics are computed automatically: held-out `R²` (primary, higher is better, can be negative — not clipped), `MAE`, `RMSE`, and `NMAE` (lower is better). The loop also provides `group_labels(X_cat)`, the constant `EPS`, and (already imported) `numpy as np` and `scipy.optimize.least_squares`. Observed training trials are mirrored read-only into `observed_trials/sld_*_train.jsonl` for inspection.

The numeric columns per family are fixed:
- `sld-vocab` → `[non_vocab_parameters, vocab_size, num_characters]`
- `sld-lrbsz` → `[lr, bsz, data_size, non_embedding_param_size]`
- `sld-dataconstrained` → `[unique_tokens, params, tokens]`

The `sld-vocab` target can be negative.

## Editable interface

Only one region is editable: the `ScalingLawModel` class body in `custom_scaling_law.py`. The contract is three methods: `__init__(benchmark_name, numeric_names, categorical_names)`, `fit(X_num, X_cat, y) -> self`, and `predict(X_num, X_cat) -> y_pred`. The same class handles all three families, dispatching on `benchmark_name`. Each family may use its own symbolic form, but the expression must be shared within a family and fit per `group`. Large pretrained models are not allowed; fitting uses the observed trials only.

The starting scaffold is a constant-mean predictor: no law, no per-group structure, no use of descriptors. Replace exactly this class body and nothing else.

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

Each family runs at seed `42` (the fit is deterministic given the split). Four metrics per family: `R²` (primary, weighted 2×, higher is better, can be negative), `MAE`, `RMSE`, and `NMAE` (lower is better, weighted 1× each). The per-family score is the weighted mean, and the task score is the **geometric mean across the three families**, so one badly negative `R²` drags the overall score down. On `sld-lrbsz`, where held-out `R²` is often negative, `MAE`/`RMSE` are the practical discriminators. Strong solutions fit coefficients per `group` and preserve sensible asymptotics on larger/denser test points.
