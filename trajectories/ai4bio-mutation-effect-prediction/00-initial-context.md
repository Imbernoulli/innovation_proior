## Research question

Deep mutational scanning measures the fitness effect of thousands of single amino-acid
substitutions in a protein, but the experiments are slow and expensive, so a computational
predictor that ranks mutations correctly would accelerate protein engineering. The single thing
being designed here is the **prediction head**: a frozen ESM-2 (650M) protein language model has
already turned each mutant sequence into a fixed 1280-dimensional mean-pooled embedding, and the
head must map that embedding (and the difference from the wild-type embedding) to a real-valued
fitness score whose *rank order* matches the assay. Everything upstream of the head — the language
model, the embeddings, the cross-validation splits, the training loop — is frozen. The only
degrees of freedom are the architecture of the head and two optimizer knobs.

## Prior art before the first rung (supervised prediction from a fixed feature vector)

The first rung is a regularized linear head, and it is the resolution of a long line of
fixed-feature supervised predictors. These are the methods the first baseline reacts to.

- **Ordinary least squares (Gauss/Legendre, early 1800s).** Fit a linear map by minimizing the
  residual sum of squares; closed form `w = (X'X)^{-1}X'y`, unbiased, and best-linear-unbiased by
  Gauss-Markov. Gap: when the feature columns are correlated — and 1280 ESM-2 dimensions are
  heavily correlated — `X'X` is ill-conditioned, its inverse blows up the variance along
  low-singular-value directions (`Cov = sigma^2 sum_j d_j^{-2} v_j v_j'`), and the coefficients
  become huge, sign-unstable, and undefined outright when features outnumber samples.
- **Subset selection / principal-component regression.** Stabilize by dropping covariates or
  projecting onto the top principal components of `X`. Gap: a hard, all-or-nothing cut — a
  direction is kept whole or discarded whole — so the estimate is a high-variance function of the
  threshold, and PCR discards directions by their variance *in `X`* with no reference to whether
  they predict `y`; a low-variance but genuinely predictive direction is thrown out with the noise.
- **Shrinkage / penalized fitting (the inadmissibility result, Stein 1956; James-Stein 1961).**
  In several dimensions the unbiased estimate is *inadmissible* under squared-error loss: a biased
  estimator that shrinks toward a point has strictly lower total error everywhere. Gap (for the
  rung): this says trading a little bias for much less variance can win, but it does not by itself
  hand over a single, well-defined, smoothly-tunable estimator for an ill-conditioned design — that
  is exactly the shape the first baseline has to land.

## The fixed substrate

A supervised regression harness is frozen and must not be touched. Per assay it loads the
precomputed ESM-2 embeddings, runs ProteinGym's pre-defined **random** 5-fold cross-validation,
and for each fold holds out 10% of the training fold as a validation set. Each fold trains a fresh
`MutationPredictor` with **AdamW** (`lr` and `weight_decay` from the config), a
**cosine-annealing** schedule over the epoch budget, **MSE loss** against the fitness score,
gradient-norm clipping at 1.0, and **early stopping** on validation Spearman with patience 20; the
best-validation checkpoint is restored and scored on the held-out fold. The per-assay number is the
mean Spearman over the 5 folds. The loop hands the head two tensors per batch and asks for one:

- `embedding`: `[B, 1280]` — mean-pooled ESM-2 (650M) representation of the mutant sequence.
- `delta_embedding`: `[B, 1280]` — the mutant-minus-wild-type embedding, i.e. the shift the
  mutation induced in representation space.

Note what the harness does *not* expose: there are no per-residue token embeddings (only the
mean-pooled vector), no zero-shot density / likelihood score, and no way to change the loss, the
optimizer family, or the schedule. The head sees one 1280-d vector and its delta, and nothing else.

## The editable interface

Exactly one region is editable — the `MutationPredictor` class between the `EDITABLE SECTION START`
and `EDITABLE SECTION END` markers in `custom_mutation_pred.py` — plus a tiny `CONFIG_OVERRIDES`
dict in `main()` that may set only `learning_rate` and `weight_decay`. The contract is fixed:
`__init__(self, embed_dim)` with `embed_dim = 1280`, and
`forward(self, embedding, delta_embedding) -> Tensor` returning shape `[B]`. Helper classes,
layers, and functions may be defined inside the region. Every method on the ladder is a fill of
exactly this contract; each later method replaces only the class body (and optionally the two
config knobs) and nothing else.

The starting point is the scaffold default: a single linear layer over the raw mutant embedding.

```python
# EDITABLE region of custom_mutation_pred.py — default fill (linear head)
class MutationPredictor(nn.Module):
    """Starter model: a single linear layer over the mutant embedding."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, embedding, delta_embedding):
        # embedding:       [B, EMBED_DIM] mutant ESM-2 embedding
        # delta_embedding: [B, EMBED_DIM] mutant - wildtype embedding
        return self.linear(embedding).squeeze(-1)   # [B] predicted fitness


# CONFIG_OVERRIDES in main() — allowed keys: learning_rate, weight_decay
CONFIG_OVERRIDES = {}
```

## Evaluation settings

Three ProteinGym single-mutant DMS assays span distinct biology: **BLAT_ECOLX** (beta-lactamase,
organismal fitness / antibiotic resistance, 4783 singles), **ESTA_BACSU** (a B. subtilis esterase,
thermostability, 2172 singles), and **RASH_HUMAN** (K-Ras GTPase, oncogene activity, 3134 singles).
The metric is **Spearman rank correlation** between predicted and true fitness, averaged over the
random 5-fold CV, reported per assay; higher is better. One seed (42). This task uses only the
*random* fold strategy — the easiest of ProteinGym's three — so the numbers are within-benchmark
relative scores, not comparable to the published supervised-leaderboard averages.
