**Problem.** Map a 1280-d frozen ESM-2 embedding (and its wild-type delta) to a fitness score. The
features are heavily correlated, so an unregularized linear head — OLS, `w = (X'X)^{-1}X'y` — has
covariance `sigma^2 sum_j d_j^{-2} v_j v_j'` that explodes along the small-singular-value directions
the redundancy creates: huge, sign-unstable coefficients, undefined when features outnumber samples.

**Key idea.** Ridge regression: add an L2 penalty on the coefficients,
`min ||y - Xw||^2 + lambda ||w||^2`, equivalently floor the spectrum at `lambda`
(`w = (X'X + lambda I)^{-1}X'y`). `X'X + lambda I` is positive definite, hence always invertible, and
each principal direction is shrunk by `d_j^2/(d_j^2+lambda)` in proportion to its instability — a soft
version of dropping principal components. Biased, but the variance saving is first-order in `lambda`
while the bias cost is second-order, so a positive `lambda` strictly beats OLS in total error.

**Why this fill.** The harness trains an `nn.Module` end-to-end with AdamW + MSE, not a closed-form
solver, so the ridge penalty is realized as **AdamW decoupled weight decay** on a single linear
layer's weights (weight decay = the gradient of `lambda||w||^2` applied as shrinkage). The layer
reads `delta_embedding`, not the raw embedding: the delta has the constant protein identity already
subtracted, so a linear map from the mutation-induced shift to fitness is a direct readout — the raw
embedding would force the head to cancel a within-assay constant and worsen the conditioning.

**Hyperparameters.** Head: `nn.Linear(1280, 1)` on `delta_embedding`. `weight_decay = 5e-2` (an order
of magnitude above the scaffold default — the correlated features need real shrinkage); `lr` left at
the loop default; MSE loss, cosine schedule, early stopping on val Spearman all fixed by the loop.

```python
# EDITABLE region of custom_mutation_pred.py — step 1: ridge (linear head + weight decay)
class MutationPredictor(nn.Module):
    """Ridge regression as a single nn.Linear, trained with AdamW (wd=5e-2).

    Uses delta_embedding (mutant - wildtype) as the input feature, so the
    model learns a linear mapping from the mutation-induced embedding shift
    to the fitness score.
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, embedding, delta_embedding):
        return self.linear(delta_embedding).squeeze(-1)


# CONFIG_OVERRIDES in main() — the ridge penalty as AdamW weight decay
CONFIG_OVERRIDES = {'weight_decay': 5e-2}
```
