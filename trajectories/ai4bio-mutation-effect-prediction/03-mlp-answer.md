**Problem.** The reshape-CNN proved nonlinearity rescues the stability cliff (ESTA −0.107 → 0.641) and
lifts the linear assays, but it bundled the nonlinearity with a convolutional weight-sharing bias over
a fake `(64,40)` grid that has no real adjacency. This rung strips the structure out: keep only a
plain nonlinear dense head and test whether the convolution was carrying any weight.

**Key idea.** A single-hidden-layer MLP over `delta_embedding`. A linear head answers with one
hyperplane and cannot bend around a threshold phenotype; a hidden layer with a nonlinearity recodes
the delta into a space where the curved fitness function is a simple readout — `ReLU(delta·W1+b1)` is a
piecewise-linear tiling of delta-space, enough to place a stability cliff. Hidden units have no target;
the loop's backpropagated MSE gradient (AdamW) supplies their error signal.

**Why these choices.**
- **`delta_embedding` only** (not the CNN's concatenation): the raw embedding's bulk is the constant
  protein identity, nuisance dimensions a single hidden layer must learn to ignore; the delta isolates
  what the mutation did. If the dense delta-only head matches the both-inputs CNN, the raw embedding
  was inert too.
- **ReLU, not logistic:** nonlinear and differentiable, no saturation for large positive inputs, so
  gradients stay alive in the active region — the right modern default for a regression head.
- **One hidden layer, width 512:** the *minimal* object that buys curvature — wide enough to tile a
  1280-d delta-space, single-layer so the test is "plain nonlinear head," not depth.
- **Dropout 0.1 + early stopping:** with ~2k–5k mutants per assay and a 512-wide layer, regularize
  against memorizing the training fold. (The literal fill applies dropout to the pre-activation:
  `Linear → Dropout → ReLU → Linear`; for ReLU this is near-equivalent to post-activation dropout.)

**Hyperparameters.** `Linear(1280, 512) → Dropout(0.1) → ReLU → Linear(512, 1)` on `delta_embedding`.
`lr`/`weight_decay` left at the loop defaults (no `CONFIG_OVERRIDES`; the nonlinearity + dropout +
early stopping control capacity, so the ridge head's heavy weight decay is unnecessary). MSE loss,
cosine schedule, early stopping on val Spearman fixed by the loop.

```python
# EDITABLE region of custom_mutation_pred.py — step 3: single-hidden-layer MLP head
class MutationPredictor(nn.Module):
    """Single-hidden-layer MLP over delta_embedding (mutant - WT).

    Architecture: Linear(embed_dim, hidden) -> Dropout -> ReLU -> Linear(hidden, 1)
    Uses delta_embedding so the network sees the mutation-induced shift
    in PLM representation space directly.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, embedding, delta_embedding):
        x = self.fc1(delta_embedding)
        x = self.dropout(x)
        x = F.relu(x)
        return self.fc2(x).squeeze(-1)
```
