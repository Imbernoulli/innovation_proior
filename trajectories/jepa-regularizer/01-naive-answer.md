**Problem.** Joint-embedding SSL trains the encoder so two augmented views of an image land at the same
embedding. The bare invariance objective — "make the views agree" — has the constant map as a global
minimizer: send every input to one vector and every paired distance is zero. Collapse. The online
linear probe reads the *frozen, detached* backbone features, so a collapsed encoder leaves the probe
nothing to separate and `val_acc` sits near the 10-class chance floor.

**Key idea (the floor).** Fill the contract with the barest invariance term and *no* anti-collapse
mechanism: the mean squared distance between the two views' projected embeddings, `F.mse_loss(z1, z2)`.
This is exactly the attract-only half of a contrastive pair loss (the margin repulsion deleted) — a web
of zero-rest-length springs whose lowest-energy state is everything coincident. It is the lower bound by
construction; running it converts "collapse is a theoretical worry" into a measured hole in `val_acc`
that every later regularizer must fill.

**Step-1 edit.** Replace the placeholder `CustomRegularizer` with the MSE-only loss. No
`CONFIG_OVERRIDES` (the default `2048 → 2048` projector; reshaping the projector cannot rescue an
objective with no anti-collapse term). No embedding normalization, no epsilon, no stabilizer — the
floor must carry no crutch.

**What to watch.** All three backbones run the identical loss; collapse is an objective property, not a
capacity one, so I expect a tight low band near chance across ResNet-18/34/50, not one architecture
rescued. The gap below the real regularizers is the budget step 2 has to recover — and its kind is
already forced: add a term that makes the collapsed (and dimensionally collapsed) configuration a
*high*-loss state.

```python
class CustomRegularizer(nn.Module):
    """Naive MSE-only regularizer (no anti-collapse). Lower-bound baseline."""

    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        loss = F.mse_loss(z1, z2)
        return {"loss": loss, "invariance_loss": loss}


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
CONFIG_OVERRIDES = {}
```
