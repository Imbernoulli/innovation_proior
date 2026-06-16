## Research question

Finite-sum optimization, `min_x F(x) = (1/n) Σ_{i=1}^n f_i(x)`, run by stochastic first-order steps.
Vanilla mini-batch SGD draws a random batch each step and follows its gradient, whose variance scales
like `1/b` in the batch size and — this is the part that bites — does **not** vanish as the iterate
approaches a solution: at a minimizer the per-example gradients `∇f_i(x*)` are nonzero even though their
average is zero, so a constant-step SGD rattles inside a noise ball and is forced to anneal its step to
`O(1/t)`, collapsing the rate to sublinear. The single thing being designed is the **variance-reduction
mechanism** — what auxiliary state (a snapshot, a recursive correction, a momentum estimate) the
optimizer keeps so that its update direction has lower variance than a raw mini-batch gradient, ideally
with variance that shrinks toward zero as the iterate settles. Everything else — the models, the losses,
the data, the learning rates, the epoch budgets, the gradient helpers — is fixed.

## Prior art before the first rung (the variance-reduction lineage)

The first rung reacts to the family of finite-sum accelerators that preceded it; the fixed substrate
below is the harness they all plug into.

- **Full gradient descent.** Steps along `∇F(x) = (1/n) Σ_i ∇f_i(x)`; with a constant step it contracts
  the suboptimality by a fixed factor each iteration (linear rate for strongly convex `F`). Gap: every
  single step touches all `n` examples, so each iteration costs a full pass — unaffordable per step at
  large `n`.
- **Mini-batch SGD (the scaffold default).** Draws a batch, follows its gradient, costs `O(b)` per step
  independent of `n`, and is an unbiased estimate of `∇F`. Gap: the gradient variance has a floor that
  survives at the optimum, so a constant step leaves a noise ball and convergence degrades to `O(1/t)`
  unless the step is annealed — fast steps, slow convergence.
- **SAG / SDCA (Le Roux–Schmidt–Bach 2012; Shalev-Shwartz–Zhang 2013).** First to get a *linear* rate at
  SGD-like per-step cost on finite sums, by storing per-example information — a table of `n` past
  gradients (SAG) or `n` dual variables (SDCA) — so each cheap step carries information about all `n`
  examples. Gap: an `O(n)`-sized table, which is infeasible for large `n` or for models where the
  per-example gradient is not a cached scalar (neural nets), and a tangled convergence story.
- **GD-with-momentum (heavy ball / Nesterov).** Keeps an exponential moving average of past gradients to
  smooth the descent direction. Widely used and effective in practice, but in the *stochastic* setting it
  has no general theorem improving the convergence rate over plain SGD — the noise nullifies the
  averaging benefit. Gap: a heuristic with unexplained success under noise.

## The fixed substrate

A single epoch-based training driver is frozen and must not be touched. It builds the model and data for
one of three problems, constructs the `VarianceReductionOptimizer`, then for each epoch calls
`train_one_epoch`, evaluates on a held-out test set at a fixed interval, and tracks the best and final
test metric. The driver counts gradient computations: one epoch of mini-batch steps costs `n/b`, and the
optimizer can report a `full_grad_count` (extra full-gradient passes) that the driver charges as another
`n/b` each. Three FIXED helpers do all gradient and loss computation; the optimizer must route through
them so the cost accounting is honest:

- `compute_full_gradient(model, X, y, loss_type, l2_reg, device)` — the exact `(1/n) Σ_i ∇f_i(x)`, one
  full pass, returned as a list of per-parameter tensors.
- `compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)` — the mini-batch gradient,
  list of per-parameter tensors.
- `compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)` — the scalar batch loss (used for
  the reported `avg_loss`).

The learning rate `self.lr` and the L2 coefficient `self.l2_reg` are handed in per problem and are fixed;
the same code must run all three problems. Parameter updates must be in-place (`p.data.add_(...)`).

## The editable interface

Exactly one region is editable — the `VarianceReductionOptimizer` class in `custom_vr.py` (lines
286–370). The contract is two methods:

- `__init__(self, model, lr, l2_reg, loss_type, n_train, batch_size, device)` — set up whatever state the
  variance-reduction mechanism needs (snapshot parameters, running gradient estimates, buffers,
  counters).
- `train_one_epoch(self, X_train, y_train)` — train for one pass over the data and return a dict with at
  least `'avg_loss'`, optionally `'full_grad_count'`. Hard constraint: `compute_full_gradient` may be
  called **at most once per epoch**.

Every method on the ladder is a fill of this same contract. The starting point is the scaffold default:
**vanilla mini-batch SGD**, no variance reduction. Each later method replaces exactly this class.

```python
# EDITABLE region of custom_vr.py (lines 286-370) -- default fill: vanilla mini-batch SGD
class VarianceReductionOptimizer:
    """Variance reduction strategy for finite-sum optimization.

    Default implementation: vanilla mini-batch SGD (no variance reduction).
    """

    def __init__(self, model: nn.Module, lr: float, l2_reg: float,
                 loss_type: str, n_train: int, batch_size: int,
                 device: torch.device):
        self.model = model
        self.lr = lr
        self.l2_reg = l2_reg
        self.loss_type = loss_type
        self.n_train = n_train
        self.batch_size = batch_size
        self.device = device
        self.params = list(model.parameters())

    def train_one_epoch(self, X_train: torch.Tensor,
                        y_train: torch.Tensor) -> dict:
        self.model.train()
        n = X_train.size(0)
        indices = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            idx = indices[start:end]
            Xb = X_train[idx].to(self.device)
            yb = y_train[idx].to(self.device)

            # Standard SGD: compute stochastic gradient and update
            self.model.zero_grad()
            loss = compute_loss_on_batch(
                self.model, Xb, yb, self.loss_type, self.l2_reg
            )
            loss.backward()
            with torch.no_grad():
                for p in self.params:
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=-self.lr)

            total_loss += loss.item()
            n_batches += 1

        return {"avg_loss": total_loss / max(n_batches, 1)}
```

## Evaluation settings

Three problems spanning the convexity spectrum, the same optimizer code on all three:

- **logistic** — L2-regularized multinomial logistic regression on MNIST (convex, `n = 60K`, 20 epochs,
  `lr = 0.1`, `l2_reg = 1e-4`, batch 128).
- **mlp** — a 2-layer MLP on CIFAR-10 (non-convex, `n = 50K`, 40 epochs, `lr = 0.05`, `l2_reg = 1e-4`,
  batch 128).
- **conditioned** — L2-regularized linear regression on synthetic ill-conditioned data (strongly convex,
  condition number `κ = 100`, `n = 10K`, 30 epochs, `lr = 0.001`, `l2_reg = 1e-3`, batch 128).

Metrics: `best_test_accuracy` and `final_test_accuracy` for logistic and mlp (higher is better);
`best_test_mse` and `final_test_mse` for conditioned (lower is better). Each problem is run over three
seeds {42, 123, 456}. The task score is the geometric mean across the three problems' settings.
