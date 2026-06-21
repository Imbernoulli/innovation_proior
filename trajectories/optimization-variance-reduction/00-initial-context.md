## Research question

Finite-sum optimization: minimize `F(x) = (1/n) Σ_{i=1}^n f_i(x)` with stochastic first-order steps. Mini-batch SGD samples a batch each step and follows its gradient. The gradient variance scales like `1/b` in the batch size; at a solution the per-example gradients `∇f_i(x*)` average to zero but are not zero individually. Models, losses, data, learning rates, epoch budgets, and gradient helpers stay fixed. The question is how to improve convergence behavior in this finite-sum setting using the available first-order gradient helpers.

## Prior art / Background / Baselines

- **Full gradient descent.** Steps along the exact average gradient `∇F(x) = (1/n) Σ_i ∇f_i(x)`; with a constant step it contracts suboptimality by a fixed factor each iteration for strongly convex problems. Each step touches all `n` examples.

- **Mini-batch SGD.** Draws a batch and follows its gradient; each step costs `O(b)` regardless of `n` and gives an unbiased estimate of `∇F`.

- **SAG / SDCA.** Stores per-example information—a table of past gradients or dual variables—so each cheap step carries information about all `n` examples and achieves a linear rate at SGD-like cost. Requires an `O(n)`-sized table.

- **Heavy-ball / Nesterov momentum.** Keeps an exponential moving average of past gradients to smooth the descent direction. Widely used in practice in both deterministic and stochastic settings.

## Fixed substrate / Code framework

A single epoch-based training driver is frozen. It builds the model and data for one of three problems, constructs the `VarianceReductionOptimizer`, then calls `train_one_epoch` each epoch, evaluates on a held-out test set at fixed intervals, and tracks the best and final test metric. The driver counts gradient computations: one epoch of mini-batch steps costs `n/b`, and the optimizer reports `full_grad_count` (extra full-gradient passes) that the driver charges as another `n/b` each.

Three fixed helpers handle all gradient and loss computation; the optimizer must route through them for honest cost accounting:

- `compute_full_gradient(model, X, y, loss_type, l2_reg, device)` — exact `(1/n) Σ_i ∇f_i(x)`, one full pass, returned as per-parameter tensors.
- `compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)` — mini-batch gradient as per-parameter tensors.
- `compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)` — scalar batch loss for the reported `avg_loss`.

The learning rate `self.lr` and L2 coefficient `self.l2_reg` are fixed per problem. The same code must run all three problems. Parameter updates must be in-place (`p.data.add_(...)`).

## Editable interface

Only one region is editable: the `VarianceReductionOptimizer` class in `custom_vr.py`. The contract is two methods:

- `__init__(self, model, lr, l2_reg, loss_type, n_train, batch_size, device)` — set up whatever state the optimizer needs.
- `train_one_epoch(self, X_train, y_train)` — train for one pass over the data and return `{'avg_loss': ..., 'full_grad_count': ...}` (the latter optional). Hard constraint: `compute_full_gradient` may be called at most once per epoch.

The starting fill is vanilla mini-batch SGD:

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

Three problems; the same optimizer code runs on all:

- **logistic** — L2-regularized multinomial logistic regression on MNIST (convex, `n = 60K`, 20 epochs, `lr = 0.1`, `l2_reg = 1e-4`, batch 128).
- **mlp** — a 2-layer MLP on CIFAR-10 (non-convex, `n = 50K`, 40 epochs, `lr = 0.05`, `l2_reg = 1e-4`, batch 128).
- **conditioned** — L2-regularized linear regression on synthetic ill-conditioned data (strongly convex, condition number `κ = 100`, `n = 10K`, 30 epochs, `lr = 0.001`, `l2_reg = 1e-3`, batch 128).

Metrics: `best_test_accuracy` and `final_test_accuracy` for logistic and mlp (higher is better); `best_test_mse` and `final_test_mse` for conditioned (lower is better). Each problem runs over three seeds {42, 123, 456}. The task score is the geometric mean across the three problem settings.
