We are solving online convex optimization with subgradients: at each round the learner commits to a point, suffers a convex loss, and receives a subgradient. The goal is sublinear regret against the best fixed decision in hindsight. On the high-dimensional sparse problems that matter here, such as text classification with bag-of-words features, each example only activates a few coordinates, and the rare features are often the most informative. Existing subgradient methods use a single step size for every coordinate, so a feature seen once receives the same update as a feature seen thousands of times. That global rate cannot take larger steps on rare, informative coordinates, and the isotropic regret bound scales with the full ambient dimension even though the gradient mass is concentrated on a handful of coordinates.

The leverage is the proximal function in mirror descent. For a fixed proximal function, the regret bound is governed by the dual norms of the gradients, which means it is governed entirely by the choice of that proximal function. Instead of fixing it in advance, we can treat it as a data-dependent object and choose it to make the gradient term of the bound small. Restricting to a diagonal Mahalanobis metric makes the problem tractable: the hindsight-optimal diagonal preconditioner scales each coordinate by the accumulated L2 norm of its gradient history, which is the square root of the running sum of squared gradients. Using the causal running norm online only costs a factor of two compared with the best fixed preconditioner chosen in hindsight, so the regret becomes proportional to the sum of those coordinate-wise accumulated norms rather than to the square root of the dimension.

The method is AdaGrad, the adaptive subgradient method. It maintains one scalar accumulator per coordinate, adding the square of each observed subgradient. The update for coordinate i divides the current subgradient by the square root of that accumulator plus a small floor. Frequently active coordinates accumulate large sums and therefore receive small effective steps; rarely active coordinates accumulate small sums and receive large effective steps when they finally fire. This per-coordinate scaling is not a heuristic: it is the diagonal preconditioner that minimizes the mirror-descent regret bound, and the online version is provably competitive with the best such preconditioner in hindsight.

The same idea extends to a full-matrix preconditioner based on the matrix square root of the outer-product matrix of gradients, which is the optimal trace-budget preconditioner and can capture correlations between coordinates. But maintaining, square-rooting, and inverting a d-by-d matrix is infeasible when the dimension is in the millions, so the diagonal form is the one that ships. The diagonal version preserves the main benefit: it adapts to sparse, heavy-tailed feature geometry while remaining linear in both time and memory. It also recovers standard methods at the extremes: an isotropic time-scaled metric gives ordinary projected gradient descent with its 1/sqrt(t) schedule, and the dual-averaging form replaces the single global sqrt(t) rate with per-coordinate rates while keeping the sparse soft-thresholding structure intact.

For an unconstrained problem with no composite regularizer, the implementation is a single accumulator, a square root, and a coordinate-wise divide. Here is a compact AdaGrad optimizer that fits into the standard online subgradient harness:

```python
import torch


class AdaGrad:
    """Diagonal AdaGrad.

    Maintains a per-coordinate sum of squared gradients and applies
    x <- x - lr * g / (sqrt(sum_of_squares) + eps).
    """

    def __init__(self, params, lr=1e-2, eps=1e-10,
                 initial_accumulator_value=0.0):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.state = {
            id(p): {
                "step": 0,
                "sum": torch.full_like(p, initial_accumulator_value),
            }
            for p in self.params
        }

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[id(p)]
            st["step"] += 1
            st["sum"].addcmul_(g, g, value=1.0)
            std = st["sum"].sqrt().add_(self.eps)
            p.addcdiv_(g, std, value=-self.lr)


# Example online loop using the optimizer above:
def run_online(model, loss_fn, data_stream, learner):
    for inputs, targets in data_stream:
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        learner.step()
```

The small additive constant eps is only there to avoid division by zero before a coordinate has accumulated any gradient mass; active coordinates quickly dominate it. The global learning rate remains a single scalar because the per-coordinate adaptation already lives in the accumulated denominator. On sparse heavy-tailed data, this turns a dimension-dependent sqrt(d T) regret into a bound closer to (log d) sqrt(T), while retaining the simplicity and cost of vanilla subgradient descent.
