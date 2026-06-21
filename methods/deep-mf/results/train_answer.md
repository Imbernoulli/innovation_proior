I have a matrix $W^* \in \mathbb{R}^{d \times d'}$ and only a small, uniformly random handful of its entries, and I want to recover the rest. Fitting the observed entries is not the difficulty: because the observed set is a tiny fraction of the $n^2$ coordinates, the matching constraints define a large affine set of interpolants, and almost none of them agree with $W^*$ on the entries I never saw. The useful prior is that $W^*$ is low rank, so the right completion is the low-rank one — but I want that bias to emerge from the parameterization and the optimizer rather than from an explicit rank cap or an added penalty. The classical convex move is to relax rank to the nuclear norm $\|W\|_* = \sum_r \sigma_r(W)$ and minimize it subject to matching the observations; under incoherence and enough samples this recovers $W^*$ exactly. But the regime where implicit regularization matters is precisely the data-poor one, where the convex surrogate stops tracking minimum rank, so nuclear norm is a reference point, not the answer. The empirical fact that keeps the question open is shallow factorization: writing $W = W_2 W_1$ with the hidden dimension left full, initializing near zero, and running small-step gradient descent on observed-entry squared loss tends to return a low-rank completion with no rank cap at all. Gunasekar and coauthors conjectured this depth-2 bias is minimum nuclear norm and proved it in a commuting positive-semidefinite sensing case. The tempting next guess — that more factors act like a Schatten-$p$ quasi-norm with $p < 1$, closer to rank — turns out to be wrong, and the way it fails points to the real mechanism.

I propose Deep Matrix Factorization: complete the matrix by fitting the observed entries with a full-dimensional depth-$N$ linear product $W = W_N W_{N-1} \cdots W_1$, started near zero and trained to tiny training loss with no explicit rank constraint, and rely on the fact that depth reshapes the optimization trajectory itself toward low rank. The load-bearing object is the singular-value dynamics. Under a balanced initialization (the factors satisfy $W_{j+1}^\top W_{j+1} = W_j W_j^\top$, exact for identity-style starts and well approximated by small Gaussian factors), the end-to-end gradient flow of the product obeys $\dot W = -\sum_{j=1}^N [WW^\top]^{(j-1)/N}\,\nabla \ell(W)\,[W^\top W]^{(N-j)/N}$. Taking an analytic SVD $W(t) = U(t) S(t) V(t)^\top$ and using that the singular vectors keep unit norm, so $\langle u_r, \dot u_r\rangle = \langle \dot v_r, v_r\rangle = 0$, the value velocity is simply $\dot\sigma_r = u_r^\top \dot W\, v_r$. Substituting the product ODE, the $j$th term contributes $-(\sigma_r^2)^{(j-1)/N}\langle \nabla\ell(W), u_r v_r^\top\rangle (\sigma_r^2)^{(N-j)/N}$; the two exponents always sum to $(N-1)/N$ independent of $j$, so all $N$ terms collapse into one,
$$\dot\sigma_r = -N\,(\sigma_r^2)^{1 - 1/N}\,\langle \nabla\ell(W),\, u_r v_r^\top\rangle.$$
This is the entire engine. For $N = 1$ the multiplier is $1$ and direct gradient flow throttles no mode by its size. For $N \ge 2$ the multiplier is $N\,\sigma_r^{\,2 - 2/N}$: a mode's growth rate is suppressed when it is small and amplified when it is already large, and the exponent grows with depth, so the separation between strong and weak modes sharpens as $N$ increases. Near-zero initialization places every singular value in the throttled region, and only modes that maintain alignment with the gradient ever switch on. A sign lemma — the diagonal scalar ODE $\dot s = (s^2)^\beta g(t)$ with $\beta = (N-1)/N$ cannot cross zero before any finite-time blow-up — lets me keep the $\sigma_r$ nonnegative.

Why depth at least three rather than two is settled by a one-measurement toy. Once the singular vectors are stationary and aligned (the converse alignment statement, that stationary vectors force the off-diagonal part of $U^\top\nabla\ell V$ to vanish, follows from the off-diagonal SVD dynamics $U^\top\dot U\,S - S\,V^\top\dot V = -\bar I \circ G \circ [U^\top\nabla\ell\, V]$ whose coupling factor $G$ is nonzero away from singular-value collisions), the common time factor cancels between two modes, and integration gives $\sigma_1 = \alpha_{12}\sigma_2 + \text{const}$ for $N=1$, $\sigma_1 = \text{const}\cdot\sigma_2^{\alpha_{12}}$ for $N=2$, and $\sigma_1 = (\alpha_{12}\sigma_2^{-(N-2)/N} + \text{const})^{-N/(N-2)}$ for $N \ge 3$, where $\alpha_{12}$ is the ratio of measurement strengths felt by the two modes. When the first mode is weaker ($0 < \alpha_{12} < 1$), depth one lets it grow linearly with the stronger mode, depth two only polynomially, and depth three or more drives it to a finite asymptote as the stronger mode runs away — true saturation, which is the low-rank bias in motion. That is why I take $N = 3$: it is the cheapest depth that produces saturation rather than a mere polynomial gap, and depth four was empirically similar, not worth the extra cost.

I deliberately do not present this as minimizing some fixed norm, because that story breaks. In commuting PSD sensing the same proof line that gives the depth-2 nuclear-norm result extends to every $N \ge 3$: the diagonal product integrates to $\widetilde W_{kk}(t) = \alpha^N(1 + (N-2)\alpha^{N-2}\widetilde A^\dagger_{kk}(s(t)))^{-N/(N-2)}$, and letting $\alpha \to 0^+$, any surviving entry forces its complementary-slackness slack to zero, yielding a primal-feasible PSD matrix with a dual-feasible sequence and vanishing duality gap for $\min\langle I, W\rangle$ subject to $\mathcal A(W) = y,\, W \succeq 0$ — i.e. a minimum-nuclear-norm point, since trace equals nuclear norm on PSD matrices. So Schatten-$p$ with $p < 1$ is already excluded as the depth bias here. Worse, that nuclear-norm limit need not even be a local minimizer of any Schatten-$p$ quasi-norm: with constraints $W_{11} = W_{22}$ and $W_{11} = W_{kk} + 1$ for $k \ge 3$, the min-trace solution is $\mathrm{diag}(1,1,0,\dots,0)$, but adding $\epsilon$ to the $(1,2)$ and $(2,1)$ entries keeps the constraints and PSD-ness while moving the nonzero eigenvalues to $1\pm\epsilon$, and strict concavity of $x^p$ gives $(1+\epsilon)^p + (1-\epsilon)^p < 2$. The bias is a property of the trajectory, not of any single penalty, which is exactly why I build it from the dynamics.

The remaining design choices fall straight out of the derivation. I keep the hidden dimension full, because capping it would impose low rank explicitly and destroy the very phenomenon I am isolating. I use near-zero initialization so every mode starts in the throttled regime; concretely each factor is Gaussian with per-factor standard deviation $\text{init\_scale}^{1/N}\, n^{-1/2}$, which puts the end-to-end product near the origin and keeps the factors approximately balanced so the end-to-end ODE I relied on actually governs the run. I use bias-free linear factors and form the product by starting from the first weight transposed and applying each later layer in turn. I train full-batch observed-entry mean squared error down to about $10^{-6}$, since the choice among interpolating solutions is made precisely as the loss is driven to zero. The canonical optimizer path is a $\mathrm{GroupRMSprop}$ with global gradient-norm scaling, $\epsilon = 10^{-4}$, and learning rate $10^{-3}$; SGD and Adam exist as options but Adam is not the default recipe.

```python
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class GroupRMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-6):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        defaults = dict(lr=lr, alpha=alpha, eps=eps, adjusted_lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            state = self.state
            if len(state) == 0:
                state["step"] = 0
                state["square_avg"] = torch.tensor(0.0)

            square_avg = state["square_avg"]
            alpha = group["alpha"]
            square_avg.mul_(alpha)
            state["step"] += 1

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("GroupRMSprop does not support sparse gradients")
                square_avg.add_((1 - alpha) * grad.pow(2).sum().cpu().float())

            avg = square_avg.div(1 - alpha ** state["step"]).sqrt_().add_(group["eps"])
            lr = group["lr"] / avg
            group["adjusted_lr"] = lr

            for p in group["params"]:
                if p.grad is not None:
                    p.data.add_(-lr.to(p.grad.data.device) * p.grad.data)

        return loss


class DeepMatrixFactorization:
    def __init__(self, depth=3, init_scale=1e-3, lr=1e-3, train_thres=1e-6):
        if depth < 2:
            raise ValueError("depth must be at least 2")
        self.depth = int(depth)
        self.init_scale = float(init_scale)
        self.lr = float(lr)
        self.train_thres = float(train_thres)

    def _build(self, n, device):
        layers = [nn.Linear(n, n, bias=False) for _ in range(self.depth)]
        scale = (self.init_scale ** (1.0 / self.depth)) * (n ** -0.5)
        for layer in layers:
            nn.init.normal_(layer.weight, std=scale)
        return nn.Sequential(*layers).to(device)

    @staticmethod
    def _e2e(model):
        weight = None
        for layer in model.children():
            weight = layer.weight.t() if weight is None else layer(weight)
        return weight

    def recover(self, observed_values, observed_mask, n, rank_hint, device, max_iters, log_iters):
        model = self._build(n, device)
        optimizer = GroupRMSprop(model.parameters(), self.lr, eps=1e-4)
        mask = observed_mask.to(device)
        target = observed_values.to(device)
        denom = max(int(mask.sum().item()), 1)

        for it in range(max_iters):
            e2e = self._e2e(model)
            loss = ((e2e - target) * mask).pow(2).sum() / denom
            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                if loss.item() <= self.train_thres:
                    break

            optimizer.step()

        with torch.no_grad():
            return self._e2e(model).detach().cpu()


def build_strategy():
    return DeepMatrixFactorization(depth=3, init_scale=1e-3, lr=1e-3, train_thres=1e-6)
```
