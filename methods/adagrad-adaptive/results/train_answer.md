The problem is online and stochastic subgradient learning on high-dimensional, sparse, heavy-tailed data such as text and ranking problems. In these settings a single global learning rate like η/√t is forced to be the same for every coordinate, which is structurally mismatched to the data: rare-but-informative features finally fire after many rounds and receive a vanishingly small step, while common features keep jittering with the same decayed rate. The worst-case regret bound for ordinary online gradient descent is minimax optimal, so improvement cannot come from a better universal rate; it has to come from adapting to the actual geometry of the gradients.

The right move is to let the optimization geometry itself be data-driven. Instead of a fixed strongly convex proximal function chosen by hand, we grow a per-coordinate Mahalanobis metric from the gradients we have seen. This changes the regret bound so that it depends on the accumulated per-coordinate gradient mass rather than the isotropic Euclidean norm, and it naturally gives large steps to rarely seen features and small steps to frequently seen ones.

The method is AdaGrad, short for adaptive subgradient methods. The core idea is to maintain a running sum of squared gradients per coordinate and divide each coordinate's update by the square root of that accumulator. Concretely, at round t the update for coordinate i is x_{t+1,i} = x_{t,i} − η · g_{t,i} / √(Σ_{τ≤t} g_{τ,i}²). This diagonal preconditioner is not an empirical heuristic; it is the regret-minimizing diagonal metric under a trace budget. Minimizing the gradient term of the mirror-descent regret bound over diagonal matrices forces the preconditioner to be proportional to the per-coordinate root-sum-of-squares of gradients. Using the causal running version costs only a factor of two in the online gradient sum, yielding the regret bound R(T) ≤ √2 · D_∞ · Σ_i √(Σ_t g_{t,i}²), where D_∞ is the ℓ_∞ diameter of the feasible set. On heavy-tailed sparse data this sum can be O(log d) rather than O(√d), so the bound improves sharply over isotropic methods while degrading gracefully back to the Euclidean rate when the data is unstructured. The step size η can be set from the diameter alone and does not need to be tuned to the unknown gradient magnitudes.

A full-matrix variant replaces the diagonal accumulator with the matrix square root of the gradient outer-product matrix G_t = Σ_τ g_τ g_τ^T, giving a regret bound in terms of tr(G_T^{1/2}). The diagonal version is the practical O(d) instantiation used at scale, while the full-matrix version shows what structure is being exploited. In code the diagonal form is almost trivial: accumulate g² and divide by its square root.

```python
import numpy as np

class AdaGrad:
    """Diagonal AdaGrad: per-coordinate adaptive subgradient method.

    Maintains the running sum of squared gradients per coordinate and
    updates each coordinate by eta / sqrt(accumulated squared gradient).
    """

    def __init__(self, d, eta=1.0, eps=1e-10):
        self.eta = eta
        self.eps = eps
        self.state_sum = np.zeros(d)  # accumulator of squared gradients

    def step(self, x, g):
        self.state_sum += g * g
        return x - self.eta * g / (np.sqrt(self.state_sum) + self.eps)


def project(x, lower=-1.0, upper=1.0):
    """Project x onto an L-infinity box [lower, upper]^d."""
    return np.clip(x, lower, upper)


def online_learn(stream, d, eta=1.0):
    """Run diagonal AdaGrad on a stream of loss objects exposing .subgradient(x).

    stream: iterable of loss objects with method subgradient(x) -> np.ndarray
    d: dimension of the weight vector
    eta: global step size, set from the diameter of the feasible set
    """
    x = np.zeros(d)
    opt = AdaGrad(d, eta=eta)
    for loss in stream:
        g = loss.subgradient(x)
        x = opt.step(x, g)
        x = project(x)
    return x
```
