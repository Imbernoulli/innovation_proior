Back to the 94% bar, where the headline record lives and where I left it at airbench94: 3.83 A100-seconds, 3.29 compiled. Every algorithmic lever I pulled to get there — whitening, Dirac init, the 64× bias learning rate, Lookahead, multi-crop TTA, alternating flip — has been about *initialization*, *data*, or *inference*. The one thing left almost untouched is the *update rule itself*. The conv weights, which are the vast bulk of the parameters and the part that actually learns the features, are still updated by plain Nesterov SGD: take the momentum-smoothed gradient and step in its direction. The geometric issue is that a convolution's weight, reshaped to 2-D, is a matrix mapping input features to output features, and its gradient is also a matrix with a *spectrum* — singular values that can be wildly unequal. SGD steps in the raw gradient direction, so the update is dominated by the gradient's top singular directions: the few directions where the gradient happens to be large get most of the step, and the many directions where it is small barely move. Early training therefore over-updates a handful of dominant directions in each weight matrix and starves the rest. In a long run the starved directions eventually get their turn; in a ~10-epoch run they may never catch up. What I want instead is an update that treats every direction of the weight matrix evenly — that moves the same distance along every direction the gradient points in, regardless of that direction's gradient magnitude.

I propose the **Muon** optimizer — MomentUm Orthogonalized by Newton-schulz. The mathematical object that equalizes the directions is the *orthogonalization* of the gradient: if the momentum-smoothed gradient matrix $G$ has SVD $G = U S V^\top$, then $U V^\top$ — $G$ with all its singular values flattened to one — points in the same directions as $G$ but gives every direction equal step length. Updating with $U V^\top$ instead of $G$ is a steepest-descent step under the spectral (operator) norm rather than the Euclidean norm, the natural geometry for a *matrix* parameter, and it equalizes per-direction learning the way the 64× bias learning rate equalized learning between bias and weight parameters — the same theme on a different axis. The obstacle is that an SVD of every conv weight every step is far too slow for a 2.59-second budget; the record would drown in eigendecompositions. So I compute $U V^\top$ SVD-free with a **Newton–Schulz iteration**: a fixed degree-5 polynomial in $G$, applied a few times, that drives the singular values toward 1 while leaving the singular vectors alone. Starting from $G$ normalized so its top singular value is $\le 1$, iterating $X \leftarrow aX + (b\,XX^\top + c\,(XX^\top)^2)X$ with coefficients $(a,b,c) = (3.4445,\,-4.7750,\,2.0315)$ pushes the spectrum toward 1 in just a handful of steps, and it is all matmuls, which the GPU eats for breakfast. I do not need clean convergence to exactly $U V^\top$: I pick the coefficients to *maximize the slope at zero* (so small singular values shoot up fast) even past the point where the iteration stops converging to exactly 1 everywhere, which leaves something like $U S' V^\top$ with $S'$ spread roughly over $(0.5, 1.5)$ — not a clean orthogonalization, but in practice it does not hurt at all, and three iterations suffice.

So each Muon step is: accumulate momentum on the gradient, orthogonalize the momentum buffer via Newton–Schulz, then step. Two scoping choices make it work. First, I apply Muon only to the 4-D conv filters (reshaped to 2-D) — the matrices where the spectral argument bites — and keep the cheap, well-behaved scalars (the whitening bias, the norm biases, the linear head) on plain SGD, since orthogonalization is meaningless for them. Second, one piece pairs naturally with orthogonalized updates: because the update direction is now scale-free (singular values $\approx 1$), the *magnitude* of each weight matters independently, so before each step I renormalize the weight to a fixed norm, `p.data.mul_(len(p.data)**0.5 / p.data.norm())`, keeping every filter matrix at a controlled scale so the equal-length updates land consistently. Orthogonalized updates equalize per-direction learning across each weight matrix, so the low-singular-value directions a short SGD run leaves starved all train at once — clearing 94% in fewer steps and dropping the record below 3.29 seconds. The two hedges are real and both repay themselves: the Newton–Schulz iteration is approximate (it yields $U S' V^\top$, not $U V^\top$), and it adds a few matmuls per weight per step, so the per-step cost rises and must be bought back by needing fewer steps. The wager of this whole ladder has been that the right *structural* insight beats brute force, and the update geometry is the last structure left to fix; orthogonalizing the matrix updates buys back more steps than the iteration costs, making this the fastest documented training to 94% on CIFAR-10, and that is where the ladder ends.

```python
@torch.compile
def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)            # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def step(self):
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None: continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                p.data.mul_(len(p.data)**0.5 / p.data.norm())                          # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape)  # whiten the update
                p.data.add_(update, alpha=-lr)                                         # take a step

# conv filters -> Muon; biases/head -> SGD
filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True)
```
