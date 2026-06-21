The whole task is a training recipe bolted onto a fixed two-layer ReLU MLP, so before I can claim anything clever I have to know what the *obvious* recipe does — that is the floor everything else is measured against. The target reads the isotropic Gaussian input through $r$ unknown orthonormal directions, $y = (1/\sqrt{r})\sum_i \mathrm{He}_3((U^{*\top}x)_i)$, with $\mathrm{He}_3(z)=z^3-3z$. Nothing in $x$ itself is informative; the entire problem is *which* directions the label looks along, and then the cubic link on those projections. The net can in principle do both jobs — its first-layer rows can rotate into $V^*=\mathrm{span}(U^*)$, and once they do the readout fits an $r$-dimensional cubic — so the game reduces to one question: can plain gradient descent get the first-layer rows to align with $V^*$ inside the 8000-step budget, and how badly does the link's information exponent fight it?

I propose, as the deliberate floor, **vanilla joint SGD**: leave every hook at the scaffold default — standard Kaiming-uniform init on both linear layers with zero biases, a fixed $n=4096$ Gaussian training set, plain SGD on both layers at $\eta=5\times10^{-2}$ with no momentum, no weight decay, no noise, and a single joint squared-loss update per step. Trivial is the *right* choice here precisely because its failure must be clean and legible, so the harder rungs can turn on exactly the mechanisms this one leaves off (frozen readout, spherical projection, closed-form solve) and have their effect be measurable against it.

What makes this the floor is not laziness but the information-exponent structure of the link, and it is worth tracing because it predicts the numbers. At a random Kaiming start the first-layer rows are essentially isotropic in $\mathbb{R}^d$, so a row's squared projection onto the fixed $r$-dimensional teacher subspace is about $r/d$ — with $d=128$ that means $\|\Pi^* w_i\|/\|w_i\| \sim \sqrt{r/d} = O(1/\sqrt{d})$, a neuron almost entirely orthogonal to the subspace it must find. The leaderboard's $\mathrm{subspace\_err}=\|P_{\hat U}-P_{U^*}\|_F$ measures exactly this failure at the matrix level: two rank-$r$ orthogonal projectors sharing no directions sit at Frobenius distance $\sqrt{2r}$ — about $2.0$, $2.4$, $2.8$ for $r\in\{2,3,4\}$ — which is what a recipe that learns *nothing* about the subspace should print.

Now the gradient that is supposed to rotate the rows. The first-layer row $w_i$ receives a step proportional to the correlation of the activation-weighted input with the label residual, $E[x\,\sigma'(\langle w_i,x\rangle)\,y]$. Stein's lemma, $E[x\,h(x)]=E[\nabla_x h(x)]$, splits this into a piece along $w_i$ itself (a rescaling, not alignment) plus $E[\sigma'(\langle w_i,x\rangle)\,\nabla_x y]$, the part that can move the row *toward* $V^*$. Expanding in Hermite tensors, the teacher's content lives entirely in $V^*$, so its $k$-th Hermite tensor is built from $k$ teacher directions; contracting it against $k$ copies of a row whose foot in $V^*$ is only $O(1/\sqrt{d})$ produces a scalar of size $O((1/\sqrt{d})^k)$. Every Hermite order costs another factor of $1/\sqrt{d}$, so the *lowest* surviving order dominates — and that order is the information exponent of the link. Here $g$ is a *pure cubic* in each coordinate, no linear or quadratic part, so the information exponent is exactly $3$. The useful part of the first gradient therefore has size $O((1/\sqrt{d})^{3-1})=O(1/d)$, and a step of size $\eta$ moves the row by $O(\eta/d)$ into the subspace, which against $d=128$ is essentially nothing.

The single-index theory of Ben Arous, Gheissari and Jagannath (2021) makes the consequence precise: for a target with information exponent $s$, one-pass SGD needs $\approx d^{s-1}$ steps to escape the uninformative equator, because the overlap starts at $O(1/\sqrt{d})$, the population correlation behaves like $\mathrm{overlap}^{s}$, and its derivative — the actual drift — is $O(\mathrm{overlap}^{s-1})$. For $s=3$ that is $d^2 = 128^2 \approx 16{,}384$ steps to find *one* direction, more than the entire 8000-step budget. The multi-direction story of Abbe, Boix-Adserà and Misiakiewicz (2023) is no kinder and explains why the floor should be flat rather than merely slow: SGD climbs teacher directions saddle-to-saddle, at cost $\approx d^{\max(\mathrm{Leap},2)}$, and a direction only leaves the equator once a lower-degree monomial couples it to an already-found one. But $g=(1/\sqrt{r})\sum_i \mathrm{He}_3((U^{*\top}x)_i)$ is a sum of *decoupled* cubics with no cross terms and no lower-degree terms at all — there is no $z_1z_2$ rung, no staircase to climb. Each of the $r$ directions must be found cold from its own degree-3 correlation; this is the worst case for the dynamic, and $r4$ is no easier per-direction than $r2$.

I should be honest about the second layer too, since the scaffold trains both jointly. While the first layer flails near the equator, the readout fits whatever near-random features it has. Random ReLU features form a fixed kernel, and a degree-3 target cannot be represented by $256$ such features in $d=128$ without paying the ambient dimension, so even a perfect convex fit captures almost none of the cubic. The MSE should therefore sit near the variance of the label: $\mathrm{Var}[\mathrm{He}_3(z)] = E[z^6-6z^4+9z^2] = 15-18+9 = 6$, and averaging $r$ independent such terms scaled by $1/\sqrt{r}$ gives $\mathrm{Var}[g]=(1/r)\cdot r\cdot 6 = 6$. So the signature of the floor is a $\mathrm{test\_mse}$ near $6$ (or worse, since untamed joint dynamics can inflate prediction variance) paired with a $\mathrm{subspace\_err}$ near $\sqrt{2r}$, collapsing $\mathrm{score}=\exp(-\mathrm{subspace\_err}^2/r)\cdot\exp(-\mathrm{test\_mse})$ to near zero — roughly $\exp(-2)\cdot\exp(-6)\approx 3\times10^{-4}$ as an optimistic bound. This is the weakest recipe I can run by construction: it has no mechanism to manufacture the missing third-order signal inside budget, and on a problem where finding the subspace *is* the task, it finds essentially nothing. The diagnosis already points at the next rung — this is an *optimization* failure, not a representation one, and the fix is to stop asking joint SGD to find the subspace through the cubic at all.

```python
# EDITABLE region of custom_strategy.py — step 1: vanilla joint SGD (the floor)
def init_model(model: nn.Sequential, config: TaskConfig) -> None:
    """Default Kaiming-uniform initialization for both linear layers."""
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def make_dataset(
    config: TaskConfig,
    teacher: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed Gaussian training set of size n=4096."""
    g = torch.Generator().manual_seed(seed)
    num_examples = 4_096
    x = torch.randn(num_examples, config.n_features, generator=g)
    y = teacher_outputs(x, teacher)
    return x, y


def get_optimizer_config(config: TaskConfig) -> dict[str, object]:
    """Plain SGD on both layers, no momentum, no weight decay."""
    return {
        "optimizer": "sgd",
        "lr_inner": 5e-2,
        "lr_outer": 5e-2,
        "wd_inner": 0.0,
        "wd_outer": 0.0,
        "momentum": 0.0,
        "noise_std": 0.0,
    }


def training_step(
    model: nn.Sequential,
    optimizer: torch.optim.Optimizer,
    optimizer_config: OptimizerConfig,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    step: int,
    config: TaskConfig,
) -> dict[str, float]:
    """Single squared-loss update applied to both layers jointly."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    preds = model(batch_x).view(-1)
    loss = ((preds - batch_y) ** 2).mean()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}
```
