SI's measured line split exactly where its trajectory importance should have. On Split-MNIST it was excellent — 0.9852 average, five short binary tasks held almost perfectly — and on Split-CIFAR100 it was mid-pack at 0.5363. But on Permuted-MNIST it collapsed to 0.4474, and the per-context line is a monotone slide $\{0.79,0.72,0.66,0.61,0.53,0.40,0.29,0.19,0.14,0.14\}$: the late contexts fall to chance for a 10-way readout *and* the earliest erode (context 1 ends at 0.79, not the $\sim0.97$ a held task should keep). That is both failure modes at once, and the cause is structural — SI's path-integral $\Omega$ only ever *adds* across ten contexts, the SGD overestimate compounds, and there is no decay, so the accumulated, scale-mismatched springs over-constrain the net while no longer holding the early contexts. The diagnosis is clean: SI's importance is *unbounded and grows*. The fix has to be the importance estimator itself — a per-parameter curvature that is bounded, anchored at a point, and PSD by construction, not a trajectory sum that runs away.

I propose **EWC (Elastic Weight Consolidation)**: derive importance as the precision of an approximate posterior instead of reading it off the path. Training is MAP estimation — by Bayes $\log p(\theta\mid D)=\log p(D\mid\theta)+\log p(\theta)-\log p(D)$, and cross-entropy *is* a negative log-likelihood, so minimizing loss and finding the most probable $\theta$ are the same act. Splitting the data into the finished context $A$ and the current $B$ and applying Bayes again,
$$\log p(\theta\mid D_A,D_B)=\log p(D_B\mid\theta)+\log p(\theta\mid D_A)-\log p(D_B\mid D_A),$$
the last term is $\theta$-independent and vanishes under any gradient. The first term is the context-$B$ loss I would minimize anyway; the middle term $\log p(\theta\mid D_A)$ is the *posterior over weights given $A$*, and **all** of $A$'s information has been absorbed into that one object. So if I had it I would be done. The catch is that it is a distribution over millions of weights, intractable — but a Gaussian approximation gives me everything I need at once: its negative log density is a quadratic in $\theta$ (the spring penalty I already want), and once diagonal, its precision is exactly per-weight stiffness.

To get the Gaussian I use Laplace's method, which is natural here because I trained $A$ to a (local) optimum $\theta^*_A$ where the gradient of $-\log p(\theta\mid D_A)$ vanishes. Taylor-expanding to second order around $\theta^*_A$, the constant drops, the first-order term is zero, and
$$-\log p(\theta\mid D_A)\approx \text{const}+\tfrac12(\theta-\theta^*_A)^\top H(\theta-\theta^*_A),$$
a Gaussian with mean $\theta^*_A$ and precision $H$ — and because $\theta^*_A$ is a minimum, $H$ is PSD, which a precision had better be. The diagonal of $H$ reads, per weight, how sharply $A$'s negative log posterior rises as I move it: large curvature means $A$ is sensitive, so hold it stiff; near-zero means $A$ does not care, so leave it loose for $B$. Unlike SI's running sum, this is evaluated *at one point* with bounded curvature — nothing compounds with how many steps a context took.

The full Hessian is a millions-by-millions matrix I cannot form, and replacing it is where the Fisher information earns its place. The Fisher $F=\mathbb{E}_{y\sim p_\theta(y\mid x)}[(\nabla_\theta\log p_\theta(y\mid x))(\nabla_\theta\log p_\theta(y\mid x))^\top]$ — the expected outer product of the score, with $y$ drawn from the *model's own* predictive distribution — equals the expected Hessian of the NLL. The identity is one line: for a single coordinate, $\partial^2\log p/\partial\theta^2=(1/p)\,\partial^2 p/\partial\theta^2-(\partial\log p/\partial\theta)^2$; taking the expectation over $y\sim p$, the first piece is $\partial^2/\partial\theta^2\!\sum_y p=\partial^2/\partial\theta^2(1)=0$ because probabilities sum to one for every $\theta$, leaving $\mathbb{E}[-\partial^2\log p/\partial\theta^2]=\mathbb{E}[(\partial\log p/\partial\theta)^2]$. So under the model distribution the curvature is an *average of squared first-order gradients* — no second derivatives — and the outer-product form is automatically PSD, where the raw empirical Hessian can be indefinite and would hand me negative, meaningless "stiffnesses." (This is also why EWC needs no floor: SI's running sum could go negative under noise and required $\epsilon$; a sum of squares cannot.) I then keep only the *diagonal*, asserting a factorized Gaussian — one independent quadratic per weight. It is lossy, since weights surely covary in their effect on $A$, but it is the price of staying linear in the number of parameters, and the diagonal already carries the per-weight curvature $F_i$ I most need.

Assembling, minimizing $-[\log p(D_B\mid\theta)+\log p(\theta\mid D_A)]$ under the Laplace-Fisher approximation gives the loss I descend while training $B$,
$$L(\theta)=L_B(\theta)+\frac{\lambda}{2}\sum_i F_i\,(\theta_i-\theta^*_{A,i})^2,$$
a quadratic anchor to $\theta^*_A$ with per-coordinate stiffness the diagonal Fisher — important weights held nearly rigid, unimportant ones free for $B$. The leading $\tfrac12$ is the honest Gaussian quadratic-form factor that fell out of the Taylor expansion; this is precisely the half SI deliberately dropped (SI had absorbed it into its $\Delta^2$ normalization, whereas here it is the genuine Gaussian coefficient). The clean derivation says $\lambda$ should be $A$'s sample size $N$, but the diagonal Laplace approximation is overconfident, so I let $\lambda$ be the tunable knob — the per-benchmark $\texttt{reg\_strength}$. Computing the diagonal Fisher is the one place I want to be exact about the expectation over $y$: it averages the squared score over $y$ drawn from the *model's own* predictive distribution, not over the true labels. So per input I run a forward pass to get $p=\text{softmax}(\text{logits})$, and for each class $k$ I backprop the per-class NLL $-\log p_\theta(y=k\mid x)$, square the gradient coordinatewise, weight by $p_k$, and sum over $k$ — the exact inner expectation, since the expectation over a categorical is the probability-weighted sum over outcomes. Averaged over a couple hundred eval-mode single-example passes (the harness caps at 200, with eval mode so dropout and BN do not corrupt the estimate), that is the diagonal Fisher, which is exactly the scaffold's default fill.

One wrinkle the harness forces me to state plainly. Canonical EWC keeps a *separate* Fisher and anchor per past context and sums explicit springs, each anchored at its own $\theta^*_t$. This loop instead sums every returned Fisher into a single $\texttt{\_custom\_importance}$ buffer and re-snapshots $\texttt{\_custom\_prev\_params}$ to the *latest* boundary, so all the accumulated stiffness is anchored at the most recent optimum, not at each context's own. For two contexts these coincide; from the third on they differ, and that divergence is exactly what the next rung is built around. Against SI's line I expect EWC to land a hair under SI on Split-MNIST (the endpoint Fisher is near zero at a cleanly converged binary task, so it under-weights what SI's trajectory held perfectly), to *not* exhibit SI's slide on Permuted-MNIST but instead stay roughly flat and jump well above 0.4474, and to near-tie SI in the mid-0.5s on Split-CIFAR100. The remaining weakness, already visible in the derivation, is that the loop sums Fishers without bound and re-anchors only at the latest — which over a long sequence will itself start to over-constrain, and that is the opening for the rung after this.

```python
# EDITABLE region of custom_regularization.py — step 2: EWC (diagonal Fisher + quadratic penalty)
def estimate_importance(model, dataset, prev_params, device):
    """EWC: Diagonal Fisher Information matrix via squared gradients."""
    est_fisher = {}
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher[n] = p.detach().clone().zero_()

    mode = model.training
    model.eval()

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_samples = min(len(data_loader), 200)

    for idx, (x, y) in enumerate(data_loader):
        if idx >= n_samples:
            break
        x = x.to(device)
        output = model(x)
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1)
        for label_index in range(output.shape[1]):
            label = torch.LongTensor([label_index]).to(device)
            negloglikelihood = F.cross_entropy(output, label)
            model.zero_grad()
            negloglikelihood.backward(
                retain_graph=True if (label_index + 1) < output.shape[1] else False
            )
            for gen_params in model.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            est_fisher[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)

    est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}

    model.train(mode=mode)
    return est_fisher


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """EWC: 0.5 * sum(fisher * (param - prev_param)^2)."""
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    fisher = importance_dict[n]
                    prev = prev_params_dict[n]
                    losses.append((fisher * (p - prev) ** 2).sum())
    if losses:
        return 0.5 * sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)
```
