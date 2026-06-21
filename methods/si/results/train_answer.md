A network trained on a sequence of tasks $\mu = 1, 2, 3, \dots$ that arrive one at a time, with only the current task's loss $L_\mu$ ever in front of it, forgets catastrophically: descending $L_\mu$ drags the shared weights wherever lowering $L_\mu$ demands, and those same weights were carrying the earlier tasks, so the earlier losses $L_\nu$ quietly climb back up. After a handful of tasks the network solves only the most recent one and has reverted to chance on the first. What I actually want to minimize is the sum over everything ever seen, $L = \sum_\mu L_\mu$, but I am forced to descend one summand at a time, and the obvious cure — throw all the data back in and train jointly — is exactly what the continual setting forbids, since the old data is gone and stockpiling it would cost memory growing with the number of tasks. McCloskey and Cohen pinned this failure down for backprop networks in 1989, and French in 1999 traced why it is *catastrophic* rather than gradual: the representation is shared and distributed, so any weight change that helps the new task tends to corrupt the old one.

I cannot change the data and would rather not change the architecture, so I change the objective: add to $L_\mu$ something that discourages clobbering what was already learned. The crude version anchors every weight to where it was, $L_\mu + \sum_k (\theta_k - \tilde\theta_k)^2$, but it is hopeless on inspection, because one global stiffness cannot be right — large enough to actually hold task 1 in place and the network is too rigid to learn task 2; small enough to learn task 2 and it does not hold task 1. The stiffness must be *per parameter*: some weights mattered enormously and must barely move, others were irrelevant and should be free to absorb the new task. That splits the problem into two coupled questions — which parameters were important, and how to turn "important" into a penalty. The probabilistic-curvature route had already been taken by elastic weight consolidation, which approximates the old-task posterior as a Gaussian centered at the converged $\theta^*_A$ with diagonal precision equal to the diagonal Fisher $F$, giving the spring $L_B(\theta) + \sum_i (\lambda/2)\, F_i (\theta_i - \theta^*_{A,i})^2$, stiff exactly on the weights $A$ cared about. That is the right shape, but stare at how $F$ is obtained: it is a *point* estimate evaluated *at the converged endpoint*, computed in a *separate phase after the task is over* — an extra sweep that is not part of training and throws away the whole trajectory. The exact diagonal Fisher also needs a sum over all output labels, so its cost scales with the number of outputs, and for a third task one stores another Fisher or folds them, with bookkeeping that grows with the task count. I want importance that comes for free, online, while I am already training — no extra phase, no extra backward pass, computed locally per parameter from the optimization itself.

I propose Synaptic Intelligence. The shift is to throw out "evaluate curvature at the endpoint" and ask instead: over the whole course of training task $\mu$, how much did each individual parameter contribute to making that task's loss go *down*? The parameters that did the heavy lifting in driving the loss down are exactly the ones I must not let drift back up, and contribution-to-loss-change can be read straight off the trajectory, step by step. Make it precise. Picture training as a path $\theta(t)$ from $\theta(t_{\mu-1})$ to $\theta(t_\mu)$; to first order an infinitesimal step changes the loss by $L(\theta+\delta) - L(\theta) \approx \sum_k g_k \delta_k$ with $g_k = \partial L/\partial\theta_k$, so coordinate $k$'s little move contributes $g_k\,\delta_k$. Summing these infinitesimal contributions over the whole path, with $\delta_k = \theta'_k\,dt$, gives the line integral of the gradient field,
$$\int_C g(\theta)\cdot d\theta = \int_{t_{\mu-1}}^{t_\mu} g(\theta(t))\cdot\theta'(t)\,dt .$$
Because the gradient of a scalar loss is a *conservative* field, this integral is path-independent and equals the net potential difference $L(\theta(t_\mu)) - L(\theta(t_{\mu-1}))$ — negative during successful descent, which is the sanity anchor that the per-step pieces summed up must reproduce the total loss drop with the right sign. Crucially the dot product is a sum, so the integral decomposes coordinate by coordinate, and the per-parameter piece is what I define as importance,
$$\omega_k^\mu \equiv -\int_{t_{\mu-1}}^{t_\mu} g_k(\theta(t))\,\theta'_k(t)\,dt .$$
The minus sign is deliberate: under descent a parameter that drives the loss down has $g_k\theta'_k < 0$ (it moves against its own gradient), so flipping the sign makes $\omega_k^\mu$ a positive credit for how much that parameter helped. This is exactly what EWC could not get without a separate phase, and it accumulates cheaply: the integrand $g_k\,\theta'_k$ is the gradient times the parameter velocity, and at each optimizer step I already have $g_k$, while the update $\Delta\theta_k = \theta_k^{\text{new}} - \theta_k^{\text{old}}$ is the discrete stand-in for $\theta'_k\,dt$. So I keep a running, per-parameter sum of $-g_k\cdot\Delta\theta_k$ — one multiply and one add per parameter per step, reusing numbers the optimizer already computed. For full-batch descent with an infinitesimal learning rate this running sum *is* the path integral.

To turn importance into a penalty I want a surrogate, added while training a future task, that re-creates the effect of the unavailable past losses — pulling the weights as if those losses were still present. I take a per-parameter quadratic anchored at the reference weights $\tilde\theta_k = \theta_k(t_{\mu-1})$, and I fix its strength by demanding faithfulness to the descent in a strong sense: a quadratic $E_k(\theta) = s_k(\tilde\theta_k - \theta_k)^2$ should reproduce, over the same motion the real loss produced, the same credit. The motion over the task is $\Delta_k = \theta_k(t_\mu) - \theta_k(t_{\mu-1})$, and the loss drop the quadratic produces over it is $s_k\Delta_k^2$, so $s_k\Delta_k^2 = \omega_k^\mu$ forces $s_k = \omega_k^\mu/\Delta_k^2$. That is the normalization: divide the path-integral importance by the square of how far the parameter actually moved. Accumulating over all past tasks,
$$\Omega_k^\mu = \sum_{\nu<\mu} \frac{\omega_k^\nu}{(\Delta_k^\nu)^2 + \xi} ,$$
and the consolidation penalty added to every later task's loss is
$$\tilde L_\mu = L_\mu + c\sum_k \Omega_k^\mu (\tilde\theta_k - \theta_k)^2 ,$$
with $c$ a single dimensionless knob trading old against new. The $\Delta^2$ in the denominator is not a fudge: it is what makes the surrogate yield the *same* $\omega$ over the *same* distance, and it fixes the units — $\omega$ has units of loss, $\Delta^2$ has units of parameter-squared, so $\omega/\Delta^2$ times $(\tilde\theta-\theta)^2$ comes out in units of loss, matching $L_\mu$ exactly. The small damping constant $\xi$ is there for a concrete reason: if a parameter barely moved, $\Delta_k\to 0$ and $\omega/\Delta^2$ would blow up, handing a stationary weight infinite importance, so $\xi$ floors the denominator. The reference $\tilde\theta_k$ and the accumulated $\Omega_k$ update only at task boundaries; during a task the running $\omega$ keeps accumulating, and once folded into $\Omega$ it is reset, so the bookkeeping is constant in the number of tasks — one running $\Omega$ per parameter, one reference per parameter, no per-task list.

Two consequences of using noisy minibatch gradients shape the final form. First, a noisy $g_k$ summed up tends to *overestimate* the magnitude of $\omega$, because the cross terms between noise and update do not fully cancel; so although $c = 1$ is the honest in-the-limit value, in practice $c$ is taken below one to compensate — smaller on noisier or harder problems (about $1$ on split MNIST, about $0.1$ on permuted MNIST, swept over $10^{-3}$–$0.1$ on CIFAR). Second, the noisy integral can come out slightly negative for some parameter, and a negative final stiffness would *reward* moving the weight away from its anchor, actively encouraging forgetting; importance must be nonnegative. The clean place to floor is the running stiffness after adding the new increment, $\Omega_k \leftarrow \mathrm{ReLU}(\Omega_k + W_k/(\Delta_k^2 + \xi))$, so a noisy negative increment can correct an earlier overestimate but the stored stiffness can never cross below zero.

To check this is genuinely curvature and not just a plausible running sum, take the one solvable case, a quadratic loss $E(\theta) = \tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*)$ under continuous gradient descent $\tau\,d\theta/dt = -H(\theta-\theta^*)$, whose solution is $\theta(t) = \theta^* + e^{-Ht/\tau}(\theta(0)-\theta^*)$. Since $\tau\,d\theta/dt = -g$, the importance $\omega_k = -\int g_k\theta'_k\,dt$ is the diagonal of the positive matrix $Q = \tau\int_0^\infty (d\theta/dt)(d\theta/dt)^\top dt$. Working in the Hessian eigenbasis $(\lambda^\alpha, u^\alpha)$ with $d^\alpha = u^\alpha\cdot(\theta(0)-\theta^*)$, the time integral of $e^{-(\lambda^\alpha+\lambda^\beta)t/\tau}$ is $\tau/(\lambda^\alpha+\lambda^\beta)$, and the prefactors collapse to
$$Q_{ij} = \sum_{\alpha\beta} u_i^\alpha d^\alpha \frac{\lambda^\alpha\lambda^\beta}{\lambda^\alpha+\lambda^\beta} d^\beta u_j^\beta ,$$
with $\tau$ dropping out entirely — importance does not depend on how fast I descend. Averaging over random initial conditions, $\langle d^\alpha d^\beta\rangle = \sigma^2\delta_{\alpha\beta}$, kills the off-diagonals and leaves $\lambda^\alpha/2$ on the diagonal, giving
$$\langle Q_{ij}\rangle = \tfrac12\sigma^2\sum_\alpha u_i^\alpha \lambda^\alpha u_j^\alpha = \tfrac12\sigma^2 H_{ij} ,$$
so on average the path-integral matrix is one half of the Hessian up to the displacement scale $\sigma^2$. The half is not a nuisance — it is exactly the half in the loss drop of a bowl, $\tfrac12 d^\top H d$ — and the $\Delta^2$ denominator, which averages to $\sigma^2$, strips the displacement scale and leaves $\Omega_k = \tfrac12 H_{kk}$; because the penalty is written with no leading $\tfrac12$, its local curvature is $2\Omega_k = H_{kk}$, the right convention. Without averaging the stored diagonal entries stay clean: for a diagonal Hessian $Q_{ii} = \tfrac12(d^i)^2 H_{ii}$, so normalizing leaves the same $\tfrac12 H_{ii}$; for a rank-1 Hessian the full matrix is $\tfrac12(d^1)^2 H_{ij}$, concentrating importance on the active low-rank curvature direction while the rest of parameter space stays flat — precisely the geometry that leaves room for later tasks. The deepest contrast with EWC is sharpest here: the empirical Fisher $\bar F = E[g g^\top]$ evaluated at the endpoint *vanishes* for a quadratic at its minimum, where the gradient is zero, so it has thrown away all the curvature by the time it looks, whereas the path integral remembers the curvature it descended through, and does so with no extra backward pass.

In code the method fills the two slots a continual-learning loop exposes. `estimate_importance` runs once after each task and turns the per-step accumulated path integral $W_k = \sum(-g_k\,\Delta\theta_k)$ into the normalized increment $W_k/(\Delta_k^2+\xi)$; `compute_regularization_loss` runs every step and returns the quadratic penalty. The per-step accumulation of $W$ is maintained by the loop and read here, and the surrounding protocol performs the cross-task summing and applies the nonnegative floor at the task boundary.

```python
import torch


def estimate_importance(model, dataset, prev_params, device):
    """SI importance, computed once after a task finishes.

    The surrounding loop adds this return value into model._custom_importance.
    In the pathint optimizer protocol the nonnegative floor is applied to the
    running task-boundary update; this hook returns the current context's raw
    normalized increment.
    """
    epsilon = getattr(model, 'epsilon', 0.1)     # damping ξ: bounds ω when Δ_k -> 0
    omega = {}
    W = getattr(model, '_custom_W', {})          # accumulated Σ -g·Δθ over the task

    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                theta = p.detach().clone()
                theta_ref = prev_params.get(n, theta)        # θ̃: snapshot at task start
                delta = theta - theta_ref                    # Δ_k: net motion over the task
                w = W.get(n, torch.zeros_like(theta))
                omega[n] = w / (delta ** 2 + epsilon)        # W/(Δ^2+ξ)

    return omega


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """SI consolidation penalty, added at every training step (scaled by c outside):
        Σ_k Ω_k (θ_k - θ̃_k)^2 ,
    with Ω_k the importance accumulated across past tasks and θ̃_k the reference weights.
    """
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    omega = importance_dict[n]               # Ω_k: consolidation strength
                    ref = prev_params_dict[n]                # θ̃_k: reference weights
                    losses.append((omega * (p - ref) ** 2).sum())
    if losses:
        return sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)
```
