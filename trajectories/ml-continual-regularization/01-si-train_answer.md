A single fixed-capacity network trains on contexts one at a time and, with the old data gone and replay forbidden, descending the current loss drags the shared weights that encoded earlier contexts wherever the new context demands — so old accuracy collapses. The crude cure is a quadratic anchor to each weight's boundary value, $L_\mu + c\sum_k(\theta_k-\tilde\theta_k)^2$, but a single global stiffness cannot win: large enough to actually hold an old context freezes the net so the new one cannot be learned, small enough to learn the new one fails to hold the old. The stiffness has to be *per parameter* — stiff on the weights an old context relied on, slack on the ones it did not — which turns the whole problem into estimating, per weight, how much each one mattered. The harness ships a default that answers this with an after-the-fact diagonal Fisher recomputed in a separate post-context sweep, while the training loop already maintains, for free, a per-step accumulator $\texttt{\_custom\_W}[n]\mathrel{+}=-\,\text{grad}\cdot(p-p_{\text{old}}[n])$ that nobody is spending. That free accumulator is the opening.

I propose **SI (Synaptic Intelligence)**: read each parameter's importance off the *training trajectory* the optimizer already walked, rather than off a final-point Fisher I would have to recompute. The derivation is a line integral. Picture training context $\mu$ as a path $\theta(t)$ from the boundary snapshot to the endpoint. An infinitesimal step changes the loss to first order by $\sum_k g_k\,\delta_k$ with $g_k=\partial L/\partial\theta_k$, so each coordinate's little move contributes $g_k(t)\,\delta_k(t)$ to the loss change right then. Summing these contributions along the path, and writing $\delta_k = \theta_k'\,dt$, the total loss change is the line integral of the gradient field $\int g(\theta(t))\cdot\theta'(t)\,dt$. Because the gradient of a scalar loss is a conservative field, this integral is path-independent and equals $L(\theta(t_\mu))-L(\theta(t_{\mu-1}))$ — negative during successful descent, which is the sanity anchor the per-parameter pieces must sum back to. And crucially the dot product is a sum, so the integral *decomposes coordinate by coordinate*: each term $\int g_k\,\theta_k'\,dt$ is one parameter's contribution to the whole context's loss drop. That per-parameter line integral, sign-flipped so a parameter that drove the loss down scores positive, is the importance
$$\omega_k^\mu = -\int g_k(\theta(t))\,\theta_k'(t)\,dt .$$
Discretized over optimizer steps with $\delta\theta_k$ standing in for $\theta_k'\,dt$, this running sum of $-g_k\cdot\delta\theta_k$ is *exactly* what $\texttt{\_custom\_W}$ already holds — the loop's per-step hook is the discrete path integral, computed from gradients training needed anyway.

The second half is turning $\omega_k^\mu$ into a penalty, and here the load-bearing design choice is the normalization. I want the spring added while training a later context to recreate, as far as descent dynamics can tell, the effect of the now-unavailable past loss. So I demand faithfulness: if I had descended the quadratic surrogate $s_k(\tilde\theta_k-\theta_k)^2$ *instead of* the real old loss, it should accrue the same per-parameter loss drop over the same motion. Over the context a parameter moved by $\Delta_k=\theta_k(t_\mu)-\theta_k(t_{\mu-1})$, the surrogate's loss drop over that motion is $s_k\Delta_k^2$, and setting it equal to the credit the real loss actually earned, $s_k\Delta_k^2=\omega_k^\mu$, forces
$$\Omega_k = \frac{\omega_k^\mu}{\Delta_k^2} \;\longrightarrow\; \frac{W_k}{\Delta_k^2+\xi}.$$
That $\Delta^2$ in the denominator is not a fudge: it makes the surrogate reproduce $\omega$ over the same distance, and it fixes the units — $\omega$ is in loss, $\Delta^2$ in parameter-squared, so $\omega/\Delta^2$ times $(\tilde\theta-\theta)^2$ lands back in loss, matching $L_\mu$ exactly and leaving $c$ a clean dimensionless knob. The small damping $\xi$ (the harness's $\texttt{model.epsilon}=0.1$) is there for a concrete failure: a weight that barely moved has $\Delta_k\to0$ and $\omega/\Delta^2$ blows up to infinite importance for a weight that did nothing, so $\xi$ floors it.

What convinced me this is the right place to start the ladder — rather than the default Fisher — is that the $\Delta^2$ normalization I introduced for faithfulness also recovers genuine curvature, and on the one case I can solve exactly it beats the default on its own terms. On a quadratic loss $E(\theta)=\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*)$, continuous gradient descent gives $-g_k\theta_k'=\tau(d\theta_k/dt)^2$, so the per-parameter $\omega$ are the diagonal of $Q=\tau\int(d\theta/dt)(d\theta/dt)^\top dt$; the learning-rate $\tau$ cancels out of the prefactor, and averaging over random initial displacements with $\langle d^a d^b\rangle=\sigma^2\delta_{ab}$ gives $\langle Q\rangle=\tfrac12\sigma^2 H$. Dividing by $\Delta_k^2$ (which averages to $\sigma^2$) strips the trajectory scale and leaves $\Omega_k=\tfrac12 H_{kk}$, so with a no-half penalty $\Omega_k(\theta-\tilde\theta)^2$ the effective curvature is $2\Omega_k=H_{kk}$ — the right convention. The sharpest contrast is that the default empirical Fisher is evaluated *at the converged endpoint*, where the gradient vanishes, so on a quadratic it reads *zero curvature* — it has thrown away everything by the time it looks — whereas the path integral accumulated curvature-flavored information along the way, while the gradients were still nonzero, at no extra cost. This is also why SI's penalty drops the leading half that the Gaussian story keeps: the half was already absorbed into the $\Delta^2$ normalization.

The one caveat I carry forward is noise. Under SGD the gradient $g_k$ is a noisy minibatch estimate, and a noisy $g_k$ summed up tends to *overestimate* the importance magnitude, because the cross terms between noise and update do not fully cancel — and critically the running $\Omega$ only ever *adds* across contexts, with no decay to relieve it. On a short clean task-incremental sequence like Split-MNIST that overestimate is mild and the springs should hold the early tasks almost perfectly; on a long domain-incremental sequence of uncorrelated contexts like Permuted-MNIST the undecayed sum can rigidify the fixed-capacity net so later contexts cannot be learned *and* the scale-mismatched importance stops holding the early ones — the falsifiable signature pointing the next rung at a bounded, re-centered importance. The strength $c$ is the per-benchmark $\texttt{reg\_strength}$; I leave $\texttt{CONFIG\_OVERRIDES}$ empty and take $c=1$, the honest in-the-limit value. Cross-context summation is done by the loop, so $\texttt{estimate\_importance}$ returns the raw normalized increment $W_k/(\Delta_k^2+\epsilon)$ and the penalty is the cheap no-half quadratic.

```python
# EDITABLE region of custom_regularization.py — step 1: SI (path-integral importance)
def estimate_importance(model, dataset, prev_params, device):
    """SI: Compute omega from accumulated path integral W and parameter changes.

    omega_k = W_k / (delta_k^2 + epsilon)

    where W_k is the accumulated per-step gradient-weighted parameter change
    (tracked in model._custom_W by the training loop) and delta_k is the
    total parameter change over the context.
    """
    epsilon = getattr(model, 'epsilon', 0.1)
    omega = {}

    # Get the accumulated W from the per-step tracking
    W = getattr(model, '_custom_W', {})

    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                p_current = p.detach().clone()
                p_prev = prev_params.get(n, p_current)
                p_change = p_current - p_prev
                w_val = W.get(n, torch.zeros_like(p_current))
                omega[n] = w_val / (p_change ** 2 + epsilon)

    return omega


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """SI: sum(omega * (param - prev_param)^2)."""
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    omega = importance_dict[n]
                    prev = prev_params_dict[n]
                    losses.append((omega * (p - prev) ** 2).sum())
    if losses:
        return sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).
CONFIG_OVERRIDES = {}
```
