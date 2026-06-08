# Perturbed Gradient Descent — escaping saddle points efficiently

## Problem

Minimize a nonconvex $f:\mathbb{R}^d\to\mathbb{R}$ that is $\ell$-gradient-Lipschitz and $\rho$-Hessian-Lipschitz, using only a gradient oracle. Plain gradient descent reaches an $\epsilon$-**first**-order stationary point ($\|\nabla f\|\le\epsilon$) in a *dimension-free* $\ell(f(\mathbf{x}_0)-f^\star)/\epsilon^2$ steps, but in a nonconvex landscape that point may be a saddle. When every saddle is strict ($\lambda_{\min}(\nabla^2 f)<0$ there), the local minima are exactly the **second**-order stationary points, so the goal becomes: find an $\epsilon$-second-order stationary point,
$$\|\nabla f(\mathbf{x})\|\le\epsilon\quad\text{and}\quad\lambda_{\min}(\nabla^2 f(\mathbf{x}))\ge-\sqrt{\rho\epsilon},$$
using gradients only, at the natural step size, with at most polylogarithmic dependence on $d$.

## Key idea

Saddles, not poor local minima, are the obstacle, and plain gradient descent stalls at them only because its gradient sensor reads zero. Near a strict saddle the gradient step is power iteration on $\mathbf{I}-\eta\nabla^2 f$: it geometrically amplifies the most-negative-curvature direction $\mathbf{e}_1$ — provided the iterate has a nonzero component there. So inject randomness **once**, exactly when the gradient is small, to plant a component along $\mathbf{e}_1$; deterministic gradient descent then does the escaping. Keeping the natural large step $\eta=\Theta(1/\ell)$ is what preserves dimension-freeness.

The crux is the volume of the **stuck region** $\mathcal{X}_{\mathrm{stuck}}$ inside the perturbation ball $\mathbb{B}_{\tilde{\mathbf{x}}}(r)$ — the starts that fail to escape. At the large step it is a curved "thin pancake" with no closed form. One never needs its shape, only its **width** along $\mathbf{e}_1$: a two-point coupling shows that two starts on a line parallel to $\mathbf{e}_1$ and at least $\delta r/(2\sqrt{d})$ apart cannot both be stuck, so the stuck set on each such line is an interval of length $\le\delta r/\sqrt{d}$. Integrating, $\mathrm{Vol}(\mathcal{X}_{\mathrm{stuck}})/\mathrm{Vol}(\mathbb{B}^{(d)}(r))\le\frac{\delta}{\sqrt{\pi d}}\frac{\Gamma(d/2+1)}{\Gamma(d/2+1/2)}\le\frac{\delta}{\sqrt{\pi d}}\sqrt{\tfrac d2+\tfrac12}\le\delta$: the $\sqrt d$ in the width cancels the $\sqrt d$ in the ball cross-section, leaving $d$ only inside a logarithm.

## Algorithm

`PGD`$(\mathbf{x}_0,\ell,\rho,\epsilon,c,\delta,\Delta_f)$, with $\chi=3\max\{\log(\tfrac{d\ell\Delta_f}{c\epsilon^2\delta}),4\}$ and
$$\eta=\tfrac c\ell,\quad r=\tfrac{\sqrt c}{\chi^2}\tfrac{\epsilon}{\ell},\quad g_{\mathrm{thres}}=\tfrac{\sqrt c}{\chi^2}\epsilon,\quad f_{\mathrm{thres}}=\tfrac{c}{\chi^3}\sqrt{\tfrac{\epsilon^3}{\rho}},\quad t_{\mathrm{thres}}=\tfrac{\chi}{c^2}\tfrac{\ell}{\sqrt{\rho\epsilon}}.$$
Run $\mathbf{x}_{t+1}=\mathbf{x}_t-\eta\nabla f(\mathbf{x}_t)$. When $\|\nabla f(\mathbf{x}_t)\|\le g_{\mathrm{thres}}$ and no perturbation occurred in the last $t_{\mathrm{thres}}$ steps, snapshot $\tilde{\mathbf{x}}_t\leftarrow\mathbf{x}_t$ and add $\xi_t$ uniform on $\mathbb{B}_0(r)$. Exactly $t_{\mathrm{thres}}$ steps after a perturbation, if $f(\mathbf{x}_t)-f(\tilde{\mathbf{x}})>-f_{\mathrm{thres}}$, output $\tilde{\mathbf{x}}$.

## Main theorem

If $f$ is $\ell$-smooth and $\rho$-Hessian-Lipschitz, then for any $\delta>0$, $\epsilon\le\ell^2/\rho$, $\Delta_f\ge f(\mathbf{x}_0)-f^\star$, and $c\le c_{\max}$, with probability $1-\delta$ `PGD` outputs an $\epsilon$-second-order stationary point in
$$O\!\left(\frac{\ell(f(\mathbf{x}_0)-f^\star)}{\epsilon^2}\,\log^{4}\!\Big(\frac{d\ell\Delta_f}{\epsilon^2\delta}\Big)\right)\text{ iterations}$$
— the dimension-free first-order rate, up to a $\log^4 d$ factor.

### Proof sketch

*Descent (large gradient).* $\ell$-smoothness gives $f(\mathbf{x}_{t+1})\le f(\mathbf{x}_t)-\eta\|\nabla f\|^2+\tfrac{\eta^2\ell}{2}\|\nabla f\|^2\le f(\mathbf{x}_t)-\tfrac\eta2\|\nabla f\|^2$ for $\eta\le1/\ell$; so $\|\nabla f\|\ge g_{\mathrm{thres}}$ drops $f$ by $\ge\tfrac\eta2 g_{\mathrm{thres}}^2$ per step.

*Escape (saddle).* At $\tilde{\mathbf{x}}$ with $\|\nabla f\|\le g_{\mathrm{thres}}$ and $\lambda_{\min}(\nabla^2 f(\tilde{\mathbf{x}}))\le-\sqrt{\rho\epsilon}$, a perturbation followed by $t_{\mathrm{thres}}$ steps drops $f$ by $f_{\mathrm{thres}}$ with probability $\ge1-\tfrac{d\ell}{\sqrt{\rho\epsilon}}e^{-\chi}$. Locally set $\gamma=-\lambda_{\min}(\nabla^2 f(\tilde{\mathbf{x}}))$, $\kappa=\ell/\gamma$, and $L=\log(d\kappa/\delta)$; the proof units are $\mathscr{S}=\sqrt{\eta\ell}\gamma/(\rho L)$, $\mathscr{G}=\sqrt{\eta\ell}\gamma^2/(\rho L^2)$, and $\mathscr{F}=\eta\ell\gamma^3/(\rho^2 L^3)$, so $\mathscr{G}L/\gamma=\sqrt{\mathscr{F}L/\gamma}=\mathscr{S}$. The stuck-region coupling is proved in two parts: **(i) confinement** — a stuck trajectory stays within $O(\mathscr{S}\hat c)$ of $\tilde{\mathbf{x}}$, shown by decomposing $\mathbf{u}_{t+1}=(\mathbf{I}-\eta\mathbf{H}-\eta\Delta_t)\mathbf{u}_t-\eta\nabla f(0)$ into the strongly-negative subspace $\mathcal{S}$ (eigenvalues $<-\gamma/L$) and its complement, where $\|\bm\beta_{t+1}\|\le6\mathscr{S}\hat c$ and $\bm\beta_{t+1}^\top\mathbf{H}\bm\beta_{t+1}\le8\eta T\mathscr{G}^2$ using the largest-eigenvalue bound from the maximizer $\lambda^\star_t=\tfrac1{(1+t)\eta}$ of $g_t(\lambda)=\lambda(1-\eta\lambda)^t$ and $\sum_{\tau_1,\tau_2}1/(1+\tau_1+\tau_2)\le2T$; **(ii) separation** — for $\mathbf{v}_t=\mathbf{w}_t-\mathbf{u}_t$ initially along $\mathbf{e}_1$, the $\mathbf{e}_1$-component obeys $\psi_{t+1}\ge(1+\tfrac{\eta\gamma}{2})\psi_t$ (after showing the off-axis part stays $\le\psi_t$), so the gap grows geometrically and overruns the confinement radius before the time budget — hence if $\mathbf{u}$ is confined, $\mathbf{w}$ escapes. The model decrease converts to a true decrease up to $O(\rho\mathscr{S}^3)=O(\sqrt{\eta\ell}\,\mathscr{F})$ by Hessian-Lipschitzness.

*Global count.* Both regimes give per-step decrease $\Theta(\tfrac{c^3}{\chi^4}\tfrac{\epsilon^2}{\ell})$; dividing $f(\mathbf{x}_0)-f^\star$ by it gives $\tfrac{\chi^4}{c^3}\tfrac{\ell(f_0-f^\star)}{\epsilon^2}=O(\tfrac{\ell(f_0-f^\star)}{\epsilon^2}\log^4(\cdot))$. The output is correct because the algorithm stops only when a perturbation fails to produce the certified drop, which (by the escape lemma, union-bounded over $\le\tfrac{\chi^3}{c}\tfrac{\sqrt{\rho\epsilon}(f_0-f^\star)}{\epsilon^2}$ perturbations, using $\chi^3 e^{-\chi}\le e^{-\chi/3}$ for $\chi\ge12$) means $\lambda_{\min}\ge-\sqrt{\rho\epsilon}$ with probability $\ge1-\delta$.

## Consequences

**Local minima (strict saddle).** If $f$ is robustly $(\theta,\gamma,\zeta)$-strict-saddle, set $\tilde\epsilon=\min(\theta,\gamma^2/\rho)$; an $\tilde\epsilon$-second-order stationary point is then forced to be $\zeta$-close to a local minimum, reached in $O(\tfrac{\ell(f_0-f^\star)}{\tilde\epsilon^2}\log^4(\cdot))$ steps.

**Linear local rate.** If a $\zeta$-neighborhood satisfies $(\alpha,\beta)$-regularity, $\langle\nabla f(\mathbf{x}),\mathbf{x}-\mathrm{proj}_{\mathcal{X}^\star}\mathbf{x}\rangle\ge\tfrac\alpha2\|\mathbf{x}-\mathrm{proj}\|^2+\tfrac1{2\beta}\|\nabla f\|^2$, then plain gradient descent with step $\tfrac1\beta$ contracts $\|\mathbf{x}-\mathrm{proj}\|^2$ by $(1-\tfrac\alpha\beta)$ each step (and never leaves the neighborhood), giving $\epsilon$-closeness in $O(\tfrac\beta\alpha\log\tfrac\zeta\epsilon)$. Two-phase `PGDli` (perturbed phase then this local phase) costs $O(\tfrac{\ell(f_0-f^\star)}{\tilde\epsilon^2}\log^4(\cdot)+\tfrac\beta\alpha\log\tfrac\zeta\epsilon)$.

## Algorithm code

```python
import numpy as np

def perturbed_gradient_descent(grad, f, x0, ell, rho, eps, c, delta, Delta_f):
    """Returns an eps-second-order stationary point of f w.h.p."""
    x = np.array(x0, dtype=float)
    d = x.size
    chi   = 3.0 * max(np.log(d * ell * Delta_f / (c * eps**2 * delta)), 4.0)
    eta   = c / ell
    r     = (np.sqrt(c) / chi**2) * eps / ell
    g_thr = (np.sqrt(c) / chi**2) * eps
    f_thr = (c / chi**3) * np.sqrt(eps**3 / rho)
    t_thr = int(np.ceil((chi / c**2) * ell / np.sqrt(rho * eps)))

    t_noise = -t_thr - 1
    x_tilde, f_tilde, t = None, None, 0
    while True:
        g = grad(x)
        if np.linalg.norm(g) <= g_thr and t - t_noise > t_thr:
            x_tilde, f_tilde, t_noise = x.copy(), f(x), t
            y = np.random.randn(d); u = np.random.rand() ** (1.0 / d)
            x = x + r * u * y / np.linalg.norm(y)          # uniform on B_0(r)
        if x_tilde is not None and t - t_noise == t_thr:
            if f(x) - f_tilde > -f_thr:                    # no certified drop -> eps-SOSP
                return x_tilde
        x = x - eta * grad(x)
        t += 1


def pgd_local(grad, f, x0, ell, rho, eps_tilde, c, delta, Delta_f, beta, num_local):
    """Perturbed phase, then plain GD with step 1/beta for linear local convergence."""
    x = perturbed_gradient_descent(grad, f, x0, ell, rho, eps_tilde, c, delta, Delta_f)
    for _ in range(num_local):
        x = x - (1.0 / beta) * grad(x)
    return x
```
