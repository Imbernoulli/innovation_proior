We want to solve a bilevel problem
$$\min_{x,y} f(x,y) \quad \text{s.t.}\quad x \in C,\; y \in S(x) := \arg\min_{y} g(x,y),$$
where the admissible $y$ are exactly the minimizers of the inner objective $g(x,\cdot)$, so as $x$ moves the whole solution set $S(x)$ moves with it. That coupling is the entire difficulty. The mature theory handles only the easy case: when $g(x,\cdot)$ is strongly convex and unconstrained, $S(x)=\{y^*(x)\}$ is a single point, the problem collapses to $\min_x f(x,y^*(x))$, and differentiating the lower stationarity equation $\nabla_y g(x,y^*(x))=0$ hands over the hypergradient $\nabla_x f - \nabla_{xy} g\,(\nabla_{yy} g)^{-1}\nabla_y f$. That formula leans on exactly two things I cannot assume: it needs $\nabla_{yy} g$ invertible (strong convexity), and it needs second-order information, at minimum a Hessian-inverse-vector product. The instant $g(x,\cdot)$ loses strong convexity the Hessian can be singular and the formula is undefined; the instant $S(x)$ becomes a flat valley of many minimizers there is no single $y^*(x)$ to differentiate. The regime I care about — Polyak-Łojasiewicz-but-nonconvex lower levels, possibly non-singleton $S(x)$, possibly constrained — is exactly where this machinery does not reach.

The alternatives each give up something I need. Unrolling — replacing $S(x)$ by $T$ steps of inner gradient descent $y_T(x)$ and backpropagating the outer loss through the whole trajectory — sidesteps the Hessian inverse, but I must store the entire trajectory for the reverse pass so memory and compute grow with $T$, a projection inside the inner loop is awkward to differentiate so I am pinned to the unconstrained case, and I get no clean finite-time relation to the true problem; it is still secretly second-order through the Jacobians of the inner update. Folding lower-optimality into a penalty and descending one joint objective over $(x,y)$ is the natural move, and the tempting first penalty is the squared lower-gradient norm $\|\nabla_y g\|^2$ because it is directly differentiable. But it fails, and the failure is instructive. Take $g(y)=y^2+2\sin^2 y$, $f(y)=\sin^2(y-2\pi/3)$, whose only bilevel solution is $y^*=0$; here $\nabla_y g = 2y+2\sin 2y$ vanishes exactly on $S(x)$, so $\|\nabla_y g\|^2 = 4(y+\sin 2y)^2$ is a legitimate optimality metric. The derivative of $(y+\sin 2y)^2$ is $2(y+\sin 2y)(1+2\cos 2y)$; at $y=2\pi/3$ the factor $1+2\cos 2y$ is zero (since $\cos(4\pi/3)=-\tfrac12$), so the penalty gradient vanishes for *every* penalty strength $\gamma$ even though $\nabla_y g$ itself is nonzero there. Gradient descent on the naive penalty parks at junk. The penalty's zero set is right; what is wrong is that its *local stationarity* can be killed by a degenerate Hessian factor — a piece of $\nabla_{yy} g$ — before lower feasibility is ever reached.

That diagnosis points at the correct abstraction. The exact constraint $y\in S(x)$ is equivalent not to "some optimality metric is zero" but to $d_{S(x)}^2(y)=0$, the squared distance from $y$ to the minimizer set. That distance is uncomputable, so I want a computable, smooth $p(x,y)$ that *behaves like* it: nonnegative, zero exactly on $S(x)$, and with $\rho\,p(x,y)\ge d_{S(x)}^2(y)$ for some $\rho$. Call any such $p$ a squared-distance bound. The domination is precisely what the naive penalty lacked: once $p$ dominates $d^2$, a small penalty residual is no longer just a small number but a certificate that $y$ is close to lower feasibility.

I propose **V-PBGD** — value-function penalty-based bilevel gradient descent — built on the choice $p(x,y)=g(x,y)-v(x)$, the lower-level value gap, with $v(x):=\min_y g(x,y)$. The reformulation it descends is the single-level objective $F_\gamma = f + \gamma\,p$, and three things make it work.

First, the value gap really is a squared-distance bound under the right weak assumption. If $g(x,\cdot)$ satisfies the $1/\mu$-PL inequality $\|\nabla_y g\|^2 \ge \tfrac1\mu (g-v)$ and is $L_g$-smooth, then PL implies quadratic growth $g(x,y)-v(x)\ge \tfrac1\mu d_{S(x)}^2(y)$, so $p$ is nonnegative, vanishes exactly on $S(x)$, and $\mu\,p \ge d^2$. PL is strictly weaker than strong convexity — it permits nonconvex landscapes and non-singleton $S(x)$ — which is the whole point. The squared-gradient-norm could also be made a squared-distance bound, but only under the stronger $1/\sqrt{\mu}$ PL constant, and it remains a useful sibling rather than the robust branch.

Second, the global faithfulness of the penalty rests on a single scalar calmness inequality, and it is worth seeing because it explains the role of the domination. Project $y$ to $y_x\in S(x)$ and assume $f(x,\cdot)$ is $L$-Lipschitz, so $f(x,y)-f(x,y_x)\ge -L\,d_{S(x)}(y)$. Adding $\gamma^*p$ and using $p\ge d^2/\rho$ gives
$$f(x,y)+\gamma^* p(x,y)-f(x,y_x) \;\ge\; -L\,d_{S(x)}(y) + \frac{\gamma^*}{\rho}\,d_{S(x)}^2(y).$$
The right side is the scalar $-Lz+\tfrac{\gamma^*}{\rho}z^2$ in $z=d_{S(x)}(y)\ge 0$, minimized at $z^*=L\rho/(2\gamma^*)$ with value $-L^2\rho/(4\gamma^*)$. Setting $\gamma^*=L^2\rho/(4\varepsilon_1)$ makes that floor exactly $-\varepsilon_1$, so $f(x,y)+\gamma^*p(x,y)\ge f^*-\varepsilon_1$ everywhere: the bilevel global solution is an $\varepsilon_1$-global-min of the penalized problem, and the same line gives the converse, that an $\varepsilon_2$-global-min with $\gamma>\gamma^*$ has $p\le(\varepsilon_1+\varepsilon_2)/(\gamma-\gamma^*)$ and hence small distance to $S(x)$. The quadratic term — present only because $p$ *dominates* $d^2$ — overpowers the linear Lipschitz slack once $\gamma$ is large enough. This is exactly tight: for $f=y,\,g=y^2$ the penalized minimizer is $-1/(2\gamma)$ with gap $1/(4\gamma^2)$, so forcing $p\le\delta$ needs $\gamma=\Omega(\delta^{-1/2})$, and that square-root scaling is the genuine price of a quadratic penalty, with a corresponding stiffness cost in the algorithm.

The reason to use the value gap rather than the gradient norm lives at the *local* level, where the naive penalty's spurious solution lived. A local solution of the value-gap penalty satisfies $\nabla_y f + \gamma\nabla_y g = 0$ (since $v$ carries no $y$-dependence), so $\|\nabla_y g\|\le L/\gamma$, and PL gives $p=g-v\le \mu L^2/\gamma^2 = O(1/\gamma^2)$ — with no division by $\nabla_{yy} g$ anywhere. Run the same accounting for $\|\nabla_y g\|^2$ and it breaks: stationarity is $\nabla_y f + 2\gamma\,\nabla_{yy} g\,\nabla_y g=0$, so I only control $\|\nabla_{yy} g\,\nabla_y g\|\le L/(2\gamma)$, and to bound $\|\nabla_y g\|$ I must divide by $\nabla_{yy} g$, i.e. assume its singular values are bounded below by some $\sigma>0$. At the spurious $y=2\pi/3$ of the failing example $\nabla_{yy} g=0$, so $\sigma=0$ and the bound is vacuous — the same degenerate-Hessian factor that killed the penalty gradient. The value gap has no such requirement, so it is not fooled by saddles of $g$.

Third, the one algorithmic obstacle — descending $F_\gamma$ needs $\nabla v(x)$ — dissolves through a PL Danskin lemma. At an arbitrary $y$, $\nabla_x g(x,y)$ is *not* $\nabla v(x)$, and $v$ can even be nonsmooth, so a literal gradient seems to demand solving and differentiating the lower level. But writing $v(x)=g(x,y^*(x))$, the chain rule gives $\nabla v=\nabla_x g(x,y^*) + (dy^*/dx)^\top \nabla_y g(x,y^*)$, and at an *unconstrained* lower minimum $\nabla_y g(x,y^*)=0$, so the scary implicit term multiplies by zero and vanishes. Under PL and $L_g$-smoothness the result holds without a unique minimizer: $\nabla v(x)=\nabla_x g(x,y^*)$ for *any* $y^*\in S(x)$, and $v$ is $(L_g+L_g^2\mu)$-smooth. That turns the value-function gradient into an ordinary first-order gradient of $g$ at a lower minimizer — no Hessian, no implicit-function theorem, no strong convexity. I do not have $y^*(x_k)$ exactly, so at outer iteration $k$ I run $T_k$ steps of inner gradient descent on $g(x_k,\cdot)$ from the warm start $\omega_1=y_k$, call the result $\hat y_k$, and approximate $\nabla v(x_k)\approx \nabla_x g(x_k,\hat y_k)$. The update direction is
$$h_k = \nabla g(x_k,y_k) - \big(\nabla_x g(x_k,\hat y_k),\,0\big),$$
where the subtraction lives only in the $x$-coordinates because $v$ depends only on $x$ (in $y$ the value-gap gradient is just $\nabla_y g(x_k,y_k)$), and the outer step is the projected gradient step
$$(x_{k+1},y_{k+1}) = \mathrm{Proj}_{C\times\mathbb{R}^{d_y}}\!\Big((x_k,y_k)-\alpha\big(\nabla f(x_k,y_k)+\gamma h_k\big)\Big).$$
Everything is first-order.

Two design choices in the schedule are load-bearing. The inner loop only needs $T_k=O(\log(\cdot))$ steps because PL makes inner GD linearly convergent, $g(x,\omega_{t+1})-v\le(1-\beta/(2\mu))(g(x,\omega_t)-v)$, and the error bound converts the contracted gap into $d_{S(x)}^2(\hat y)\le \mu(1-\beta/(2\mu))^{T}(g(x,\omega_1)-v)$. But that bound carries the *initial* gap, and $x_k$ drifts across the outer loop, so a cold start could let it grow without bound and no fixed $T_k$ would control the error. Warm-starting at $\omega_1=y_k$ fixes this: PL gives $g(x_k,y_k)-v(x_k)\le\tfrac1\mu\|\nabla_y g(x_k,y_k)\|^2$, and substituting the outer step relation yields $g(x_k,\omega_1)-v(x_k)\le \tfrac{2}{\mu\gamma^2\alpha^2}\|y_{k+1}-y_k\|^2 + \tfrac{2L^2}{\mu\gamma^2}$, tying the inner difficulty to outer progress plus an $O(1/\gamma^2)$ floor. The step size must shrink as $\alpha\sim 1/\gamma$ because $F_\gamma$ is $L_\gamma$-smooth with $L_\gamma = L_f + \gamma(2L_g + L_g^2\mu)$ — $\nabla f$ contributes $L_f$, $\nabla g$ contributes $\gamma L_g$, and $\nabla v$, the gradient of the $(L_g+L_g^2\mu)$-smooth value function, contributes $\gamma(L_g+L_g^2\mu)$. That $\alpha\sim 1/\gamma$ is the algorithmic shadow of the accuracy-$\gamma$ tradeoff.

The outer rate is projected-gradient descent plus a controlled estimation error. With $G_\gamma(z)=\tfrac1\alpha(z-\mathrm{Proj}_Z(z-\alpha\nabla F_\gamma(z)))$ the projected-gradient mapping and $C_f=\inf f$, the estimation error is the inner error in disguise, $\|\nabla F_\gamma-\widehat{\nabla F_\gamma}\|^2=\gamma^2\|\nabla v(x_k)-\nabla_x g(x_k,\hat y_k)\|^2\le \gamma^2 L_g^2 d_{S(x_k)}^2(\hat y_k)$, so choosing $T_k=\Omega(\log(\alpha k))$ drives it below $O(1/k^2)$, which is summable. Descending the $L_\gamma$-smooth $F_\gamma$ via the projection optimality $\mathrm{Proj}_Z(z-\alpha q)=\arg\min_{z'\in Z}\langle q,z'\rangle+\tfrac{1}{2\alpha}\|z-z'\|^2$ plus Young, and telescoping with $\sum 1/k^2\le 1$, gives
$$\frac1K\sum_{k=1}^{K}\|G_\gamma(x_k,y_k)\|^2 \;\le\; \frac{18\,(F_\gamma(x_1,y_1)-C_f)}{\alpha K} + \frac{10\,L^2 L_g^2}{K}.$$
Since $\alpha=\Theta(1/\gamma)$, reaching an $\varepsilon$-stationary point of the penalized problem costs $\tilde O(\gamma\varepsilon^{-1})$ outer iterations; choosing $\delta=\varepsilon$ forces $\gamma=O(\varepsilon^{-1/2})$ and a combined $\tilde O(\varepsilon^{-3/2})$. This is a finite-time *stationarity* result for $F_\gamma$ combined with the separate penalty-to-original relations, not a claim that every stationary point is a local minimum. A penalized stationary point maps to a bilevel KKT point with no constraint qualification, provided the tolerances are chosen consistently: an $\eta$-stationary point gives $\|\nabla_y g\|\le(L+\eta)/\gamma$, so $\gamma=\Omega(\delta^{-1/2})$ with $\eta=O(\delta)$ yields only an $O(\sqrt\delta)$ feasibility residual; an $O(\varepsilon)$ KKT residual requires picking $\gamma$ and $\eta$ so that $(L+\eta)/\gamma=O(\varepsilon)$, after which the multiplier candidate $w=\gamma(y-y^*)$ aligns the penalized conditions with the KKT blocks up to $O(\delta)$ via a Taylor expansion of $\nabla_y g$ and $\nabla_x g$ around $y^*$.

The method extends cleanly. For a compact convex lower constraint $U$ the inner loop becomes projected GD and a generalized Danskin proposition (under quadratic growth plus an error bound, or plus convexity) keeps $\nabla v(x)=\nabla_x g(x,y^*)$ — now the implicit term vanishes by the variational-inequality optimality of $y^*$ rather than by $\nabla_y g=0$ — giving $\tilde O(\varepsilon^{-3/2})$ (quadratic growth + convex) or $\tilde O(\varepsilon^{-2})$ (error-bound branch). A stochastic variant V-PBSGD uses inner SGD with $\beta_t=1/(L_g\sqrt t)$, samples $\hat y$ from the trajectory with $P(i=t)\propto\beta_t$, and an $M$-sample outer minibatch, at the cost of $O(\gamma^2 c^2/M)+O(\gamma^2\ln T/\sqrt T)$ noise terms — the $\gamma^2$ amplification is expected since the penalty scales the gradient noise. And the nagging fact that the value gap is only ever an $\varepsilon$-approximation, traceable to $p\sim d^2$ being *quadratic* near $S(x)$, suggests a penalty that grows *linearly* there: $p=\|\nabla_y g\|$, the norm not its square. For any $\gamma>L\sqrt\mu$ the global solutions of $\min f+\gamma\|\nabla_y g\|$ coincide *exactly* with the bilevel global solutions — constant $\gamma=O(1)$, no $\varepsilon$ gap — by the classical exact-penalty kink, because the error bound makes $\|\nabla_y g\|$ grow linearly like $d_{S(x)}$. The price is nonsmoothness at the optimum; the smooth-plus-norm composite is solved by a Prox-linear method at the better $\tilde O(\varepsilon^{-1})$ rate, but its subproblem involves the Jacobian of $\nabla_y g$ and rarely has a closed form, so the value gap remains the simple fully-first-order workhorse and the nonsmooth penalty the sharper-rate sibling.

Two implementations realize the value-gap direction directly. The toy verification collapses the inner loop, since for $f(x,y)=\cos(4y+2)/(1+e^{2-4x})+\tfrac12\ln((4x-2)^2+1)$ over $x\in[0,3]$ and $g(x,y)=(x+y)^2+x\sin^2(x+y)$ the solution map is $S(x)=\{-x\}$, so $v(x)=g(x,-x)=0$ identically, $p=g-v=g$, and $\nabla p=\nabla g$ with no inner solve. I take care to use the correct $x$-derivative of $f$: with $u=e^{2-4x}$, the term differentiates to $4u\cos(4y+2)/(1+u)^2$, the denominator being $(1+e^{2-4x})^2$:

```python
import numpy as np

def g(x, y):
    return (x + y) ** 2 + x * np.sin(x + y) ** 2

def dg(x, y):                                       # gradient of g in (x, y)
    return 2 * np.array([x + y + 0.5 * np.sin(x + y) ** 2 + x * np.sin(x + y) * np.cos(x + y),
                         x + y + x * np.sin(x + y) * np.cos(x + y)])

def f(x, y):
    return np.cos(4 * y + 2) / (1 + np.exp(2 - 4 * x)) + 0.5 * np.log((4 * x - 2) ** 2 + 1)

def df(x, y):
    return np.array([4 * np.exp(2 - 4 * x) * np.cos(4 * y + 2) / (1 + np.exp(2 - 4 * x)) ** 2
                     + (16 * x - 8) / ((4 * x - 2) ** 2 + 1),
                     -4 * np.sin(4 * y + 2) / (1 + np.exp(2 - 4 * x))])

def box(x, xlim):                                   # projection onto C = [0, 3]
    return min(max(x, min(xlim)), max(xlim))

def solve_vpbgd(x, y, alpha, gam, xlim, eps=1e-5):
    dF = df(x, y) + gam * dg(x, y)                  # grad F_gamma = grad f + gamma*grad(g - v)
    x_, y_ = box(x - alpha * dF[0], xlim), y - alpha * dF[1]
    pg = (1 / alpha) * np.array([x - x_, y - y_])   # projected-gradient mapping G_gamma
    k = 0
    while np.linalg.norm(pg) > eps:                 # stop at ||G_gamma|| <= eps
        x, y = x_, y_
        dF = df(x, y) + gam * dg(x, y)
        x_, y_ = box(x - alpha * dF[0], xlim), y - alpha * dF[1]
        pg = (1 / alpha) * np.array([x - x_, y - y_])
        k += 1
    return x, y, k

if __name__ == '__main__':
    xlim, gam, alpha0, N = [0., 3.], 10.0, 0.1, 1000
    for n in range(N):
        x0, y0 = np.random.uniform(0, 3.5), np.random.uniform(-5, 8.5)
        x, y, k = solve_vpbgd(x0, y0, alpha0 / gam, gam, xlim)   # alpha = alpha0 / gamma
        print('run', n, 'steps', k, '(x,y)=({:.3f},{:.3f})'.format(x, y))
```

The data hyper-cleaning implementation has a real inner loop. The upper variable $x$ is a per-example logit with weight $\sigma(x_i)$, the lower variable $y$ is the classifier, $g$ is the sigmoid-weighted cross-entropy on the corrupted training set and $f$ the cross-entropy on the clean validation set. A persistent auxiliary `net_inner`, initialized from `net` and warm-started across outer iterations, runs the inner GD to get $\hat y\approx y^*(x)$, and $v(x)\approx g(x,\hat y)$ is evaluated with the inner outputs *detached* — because $v$ carries no gradient through the inner iterate's $y$; its only live dependence is on $x$ through $\sigma(x)$. Forming $f+\gamma(g-v)$ and backpropagating then produces exactly $h_k$: the $y$-gradient of $g-v$ is $\gamma\nabla_y g$ (since $v$ is $y$-independent) and the $x$-gradient is $\gamma(\nabla_x g(x,y)-\nabla_x g(x,\hat y))$. I ramp $\gamma$ linearly from $0$ to $\gamma_{\max}$ and scale the objective by $\min(1/\gamma,1)$, realizing the $\alpha\sim1/\gamma$ shrinkage that $L_\gamma=\Theta(\gamma)$ demands:

```python
import copy
import torch
import torch.nn.functional as F

def sq_norm(params):
    return sum(torch.linalg.norm(w) ** 2 for w in params)

def run_vpbgd_hyperclean(net, x, tr, val, lrx, lry, lr_inner, inner_itr,
                         gamma_init, gamma_max, gamma_argmax_step, outer_itr, reg=0.0):
    # net: classifier (lower variable y); x: per-example logits (upper variable), weight=sigmoid(x)
    y_opt = torch.optim.SGD(net.parameters(), lr=lry)
    x_opt = torch.optim.SGD([x], lr=lrx)
    gam = gamma_init
    step_gam = (gamma_max - gamma_init) / gamma_argmax_step          # linear gamma ramp

    net_inner = copy.deepcopy(net); net_inner.train()               # persistent auxiliary inner net
    opt_inner = torch.optim.SGD(net_inner.parameters(), lr=lr_inner)

    for k in range(outer_itr):
        net.train()
        with torch.no_grad():
            sigx = torch.sigmoid(x)                                  # frozen weights for inner loop
        for _ in range(inner_itr):                                   # inner GD on g(x_k, .) -> y_hat
            opt_inner.zero_grad()
            ce = F.cross_entropy(net_inner(tr.data), tr.dirty_target, reduction='none')
            inner_loss = (sigx * ce).mean() + reg * sq_norm(net_inner.parameters())
            inner_loss.backward()
            opt_inner.step()

        y_opt.zero_grad(); x_opt.zero_grad()
        fy = F.cross_entropy(net(val.data), val.clean_target)                          # f(x, y)
        ce_tr = F.cross_entropy(net(tr.data), tr.dirty_target, reduction='none')
        gxy = (torch.sigmoid(x) * ce_tr).mean() + reg * sq_norm(net.parameters())      # g(x, y)
        ce_in = F.cross_entropy(net_inner(tr.data), tr.dirty_target, reduction='none').detach()
        vx = (torch.sigmoid(x) * ce_in).mean() + reg * sq_norm(net_inner.parameters()).detach()
        # v(x) ~ g(x, y_hat): y_hat side detached, x side live through sigmoid(x)

        lr_decay = min(1.0 / (gam + 1e-8), 1.0)                      # realizes alpha ~ 1/gamma
        loss = lr_decay * (fy + gam * (gxy - vx))                    # F_gamma = f + gamma*(g - v)
        loss.backward()                            # -> grad f + gamma*(grad g - (grad_x g(.,y_hat),0))
        x_opt.step(); y_opt.step()
        gam = min(gamma_max, gam + step_gam)
```
