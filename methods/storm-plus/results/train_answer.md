We want to minimize a smooth but non-convex objective seen only through noise, $F(x) = \mathbb{E}_{\xi\sim D}[f(x;\xi)]$, with access to nothing but first-order stochastic gradients $\nabla f(x;\xi)$. Since global minimization is intractable here, the honest target is an approximate stationary point: after $T$ oracle calls, output $\bar x$ with $\mathbb{E}\lVert\nabla F(\bar x)\rVert$ as small as possible. For general smooth non-convex $F$, tuned SGD drives this down at $O(1/T^{1/4})$ and that is tight; but on the narrower and very common class where each sampled loss $f(\cdot,\xi)$ is itself $L$-smooth — expectation over smooth losses — the optimal rate improves to $O(1/T^{1/3})$, and variance reduction is what buys the gap. The trouble is that every existing way of reaching $1/T^{1/3}$ has a sting. The SVRG/SARAH/SPIDER/PAGE family does it with checkpoints: every so often it burns a mega-batch of $N$ samples (with $N$ as large as $O(T)$, never smaller than about $T^{2/3}$) to compute a low-noise anchor gradient, then runs cheap recursive steps off it. That mega-batch makes no progress while it computes, and the batch size, the checkpoint frequency, and an $L$-tied step cap are all knobs that must be balanced against constants one never knows. STORM (Cutkosky and Orabona 2019) is the method I actually like, because it reaches the optimal rate with no checkpoint and no mega-batch at all, by keeping a corrected-momentum gradient estimate. But it sets its step $\eta_t \propto (\sum\lVert g\rVert^2)^{-1/3}$ and its momentum $a_{t+1} = c\,\eta_t^2$ with constants $k = bG^{2/3}/L$, $c = 28L^2 + \dots$, $w = \max((4Lk)^3, 2G^2,\dots)$ — so it must be told the smoothness $L$ and a gradient-norm bound $G$, and its proof needs $a_{t+1} \propto L^2\eta_t^2$ to balance, collapsing if those constants are stripped out. So the goal I cannot let go of: keep STORM's checkpoint-free variance reduction and its $O(1/T^{1/3})$ rate, but set $\eta_t$ and $a_t$ from observed quantities alone — no $L$, no $G$, no $\sigma$ — and additionally adapt to the noise level, running at the faster $O(1/\sqrt T)$ when the problem is nearly noiseless.

I propose STORM+, a fully parameter-free, checkpoint-free variance-reduced optimizer. It keeps STORM's corrected-momentum estimate, refreshed with two gradient calls on the same fresh sample $\xi_t$ at the current and previous iterate,
$$d_t = \nabla f(x_t;\xi_t) + (1-a_t)\big(d_{t-1} - \nabla f(x_{t-1};\xi_t)\big), \qquad x_{t+1} = x_t - \eta_t\,d_t.$$
Expanded, $d_t = a_t g_t + (1-a_t)d_{t-1} + (1-a_t)(g_t - \tilde g_{t-1})$ with $g_t = \nabla f(x_t;\xi_t)$ and $\tilde g_{t-1} = \nabla f(x_{t-1};\xi_t)$: plain momentum plus one within-sample gradient difference. The reason that difference earns its second oracle call shows up in the error $\varepsilon_t = d_t - \nabla F(x_t)$, which obeys
$$\varepsilon_t = (1-a_t)\varepsilon_{t-1} + a_t\big(g_t - \nabla F(x_t)\big) + (1-a_t)Z_t, \qquad Z_t = \big(g_t - \tilde g_{t-1}\big) - \big(\nabla F(x_t) - \nabla F(x_{t-1})\big).$$
The first term contracts the old error by $(1-a_t)$; the second is fresh zero-mean noise of size $\sigma$, premultiplied by $a_t$ so a small momentum squashes it; the third is also zero-mean given the past, and by per-sample $L$-smoothness $\lVert Z_t\rVert \le 2L\lVert x_t - x_{t-1}\rVert = 2L\,\eta_{t-1}\lVert d_{t-1}\rVert$, so it shrinks with the step. That is the whole engine: the error contracts toward a small equilibrium, and the noise decreases without ever computing an anchor.

What makes STORM+ work is the parameter-free schedule that replaces STORM's $L$- and $G$-laden constants. I set the momentum from the cumulative raw-gradient energy and the step from the cumulative estimate energy:
$$a_{t+1} = \frac{1}{\big(1 + \sum_{i\le t}\lVert g_i\rVert^2\big)^{2/3}}, \qquad \eta_t = \frac{1}{\big(\sum_{i\le t}\lVert d_i\rVert^2 / a_{i+1}\big)^{1/3}}.$$
Each piece is forced, not free. The AdaGrad shape — a schedule inversely proportional to a power of accumulated squared norms — is what makes the relevant sums summable, through the telescoping identity $\sum_i b_i / (\sum_{j\le i} b_j)^p \le \frac{1}{1-p}(\sum_i b_i)^{1-p}$ for $p\in(0,1)$ (proved by induction: $h(x) = \frac{1}{1-p}(Z-x)^{1-p} + x/Z^p$ is concave and maximized at $x=0$). For the step I use exponent $1/3$, not AdaGrad's $1/2$: the curvature tax $\frac{L\eta_t^2}{2}\lVert d_t\rVert^2$ and the smoothness term both carry $\eta_{t-1}^2\lVert d_{t-1}\rVert^2$, and with $\eta_t \propto (\sum\lVert d_i\rVert^2)^{-1/3}$ this is exactly the $b/(\sum b)^{2/3}$ shape with $p=2/3$, telescoping to a cube root that lands $1/T^{1/3}$; $p=1/2$ would only deliver the SGD-flavored $1/T^{1/4}$. Crucially the denominator accumulates the estimate norms $\lVert d_i\rVert^2$, not the raw $\lVert g_i\rVert^2$, because for the telescope to fire the quantity in the denominator must be the same one sitting in the numerator — the terms I am taming are literally built from $\lVert d_{t-1}\rVert^2$, so an $\lVert g\rVert^2$ denominator would leave nothing to cancel. The momentum uses exponent $2/3$ on the raw-gradient energy because once a noise floor keeps $\sum\lVert g_i\rVert^2$ growing like $t$, this gives $a_t \approx t^{-2/3}$, exactly the decay STORM wanted from $a_t \propto L^2\eta_t^2$ but with no $L^2$ anywhere; its square $a_t^2 \propto (\cdot)^{-4/3}$ then lands the fresh-noise term at a constant rather than a logarithm (exponent $4/3 > 1$ is the threshold). The $+1$ inside makes $a_1 = 1$ — honestly trusting the first sample fully when there is no history — and keeps $a_t \in (0,1]$, which the convex-weight recursion needs.

The headline is the $1/a_{i+1}$ reweighting inside $\eta_t$. STORM coupled step and momentum as $a \propto L^2\eta^2$, using its knowledge of $L$; I refuse $L$, so I cannot write $a$ in terms of $\eta$ that way, yet the proof's balance between the smoothness term and the variance term still demands a coupling. I forge it the only computable way — by reversing the dependence, making the step depend on the momentum. The factor $1/a_{i+1} \approx (1 + \sum\lVert g\rVert^2)^{2/3} \approx t^{2/3}$ inflates the step's denominator, so $\eta_t$ shrinks faster than a plain $(\sum\lVert d\rVert^2)^{-1/3}$ would — fast enough to control $Z_t$ because that extra $t^{2/3}$ of shrinkage does the job STORM's explicit $L^2$ used to do, parameter-free. The index $a_{i+1}$ is not really the future: at iteration $t$, $a_{t+1}$ is formed from $g_1,\dots,g_t$, all already in hand before the step is taken, so it is fully computable.

Three layers verify the construction. With $\sigma=0$ one shows $d_t = \nabla F(x_t)$ exactly, and smoothness plus the $p=1/3$ telescope (and summation by parts using $\Delta_t = F(x_t)-F(x^*)\in[0,B]$) give $\sum_t\lVert\nabla F(x_t)\rVert^2 \le O(1 + L^3 + B^{9/4})$, hence $O(1/\sqrt T)$ — the step alone, with its $1/a$ reweighting, already does the right thing. In the stochastic case one bounds $\mathbb{E}\sum\lVert\varepsilon_t\rVert^2$ from the error recursion (the noise and $Z_t$ pieces being zero-mean given the past), then converts it through a two-case split on whether $\mathbb{E}\sum\lVert\varepsilon\rVert^2$ exceeds $\tfrac12\mathbb{E}\sum\lVert\nabla F\rVert^2$ into $\mathbb{E}\sum_t\lVert\nabla F(x_t)\rVert^2 \le O(M^2 + \kappa^2\sigma^{2/3}T^{1/3})$. The genuinely random $a_t$ — whose reciprocal increments $1/a_{t+1}-1/a_t$ are no longer bounded by $2/3$, since one large $\lVert g_t\rVert^2$ can make them jump — is tamed by a stopping time. With $\beta = \min\{1, 1/G^4\}$ and $\tau^* = \max\{t: a_t \ge \beta\}$, concavity of $y^{2/3}$ restores $1/a_{t+1}-1/a_t \le 2/3$ once $a_t < \beta$, while before $\tau^*$ one has $1/a_{t+1} \le 1/\tilde\beta = (1/\beta^{3/2}+G^2)^{2/3}$; the cross terms are martingale differences killed by Doob's optional stopping; the variance term is pinned at a constant by the exponent-$4/3$ dyadic-blocking lemma $\sum a_i/(1+\sum_{j\le i}a_j)^{4/3} \le 12$; and Young's inequality with $\rho = (512L^2)^{1/3}$ (exponents $3/2$ and $3$) splits the mixed gradient-energy/estimate-energy products. Taking the worse case and a uniformly random iterate $\bar x_T$ with Jensen gives
$$\mathbb{E}\lVert\nabla F(\bar x_T)\rVert \le O\!\left(\frac{M}{\sqrt T} + \frac{\kappa\,\sigma^{1/3}}{T^{1/3}}\right),$$
with $\kappa = O(B^{3/4} + L^{3/2})$ and $M = O(1 + L^{9/4} + B^{9/8} + G^5 + (LG^4)^{3/2})$. The rate is the optimal $O(1/T^{1/3})$ when noisy, the second term vanishes when $\sigma=0$ to leave $O(1/\sqrt T)$, and $L,G,B,\sigma$ live only in the constants — never in the schedules the algorithm computes. The method needs only $d_1 = g_1$ from a single sample, two oracle calls per step (which share the sample $\xi_{t+1}$), one scalar step size and one scalar momentum per iteration, no batch and no anchor, and nothing to tune but an overall step-size scale.

```python
import torch


def grad(model, loss_fn, x, batch):
    """One stochastic gradient grad f(x; xi), flattened to a single vector."""
    set_flat_params(model, x)
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model, batch)
    loss.backward()
    return flat_grad(model)                       # 1-D tensor over all parameters


class StormPlus:
    """STORM+ : fully parameter-free, no-checkpoint variance reduction.

        a_{t+1} = 1 / (1 + sum_{i<=t} ||g_i||^2) ^ (2/3)
        eta_t   = 1 / (sum_{i<=t} ||d_i||^2 / a_{i+1}) ^ (1/3)
        x_{t+1} = x_t - eta_t d_t
        d_{t+1} = g_{t+1} + (1 - a_{t+1}) (d_t - grad f(x_t; xi_{t+1}))
    """

    def __init__(self, model, loss_fn, x_init):
        self.model = model
        self.loss_fn = loss_fn
        self.x = x_init.clone()              # current iterate x_t
        self.g = None                        # current stochastic gradient g_t
        self.d = None                        # gradient estimate d_t
        self.sum_g2 = 0.0                    # sum ||g_i||^2  (gradient energy)
        self.sum_d2_over_a = 0.0             # sum ||d_i||^2 / a_{i+1}  (estimate energy)

    def step(self, batch=None):
        if self.d is None:                                   # warm start: d_1 = g_1 (one sample)
            if batch is None:
                batch = draw_sample()
            self.g = grad(self.model, self.loss_fn, self.x, batch)
            self.d = self.g.clone()

        g_t = self.g                                         # grad f(x_t; xi_t)

        # momentum a_{t+1} from cumulative gradient norms -> ~ t^{-2/3}, in (0, 1]
        self.sum_g2 += g_t.pow(2).sum().item()
        a_next = 1.0 / (1.0 + self.sum_g2) ** (2.0 / 3.0)

        # step eta_t from cumulative estimate norms reweighted by 1/a (lr <-> momentum coupling)
        self.sum_d2_over_a += self.d.pow(2).sum().item() / a_next
        eta_t = 1.0 / (self.sum_d2_over_a) ** (1.0 / 3.0)

        x_old = self.x.clone()
        with torch.no_grad():
            self.x = self.x - eta_t * self.d                 # x_{t+1} = x_t - eta_t d_t

        # refresh on a FRESH sample: two oracle calls (current and previous iterate), no checkpoint
        next_batch = draw_sample()
        g_next = grad(self.model, self.loss_fn, self.x, next_batch)    # grad f(x_{t+1}; xi_{t+1})
        g_tilde = grad(self.model, self.loss_fn, x_old, next_batch)    # grad f(x_t;     xi_{t+1})
        self.d = g_next + (1.0 - a_next) * (self.d - g_tilde)          # corrected momentum
        self.g = g_next                                                # g_{t+1} for next iteration
        return eta_t, a_next
```
