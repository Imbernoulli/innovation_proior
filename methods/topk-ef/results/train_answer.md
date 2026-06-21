When I train a model data-parallel across many workers, the thing that actually hurts is not the arithmetic, it is the synchronization. Every step each worker computes a stochastic gradient $g_t$ on its local minibatch, and before the optimizer can move, the workers must all-reduce that gradient into a single global sum. But that gradient is the whole model laid flat — millions of 32-bit floats shipped over the wire every iteration — and when I profile it the GPUs sit idle waiting on the network. So the goal is brutally simple to state: send a hundred or a thousand times fewer bits per step and still land on the same model full-gradient SGD would have reached. The difficulty is equally simple: the instant I send a lossy version of $g_t$ I perturb the descent direction, and a perturbed direction can slow convergence, bias where I end up, or stop it converging at all. I need a compressor aggressive enough to matter and a reason to believe the optimizer still arrives where SGD would.

There is a structural fact that invites compression: gradients in deep networks are strongly positively skewed. Most coordinates are near zero and a handful are large — in a translation model's embedding matrix, a minibatch only touches a few vocabulary rows, so only those get a real gradient. Concretely, most of the gradient's energy $\|g\|^2$ lives in a few coordinates, so keeping the $k$ largest-magnitude coordinates and zeroing the rest preserves almost all the energy while sending only $k$ values plus $k$ indices. That is the natural compressor, $\mathrm{top}_k(g)$. The existing options all fall short on the same axis. Unbiased stochastic quantization (QSGD, TernGrad) randomizes its rounding so that $\mathbb{E}[Q(g)] = g$; that plugs straight into the SGD proof, paying only a variance factor, but it has a bits floor — even at the coarsest level it must ship the sign and index of order $\sqrt{d}$ coordinates, so it never reaches the constant-coordinates-per-step regime, and pushing harder just inflates the variance and the iteration count. Sign compression (1-bit SGD, signSGD) is one bit per coordinate and very cheap, but it is biased, $\mathbb{E}[\mathrm{sign}(g)] \neq \nabla f$, and discards all magnitude information; its guarantees need benign noise or growing batch sizes. Magnitude sparsification (gradient dropping, top-$k$) is empirically excellent at 99–99.9% drop rates, but it too is biased, $\mathbb{E}[\mathrm{top}_k(g)] \neq g$, and had no convergence guarantee — with a concrete failure mode: a coordinate whose magnitude is persistently small never enters the top-$k$, so its direction is permanently starved. Mem-SGD put the local-memory variant on a footing but only in the smooth, strongly convex case. So the field had strong empirical recipes for aggressive biased compression and only partial theory.

I should not wave away how badly a biased compressor can break, because understanding the failure tells me what my fix must protect. Take the one-dimensional gradient $g = +4$ with probability $1/4$ and $g = -1$ with probability $3/4$: the mean is $\tfrac14(4)+\tfrac34(-1) = \tfrac14 > 0$, so true descent moves left, but $\mathbb{E}[\mathrm{sign}(g)] = \tfrac14(+1)+\tfrac34(-1) = -\tfrac12 < 0$ moves right — the sign discarded the rare large magnitude that should have dominated the average. Worse, on $f(x) = \varepsilon|x_1+x_2| + |x_1-x_2|$ with $0<\varepsilon<1$, starting at $(1,1)$ signSGD only ever moves along $(1,-1)$ while $x_1+x_2$ stays frozen at $2$ for any step-size schedule, so it never reaches the origin — a fixed direction $\varepsilon(1,1)$ is thrown away at every step. Naive top-$k$ has exactly this same disease: the persistently small coordinates are dropped forever and never accumulate into a move. The information in them is not worthless, it is just below threshold at each instant.

So the proposal — I call it EF-TopK, top-$k$ sparsification with error feedback — is to stop throwing the suppressed coordinates away and instead remember them. I maintain a local residual vector $e_t$, the running total of everything I have suppressed so far. Each step, before I compress, I add the residual back into the (step-size-scaled) gradient, compress that, step with the result, and stash whatever did not make the cut back into the residual. Writing the loop with $e_0 = 0$ and a generic compressor $C$:
$$p_t = \gamma\,g_t + e_t,\qquad \Delta_t = C(p_t),\qquad x_{t+1} = x_t - \Delta_t,\qquad e_{t+1} = p_t - \Delta_t.$$
The step size $\gamma$ is folded into $p_t$ *before* compression, so what travels is $C(\gamma g_t + e_t)$ and the residual is in update units. A coordinate that is persistently small now keeps accumulating in $e_t$ until it finally crosses into the top-$k$ and is sent in one shot. Nothing is forgotten; it is only delayed.

What makes this provably work, rather than just feel right, is a change of variable. The residual is an owed-but-unapplied update, and updates are subtracted from the iterate, so I define the virtual iterate $\tilde{x}_t = x_t - e_t$, the point at which the delayed update has already been paid. Its recursion collapses beautifully:
$$\tilde{x}_{t+1} = x_{t+1} - e_{t+1} = (x_t - \Delta_t) - (p_t - \Delta_t) = x_t - p_t = (x_t - e_t) - \gamma g_t = \tilde{x}_t - \gamma g_t.$$
The $\Delta_t$ and the compression cancel exactly: the virtual iterate runs honest, uncompressed SGD. So error feedback is not approximating SGD heuristically — it is SGD on a shadow sequence $\tilde{x}$, and the only gap between $\tilde{x}_t$ and the real $x_t$ is the residual $e_t$. If $e_t$ stays bounded, then $x_t \approx \tilde{x}_t$, and on a smooth function $\nabla f(x_t) \approx \nabla f(\tilde{x}_t)$, so the descent on $\tilde{x}$ carries over to $x$. Error feedback is a delayed gradient method, and on a smooth function a small delay is cheap.

The whole argument therefore hinges on bounding $e_t$, and for that I stop talking about top-$k$ specifically and isolate the one property I use: $C$ is a $\delta$-approximate compressor, $\delta \in (0,1]$, if $\|C(x) - x\|^2 \le (1-\delta)\|x\|^2$ for all $x$ — a contraction that bounds the dropped energy as a fraction of the input. Top-$k$ satisfies this with $\delta = k/d$. The argument is a comparison to random-$k$: since top-$k$ keeps the $k$ largest coordinates, it drops the least possible energy of any size-$k$ keep set, so pointwise $\|x - \mathrm{top}_k(x)\|^2 \le \|x - \mathrm{rand}_k(x)\|^2$, and random-$k$ keeps each coordinate with probability $k/d$, giving $\mathbb{E}_\omega\|x - \mathrm{rand}_k(x)\|^2 = \sum_i x_i^2(1 - k/d) = (1-k/d)\|x\|^2$. Hence $\delta = k/d$ (and $\delta = 1/d$ for the most aggressive top-1). Now the residual: $\|e_{t+1}\|^2 = \|C(p_t) - p_t\|^2 \le (1-\delta)\|e_t + \gamma g_t\|^2$. The cross term couples $e_t$ and $g_t$, so I split with Young's inequality $\|a+b\|^2 \le (1+\eta)\|a\|^2 + (1+1/\eta)\|b\|^2$ and choose $\eta = \delta/(2(1-\delta))$, which makes the contraction factor $(1-\delta)(1+\eta) = 1 - \delta/2 < 1$ and the injection factor $1+1/\eta = (2-\delta)/\delta \le 2/\delta$. Unrolling the linear recursion $a_{t+1} \le (1-\delta/2)a_t + c$ from $e_0 = 0$ with $\mathbb{E}\|g_t\|^2 \le \sigma^2$ gives the geometric-series bound
$$\mathbb{E}\|e_t\|^2 \le \frac{4(1-\delta)\,\gamma^2\sigma^2}{\delta^2},$$
which is $O(\gamma^2)$, finite, and exactly zero at $\delta = 1$. The memory never blows up.

Cashing this in on the virtual SGD iterate, $L$-smoothness gives $\mathbb{E}_t[f(\tilde{x}_{t+1})] \le f(\tilde{x}_t) - \gamma\langle\nabla f(\tilde{x}_t), \nabla f(x_t)\rangle + \tfrac{L\gamma^2}{2}\sigma^2$. I never observe $\tilde{x}_t$, so I trade $\nabla f(\tilde{x}_t)$ for $\nabla f(x_t)$ via Young's inequality with parameter $\rho$ and the Lipschitz-gradient bound $\|\nabla f(x_t) - \nabla f(\tilde{x}_t)\|^2 \le L^2\|x_t - \tilde{x}_t\|^2 = L^2\|e_t\|^2$. Substituting the residual bound, telescoping, and dividing by $\gamma(1-\rho/2)$ yields, for any $0 < \rho < 2$,
$$\frac{1}{T+1}\sum_{t=0}^{T}\mathbb{E}\|\nabla f(x_t)\|^2 \le \frac{f_0}{\gamma(1-\rho/2)(T+1)} + \frac{L\gamma\sigma^2}{2-\rho} + \frac{4\gamma^2 L^2\sigma^2(1-\delta)}{\rho(2-\rho)\delta^2},$$
with $f_0 = f(x_0) - f^\star$. Taking $\rho = 1$ and $\gamma = 1/\sqrt{T+1}$ gives a leading $O(1/\sqrt{T})$ term with the compression quality $\delta$ appearing only in the higher-order $O(1/T)$ term; letting $\rho$ shrink slowly with $T$ drives the first two constants toward the plain-SGD constants. That is the precise "compression for free" statement: $\delta$ never enters the leading $O(1/\sqrt{T})$ term, and the starvation that killed naive top-$k$ is gone because suppressed signal is carried forward rather than discarded. This is not a fluke of smoothness — in the non-smooth convex case the Lipschitz-gradient bridge is unavailable, so a Cauchy-Schwarz cross term puts $\delta$ into the leading constant, $\mathbb{E}[f(\bar{x}_T)] - f^\star \le \|x_0-x^\star\|^2/(2\gamma(T+1)) + \gamma\sigma^2(\tfrac12 + 2\sqrt{1-\delta}/\delta)$, but the method still converges at the right $1/\sqrt{T}$ order, which the naive biased compressor could not even guarantee, and for $k=1$ this is a convergent greedy-coordinate method on non-smooth functions. The same virtual-iterate identity also restores generalization: $x_t - e_t = x_0 - \sum_{i<t}\gamma g_i$ lies exactly in the gradient span when $x_0 = 0$, so $x_t$ is within $\|e_t\|$ of the span, recovering the min-norm / max-margin solution that biased compression breaks.

In code the mechanism is just made local to each parameter tensor — per-name residual, because each tensor has its own scale and "largest magnitude" only means something within a block. The `compress` body adds the stored residual ($p_t = g_t + e_t$, with $\gamma$ left to the optimizer after decompression), flattens, takes $k = \max(1, \lfloor d\cdot\text{ratio}\rfloor)$ where the $\max(1,\cdot)$ guarantees even a tiny tensor sends at least one coordinate so nothing is silenced forever, selects the top-$k$ indices by magnitude, gathers their values, and stores everything not sent ($p_t - C(p_t)$) back into the residual. Only the `[values, indices]` payload — $2k$ numbers, the $100\times$ saving — goes on the wire; the residual and the `(numel, shape)` context stay local. The `decompress` body scatters the kept values into a zero buffer and reshapes, the zeros being the dropped coordinates that are not lost but waiting in the residual for next step.

```python
import torch


class Compressor:
    """Top-K sparsification with error feedback (EF-TopK).

    Keeps the k = max(1, int(d * compress_ratio)) largest-magnitude coordinates
    of each gradient tensor. The (d-k) suppressed coordinates are accumulated in a
    per-tensor residual and added back before the next compression, so persistent
    suppressed signal is delayed rather than erased -- which makes the biased
    top-k compressor match SGD's convergence rate."""

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        self.residuals = {}                       # e[name]: local memory, NOT communicated

    def compress(self, tensor, name):
        # error correction: p_t = g_t + e_t
        if name in self.residuals:
            tensor = tensor + self.residuals[name]

        shape = tensor.size()
        tensor_flat = tensor.flatten()
        numel = tensor_flat.numel()
        k = max(1, int(numel * self.compress_ratio))

        # top-k by magnitude
        _, indices = torch.topk(tensor_flat.abs(), k, sorted=False)
        values = torch.gather(tensor_flat, 0, indices)

        # residual = what was NOT sent: e_{t+1} = p_t - C(p_t)
        decompressed_flat = self.decompress([values, indices], (numel, shape)).flatten()
        self.residuals[name] = (tensor_flat - decompressed_flat).view(shape)

        return [values, indices], (numel, shape)

    def decompress(self, compressed_tensors, ctx):
        values, indices = compressed_tensors
        numel, shape = ctx
        tensor_decompressed = torch.zeros(
            numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)
```
