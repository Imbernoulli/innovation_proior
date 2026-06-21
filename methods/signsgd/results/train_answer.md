When I train a $d$-parameter network across $M$ workers, the thing that actually hurts is not the arithmetic — it is the gradient exchange. Every worker computes a stochastic gradient on its slice of the minibatch, and before anyone can step those gradients must go up to a parameter server, be aggregated, and come back down. The usual accounting is a 32-bit float per coordinate going up and another coming back, so $64 M d$ bits per iteration, and with modern nets at $d > 10^8$ that wall grows worse the more workers I add — exactly backwards from what I want parallelism to do. So the real problem is not "optimize faster," it is "say less per coordinate," ideally one bit in *both* directions, without giving up the $O(1/\sqrt N)$ non-convex rate that full-precision SGD enjoys, and with a guarantee that is not vacuous at scale.

The existing options each get part of this and miss the rest. Full-precision distributed SGD is the rate to match but does nothing for communication. The principled compressors — QSGD and TernGrad — stochastically round each coordinate so the compressed gradient stays an *unbiased* estimate of the true gradient, which lets them bootstrap standard SGD theory verbatim; but unbiasing a one-bit message demands a brutal randomization that inflates the variance by a factor of order $\sqrt d$, so at $d > 10^8$ the bound they inherit carries a $\sqrt d$-sized constant and says nothing, and the server's sum of quantized updates is no longer one bit, so the return path picks up log factors. Seide's 1-bit SGD goes the other way — thresholded sign with the discarded magnitude carried forward as an error-feedback residual — and is near-lossless in practice, but it is a heuristic with no convergence guarantee. Rprop and the whole RMSprop/Adam family are *already* sign-flavored (Adam's step is a mean over a root-mean-square, and shrinking both EMA timescales $\beta_1,\beta_2 \to 0$ collapses it to $g/\sqrt{g^2} = \mathrm{sign}(g)$), but they transmit full-precision rescaled gradients and so buy nothing for communication. The recurring pattern is stark: aggressive compression with no theory, or theory whose variance explosion makes it empty at deep-learning scale.

I propose signSGD, with its momentum variant Signum and its distributed majority-vote form. The move is to transmit only the sign of each stochastic gradient coordinate — one bit per parameter — and step $$x_{k+1} = x_k - \delta\,\mathrm{sign}(\tilde g_k).$$ This is the crudest possible compression of a number: throw away exponent and mantissa, keep one bit. Everyone else avoided the *biased* sign by randomizing to unbias it; I keep the bias and confront it in the analysis, because the bias turns out to be exactly controllable. A coordinate's sign is wrong only when the noise overpowers the signal, so for $g_i \ne 0$, $$P[\mathrm{sign}(\tilde g_i) \ne \mathrm{sign}(g_i)] \le P[|\tilde g_i - g_i| \ge |g_i|] \le \frac{\mathbb{E}|\tilde g_i - g_i|}{|g_i|} \le \frac{\sigma_i}{|g_i|}$$ by Markov on the absolute deviation followed by Jensen ($\mathbb{E}|X| \le \sqrt{\mathbb{E}X^2}$) to get from the second moment I control to the first absolute moment I needed. Now multiply by the cost a flipped sign actually inflicts, which is the gradient magnitude $|g_i|$ there, and the magnitude *cancels*: $|g_i|\,P[\text{wrong}] \le \sigma_i$. That cancellation is the whole trick. The damage of the bias is capped by the noise scale alone, independent of how large or small the gradient is — and if $g_i = 0$ the descent cost is zero anyway, so there is nothing to divide by. This is precisely the structure an unbiased scheme cannot use, because it spends all of its design forcing the expectation to be exact rather than letting it be approximately-right-where-it-matters.

The right geometry for a sign step is $\ell_\infty$, not $\ell_2$. The move $-\delta\,\mathrm{sign}(\tilde g)$ is a vector of $\pm\delta$, so it lives on a box of side $\delta$, and non-stochastic signSGD is exactly $\ell_\infty$ steepest descent, $\arg\min\{g^\top v : \|v\|_\infty \le 1\} = -\mathrm{sign}(g)$. The matching majorization is therefore coordinate-wise smoothness, $|f(y) - [f(x)+g(x)^\top(y-x)]| \le \tfrac12\sum_i L_i(y_i-x_i)^2$ for a vector $L \ge 0$, which is strictly finer than the scalar $L$-smoothness it recovers (set $L := \|L\|_\infty$) and which keeps the per-coordinate distribution of curvature that a scalar constant would throw away; likewise I assume a per-coordinate variance bound $\mathbb{E}[(\tilde g_i - g_i)^2] \le \sigma_i^2$ with an unbiased oracle, recovering the total variance $\sigma^2 := \|\sigma\|_2^2$ by summing, and a minibatch of size $n_k$ squashes coordinate variance to $\sigma_i^2/n_k$. Plugging the sign step into the coordinate-smoothness inequality, each $(\mathrm{sign})_i^2 = 1$, so the curvature term is just $(\delta_k^2/2)\|L\|_1$; and using the identity $g_k^\top \mathrm{sign}(\tilde g_k) = \|g_k\|_1 - 2\sum_i |g_{k,i}|\,\mathbb{1}[\text{disagree}]$ turns the progress into an $\ell_1$ term minus an error term that the key bound caps at $\sigma_i/\sqrt{n_k}$, giving $$\mathbb{E}[f_{k+1}-f_k \mid x_k] \le -\delta_k\|g_k\|_1 + 2\frac{\delta_k}{\sqrt{n_k}}\|\sigma\|_1 + \frac{\delta_k^2}{2}\|L\|_1.$$ Because the sign is biased I cannot kill the noise with a decaying learning rate alone the way SGD does — I must *reduce* the noise, which means growing the batch. With $\delta_k = 1/\sqrt{\|L\|_1 K}$ (balancing the quadratic-in-$\delta$ curvature cost against the linear-in-$\delta$ gain) and $n_k = K$, telescoping over $k=0\dots K-1$ and squaring gives $$\mathbb{E}\!\left[\tfrac1K\textstyle\sum_k \|g_k\|_1\right]^2 \le \frac{1}{\sqrt N}\Big[\sqrt{\|L\|_1}\,(f_0 - f^* + \tfrac12) + 2\|\sigma\|_1\Big]^2,$$ where $N = O(K^2)$ gradient calls turns the $1/\sqrt K$ into $1/\sqrt N$. There is the SGD-class rate for a one-bit method on a non-convex objective, with no $d$ factor on the noise. The growing batch looks wasteful but is a systems *win*: $N$ calls happen in $O(\sqrt N)$ iterations, hence $O(\sqrt N)$ rounds of the expensive communication rather than $O(N)$.

Whether the magnitude-blind sign helps or hurts versus SGD is a question of *relative density*, made precise by $\phi(v) := \|v\|_1^2/(d\|v\|_2^2)$, which is $1$ for a dense vector and $\sim 1/d$ for a sparse one. Translating the $\ell_1$ bound into the squared expected average $\ell_2$ norm through $\phi$ yields $\mathbb{E}[\tfrac1K\sum\|g_k\|_2]^2 \le (2/\sqrt N)[R_1 L (f_0-f^*+\tfrac12)^2 + 4R_2\sigma^2]$ against SGD's $(1/\sqrt N)[2L(f_0-f^*)+\sigma^2]$, with $R_1 = \sqrt{\phi(L)}/\phi(g)$ on the curvature term and $R_2 = \phi(\sigma)/\phi(g)$ on the noise. The sign matches or beats SGD when noise is no denser than the gradient ($R_2$ not $\gg 1$) — intuitively, when noise is sparse and loud, the sign cannot be screamed at, since every coordinate moves by the same $\delta$ and the loud coordinates get scaled *down* relative to the dense signal — and it loses when curvature is much denser than the gradient ($R_1 \gg 1$). Which regime are real nets in? This I can measure before committing: Welford's single-pass algorithm gives the exact gradient and per-coordinate variance, and on a Resnet-20/CIFAR-10 run $\phi(g)$ and $\phi(\sigma)$ stay of the *same order* and both dense throughout training, across architectures — the favorable regime, decisively.

A small-batch story ($n_k = 1$) needs a sharper flip bound than the crude $\sigma_i/|g_i|$, so I assume the per-coordinate noise is unimodal and symmetric, which the central-limit theorem makes mild for any non-tiny batch and which histograms of real gradient noise confirm. Gauss's inequality then gives, with $S_i := |g_i|/\sigma_i$, a flip probability $(2/9)/S_i^2$ when $S_i > 2/\sqrt3$ and $\tfrac12 - S_i/(2\sqrt3)$ otherwise — strictly below $\tfrac12$ for every nonzero signal. Redoing the one-step bound with this case split produces a *mixed* norm: high-SNR coordinates converge in $\ell_1$ ($|g|$), low-SNR ones in a variance-weighted $\ell_2$ ($g^2/\sigma$), which is structurally sensible — ride the $\ell_1$ geometry while the sign is reliable, fall back to a gentler quadratic as the SNR degrades — and again hits $1/\sqrt N$ with the noise entering *linearly* rather than quadratically.

The part that pays for the whole exercise is compressing the *return* trip. Each worker sends $\mathrm{sign}(\tilde g_m)$ up — one bit — but $\sum_m \mathrm{sign}(\tilde g_m)$ is an integer in $[-M,M]$, so the broadcast back is not one bit. The fix is to take the sign of the vote count: $$x_{k+1} = x_k - \delta\,\mathrm{sign}\!\Big[\textstyle\sum_{m=1}^M \mathrm{sign}(\tilde g_m)\Big].$$ Majority vote — one bit up, one bit down, $2Md$ bits an iteration against SGD's $64Md$. To show it costs nothing I only need the same inequality $|g_i|\,P[\text{majority wrong}] \le \sigma_i$. For $S \le 1$ it is trivial; for $S > 1$, a single worker is right more than half the time (Cantelli gives failure $q \le 1/(1+S^2) < \tfrac12$ with no shape assumption), so the server is receiving a *repetition code* — the same true bit through $M$ independent noisy channels — and strict majority is its maximum-likelihood decoder, which only lowers the error. Under unimodal symmetry I get more: the binomial tail $P[Z \le M/2] \le 1/(\sqrt M S)$ for $Z \sim \mathrm{Binom}(M,p)$ (Cantelli with margin $\epsilon = p - \tfrac12$) delivers the full $\sqrt M$ variance reduction, so $M$ workers behave like one worker with $\sqrt M$ times less noise — the same speedup full-precision distributed SGD enjoys, now at one bit each way. Symmetry is essential here: a skewed noise like $P[X=50]=0.1, P[X=-1]=0.9$ has positive mean yet $P[\text{correct sign}]=0.1$, so more workers would make things *worse*.

Folding momentum *inside* the sign gives Signum: accumulate $m_{k+1} = \beta m_k + (1-\beta)\tilde g_k$ and step $x_{k+1} = x_k - \delta\,\mathrm{sign}(m_{k+1})$. Momentum is an average of recent gradients, so it has lower variance — but it averages in *stale* gradients from earlier, curved points, so it trades variance reduction against curvature-induced bias, with $\beta$ the knob. To prove this I abstract a master lemma: for any $x_{k+1} = x_k - \delta_k\mathrm{sign}(v_k)$, if $\mathbb{E}[\sum_i |g_{k,i}| P[\mathrm{sign}(v_{k,i}) \ne \mathrm{sign}(g_{k,i}) \mid x_k]] \le \xi(k) \to 0$, the same single-step-plus-telescope argument bounds the average $\ell_1$ gradient — and a sufficient condition is that $v_k$ tracks $g_k$ in expected absolute value, $\sum_i \mathbb{E}|v_k[i]-g_k[i]| \le \xi(k)$, because the magnitude cancels exactly as before. Splitting the normalized stochastic momentum against its deterministic-gradient counterpart gives a variance piece bounded by a coordinate-wise *martingale* second-moment bound $\mathbb{E}[(\sum_l \alpha_l Z_l)^2] \le \sum_l \alpha_l^2\sigma_l^2$ (the unbiased oracle makes the cross terms vanish, so I treat the dependent noises *as if* independent for the upper bound), and a bias piece bounded by $\|g(x+\epsilon s) - g(x)\|_1 \le 2\epsilon\|L\|_1$ (Taylor with an averaged Hessian, split into psd and nsd parts, each $\preceq \mathrm{diag}(L)$) chained over steps. Together $$\sum_i \mathbb{E}|\tilde m_k[i] - g_k[i]| \le \frac{2}{\sqrt{k+1}}\Big(8\|L\|_1\delta\frac{\beta}{1-\beta} + \sqrt3\,\|\sigma\|_1\sqrt{1-\beta}\Big),$$ which is $\xi(k) = O(1/\sqrt k)$; feeding it through the master lemma yields the $O(1/\sqrt N)$ rate with $\beta$ as an explicit bias-variance knob — the $\|\sigma\|_1\sqrt{1-\beta}$ variance term shrinks as $\beta\to1$ while the $\delta\|L\|_1/(1-\beta)$ bias term blows up. A short warmup $C$ ($=54$ for $\beta=0.9$) lets the stale-momentum tail decay; during it I accumulate momentum but step with plain $\mathrm{sign}(\tilde g)$, so no iterations are wasted.

One discretization detail: $\mathrm{sign}(0)$. The wire/majority path uses $\,\ge 0\,$ so every coordinate stays binary and an exact vote tie broadcasts as $+1$, while the local-optimizer path mirrors the original code's $\mathrm{sign}(0) \to 0$ (a zero step). This lands cleanly in the parameter-server codec — encode is the sign bit, aggregate is the majority vote with the $\ge 0$ tie convention, decode is $\pm1$ — with $\delta$ the entire step size, the trust-region radius of the $\ell_\infty$ box.

```python
import torch


class SignSGDCodec:
    """signSGD with majority-vote aggregation: 1 bit per coordinate each direction."""

    def __init__(self):
        self.state = {}

    def encode(self, grad, name):
        shape = grad.shape
        bits = (grad.flatten() >= 0).to(torch.uint8)         # sign(g~), stored as uint8
        return [bits], shape

    def aggregate(self, messages):
        votes = sum(b.to(torch.float32) * 2 - 1 for [b] in messages)  # sum_m sign(g~_m)
        return [(votes >= 0).to(torch.uint8)]                # majority vote -> 1 bit back

    def decode(self, received, ctx):
        [bits] = received
        return (bits.to(torch.float32) * 2 - 1).view(ctx)    # bits -> +-1, step = -delta*this


class SignumCodec(SignSGDCodec):
    """Signum: sign of the per-parameter momentum m = beta*m + (1-beta)*g~."""

    def __init__(self, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.buf = {}

    def encode(self, grad, name):
        shape = grad.shape
        g, beta = grad.flatten(), self.momentum
        m = self.buf.get(name)
        if m is None:
            m = torch.zeros_like(g)
        m = beta * m + (1 - beta) * g                        # m_{k+1}=beta*m_k+(1-beta)*g~_k
        self.buf[name] = m
        return [(m >= 0).to(torch.uint8)], shape             # sign(m_{k+1})
```

```python
import torch


class Signum:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0,
                 decoupled_weight_decay=0.0):
        self.params = list(params)
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.wd_lh = decoupled_weight_decay
        self.state = {id(p): None for p in self.params}

    @torch.no_grad()
    def step(self):
        beta = self.momentum
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.wd != 0:
                g = g.add(p, alpha=self.wd)                   # rescaled_grad includes wd*weight
            if beta != 0:
                m = self.state[id(p)]
                if m is None:
                    m = torch.zeros_like(g)
                m.mul_(beta).add_(g, alpha=1 - beta)          # state=beta*state+(1-beta)*rescaled_grad
                self.state[id(p)] = m
                direction = torch.sign(m)
            else:
                direction = torch.sign(g)
            if self.wd_lh != 0:
                p.mul_(1 - self.lr * self.wd_lh)              # MXNet wd_lh-style decoupled decay
            p.add_(direction, alpha=-self.lr)                 # signSGD/Signum signed step
```
