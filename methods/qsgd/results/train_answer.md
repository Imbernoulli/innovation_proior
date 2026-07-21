When I scale stochastic gradient descent across $K$ workers, each holding a full copy of the parameter vector $x \in \mathbb{R}^n$ and computing a stochastic gradient on its own data shard, every iteration forces the workers to agree on an aggregate gradient before they can step, and that agreement is paid for in communication. With dense gradients each worker puts $n$ floating-point numbers — $32n$ bits — on the wire and pulls in everyone else's, every single round. For the models I care about, with tens of millions of parameters, profiling a real run reveals the ugly truth: as the model grows and as I add GPUs, the wall-clock time of an iteration is increasingly not the gradient computation but the gradient exchange. The compute bar shrinks with more parallelism while the communication bar swells, until most of an iteration is just shoving floats across the network. The bottleneck is bytes, not flops, so the whole game is to put fewer bits per gradient on the wire without wrecking the optimization — and to do it with a provable guarantee and a smooth knob, not just a heuristic that happened to work on one model.

The obvious lever is to compress before sending and decompress on the other side, and the most aggressive scheme in production does exactly this: squash each gradient coordinate down to a single bit — its sign relative to a threshold of zero — and reconstruct it from a couple of recomputed scaling values. Done raw, this diverges, because the reconstruction is a poor stand-in for the true gradient and the errors accumulate rather than wash out. The rescue is error feedback: keep a residual buffer $\Delta$ and quantize $G(t) + \Delta(t-N)$ instead of the bare gradient, with $\Delta(t) = G(t) - Q^{-1}(G_{\text{quant}}(t))$, so the discarded part is smeared into later rounds and eventually folded into the model. It scaled speech networks beautifully, but three things make me uneasy. There is no convergence guarantee, not even under strong assumptions — it is a heuristic that works under conditions nobody can fully characterize. It is stuck at roughly one bit per coordinate, with no dial to spend a few more bits to buy back stability or to push compression further. And its residual buffer is a whole extra copy of the model in memory. The first complaint is the one that matters: why is there no guarantee, and what about this scheme puts it outside the convergence theory I already trust for plain SGD?

Writing down that theory and staring at it is what cracks the problem. For a convex, $L$-smooth $f$, SGD with access to stochastic gradients $g$ satisfying $\mathbb{E}[g] = \nabla f$ and a second-moment bound $\mathbb{E}[\|g\|^2] \le B$ converges: with a good constant step, after $T$ steps the averaged iterate obeys
$$\mathbb{E}\!\left[ f\!\left(\tfrac{1}{T}\sum_t x_t\right)\right] - \min f \;\le\; R\sqrt{\tfrac{2\sigma^2}{T}} \;+\; \tfrac{L R^2}{T},\qquad R^2 = \sup_x \|x - x_0\|^2,$$
so reaching error $\epsilon$ in the variance-dominated regime needs $T = O\!\big(R^2 \sigma^2/(K\epsilon^2)\big)$ iterations, the $1/K$ coming because averaging $K$ independent worker gradients is a minibatch of size $K$. The whole content of this bound, from my point of view, is that the iteration count is linear in the variance of the stochastic gradient. Nothing else about the gradient enters the theorem — not its sparsity, not its representation, not how many bits it took to send — only its variance, and only because it is unbiased enough to count as a stochastic gradient at all. That is the answer: the 1-bit quantizer is biased, so its reconstruction has the wrong expectation, so it is not a stochastic gradient for $f$ and the theorem simply does not apply; error feedback is a patch that tries to recover over many steps the unbiasedness broken in one. The right move is therefore to never break unbiasedness in the first place. Compress with a randomized map $Q$ that is exactly unbiased, $\mathbb{E}[Q(g)] = g$, and $Q(g)$ is itself a legitimate stochastic gradient whose only effect on the bound is to inflate the variance; if it multiplies the second moment by a factor $c$, it multiplies the iteration count by about $c$ and changes nothing else. No error feedback, no new proof machinery, no residual buffer.

The method I propose is QSGD — Quantized SGD — a family of lossy gradient compressors built around an exactly unbiased randomized quantizer with a single tunable knob $s$, the number of quantization levels. The design problem is to make $Q$ unbiased, cheap to encode in few bits, and as low-variance as possible for a given bit budget. The simplest unbiased quantizer reveals where the pain is. Keep the norm $\|v\|_2$ — one shared float — and normalize each coordinate so $a_i = |v_i|/\|v\|_2 \in [0,1]$, then round each normalized magnitude stochastically to $0$ or $1$: $Q(v_i) = \|v\|_2\,\mathrm{sgn}(v_i)\,\xi_i$ with $\xi_i = 1$ at probability $a_i$ and $0$ otherwise. It is unbiased because $\mathbb{E}[\xi_i] = a_i$, and it is almost free to send: a sign bit per surviving coordinate plus a single float. But its variance is ruinous. Using $\xi_i^2 = \xi_i$ for a $0/1$ variable, $\mathbb{E}[\|Q(v)\|^2] = \|v\|_2^2 \sum_i a_i = \|v\|_2\,\|v\|_1 \le \sqrt{n}\,\|v\|_2^2$ by Cauchy–Schwarz, so the second-moment blowup is up to $\sqrt{n}$ — about three thousand for $n = 10^7$, three thousand times as many iterations. The bandwidth per step is wonderful and the steps-to-converge is a catastrophe. The source of the blowup is that each coordinate is slammed all the way to $0$ or to the full norm, an enormous swing around its true value, summed over $n$ coordinates.

The fix is to give each coordinate intermediate landing spots so random rounding only has to bridge a small gap. Lay down $s$ evenly spaced levels in $[0,1]$ and, for a coordinate with $a = a_i$, find the integer $\ell$ with $a \in [\ell/s, (\ell+1)/s]$; round to $(\ell+1)/s$ with probability $p(a,s)$ and to $\ell/s$ otherwise. Demanding exact unbiasedness forces the round-up probability: $\mathbb{E}[\xi_i] = \ell/s + p/s = a$ gives
$$p(a,s) = a\,s - \ell,$$
the fractional part of $a s$, which is automatically a valid probability in $[0,1]$ and naturally larger the closer $a$ sits to the upper level. So
$$Q_s(v_i) = \|v\|_2\,\mathrm{sgn}(v_i)\,\xi_i(v,s),\qquad \xi_i = \tfrac{\ell+1}{s}\ \text{w.p.}\ p(a,s),\ \ \tfrac{\ell}{s}\ \text{otherwise},$$
unbiased by construction and a strict generalization of the crude scheme, which is just $s = 1$. The payoff appears in the variance. Each $\xi_i$ randomizes only between two adjacent levels a distance $1/s$ apart, so $\mathbb{E}[\xi_i^2] = a^2 + \tfrac{1}{s^2}p(1-p) \le a^2 + p/s^2$ — the $1/s^2$ is the whole point, since more levels make the gap, and hence the per-coordinate variance, quadratically smaller. Summing and using $\sum_i a_i^2 = 1$,
$$\mathbb{E}\big[\|Q_s(v)\|^2\big] \le \|v\|_2^2\Big(1 + \tfrac{1}{s^2}\textstyle\sum_i p(a_i,s)\Big).$$
The two ways to bound $\sum_i p$ produce the characteristic $\min$. From $p \le 1$ comes $\sum_i p \le n$, hence $n/s^2$; from $p \le a s$ comes $\sum_i p \le s\,\|v\|_1/\|v\|_2 \le s\sqrt{n}$, hence $\sqrt{n}/s$. Taking the smaller,
$$\mathbb{E}\big[\|Q_s(v) - v\|^2\big] \le \min\!\Big(\tfrac{n}{s^2}, \tfrac{\sqrt{n}}{s}\Big)\|v\|_2^2.$$
At $s = 1$ this is $\sqrt{n}$, consistent with the crude scheme; for $1 \le s \le \sqrt{n}$ the active branch is $\sqrt{n}/s$, dropping like $1/s$; and at $s = \sqrt{n}$ both branches equal $1$, so the added variance is at most $\|v\|_2^2$ and the second moment at most $2\|v\|_2^2$. That is the sweet spot: spending $s = \sqrt{n}$ levels takes the dimension-dependent $\sqrt{n}$ penalty all the way down to a constant factor of $2$ on iteration count. The knob I went looking for is now quantitative.

The bits, however, threaten to undo this. If each of $n$ level indices naively took $\log s = \tfrac12\log n$ bits, that is about $\tfrac{n}{2}\log n$ bits, worse than $32n$. The structural fact that saves it is that the integer levels $s\,\xi_i$ are not uniform: a large level means a coordinate carries a large fraction of the norm, and a unit-norm vector supports only a few large coordinates, so large integers are rare and small ones common — exactly what a universal integer code is for. Encode each level with Elias recursive (omega) coding, $|\mathrm{Elias}(k)| \le (1+o(1))\log k + 1$, and send the tuple $(\|v\|_2,\ \text{signs},\ \text{levels})$, coding for each nonzero the gap to the next nonzero (to locate it without a full bitmap), then its sign, then $\mathrm{Elias}(s\,\xi_i)$. The load-bearing lemma is that for a length-$m$ positive-integer vector with $\|q\|_p^p \le \rho$,
$$\sum_i |\mathrm{Elias}(q_i)| \le \Big(\tfrac{1+o(1)}{p}\log\tfrac{\rho}{m} + 1\Big) m,$$
which follows by writing $\log q_i = \tfrac1p\log q_i^p$ and applying Jensen to the concave $\log$. Feeding in the density — at most $s^2$ large coordinates (since $\sum_i u_i^2 = 1 \ge (\text{count})/s^2$) plus at most $s\sqrt{n}$ small ones that round up, giving $O(s(s+\sqrt{n}))$ nonzeros — and the level-norm $\|\zeta\|_2^2 \le 2(s^2 + n)$, the per-iteration cost becomes $\big(3 + (\tfrac32 + o(1))\log\tfrac{2(s^2+n)}{s(s+\sqrt{n})}\big)\,s(s+\sqrt{n}) + 32$. In the sparse regime $s=1$ this is $O(\sqrt{n}\log n)$ bits with the ruinous $\sqrt{n}$ variance. In the dense regime $s=\sqrt{n}$ every coordinate is nonzero, so positions buy nothing — drop them and code every coordinate, zeros included, with a shifted code $\mathrm{Elias}'(k) = \mathrm{Elias}(k+1)$; with $\mathbb{E}[\|\zeta\|_2^2] \le s^2 + \min(n, s\sqrt{n}) = 2n$, the logarithm sees $1 + 2n/n = 3$ and, since $2 + \tfrac12\log_2 3 = 2.79\ldots$, the cost is at most $2.8\,n + 32$ bits against $32n$ — roughly an $11\times$ reduction while the second-moment blowup stays at most $2$. This is essentially optimal: any scheme with constant variance blowup must send $\Omega(n)$ bits per round, the distributed-mean-estimation lower bound.

The convergence guarantee now costs nothing extra, which is the entire reward for insisting on unbiasedness. With $\alpha = \min(n/s^2, \sqrt{n}/s)$, original second moment $B$ becomes $B_q = (1+\alpha)B$ for the quantized gradients, and substituting into the standard theorem, parallel $Q_s$-SGD on $K$ workers reaches error $\epsilon$ in
$$T = O\!\Big(R^2 \max\!\big(\tfrac{2 B_q}{K\epsilon^2}, \tfrac{L}{\epsilon}\big)\Big)$$
iterations — identical in form to full-precision SGD, with only the noise scale changed. Because the argument rides on top of any SGD analysis, the non-convex version that actually matters for neural nets follows too: by Ghadimi–Lan, with constant step $\eta = O(1/L)$ and a random stopping iterate $R$, $\tfrac1L\mathbb{E}[\|\nabla f(x_R)\|^2] \le O\!\big(\sqrt{L(f(x_1)-f^*)}/N + (1+\alpha)B/L\big)$, the compression entering only through the same noise floor, which I shrink by spending more levels.

Two practical choices bridge the clean theory to a GPU. The variance bound is tied to the full dimension $n$, forcing $s = \sqrt{n}$ for a constant blowup, which is more bits than a 2- or 4-bit budget allows; so I decouple variance from dimension by flattening the gradient and quantizing it in buckets of $d$ consecutive entries, each with its own norm and levels. Every bucket obeys the lemmas with $d$ in place of $n$, so the blowup is $\min(d/s^2, \sqrt{d}/s)$ with $d$ a quantity I choose — for instance $d = 512$ with $4$ bits ($s = 16$) gives blowup at most $\sqrt{512}/16 \approx 1.41$ at only 4 bits per coordinate, costing one extra scaling float per bucket; $d = 1$ is no quantization and $d = n$ is the full-tensor scheme. Separately, one may normalize by $\max_i |v_i|$ rather than $\|v\|_2$, mapping the largest coordinate to exactly $1$ and spending levels on the occupied range — a dense-code variant that forfeits the sparsity proof (irrelevant when every coordinate is already nonzero) in exchange for stronger fixed-width behavior. The operator I would ship is the per-tensor $L_2$-normalized form: flatten, take the norm, scale each absolute coordinate into level space $s|v_i|/\|v\|$, floor to the lower level, round up with probability equal to the fractional part, carry the sign on the integer level, and send the signed integer levels plus the single norm float; decompression is just $(\|v\|/s)\times$ level. Memory is exactly one tensor of levels and a scalar, with no residual buffer because unbiasedness made error feedback unnecessary.

```python
import torch

from grace_dl.torch import Compressor


class QSGDCompressor(Compressor):

    def __init__(self, quantum_num):
        super().__init__()
        self.quantum_num = quantum_num

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        norm = tensor.norm()
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = previous_level + is_next_level

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(
            torch.int8 if self.quantum_num < 128 else torch.half
        )
        tensor_compressed = tensor_compressed, norm

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, norm = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed
```
