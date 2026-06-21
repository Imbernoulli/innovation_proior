Error-feedback top-$k$ did exactly what the QSGD collapse said it should, and the residual numbers tell me both that the memory was right and where it still leaves accuracy on the table. The decisive VGG-11-BN/CIFAR-100 result is fixed: $\{70.78, 69.75, 70.20\}$, mean $70.24$ — the seed-dependent collapse is gone, all three seeds within a point. ResNet-20 tightened too, $\{92.05, 92.40, 92.29\}$, mean $92.25$, the weak seed pulled up. But read ResNet-56 against QSGD: EF-TopK is $\{93.78, 93.79, 93.98\}$, mean $93.85$ — *below* QSGD's $94.08$, and below uncompressed SGD. That is the tell. On the setting where QSGD did not collapse, the aggressive 99%-drop top-$k$ is slightly *worse* than the gentler quantizer: keeping only 1% of coordinates, even with the residual paying the rest back later, means the model is always working from a stale, delayed version of the gradient, and at 100× the residual carries a lot of lag. EF-TopK's weakness is *delay* — it transmits a tiny dense-in-energy slice now and owes the rest. The fix keeps the error-feedback memory but stops betting on sparsity.

I propose **SignSGD with error feedback and mean-magnitude scaling**. Top-$k$'s lag comes from sending *few coordinates*; instead send *every* coordinate but spend almost nothing on each. The crudest per-coordinate message is one bit — the sign. $\text{sign}(g)$ keeps the direction of every coordinate and throws away only magnitude, the opposite trade from top-$k$, which keeps full magnitude on a tiny support and drops the rest. A sign message touches all $d$ coordinates each step, so there is no stale owed-for-later mass; every direction gets a vote now. The cost is 1 bit per coordinate against float32's 32, a 32× compression — less than top-$k$'s nominal 100× on the wire, but with no sparse-support lag.

Sign's danger is the mirror image of top-$k$'s. Where top-$k$ is biased by starving directions, sign is biased by forgetting magnitude: a coordinate that gets $+0.1$ on nine minibatches and $-0.9$ on the tenth has true mean zero — it should sit still — but a pure sign rule sees nine $+$ votes and one $-$ and marches the weight a long way the wrong direction. So $\text{sign}(g)$ raw is biased and can point against the true gradient, exactly the uncorrected bias that collapsed QSGD's VGG seeds. The lesson of the last two rungs is not optional: a biased compressor needs a memory. So I use sign *with the same error feedback*, applied to the magnitude information sign discards. But error feedback needs a reconstruction to take the difference against, and sign alone reconstructs to $\pm 1$, which is the wrong scale — gradient coordinates are tiny, not order one. So I attach a magnitude on decompress, the cheapest natural choice being the *mean absolute value* of the tensor: reconstruct each coordinate as $\text{sign}(g_i)\cdot\text{mean}(|g|)$. One shared scalar carries the typical magnitude, the per-coordinate signs carry direction. This is the right reconstruction twice over. It gives error feedback something correctly-scaled to subtract — the residual $e \leftarrow (g + e_{\text{prev}}) - \text{sign}\cdot\text{mean}(|\cdot|)$ carries forward every coordinate's deviation from the shared magnitude and every mis-signed coordinate's full value, paid back next step. And the mean-magnitude scale keeps the reconstructed gradient's norm in the right ballpark, so the optimizer's step size, tuned for full gradients, still makes sense.

Why should this *beat* EF-TopK rather than merely differ? A per-coordinate signal-to-noise argument: the sign of a coordinate is only wrong when the noise overpowers the signal, the damage of a flipped sign is proportional to that coordinate's magnitude, and the probability of flipping is inversely proportional to it — so the error a flip costs is capped by the noise scale, *independent* of the gradient size. When a coordinate's true gradient is large relative to its noise the sign is almost always right and I make good progress; when it is small relative to noise, going slightly wrong is cheap because I am near stationary there anyway. For real networks the gradient and its noise turn out to be of the *same density* — both dense, neither concentrated on a sparse loud set — which is exactly the regime where the magnitude-blind sign converges about as fast as full SGD and pockets the compression. EF-TopK by contrast *bets on sparsity*: it wins big only if the energy really is in 1% of coordinates, and when it is not — a dense gradient, which is what these CNNs have — it is forced to delay 99% of a genuinely dense signal, the lag that left ResNet-56 short. Sign spends one bit on every coordinate, so a dense gradient is its home turf.

The fill departs from textbook distributed signSGD in two ways I match. The classic version compresses the *return* trip via majority vote $\text{sign}[\sum_w \text{sign}(g_w)]$, one bit each way; this single-node benchmark simulates one worker, so there is no majority vote — `aggregate` is not in the contract, the loop just compresses and decompresses per parameter. And there is no momentum inside the sign (no Signum); momentum lives in the fixed SGD optimizer outside the compressor. So the fill is plain sign-with-error-feedback-and-mean-magnitude. `self.residuals` is a per-name dict; `ef_beta = 1.0` scales how much residual is added back (here, full). On `compress`, if a residual exists, $\text{tensor} = g + \text{ef\_beta}\cdot\text{residual}[name]$; flatten; `signs = (tensor_flat >= 0)` as `uint8` so $\text{sign}(0)\to +1$ and every coordinate is a single bit; `mean_magnitude = |tensor_flat|.mean()`; reconstruct with `sign_float = signs*2 - 1` mapping $\{0,1\}\to\{-1,+1\}$; set `residual[name] = (tensor_flat - reconstructed).view(shape)`; send `[signs, mean_magnitude]`, context `shape`. `decompress` maps the bits back to $\pm 1$, multiplies by the mean magnitude, views to shape. `compress_ratio` is accepted but unused — sign is always 1 bit — exactly as QSGD's `quantum_num` overrode it.

Reading EF-TopK's shape, the falsifiable claims are specific. On ResNet-56/CIFAR-10 — where top-$k$'s delay cost it, landing at $93.85$ below QSGD's $94.08$ — sign-with-EF on a dense gradient should *recover* that, back toward $\sim\!94.1$, at or above both prior rungs; that single result decides whether the dense-vs-sparse story is right. On VGG-11-BN/CIFAR-100 it should *hold* the hard-won $\sim\!70$ and edge slightly above, since a dense one-bit signal does not strand 99% of the gradient; I expect $\sim\!70.7$. On ResNet-20 I expect a tight $\sim\!92.5$ with the lowest seed spread of the three rungs. If sign comes back below EF-TopK on ResNet-56, the dense-gradient bet was wrong — but the densities of these CNN gradients say otherwise, so I expect sign-with-EF to be the strongest of the three baselines. What it still cannot do — and what the next rung must address — is the systems problem under all three: top-$k$'s supports do not all-reduce across workers and sign's majority vote is a non-linear collective, so neither aggregates cheaply at scale.

```python
class Compressor:
    """SignSGD with error feedback.

    Compresses each gradient element to its sign (+1 or -1), achieving
    32x compression. Error feedback accumulates the magnitude information
    lost during sign extraction, improving convergence.

    The compress_ratio parameter is not used for sign compression (always
    1-bit), but the error feedback momentum can be tuned.
    """

    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio
        self.residuals = {}
        # Error feedback momentum
        self.ef_beta = 1.0

    def compress(self, tensor, name):
        # Error feedback: add accumulated residual
        if name in self.residuals:
            tensor = tensor + self.ef_beta * self.residuals[name]

        shape = tensor.shape
        tensor_flat = tensor.flatten()

        # Sign compression: 1 bit per element
        signs = (tensor_flat >= 0).to(torch.uint8)

        # Scale by mean magnitude for better reconstruction
        mean_magnitude = tensor_flat.abs().mean()

        # Update residual: original - reconstructed
        sign_float = signs.float() * 2 - 1  # map {0,1} -> {-1,+1}
        reconstructed = sign_float * mean_magnitude
        self.residuals[name] = (tensor_flat - reconstructed).view(shape)

        return [signs, mean_magnitude], shape

    def decompress(self, compressed_tensors, ctx):
        shape = ctx
        signs, mean_magnitude = compressed_tensors

        # Reconstruct: sign * mean_magnitude
        sign_float = signs.float() * 2 - 1
        tensor_decompressed = sign_float * mean_magnitude
        return tensor_decompressed.view(shape)
```
