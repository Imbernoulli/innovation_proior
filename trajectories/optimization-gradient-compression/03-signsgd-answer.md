**Problem.** EF-TopK fixed QSGD's collapse but its 99%-sparse support leaves a *stale, owed*
gradient: on a dense-gradient setting (ResNet-56) it landed at 93.85, below QSGD's 94.08, because
the residual carries a lot of lag at 100×. The fix: keep the error-feedback memory but stop
betting on sparsity.

**Key idea (sign compression + error feedback + mean-magnitude scaling).** The crudest
per-coordinate message is one bit — the sign. Send `sign(g)` for *every* coordinate (1 bit each,
32×), so no direction is delayed the way top-k delays 99% of them. Reconstruct each coordinate as
`sign(g_i)·mean(|g|)` — one shared scalar for the typical magnitude. Sign is biased
(magnitude-blind: a positive-mean bimodal gradient can be mis-signed), so carry the discarded
magnitude in a per-tensor **error-feedback** residual against the `sign·mean` reconstruction and
add it back next step.

**Why it beats top-k here.** A flipped sign costs at most the noise scale (the magnitude cancels:
damage ∝ magnitude, flip probability ∝ 1/magnitude), so on networks whose gradient and noise are
of the *same density* — dense, not sparse — sign converges about as fast as full SGD. EF-TopK
*bets on sparsity* and must delay 99% of a dense gradient; sign spends one bit on every
coordinate, so a dense gradient is its home turf, recovering top-k's lost fraction while keeping
the residual-bought robustness.

**This task's fill (departures from textbook signSGD).** Single-node simulation, so **no
majority-vote** return-trip compression (no `aggregate`), and **no momentum-inside-the-sign**
(no Signum — momentum lives in the fixed SGD optimizer). Plain sign + EF + mean-magnitude.
`signs = (tensor >= 0)` so `sign(0) → +1`; `ef_beta = 1.0` adds the full residual back.
`compress_ratio` is accepted but unused (sign is always 1 bit).

**Hyperparameters.** `ef_beta = 1.0` (full error feedback); per-name residual state;
`compress_ratio = 0.01` accepted but unused.

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
