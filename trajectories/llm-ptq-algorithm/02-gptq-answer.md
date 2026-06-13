**Problem (from step 1).** RTN held at INT4 (5.1343 g128, 5.0890 g64) but blew up at INT3-g128 to
6.7341 — degradation 1.8270, eight times the INT4 hit — because it minimizes per-*weight* error
$\lVert\mathbf W-\widehat{\mathbf W}\rVert^2$ in isolation and discards the calibration stream, so at
8 levels the rounding residual compounds across 32 blocks with nothing to absorb it.

**Key idea.** Minimize the layer's *output* error
$\lVert\mathbf{WX}-\widehat{\mathbf W}\mathbf X\rVert^2$ instead. It decomposes by output row into a
quadratic whose Hessian is $\mathbf H=\mathbf X^\top\mathbf X$ (the input second moment, accumulated in
`add_batch`). Quantize all rows in a single fixed left-to-right column order so one shared $\mathbf H$
serves every row; round each column, then push its per-row residual onto the still-free columns through
$\mathbf H^{-1}$ (the brain-surgeon/OBS update), so the output is preserved even as individual weights
drift further from their originals.

**Why it beats RTN.** The deferred error has somewhere to go: a quantized column drifts the remaining
columns to keep $\mathbf{WX}$ right, weighted by how hard real text excites each input direction —
exactly the structure RTN ignored. Most valuable where the grid is thinnest (INT3).

**This task's fill (vs the most-refined form).** It computes the dense inverse once —
`L = cholesky(H); Hinv = cholesky_inverse(L)` (`pinv` fallback) — and reads $[\mathbf H^{-1}]_{ii}$ and
the rows $[\mathbf H^{-1}]_{i,\cdot}$ directly, rather than the "Cholesky-of-the-inverse stores the scaled
row-tails" trick. Numerically stabilized by `PERCDAMP=0.01` of the mean diagonal added before inverting.
Blocked in `BLOCK_SIZE=128` columns (within-block compensation, then one GEMM to the rest). Per-group
scales are recomputed at each `col % group_size == 0` boundary from the *current* (already-compensated)
weights, so grouping and compensation reinforce; the same class runs INT4-g128 / INT3-g128 / INT4-g64.

**What to watch.** INT3-g128 is the test: RTN's 1.8270 degradation should drop substantially (perplexity
into the low-6 range). INT4-g128 a real but smaller gain under RTN's 5.1343; INT4-g64 the smallest gap
(fine grouping already overlaps compensation). Quant-time rises ~10× (an inverse + column sweep per layer).

```python
# EDITABLE region of gptq/custom_ptq.py (lines 26-157) — step 2: GPTQ (Hessian error compensation)

def quantize_tensor(x, scale, zero_point, qmin, qmax):
    """Quantize a float tensor to integers given scale and zero point."""
    x_int = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return x_int


def dequantize_tensor(x_int, scale, zero_point):
    """Dequantize integer tensor back to float."""
    return (x_int - zero_point) * scale


def find_scale_zero(weight, num_bits=4, group_size=-1, symmetric=True):
    """Compute per-channel (or per-group) quantization parameters."""
    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1

    if group_size > 0:
        out_features, in_features = weight.shape
        assert in_features % group_size == 0
        w_groups = weight.reshape(out_features, -1, group_size)
        if symmetric:
            w_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = w_groups.amin(dim=-1, keepdim=True)
            w_max = w_groups.amax(dim=-1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)
        scale = scale.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
        zero_point = zero_point.reshape(out_features, -1).repeat_interleave(group_size, dim=1)
    else:
        if symmetric:
            w_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
            scale = w_max / qmax
            zero_point = torch.zeros_like(scale)
        else:
            w_min = weight.amin(dim=1, keepdim=True)
            w_max = weight.amax(dim=1, keepdim=True)
            w_range = (w_max - w_min).clamp(min=1e-12)
            scale = w_range / (qmax - qmin)
            zero_point = torch.round(qmin - w_min / scale)

    return scale, zero_point, qmin, qmax


class LayerQuantizer:
    """GPTQ quantizer -- Hessian-based error compensation.

    Collects input activation statistics (H = X^T X), then quantizes
    weights column-by-column, compensating for quantization error using
    the Hessian inverse so that layer output error is minimized.
    """

    BLOCK_SIZE = 128
    PERCDAMP = 0.01

    def __init__(self, layer, num_bits=4, group_size=-1):
        self.layer = layer
        self.num_bits = num_bits
        self.group_size = group_size
        self.out_features, self.in_features = layer.weight.shape
        self.dev = layer.weight.device
        self.nsamples = 0
        self.H = torch.zeros(
            (self.in_features, self.in_features),
            device=self.dev, dtype=torch.float32
        )

    def add_batch(self, inp):
        """Accumulate Hessian approximation from calibration inputs."""
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        n = inp.shape[0]
        inp = inp.float()
        self.H += inp.T @ inp
        self.nsamples += n

    def quantize(self):
        """GPTQ: column-by-column quantization with Hessian error compensation."""
        W = self.layer.weight.data.clone().float()
        H = self.H.clone()

        if self.nsamples > 0:
            H /= self.nsamples

        num_bits = self.num_bits
        group_size = self.group_size
        qmin = -(1 << (num_bits - 1))
        qmax = (1 << (num_bits - 1)) - 1

        # Add dampening to diagonal for numerical stability
        damp = self.PERCDAMP * torch.mean(torch.diag(H))
        H += damp * torch.eye(self.in_features, device=self.dev)

        # Compute Hessian inverse via Cholesky decomposition
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
        except Exception:
            # Fallback to pseudo-inverse if Cholesky fails
            Hinv = torch.linalg.pinv(H)

        Q = torch.zeros_like(W)
        Err = torch.zeros_like(W)

        # Process columns in blocks
        for col_start in range(0, self.in_features, self.BLOCK_SIZE):
            col_end = min(col_start + self.BLOCK_SIZE, self.in_features)

            W_block = W[:, col_start:col_end].clone()
            Hinv_block_diag = torch.diag(
                Hinv[col_start:col_end, col_start:col_end]
            )

            for j in range(col_end - col_start):
                col = col_start + j
                w_col = W_block[:, j]

                # Compute scale: per-group if group_size > 0, else per-column
                if group_size > 0 and col % group_size == 0:
                    g_end = min(col + group_size, self.in_features)
                    W_group = W[:, col:g_end]
                    g_max = W_group.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                    group_scale = (g_max / qmax).squeeze(1)

                if group_size > 0:
                    scale = group_scale
                else:
                    w_abs_max = w_col.abs().max().clamp(min=1e-12)
                    scale = w_abs_max / qmax

                # Quantize and dequantize
                q_col = torch.clamp(
                    torch.round(w_col / scale), qmin, qmax
                ) * scale
                Q[:, col] = q_col

                # Error compensation: distribute error weighted by Hessian
                err = (w_col - q_col) / Hinv_block_diag[j].clamp(min=1e-12)
                Err[:, col] = err

                # Update remaining columns in block
                if j + 1 < col_end - col_start:
                    W_block[:, j+1:] -= (
                        err.unsqueeze(1)
                        * Hinv[col, col_start+j+1:col_end].unsqueeze(0)
                    )

            # Propagate error to remaining columns outside block
            if col_end < self.in_features:
                W[:, col_end:] -= (
                    Err[:, col_start:col_end]
                    @ Hinv[col_start:col_end, col_end:]
                )

        return Q.to(self.layer.weight.dtype)

    def free(self):
        """Release calibration buffers."""
        del self.H
        self.H = None
```
