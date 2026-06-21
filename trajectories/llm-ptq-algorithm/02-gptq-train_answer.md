RTN's floor came in as predicted: at INT4-g128 it lands $5.1343$ (degradation $0.2271$), at the finer INT4-g64 it improves slightly to $5.0890$, but at INT3-g128 it blows up to $6.7341$ — degradation $1.8270$, eight times the INT4 hit. The thin-grid case is exactly where rounding fails, and the reason is structural, not an implementation slip: RTN minimizes per-weight error $\lVert\mathbf W-\widehat{\mathbf W}\rVert^2$ element by element, when the quantity that actually propagates to the next block is the layer output $\mathbf W\mathbf X$, and it discards the calibration stream the scaffold feeds through `add_batch` for free. Both of those are what I now spend.

The method is GPTQ, second-order error compensation (in the brain-surgeon lineage, Frantar et al. 2023). I state the honest objective RTN was approximating badly: for one linear layer with weights $\mathbf W$ and calibration inputs stacked as columns of $\mathbf X$, I want quantized $\widehat{\mathbf W}$ that keeps the output close,

$$\arg\min_{\widehat{\mathbf W}}\;\lVert\mathbf W\mathbf X-\widehat{\mathbf W}\mathbf X\rVert_2^2 .$$

The grid is fixed up front; each weight is free to land on any grid value. The key structural fact is that this decomposes by *row* of $\mathbf W$ — output channels are independent linear functionals of the same input — so per row it is a quadratic, and differentiating $\lVert\mathbf w^\top\mathbf X-\widehat{\mathbf w}^\top\mathbf X\rVert^2$ twice gives a Hessian $\mathbf H=2\mathbf X\mathbf X^\top$ that depends only on the inputs, identical for every row. That is why the calibration stream matters: $\mathbf X\mathbf X^\top$ is the input second moment, telling me which input directions real text excites hard — where a weight error costs the output a lot — and which are nearly dead, where RTN was wasting its worry. So `add_batch` reshapes the input to $(\text{tokens},\text{in\_features})$ and accumulates $\mathbf X^\top\mathbf X$ into `H`; the factor of $2$ is irrelevant, and I normalize by the sample count at quantize-time (`H /= nsamples`) to keep the dampening scale sane across layers.

The compensation step is the brain-surgeon logic specialized to quantization. Fix one coordinate $q$ to its chosen grid value and move all the still-free coordinates to absorb the damage. With $\boldsymbol\delta=\widehat{\mathbf w}-\mathbf w$ and the constraint $\mathbf e_q^\top\boldsymbol\delta=\mathrm{quant}(w_q)-w_q$, minimizing $\tfrac12\boldsymbol\delta^\top\mathbf H\boldsymbol\delta$ by Lagrange multipliers gives the optimal compensating update

$$\boldsymbol\delta_F=-\,\frac{w_q-\mathrm{quant}(w_q)}{[\mathbf H^{-1}]_{qq}}\,(\mathbf H^{-1})_{:,q},\qquad \Delta\mathcal L=\tfrac12\,\frac{(\mathrm{quant}(w_q)-w_q)^2}{[\mathbf H^{-1}]_{qq}} .$$

The $q$-th component of $\boldsymbol\delta$ is exactly $-(w_q-\mathrm{quant}(w_q))$ — it snaps $w_q$ onto the grid — and the other entries spread the compensation along the input correlations encoded in $\mathbf H^{-1}$. This is the move RTN never makes: a quantized weight is allowed to drift the *remaining* weights to keep the output right, rather than each one being rounded in isolation.

The original recipe (OBQ) picks, per row, the cheapest-to-quantize free weight next — a greedy order — and so maintains a separate evolving $\mathbf H^{-1}$ per row at cost $O(d_{\text{row}}\,d_{\text{col}}^3)$, hopeless on a $4096\times4096$ (or $14336\times4096$) Mistral linear. The diagnostic that rescues it: on large over-parameterized layers, clever ordering barely beats a fixed arbitrary order, because the few high-error weights rounded early are a vanishing fraction. So I drop the greedy order and quantize *all rows in the same fixed left-to-right column order*. Then $\mathbf H^{-1}$ is shared across rows and downdated once per column instead of once per (row, column), and the cost collapses by a factor of $\min(d_{\text{row}},d_{\text{col}})$. The sweep becomes: round column $i$, form each row's scaled error $\mathbf E_{:,i}=(\mathbf W_{:,i}-\mathrm{quant}(\mathbf W_{:,i}))/[\mathbf H^{-1}]_{ii}$, and push it into the not-yet-quantized columns $j>i$ via $\mathbf W_{:,j}\mathrel{-}=\mathbf E_{:,i}\,[\mathbf H^{-1}]_{ij}$.

Two practical issues fix the exact code. For throughput, the rank-one global downdate touches a huge matrix with a couple of FLOPs per entry — pure bandwidth, tensor cores idle — repeated $d_{\text{col}}$ times. The fix is blocking: the rounding of column $i$ depends only on updates from columns before it, so I process a block of `BLOCK_SIZE = 128` consecutive columns using only the $128\times128$ corner of $\mathbf H^{-1}$, doing the column-by-column compensation *within* the block, and then apply the block's accumulated error to all columns to the right in one GEMM, $\mathbf W_{:,\text{rest}}\mathrel{-}=\mathbf E_{\text{block}}\,\mathbf H^{-1}_{\text{block},\text{rest}}$ — same arithmetic, but the heavy update now saturates the hardware. For numerics, some layers' $\mathbf H$ are near-singular (input directions calibration never excited), so before inverting I add a dampening of `PERCDAMP = 0.01` times the mean diagonal, $\mathbf H\mathrel{+}=0.01\cdot\overline{\operatorname{diag}\mathbf H}\cdot\mathbf I$.

One honest point about *this* fill versus the most refined form of the method. The most aggressive GPTQ never forms $\mathbf H^{-1}$ at all — it observes that the downdate is one step of Gaussian elimination and sweeps a Cholesky factor of $\mathbf H^{-1}$, whose rows already hold the scaled row-tails the update needs. This task takes the more direct route: it computes the dense inverse once, `L = cholesky(H); Hinv = cholesky_inverse(L)` with a `pinv` fallback, then reads $[\mathbf H^{-1}]_{ii}$ for the per-column error and the rows $[\mathbf H^{-1}]_{i,\cdot}$ for the propagation, dividing the error by $[\mathbf H^{-1}]_{ii}$ each step. Mathematically this reproduces the same OBS compensation; it just carries the full dense inverse and uses its raw entries rather than a Cholesky factor's scaled tails. A single dampening plus a single factorization is enough to keep Mistral's layers from drifting into a wrecked state the way an iterated rank-one downdate would. Grouping comes for free: the derivation only assumed a fixed grid and a rounding map, never one scale per row, so at each group boundary (`col % group_size == 0`) I recompute the symmetric per-group scale $g_{\max}/q_{\max}$ from that group's columns of the *current* weight matrix — fit to the weights as compensation has already moved them, so grouping and the second-order correction reinforce each other, and the same code path serves group 128, group 64, and per-channel.

The delta from RTN is therefore: keep the same symmetric per-group grid, but accumulate $\mathbf X^\top\mathbf X$, dampen and invert it once, and sweep columns in blocks of 128, compensating each column's residual onto the still-free columns through the inverse Hessian so the *output*, not the per-weight value, is preserved. I expect INT3-g128 to recover substantially below RTN's $6.7341$ into the low-6 range, since the deferred error now has somewhere to go; INT4-g128 a real but smaller gain under $5.1343$; and INT4-g64 the smallest gap, since fine grouping already does much of compensation's work. The cost I accept is roughly an order of magnitude more quant-time, for the inverse and the column sweep per layer.

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
