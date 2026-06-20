Round-to-nearest gave me the floor and it is exactly as advertised: fine at 4-bit g128 (Llama-2-7B at
5.73 perplexity, within a few tenths of FP16), and a wreck at 3-bit per-channel — LLaMA-7B at 25.54
against an FP16 of 5.68. That 25.54 is the number I have to beat. The thing that nags me is that RTN
throws away information it does not have to. It rounds every weight to its nearest grid point
*independently*, as if each weight stood alone. But the weights in a linear layer do not act alone:
they are summed against an input to produce an output, and what I actually care about is not whether
each weight is close to the grid but whether the *layer's output* is close to what it was in FP16. RTN
optimizes the wrong objective.

Let me write the right objective down. For a linear layer with weight W and calibration inputs X, the
quantity I want to preserve is the output WX. So the honest per-layer problem is

  min_{Ŵ on the grid}  ‖ WX − Ŵ X ‖²_F .

This is a constrained least-squares problem: find the grid-valued Ŵ that reproduces the FP16 output on
the calibration data. RTN ignores X entirely and just snaps W to the grid; it is the solution you get
if you pretend X = I. The moment I put X back in, I can do much better, because I can let the *errors
made on already-quantized weights be compensated by adjusting the weights not yet quantized*. If I
round one weight down and that pushes the output up, I can nudge a still-unquantized weight to push it
back. RTN has no mechanism for this. The least-squares objective does.

There is a classical tool for exactly this: Optimal Brain Surgeon / Optimal Brain Quantization. Quantize
the weights one at a time; at each step pick the weight whose quantization does least damage, round it,
and then apply a closed-form update to *all the remaining* weights that optimally compensates for the
error just introduced. The compensation is governed by the Hessian of the layer objective, which here
is beautifully simple: the objective is quadratic in W, so the Hessian is H = 2 X Xᵀ — it depends only
on the inputs, not on the weights. The OBQ update rounds weight w_q, incurs error (w_q − quant(w_q)),
and redistributes it onto the surviving weights weighted by the rows of H⁻¹. Run to completion and you
get the least-squares-optimal grid assignment, far better than independent rounding.

So why isn't everyone already doing this on 175B-parameter models? Because OBQ is far too slow. It
re-evaluates which weight to quantize next and downdates the inverse Hessian after every single weight,
which makes it cubic in the layer width per row and quadratic in the number of rows — fine for a small
net, hopeless for billions of parameters. If I want the OBS-quality compensation at LLM scale I have to
make it scale. I have three problems to fix, and each one is a real algorithmic change, not a tweak.

**First: kill the greedy order.** OBQ's per-weight cleverness is choosing, for each row, the order that
quantizes the least-damaging weight next. But I notice something in the regime I care about: on large,
heavily over-parameterized layers the greedy order barely beats an arbitrary fixed order — the
advantage of being clever about which weight to do next shrinks as the layer grows, because there is so
much redundancy that almost any order works about as well. That observation is liberating, because the
Hessian H = 2XXᵀ depends only on the inputs and is *shared across all rows*. If I quantize every row in
the *same* fixed left-to-right column order, then the inverse-Hessian information every row needs is
identical, so I compute one H⁻¹ and downdate it *once per column* instead of once per weight. That alone
collapses the cost from per-weight to per-column work.

**Second: batch the updates.** Even shared across rows, applying the rank-one error-compensation update
to all columns to the right after every single column is bandwidth-bound — lots of memory traffic, tiny
arithmetic. So I process columns in *blocks* of B = 128. Inside a block I do the per-column updates but
keep the compensation contained to the block; then once per block I apply the block's whole accumulated
error to all columns to the right in a *single* matrix multiply. That turns a stream of memory-bound
rank-one updates into one compute-bound GEMM per block — exactly what a GPU wants.

**Third: stop downdating the inverse.** OBS keeps an explicit inverse Hessian and downdates it in place
after each step. At this scale, with thousands of accumulating downdates, that in-place inverse drifts
numerically and eventually goes indefinite, and the algorithm falls apart. But the OBS inverse-downdate
is, when I look at it carefully, *exactly symmetric Gaussian elimination* on H⁻¹. That means all the
scaled inverse-Hessian row-tails the update needs can be read directly off a single **Cholesky factor**
of H⁻¹, computed once, up front, in a numerically stable routine — instead of thousands of in-place
downdates that drift. I add a mild dampening of about 1% of the mean diagonal to H before inverting so
the Cholesky is well-conditioned, take the upper factor U with H⁻¹ = UᵀU, and then for column j the
quantity U[j, j:] / U[j,j] *is* the sequentially-downdated inverse row I need. No running inverse at
all.

Putting the three together: the algorithm builds H = 2XXᵀ from calibration activations, dampens and
Cholesky-factors H⁻¹ once, then sweeps blocks of 128 columns left to right; inside each block it rounds
each column to the grid, forms the scaled error (w − q)/U_jj, and pushes that error onto the remaining
in-block columns via the Cholesky row-tail; after the block it applies the accumulated error to all
columns to the right with one GEMM.

```python
H = torch.linalg.cholesky(H)            # H starts as 2 X Xᵀ + damping
H = torch.cholesky_inverse(H)
U = torch.linalg.cholesky(H, upper=True)            # scaled inverse-Hessian row-tails, computed ONCE
for i1 in range(0, d_col, blocksize):               # blocks of 128 columns
    for i in range(i2 - i1):
        w = W1[:, i]; d = U1[i, i]
        q = quantizer.quantize(w.unsqueeze(1)).flatten()   # round THIS column to the grid
        err = (w - q) / d                                  # scaled error
        W1[:, i:] -= err.unsqueeze(1) * U1[i, i:].unsqueeze(0)   # compensate within block
    W[:, i2:] -= Err1.matmul(U[i1:i2, i2:])                # one GEMM updates the rest of the row
```

It is grid-agnostic, which matters: the inner `quantize` is just RTN onto whatever grid I hand it, so I
can combine this compensation with per-channel *or* g128 scales, and the group scales can even be fit
against the *already-updated* weights — which is what unlocks the extreme 2–3-bit regime where RTN died.

Here is the bar and the bet. RTN at 3-bit per-channel on LLaMA-7B is 25.54. My claim is that replacing
independent rounding with output-reconstruction-optimal compensation — round a weight, then repair the
damage on the weights downstream — should recover most of the gap to FP16 (5.68), because the layer
output, not the per-weight grid distance, is what I am now minimizing. I am betting 3-bit per-channel
comes down from 25.54 to single digits, into the 8-ish range. The two risks are real: the greedy-order
shortcut might cost accuracy on some layer (I am wagering the over-parameterization makes order
irrelevant), and the whole thing only works at scale because the Cholesky reformulation keeps the
arithmetic stable across thousands of columns. If it holds, low-bit weight-only quantization stops
being a memory trick and becomes nearly lossless — and the next question becomes which *weights*
deserve the protection, not just how to round them.
