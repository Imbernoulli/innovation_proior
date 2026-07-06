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
back. RTN has no mechanism for this. The least-squares objective does. And this is exactly the "shape
the noise so it cancels at the output rather than accumulates" lever I identified as the thing plain
rounding could not reach — the calibration activations X are the information RTN refused to use.

Let me look at one row to see the structure, because a linear layer's rows are independent under this
objective and that is going to matter. Take a single output row wᵀ (length d, the input dimension). Its
contribution to the loss is ‖(w − ŵ)ᵀX‖² = (w − ŵ)ᵀ (XXᵀ) (w − ŵ). That is a quadratic form in the
perturbation Δw = w − ŵ, so the objective is *exactly* quadratic and its Hessian is H = 2XXᵀ — and,
crucially, H depends only on the inputs, not on the weights, so it is the *same matrix for every row of
the layer*. That single fact is the seed of everything that makes this scale; I will come back to it.

Before I reach for the classical machinery, let me weigh the cruder options for minimizing this
objective, because if one of them is good enough I should not build anything heavier. The first is
straight retraining or quantization-aware training: unfreeze the weights, add fake-quantization to the
forward pass, and fine-tune with gradient descent so the model *learns* to be robust to the grid. That
would work, but it violates the whole premise — I have a fixed pretrained model, a few hundred
calibration sequences, and a few GPU-hours, not a training run over the pretraining corpus; QAT on a 7B
model is days of compute and risks moving the weights away from a checkpoint I was handed as fixed. The
second is iterative RTN: round, measure the output error, and re-round with a corrected scale, looping.
But that only adjusts the *scale*, not the individual rounding decisions, and the loss here is not a
function of one scalar per channel — it is combinatorial over which grid point each weight snaps to, so a
scale search cannot reach the compensation I need. The third is exhaustive or randomized search over
grid assignments, which is hopeless: a single 4096-wide row at 3 bits has 8^4096 assignments. So the
honest position is that I want the *exact minimizer of the quadratic under the discrete constraint*, and
I want it in closed form, one weight at a time — which is precisely what the classical second-order
pruning literature already solved for a nearly identical problem.

There is a classical tool for exactly this quadratic-with-a-discrete-constraint problem: Optimal Brain
Surgeon / Optimal Brain Quantization. Quantize the weights one at a time; at each step pick the weight
whose quantization does least damage, round it, and then apply a closed-form update to *all the
remaining* weights that optimally compensates for the error just introduced. Let me actually derive the
two formulas I need, because the whole method lives in them. Suppose I decide to fix coordinate q to its
quantized value, i.e. I impose the constraint e_qᵀΔw + (w_q − quant(w_q)) = 0. Minimizing ½Δwᵀ H Δw
subject to that one linear constraint is a Lagrange problem whose solution is

  Δw = − (w_q − quant(w_q)) / [H⁻¹]_qq · H⁻¹ e_q ,   with cost increment  ½ (w_q − quant(w_q))² / [H⁻¹]_qq .

So the damage of quantizing coordinate q is (rounding error)² divided by [H⁻¹]_qq, and the optimal
repair spreads −error/[H⁻¹]_qq times the q-th column of H⁻¹ across all the still-free weights. Greedy OBQ
reads off the first formula to pick the least-damaging q at each step, applies the second, then downdates
H⁻¹ to remove the now-frozen coordinate and repeats. Run to completion and you get a grid assignment far
better than independent rounding, because every rounding error has been partly absorbed by the weights
downstream of it.

So why isn't everyone already doing this on 175B-parameter models? Because OBQ is far too slow, and I
should count the cost precisely rather than assert it. For one row of width d, OBQ does d elimination
steps; each step must downdate the d×d inverse Hessian, an O(d²) operation, so one row is O(d³), and a
layer with d_row output rows — each, in principle, choosing its *own* greedy order and therefore its own
sequence of downdates — is O(d_row · d³). For a middling 4096×4096 layer that is 4096 · 4096³ ≈ 4096⁴ ≈
2.8×10¹⁴ operations *per layer*, and a 7B model has dozens of such layers; this is hopeless. If I want
OBS-quality compensation at LLM scale I have to make it scale, and I have three distinct problems to
fix, each a real algorithmic change rather than a tweak.

**First: kill the greedy order.** OBQ's per-weight cleverness is choosing, for each row, the order that
quantizes the least-damaging weight next. But I notice something about the regime I care about: on
large, heavily over-parameterized layers the greedy order barely beats an arbitrary fixed order — the
advantage of being clever about which weight to do next shrinks as the layer grows, because there is so
much redundancy that almost any order works about as well. That observation is liberating, and it
connects straight to the seed fact above: because H = 2XXᵀ is shared across all rows, if I quantize
every row in the *same* fixed left-to-right column order, then the inverse-Hessian information every row
needs is identical, so I compute one H⁻¹ and downdate it *once per column* instead of once per weight per
row. Recount the cost: forming and factoring H is O(d³) done once, and the per-row work drops to O(d²)
(each row just reads the shared factor), so the layer is O(d³ + d_row·d²). For 4096×4096 that is 6.9×10¹⁰
+ 4096·1.7×10⁷ ≈ 1.4×10¹¹ — about 2000× cheaper than the 2.8×10¹⁴ of naive per-row OBQ, which is the
difference between minutes and never. The greedy order was buying an accuracy improvement that the
over-parameterization has made nearly worthless, and giving it up is what unlocks the shared inverse.

**Second: batch the updates.** Even shared across rows, applying the rank-one error-compensation update
to all columns to the right after every single column is bandwidth-bound: each such update touches the
whole trailing block of the weight matrix but does only O(d) arithmetic per element it reads, so it is
memory-traffic-limited — lots of DRAM reads, tiny compute, exactly the wrong ratio for a GPU. So I
process columns in *blocks* of B = 128. Inside a block I do the per-column OBS updates but keep the
compensation contained to the block (128 columns wide); then once per block I apply the block's whole
accumulated error to all columns to the right in a *single* matrix multiply. That turns a stream of B
memory-bound rank-one updates into one compute-bound GEMM per block — the arithmetic intensity rises by
roughly B, and the operation moves from the bandwidth-bound side of the roofline to the compute-bound
side, which is where the hardware is fast. The math is identical to doing every update immediately; the
blocking only changes *when* the trailing update is applied, not its value.

**Third: stop downdating the inverse.** OBS keeps an explicit inverse Hessian and downdates it in place
after each step. At this scale, with thousands of accumulating in-place rank-one downdates, that inverse
drifts numerically and eventually loses positive-definiteness — a single downdate that pushes an
eigenvalue slightly negative makes the "cost increment" formula produce negative damage, the algorithm
starts *maximizing* error, and the whole layer diverges. So I need the sequence of downdated inverse rows
without ever forming a running inverse. Here is the observation that saves it: the OBS inverse-downdate
is, when I write it out, *exactly symmetric Gaussian elimination* on H⁻¹. That means all the scaled
inverse-Hessian row-tails the update needs can be read directly off a single **Cholesky factor** of H⁻¹,
computed once, up front, in a numerically stable routine — instead of thousands of in-place downdates
that drift. Concretely: I add a mild dampening of about 1% of the mean diagonal to H before inverting
(this shifts every eigenvalue up by the same small amount so the smallest never crosses zero, and it
costs almost nothing because H's diagonal is large where it matters), factor to get H⁻¹, then take its
upper Cholesky factor U with H⁻¹ = UᵀU. The 1% figure is a deliberate compromise I can reason about: too
small and it fails to lift a near-zero eigenvalue back to safety, so the factorization is unstable on
layers with rank-deficient calibration statistics; too large and the damping distorts H away from the
true second moment, so the compensation is solving a perturbed problem and leaves accuracy on the table.
One percent of the mean diagonal is small relative to the large diagonal entries that dominate the
compensation, yet comfortably larger than the numerical noise floor of the factorization — it buys
stability where H is degenerate without meaningfully corrupting H where it is well-conditioned. For column j, the quantity U[j, j:] / U[j, j] *is* precisely the
sequentially-downdated inverse row the OBS update at step j would have produced — the elimination has
already been done, once, stably, by the Cholesky routine. No running inverse at all.

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

One implementation detail in building H is worth reasoning about rather than copying blindly, because
it governs whether the Hessian is a faithful average over the calibration set. I do not have all
calibration activations in memory at once — they arrive as a stream of batches as the harness walks the
model forward — so I accumulate H as a running mean: on receiving a batch of `b` token-vectors I rescale
the existing H by nsamples/(nsamples + b), increment the count, and add the new outer product scaled by
2/nsamples. The effect is that H is always the average of 2XXᵀ over exactly the tokens seen so far, so
the Hessian is normalized to the *number* of calibration tokens rather than growing with it — which
keeps the 1%-of-mean-diagonal damping meaningful (a fixed fraction of a stable quantity) regardless of
how many sequences I calibrate on. The block size B = 128 is not arbitrary either: it is large enough
that the once-per-block trailing GEMM dominates the memory-bound inner updates (so the roofline argument
holds), and small enough that the in-block trailing matrix W1 stays in fast memory and the compensation
stays local; it also happens to match the natural g128 group width, so a block boundary and a group
boundary can be made to coincide when I combine this with group quantization.

Let me sanity-check the arithmetic on a two-column toy so I trust the compensation is doing what the
derivation promised. Take a single row w = [w₁, w₂] with a Hessian that couples the two columns, and
quantize column 1 first with error δ = w₁ − q₁. The OBS update sends w₂ ← w₂ − (δ/[H⁻¹]₁₁)·[H⁻¹]₁₂; in
the Cholesky reading that is exactly w₂ ← w₂ − (δ/U₁₁)·U₁₂/U₁₁ scaled through the factor. The point is
the *sign*: if rounding column 1 down (δ > 0) tends, through the positive off-diagonal coupling, to
raise the output, the update lowers w₂ to compensate, and column 2's own subsequent rounding then starts
from an already-corrected value. So the second column is not rounded to reproduce its original weight —
it is rounded to reproduce the *output the first column's error left behind*. That is the whole mechanism
in miniature, and it is why the layer output, not the per-weight grid distance, is what shrinks.

I should also confront a couple of things that could go wrong, because I am betting the layer on them.
One is dead input channels: if some calibration input channel is always zero, the corresponding diagonal
of H is zero and H⁻¹ blows up. The guard is to detect those (diag(H) == 0), set the weight column to
zero and pin that diagonal to 1 so the factorization stays finite — those channels contribute nothing to
the output anyway, so zeroing them is exact, not an approximation. The other is the greedy-order gamble
itself: on some unusually low-redundancy layer the fixed order might cost real accuracy. I am wagering
that the over-parameterization of a 7B-and-up Transformer makes the order irrelevant everywhere it
matters, and the fixed shared order is the price of the whole speedup — if that wager is wrong it will
show up as a layer that reconstructs poorly, but the redundancy argument says it holds. A quick way to
see why order should barely matter: the greedy choice buys the difference between quantizing the
least-damaging free coordinate and an arbitrary one, and that difference is the spread in the cost-
increment (error)²/[H⁻¹]_qq across candidate columns; in a heavily redundant layer the free directions
are numerous and interchangeable, so that spread is small and the order barely changes the total. In a
narrow, near-orthogonal layer it would not be — which is exactly where I would expect the fixed order to
cost something — but the layers that dominate a large Transformer's parameter count are the wide ones.

There is one discomfort I want to name honestly even as I commit, because it is the seam the next rung
will pull on. This whole method *regresses the quantized weights against a particular few hundred
calibration sequences*: H = 2XXᵀ is that calibration set's second-moment matrix, and the compensation
tunes every weight to reproduce the FP16 output specifically on those inputs. It is a small, strongly
constrained fit — the weights can only move onto grid points, which is a heavy regularizer — so I do not
expect gross overfitting, but it is a fit nonetheless, and it treats the calibration distribution as if
it were the deployment distribution. If those diverge, or if the calibration set is small and peculiar,
the compensation could be repairing errors that do not matter and creating ones that do. I flag it and
move on, because at the accuracy I am chasing the fit is worth it; but a method that got the same
recovery *without* a per-layer second-order regression against calibration data would be strictly safer,
and that is a direction I can already feel.

It is grid-agnostic, which matters and is the bridge to the next rung: the inner `quantize` is just RTN
onto whatever grid I hand it, so I can combine this compensation with per-channel *or* g128 scales, and
the group scales can even be fit against the *already-updated* weights within the block — which is what
unlocks the extreme 2–3-bit regime where RTN died, because at those bit-widths the compensation is
carrying most of the load and the scale just has to be locally reasonable.

Let me estimate how much of the gap I should expect to recover, rather than just hoping for "single
digits". The RTN failure at 3-bit was that each row's output error was the full uncancelled sum of its
per-weight rounding errors: with d ≈ 4096 columns each contributing an independent error of RMS ≈ 0.385σ
(from the floor's SQNR arithmetic), the row-output error is large and, worse, it accumulates coherently
down the 32-layer stack. What the OBS compensation does is drive the *residual* output error toward the
part that is genuinely unrepresentable — the component of the rounding that no downstream weight can
absorb because it lies outside the span the remaining columns can reach. Since a wide layer has far more
free directions than binding constraints at any step, most of each rounding error *is* absorbable, so I
expect the reconstructed output error to be a small fraction of RTN's, and the depth-compounding to run
the other way: instead of √L amplification of a large per-layer error, I get √L amplification of a much
smaller one. That is why I expect the perplexity to fall by something like the 25.54 → high-single-digits
jump rather than merely a few points — the mechanism attacks the dominant term. But I also expect it to
*stop* short of FP16 (5.68), because compensation cannot manufacture resolution that the 8-level grid
simply does not have: the unrepresentable residual is real, and it is the ~2-plus perplexity points that
will remain. I should also hold my comparison discipline from the floor — 25.54 and my target are both
3-bit *per-channel* LLaMA-7B, so they are directly comparable; I must not quietly congratulate GPTQ by
lining it up against a g128 number, which would be a different, easier column.

Here is the bar and the bet, read directly off the floor. RTN at 3-bit per-channel on LLaMA-7B is 25.54
against an FP16 of 5.68 — an 8-bit-of-quality hole. My claim is that replacing independent rounding with
output-reconstruction-optimal compensation recovers most of that hole, because I have swapped the wrong
objective (per-weight grid distance) for the right one (layer output on the calibration data), and the
depth-compounding argument from the floor now runs in reverse: errors that used to accumulate down the
stack are instead cancelled at each layer's output before they propagate. I am betting 3-bit per-channel
LLaMA-7B comes down from 25.54 into single digits, into the 8-ish range — a more-than-3× cut in the
perplexity gap — achieved post-training in a few GPU-hours on a few hundred calibration sequences with
no retraining. The two risks are the ones I named: the greedy-order shortcut, and the numerical
stability that only the Cholesky reformulation buys at scale. If it holds, low-bit weight-only
quantization stops being a memory trick and becomes nearly lossless — and the residual ~2.4 perplexity
gap to FP16 reframes the next question. GPTQ spends its bit budget *uniformly*: it minimizes total
output error but has no notion that some weights matter far more than others. The next lever is not
better rounding of all weights, but deciding *which* weights deserve the protection in the first place.
