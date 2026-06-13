RTN's numbers came back and they point straight at its successor. At INT4-g128 it lands 5.1343 — a
degradation of 0.2271 over the 4.9071 FP16 reference, which is survivable but visible. At INT4-g64 it
improves to 5.0890 (degradation 0.1819), exactly the direction I predicted: a smaller group means the
scale is set by the local max over 64 columns instead of 128, fewer typical weights get polluted by each
outlier, so finer granularity buys a small recovery. The standard INT4 setting is clearly not where RTN
breaks. INT3-g128 is. There it jumps to 6.7341 — a degradation of 1.8270, eight times the INT4 hit and an
absolute perplexity blow-up. Halving the grid (8 levels instead of 16) more than doubled the per-element
residual, the outlier-inflated scale now spends a precious third of the levels on the extremes, and across
32 transformer blocks the per-layer output errors compounded. So the diagnosis is confirmed in the
metric: RTN is fine where the grid is fat and falls apart where it is thin, and the thin-grid case is the
one I most need to fix. The reason is structural, and it is the thing I flagged when I built the floor: RTN
minimizes per-weight error, $\lVert\mathbf W-\widehat{\mathbf W}\rVert^2$, element by element, when the
quantity that actually propagates to the next block is the layer output $\mathbf W\mathbf X$. It also
throws away the calibration data the scaffold is streaming through `add_batch` for free. Both of those are
exactly what I now spend.

So let me state the honest objective RTN was approximating badly. For one linear layer with weights
$\mathbf W$ and calibration inputs stacked as columns of $\mathbf X$, I want quantized $\widehat{\mathbf W}$
that keeps the layer's output close:
$\arg\min_{\widehat{\mathbf W}}\lVert\mathbf W\mathbf X-\widehat{\mathbf W}\mathbf X\rVert_2^2$. The grid is
fixed up front; each weight is free to land on any grid value. The first useful fact is that this
decomposes by row of $\mathbf W$: output channel $r$ contributes
$\lVert\mathbf w_r^\top\mathbf X-\widehat{\mathbf w}_r^\top\mathbf X\rVert^2$ and the rows do not interact,
because the output channels of a linear layer are independent linear functionals of the same input. So per
row it is a quadratic in the weight vector, and the curvature is identical for every row: differentiate
$\lVert\mathbf w^\top\mathbf X-\widehat{\mathbf w}^\top\mathbf X\rVert^2$ twice and the Hessian is
$\mathbf H=2\mathbf X\mathbf X^\top$, which depends only on the inputs, not on any weights. This is why the
calibration stream matters: $\mathbf X\mathbf X^\top$ is precisely the input second moment, and it tells
me which input directions are excited hard by real text — the directions where a weight error costs the
output a lot — and which are nearly dead, where RTN was wasting its worry. In the scaffold I accumulate
exactly this in `add_batch`: reshape the layer input to $(\text{tokens},\text{in\_features})$ and add
$\mathbf X^\top\mathbf X$ into `self.H` (the proportionality constant 2 is irrelevant; I will normalize by
the sample count at quantize-time, `H /= nsamples`, to keep the scale of the dampening sane across
layers). RTN kept this `H` buffer purely for interface compatibility; now it is the heart of the method.

The accurate small-model idea I reach for is the brain-surgeon logic specialized to quantization. Given the
row quadratic, I fix one coordinate $q$ to its chosen grid value and move all the *other* still-free
coordinates to compensate optimally. Let $\boldsymbol\delta=\widehat{\mathbf w}-\mathbf w$ and let the
quantized target be $\mathrm{quant}(w_q)$. The row's quadratic increase is
$\tfrac12\boldsymbol\delta^\top\mathbf H\boldsymbol\delta$ subject to
$\mathbf e_q^\top\boldsymbol\delta=\mathrm{quant}(w_q)-w_q$. The Lagrangian gives
$\mathbf H\boldsymbol\delta+\lambda\mathbf e_q=0$, so $\boldsymbol\delta=-\lambda\mathbf H^{-1}_{:,q}$;
enforcing the constraint gives $\lambda=(w_q-\mathrm{quant}(w_q))/[\mathbf H^{-1}]_{qq}$, and the optimal
compensating update to the free set is
$\boldsymbol\delta_F=-\tfrac{w_q-\mathrm{quant}(w_q)}{[\mathbf H^{-1}]_{qq}}(\mathbf H^{-1})_{:,q}$, with a
loss increase $\tfrac12(\mathrm{quant}(w_q)-w_q)^2/[\mathbf H^{-1}]_{qq}$. The sign checks out: the $q$-th
component of $\boldsymbol\delta$ is exactly $-(w_q-\mathrm{quant}(w_q))$, i.e. it snaps $w_q$ onto the grid,
and the other entries spread the compensation along the input correlations encoded in $\mathbf H^{-1}$.
This is the move RTN never makes: it lets a quantized weight drift the *remaining* weights to keep the
output right, rather than rounding each one in isolation.

The original brain-surgeon recipe (OBQ) picks, per row, the cheapest-to-quantize free weight next — a
greedy order — and so maintains a *separate* evolving $\mathbf H^{-1}$ per row, costing
$O(d_{\text{row}}\cdot d_{\text{col}}^3)$, cubic in the column count and multiplied by every row. On a
Mistral linear that is $4096\times4096$ (or $14336\times4096$ for the MLP) this is hopeless. But the
diagnostic that rescues it: greedy ordering barely beats a fixed arbitrary order on large,
over-parameterized layers — the gain of being clever about order shrinks as layers grow, because the few
high-error weights rounded early are a vanishing fraction of the layer. So I drop the greedy order and
quantize *all rows in the same fixed left-to-right column order*. Then $\mathbf H$ and $\mathbf H^{-1}$
depend only on $\mathbf X$, identical across rows, and I downdate one shared inverse exactly $d_{\text{col}}$
times instead of per row — the cost collapses by a factor of $\min(d_{\text{row}},d_{\text{col}})$. The
per-column update becomes: round column $i$, form each row's scaled error
$\mathbf E_{:,i}=(\mathbf W_{:,i}-\mathrm{quant}(\mathbf W_{:,i}))/[\mathbf H^{-1}]_{ii}$, and push it into
the not-yet-quantized columns $j>i$ via $\mathbf W_{:,j}\mathrel{-}=\mathbf E_{:,i}[\mathbf H^{-1}]_{ij}$.
The algorithm is now a column sweep.

Two practical problems remain, and here is where this task's fill makes a specific choice I should be
honest about. First, throughput: the rank-one inverse downdate touches a huge matrix with a couple of
FLOPs per entry — a terrible compute-to-memory ratio, all bandwidth, tensor cores idle, repeated
$d_{\text{col}}$ times. The fix is blocking: the rounding decision for column $i$ depends only on updates
from columns *before* $i$, so updates to far-right columns can be deferred. Process a block of
`BLOCK_SIZE=128` consecutive columns using only the $128\times128$ corner of $\mathbf H^{-1}$ and the
128-wide weight slab, doing the column-by-column compensation *within* the block; then, once the block is
finished, apply the block's accumulated error to *all* columns to the right in one matrix-matrix multiply
$\mathbf W_{:,\text{rest}}\mathrel{-}=\mathbf E_{\text{block}}\,\mathbf H^{-1}_{\text{block},\text{rest}}$.
Same total arithmetic, but the heavy global update is a GEMM that saturates the hardware while the
bandwidth-bound fiddling is confined to the block. That is exactly the loop structure here: an outer loop
over `col_start` in steps of 128, an inner loop over the columns of the block computing per-row `err` and
updating `W_block[:, j+1:]` via `Hinv[col, col_start+j+1:col_end]`, and after the inner loop a single
`W[:, col_end:] -= Err[:, block] @ Hinv[block, col_end:]`.

Second, numerics. I am consuming the inverse Hessian, and at 7B scale some layers' $\mathbf H$ are
near-singular — input directions that calibration never excited leave $\mathbf H$ rank-deficient, and the
inverse blows up. Two defenses. A mild dampening: add $\lambda$ equal to `PERCDAMP=0.01` of the mean
diagonal of $\mathbf H$ before inverting, $\mathbf H\mathrel{+}=0.01\cdot\overline{\operatorname{diag}\mathbf H}\cdot\mathbf I$,
which keeps it well-conditioned. And a stable inverse: take a Cholesky factorization of the dampened
$\mathbf H$ and invert through it — `L = cholesky(H); Hinv = cholesky_inverse(L)` — with a `pinv` fallback
if even the dampened matrix fails to factor. This is the place this task's implementation *differs* from
the most aggressive form of the method, and I want to be precise so the code I land is the code that runs.
The most refined version of GPTQ never forms $\mathbf H^{-1}$ at all: it observes that the OBS downdate is
one step of Gaussian elimination, so the scaled row-tails the update needs live in the rows of a Cholesky
factor of $\mathbf H^{-1}$, and it sweeps that factor — avoiding any repeated inversion. This task's fill
takes the more direct route: it computes the dense inverse $\mathbf H^{-1}$ once via Cholesky, then reads
$[\mathbf H^{-1}]_{ii}$ for the per-column error and the rows $[\mathbf H^{-1}]_{i,\cdot}$ for the
propagation, dividing the error by $[\mathbf H^{-1}]_{ii}$ each step. Mathematically this reproduces the
same OBS compensation — error $\propto(w-q)/[\mathbf H^{-1}]_{ii}$ spread by row $i$ of the inverse — but
it carries the full dense inverse and the within-block updates use its raw entries rather than a Cholesky
factor's scaled tails. So I should not import the "Cholesky-of-the-inverse stores the row-tails" story
into how I describe what runs here; what runs is "one stable inverse, read directly, blocked." The single
dampening + single factorization is enough to keep Mistral's layers from drifting into a wrecked layer the
way an iterated rank-one downdate would.

One more piece this task exposes and I must get right: grouping. The derivation only assumed a fixed grid
and a $\mathrm{quant}(\cdot)$ that rounds onto it — it never assumed one scale per row — so I can use a
per-group scale, an independent step for every `group_size` consecutive columns, and the error-compensation
machinery is unchanged. Here, when the column sweep reaches a group boundary (`col % group_size == 0`), I
recompute the symmetric per-group scale from that group's columns of the *current* weight matrix:
$g_{\max}=\max|\mathbf W_{:,\text{group}}|$ per row, scale $=g_{\max}/q_{\max}$, zero point 0. Because the
sweep updates columns as it goes, each group's scale is fit to the weights *as compensation has already
moved them*, so grouping and the second-order correction reinforce each other — and the same code path
serves group 128, group 64, and per-channel (one scale per row), which is why the same class must run
across all three eval settings. Rounding a column is then
$\operatorname{clamp}(\operatorname{round}(w_{\text{col}}/\text{scale}),q_{\min},q_{\max})\cdot\text{scale}$,
and the quantized column is written into $\mathbf Q$, whose final cast back to the layer dtype is the
returned weight. (The full scaffold module is in the answer.)

So the delta from RTN: keep the same symmetric per-group grid, but accumulate $\mathbf X^\top\mathbf X$ in
`add_batch`, dampen and invert it once via Cholesky, and sweep columns left-to-right in blocks of 128,
rounding each column and compensating its residual onto the still-free columns through the inverse
Hessian so the layer *output* — not the per-weight value — is what is preserved. The only thing RTN and
GPTQ share is the grid; everything that decides *how* a weight rounds is now informed by the calibration
inputs RTN ignored.

Reading RTN's shape, here is what I expect, falsifiably, against its numbers. The whole point of error
compensation is to recover resolution where the grid is thinnest, so INT3-g128 is the setting I am
watching: RTN's 6.7341 (degradation 1.8270) is the blow-up, and if column-by-column compensation works at
all it should cut that degradation substantially — I expect INT3 to drop well under RTN's, into the low-6
range, because the deferred error now has somewhere to go instead of accumulating. At INT4-g128 the grid
is already fat enough that RTN was only at 0.2271 degradation, so I expect a real but smaller
improvement — into the low-5.0s, comfortably under RTN's 5.1343. At INT4-g64 RTN was already strong
(5.0890); compensation should still help, but this is the setting where I expect the *smallest* gap,
because fine grouping has already done much of the work error compensation would otherwise do — the two
mechanisms partly overlap. The cost I accept: where RTN quantized a block in ~22s, GPTQ pays for an
inverse and a column sweep per layer, so I expect quant-time to rise by roughly an order of magnitude.
The bet is that the INT3 recovery is large enough to justify it — and if INT3 does *not* move much, that
would tell me the bottleneck at 8 levels is the grid coarseness itself, not the rounding objective, which
would push the next rung toward protecting weights *before* rounding rather than compensating *after*.
