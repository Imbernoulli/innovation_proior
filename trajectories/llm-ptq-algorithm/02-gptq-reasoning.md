RTN's numbers came back and they point straight at its successor. At INT4-g128 it lands 5.1343 — a
degradation of 0.2271 over the 4.9071 FP16 reference, survivable but visible. At INT4-g64 it improves to
5.0890 (degradation 0.1819), exactly the direction I predicted: a smaller group means the scale is set by
the local max over 64 columns instead of 128, fewer typical weights get polluted by each outlier, so finer
granularity buys a small recovery — the gap is only $0.2271-0.1819=0.045$ of perplexity, confirming that
at INT4 the level count, not the group extent, is the bottleneck. The standard INT4 setting is clearly not
where RTN breaks. INT3-g128 is. There it jumps to 6.7341 — a degradation of 1.8270, and the ratio tells
the story: $1.8270/0.2271\approx8$, so the INT3 hit is eight times the INT4 hit at the same group size.
That is even worse than my $\Delta^2/12$ estimate of $\approx5.44\times$ the per-weight noise power, which
is the compounding across 32 blocks doing exactly what I warned it would — the $5.44\times$ is the
single-layer noise, and re-feeding each block's degraded output into the next inflates it further into an
$8\times$ end-metric blow-up. So the diagnosis is confirmed in the metric: RTN is fine where the grid is
fat and falls apart where it is thin, and the thin-grid case is the one I most need to fix. The reason is
structural, the thing I flagged when I built the floor: RTN minimizes per-weight error,
$\lVert\mathbf W-\widehat{\mathbf W}\rVert^2$, element by element, when the quantity that actually
propagates to the next block is the layer output $\mathbf W\mathbf X$. It also throws away the calibration
data the scaffold is streaming through `add_batch` for free. Both of those are exactly what I now spend.

So let me state the honest objective RTN was approximating badly. For one linear layer with weights
$\mathbf W$ and calibration inputs stacked as columns of $\mathbf X$, I want quantized
$\widehat{\mathbf W}$ that keeps the layer's output close:
$\arg\min_{\widehat{\mathbf W}}\lVert\mathbf W\mathbf X-\widehat{\mathbf W}\mathbf X\rVert_2^2$. The grid is
fixed up front; each weight is free to land on any grid value. The first useful fact is that this
decomposes by row of $\mathbf W$: output channel $r$ contributes
$\lVert\mathbf w_r^\top\mathbf X-\widehat{\mathbf w}_r^\top\mathbf X\rVert^2$ and the rows do not interact,
because the output channels of a linear layer are independent linear functionals of the same input. So per
row it is a quadratic in the weight vector, and the curvature is identical for every row: differentiate
$\lVert\mathbf w^\top\mathbf X-\widehat{\mathbf w}^\top\mathbf X\rVert^2$ twice and the Hessian is
$\mathbf H=2\mathbf X\mathbf X^\top$, which depends only on the inputs, not on any weights. That the
curvature is weight-independent and shared across rows is the only structural fact I have, and I want to
see how far it carries. It is why the calibration stream matters: $\mathbf X\mathbf X^\top$ is precisely
the input second moment, and it tells me which input directions are excited hard by real text — the
directions where a weight error costs the output a lot — and which are nearly dead, where RTN was wasting
its worry. In the scaffold I accumulate exactly this in `add_batch`: reshape the layer input to
$(\text{tokens},\text{in\_features})$ and add $\mathbf X^\top\mathbf X$ into `self.H`. RTN kept this buffer
purely for interface compatibility; now it is the heart of the method.

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

I have pushed enough algebra that I do not trust it on faith, so let me put numbers through it on a
two-variable quadratic and watch what the update does. Take $\mathbf H=\begin{psmallmatrix}3&1.2\\1.2&2\end{psmallmatrix}$,
so $\det\mathbf H=6-1.44=4.56$ and
$\mathbf H^{-1}=\tfrac1{4.56}\begin{psmallmatrix}2&-1.2\\-1.2&3\end{psmallmatrix}\approx\begin{psmallmatrix}0.4386&-0.2632\\-0.2632&0.6579\end{psmallmatrix}$,
and a weight vector $\mathbf w=(0.83,-0.40)$ on a grid of spacing $0.5$. Fix coordinate $0$:
$\mathrm{quant}(0.83)=1.0$ — it rounds *up*, so $w_q-\mathrm{quant}(w_q)=-0.17$. The formula gives
$\lambda=(-0.17)/0.4386=-0.3876$ and $\boldsymbol\delta=-\lambda\mathbf H^{-1}_{:,0}=(0.170,-0.102)$, so
$\mathbf w_{\text{new}}=(1.000,-0.502)$. Three things I can check directly fall out. Coordinate $0$ lands
*exactly* on $1.000$ — the constraint is satisfied, which is the whole point: the $0$-th entry of
$\boldsymbol\delta$ has to be precisely $\mathrm{quant}(w_0)-w_0=0.17$, and it is. Coordinate $1$ moved by
$-0.102$, the same sign as the off-diagonal $\mathbf H^{-1}_{10}<0$ paired with a positive bump on
coordinate $0$ — the free weight absorbs the perturbation along the curvature, which is exactly what
"compensate optimally" should mean. And the predicted loss increase
$\tfrac12(0.17)^2/0.4386=0.5\cdot0.0289/0.4386=0.032946$; evaluating the quadratic directly,
$\mathbf H\boldsymbol\delta=(3\cdot0.17+1.2\cdot(-0.102),\,1.2\cdot0.17+2\cdot(-0.102))=(0.3876,\,0.000)$,
so $\tfrac12\boldsymbol\delta^\top\mathbf H\boldsymbol\delta=\tfrac12(0.17\cdot0.3876+(-0.102)\cdot0)=0.032946$.
They agree to every digit I printed. The zero in the second component of $\mathbf H\boldsymbol\delta$ is an
extra reassurance: after the update the gradient has no component along the free coordinate, which is the
optimality condition for that coordinate sitting at its constrained minimum. So the OBS closed form is
doing what I derived, not what I hoped.

One more thing bothers me before I build on this, and it is worth thirty seconds to settle because it
governs how I normalize $\mathbf H$. The scaffold accumulates $\mathbf X^\top\mathbf X$ raw, summed over
128 sequences of 2048 tokens; the diagonal magnitudes then depend on how many tokens flowed through, which
varies with the layer's position and could make a fixed dampening constant behave differently layer to
layer. If I normalize `H /= nsamples`, does that change the compensation? Look at the update:
$\boldsymbol\delta_F$ depends on $\mathbf H$ only through the ratio $\mathbf H^{-1}_{:,q}/[\mathbf H^{-1}]_{qq}$,
and $\mathrm{quant}(w_q)$ depends only on $\mathbf W$ and the grid, not on $\mathbf H$ at all. Scale
$\mathbf H\to c\mathbf H$ and then $\mathbf H^{-1}\to\mathbf H^{-1}/c$, so both numerator and denominator
of that ratio pick up a factor $1/c$ and it is unchanged; the whole sweep is invariant to a global scaling
of $\mathbf H$. So `H /= nsamples` moves no rounding decision — and neither does the factor-of-2 I dropped
from $2\mathbf X\mathbf X^\top$. What it *does* do is keep the dampening term sane: `damp = PERCDAMP *
mean(diag(H))` scales with $\mathbf H$ too, so `damp/H` is invariant, and normalizing merely keeps the
absolute magnitude of the diagonal in a stable range across layers rather than letting it drift with token
count. Good — the normalization is cosmetic for correctness and stabilizing for numerics, which is exactly
what I want it to be.

Now the cost, because the naive version of this explodes. The original brain-surgeon recipe picks, per
row, the cheapest-to-quantize free weight next — a greedy order chosen by minimizing
$(\mathrm{quant}(w_q)-w_q)^2/[\mathbf H^{-1}]_{qq}$ — and so maintains a *separate* evolving
$\mathbf H^{-1}$ per row. Each row does $d_{\text{col}}$ quantization steps, each step's rank-one downdate
touches all $O(d_{\text{col}}^2)$ entries of the current inverse, so it is $O(d_{\text{col}}^3)$ per row,
$O(d_{\text{row}}\cdot d_{\text{col}}^3)$ for the layer — cubic in the column count and multiplied by every
row. On a Mistral linear that is brutal: $\mathbf q\_proj$ is $4096\times4096$ so
$d_{\text{row}}\cdot d_{\text{col}}^3\approx4096\cdot4096^3\approx2.8\times10^{14}$, and the MLP
$\mathbf{gate}/\mathbf{up}$ are $14336\times4096$, worse. Hopeless per layer, times seven linears times 32
blocks. The rows are paying for the privilege of each choosing its own order; what does that privilege buy?
The diagnostic that rescues it: greedy ordering barely beats a fixed arbitrary order on large,
over-parameterized layers. The intuition is that greedy gains come from rounding the few high-error weights
early while many other weights are still free to absorb the compensation; but on a huge layer the count of
"saved" early weights is a vanishing fraction of the layer, and those greedily-deferred weights end up
quantized near the *end*, when almost no free weights remain to compensate, so the deferral costs about as
much as it saved. So I drop the greedy order and quantize *all rows in the same fixed left-to-right column
order*. Then $\mathbf H$ and $\mathbf H^{-1}$ depend only on $\mathbf X$, identical across rows, and I
downdate one shared inverse exactly $d_{\text{col}}$ times instead of per row. The cost drops to
$O(\max\{d_{\text{row}}\cdot d_{\text{col}}^2,\,d_{\text{col}}^3\})$ — for $4096\times4096$ that is
$4096^3\approx6.9\times10^{10}$, a factor of $\min\{d_{\text{row}},d_{\text{col}}\}=4096$ off the per-row
cost. That is the collapse that makes this run at 7B.

Concretely the per-column update becomes: round column $i$, form each row's scaled error
$\mathbf E_{:,i}=(\mathbf W_{:,i}-\mathrm{quant}(\mathbf W_{:,i}))/[\mathbf H^{-1}]_{ii}$, and push it into
the not-yet-quantized columns $j>i$ via $\mathbf W_{:,j}\mathrel{-}=\mathbf E_{:,i}[\mathbf H^{-1}]_{ij}$.
This is the OBS $\boldsymbol\delta$ written for the shared inverse and applied to every row at once — the
same closed form I just verified on the $2\times2$, now with $\mathbf E_{:,i}$ playing the per-row role of
$(w_q-\mathrm{quant}(w_q))/[\mathbf H^{-1}]_{qq}$ and row $i$ of the inverse spreading it. The algorithm is
now a column sweep.

Two practical problems remain, and here is where this task's fill makes a specific choice I should be
honest about. First, throughput. The per-column rank-one operation touches a huge matrix with a couple of
FLOPs per entry — a terrible compute-to-memory ratio, all bandwidth, tensor cores idle, repeated
$d_{\text{col}}$ times. The fix is blocking, and it is exact rather than an approximation: the rounding
decision for column $i$ depends only on updates from columns *before* $i$, so updates to far-right columns
can be deferred without changing any decision. I process a block of `BLOCK_SIZE=128` consecutive columns
using only the $128\times128$ corner of $\mathbf H^{-1}$ and the 128-wide weight slab, doing the
column-by-column compensation *within* the block; then, once the block is finished, I apply the block's
accumulated error to *all* columns to the right in one matrix-matrix multiply
$\mathbf W_{:,\text{rest}}\mathrel{-}=\mathbf E_{\text{block}}\,\mathbf H^{-1}_{\text{block},\text{rest}}$.
Same total arithmetic, but the heavy global update is a GEMM that saturates the hardware while the
bandwidth-bound fiddling is confined to the block. That is exactly the loop structure here: an outer loop
over `col_start` in steps of 128, an inner loop over the columns of the block computing per-row `err` and
updating `W_block[:, j+1:]` via `Hinv[col, col_start+j+1:col_end]`, and after the inner loop a single
`W[:, col_end:] -= Err[:, block] @ Hinv[block, col_end:]`.

Second, numerics, and this is where the implementation on this surface diverges from the most aggressive
form of the method, so I want to be precise and land the code that runs. I am consuming the inverse
Hessian, and at 7B scale some layers' $\mathbf H$ are near-singular — input directions that calibration
never excited leave $\mathbf H$ rank-deficient, and the inverse blows up, or $[\mathbf H^{-1}]_{ii}$ goes
tiny and the error $(w-q)/[\mathbf H^{-1}]_{ii}$ shoves the remaining weights off in a wrong direction,
poisoning every downstream block. Two defenses. A mild dampening: add $\lambda$ equal to `PERCDAMP=0.01`
of the mean diagonal of $\mathbf H$ before inverting,
$\mathbf H\mathrel{+}=0.01\cdot\overline{\operatorname{diag}\mathbf H}\cdot\mathbf I$, which keeps it
well-conditioned. Why $0.01$ of the mean diagonal specifically, and why that constant is a principled
regularizer rather than a fudge: the diagonal $\mathbf H_{ii}=\sum_t x_{t,i}^2$ is the energy calibration
text puts through input channel $i$, and it splits the channels into two populations. A well-excited
channel has $\mathbf H_{ii}$ of order the mean $m=\overline{\operatorname{diag}\mathbf H}$ or larger; a
channel real text never lights up has $\mathbf H_{ii}\approx0$, and it is those near-dead directions that
make $\mathbf H$ singular and its inverse explode. Adding $\lambda=0.01\,m$ to every diagonal lifts the
smallest eigenvalue to at least $\lambda$, so $\lVert\mathbf H^{-1}\rVert_2\le1/\lambda$ is bounded and no
entry of the inverse — neither the $[\mathbf H^{-1}]_{ii}$ I divide by nor the row $[\mathbf H^{-1}]_{i,\cdot}$
I propagate along — can run away. And the constant is *direction-selective* by construction: for an active
channel with $\mathbf H_{ii}=5m$ the damp changes it to $5.01m$, a $0.2\%$ perturbation its inverse barely
feels; for a dead channel with $\mathbf H_{ii}\approx0$ the damp is the entire diagonal, turning an
unbounded inverse entry into a bounded $\approx100/m$. So $0.01$ is small enough to be transparent to the
directions that carry the output and large enough to be decisive for the ones that would otherwise wreck
the sweep — a regularizer that touches only the columns it must. And a stable inverse: take a Cholesky
factorization of the dampened $\mathbf H$ and
invert through it — `L = cholesky(H); Hinv = cholesky_inverse(L)` — with a `pinv` fallback if even the
dampened matrix fails to factor. The most refined version of the method never forms $\mathbf H^{-1}$ at
all: it observes that the OBS downdate is one step of Gaussian elimination, so the scaled row-tails the
update needs live in the rows of a Cholesky factor of $\mathbf H^{-1}$, and it sweeps that factor, avoiding
any inversion. This task's fill takes the more direct route: it computes the dense inverse
$\mathbf H^{-1}$ once via Cholesky, then reads $[\mathbf H^{-1}]_{ii}$ for the per-column error and the
rows $[\mathbf H^{-1}]_{i,\cdot}$ for the propagation, dividing the error by $[\mathbf H^{-1}]_{ii}$ each
step. Mathematically this reproduces the same OBS compensation — error
$\propto(w-q)/[\mathbf H^{-1}]_{ii}$ spread by row $i$ of the inverse, precisely the $2\times2$ update I
checked — but it carries the full dense inverse and the within-block updates use its raw entries rather
than a Cholesky factor's scaled tails. So I should not import the "Cholesky-of-the-inverse stores the
row-tails" story into how I describe what runs here; what runs is "one stable inverse, read directly,
blocked." The single dampening plus single factorization is enough to keep Mistral's layers from drifting
into a wrecked layer the way an iterated rank-one downdate would — there is no iterated inverse to drift,
because I form it once.

One more piece this task exposes and I must get right: grouping. The derivation only assumed a fixed grid
and a $\mathrm{quant}(\cdot)$ that rounds onto it — it never assumed one scale per row — so I can use a
per-group scale, an independent step for every `group_size` consecutive columns, and the
error-compensation machinery is unchanged. Here, when the column sweep reaches a group boundary
(`col % group_size == 0`), I recompute the symmetric per-group scale from that group's columns of the
*current* weight matrix: $g_{\max}=\max|\mathbf W_{:,\text{group}}|$ per row, scale $=g_{\max}/q_{\max}$,
zero point 0. Because the sweep updates columns as it goes, each group's scale is fit to the weights *as
compensation has already moved them* — the group boundaries to the left have been rounded and their error
already pushed rightward into this group before I measure its max — so grouping and the second-order
correction reinforce each other rather than fighting. And the same code path serves group 128, group 64,
and per-channel (one scale per row), which is why the same class must run across all three eval settings.
Rounding a column is then
$\operatorname{clamp}(\operatorname{round}(w_{\text{col}}/\text{scale}),q_{\min},q_{\max})\cdot\text{scale}$,
and the quantized column is written into $\mathbf Q$, whose final cast back to the layer dtype is the
returned weight. The full scaffold module is in the answer.

So the delta from RTN: keep the same symmetric per-group grid, but accumulate $\mathbf X^\top\mathbf X$ in
`add_batch`, dampen and invert it once via Cholesky, and sweep columns left-to-right in blocks of 128,
rounding each column and compensating its residual onto the still-free columns through the inverse Hessian
so the layer *output* — not the per-weight value — is what is preserved. The only thing RTN and GPTQ share
is the grid; everything that decides *how* a weight rounds is now informed by the calibration inputs RTN
ignored.

Reading RTN's shape, here is what I expect, falsifiably, against its numbers. The whole point of error
compensation is to recover resolution where the grid is thinnest, so INT3-g128 is the setting I am
watching: RTN's 6.7341 (degradation 1.8270) is the blow-up, and if column-by-column compensation works at
all it should cut that degradation substantially — I expect INT3 to drop well under RTN's, into the low-6
range, because the deferred error now has somewhere to go instead of accumulating. But "somewhere to go" is
not "nowhere left to lose," and I should be honest about the floor compensation cannot cross, because it
sets how much of INT3 I expect to remain. The residual of the *last* column in each sweep has no free
columns to its right to absorb it — it is pure RTN error, uncompensated by construction — and every
compensating move pushes the still-free weights further from their FP values onto a grid that will itself
re-round them, injecting fresh residual whenever the optimal compensation lands off a grid point. So the
deferred error has a destination, but the destination is quantized too; compensation recycles error into
progressively harder-to-place weights rather than dissolving it. At 16 levels that recycling is cheap —
there is slack in the grid — which is why I expect INT4 to improve only modestly. At 8 levels the grid has
almost no slack, so I expect compensation to soften INT3 substantially yet leave it *still* well above
INT4, the ratio between them barely moved. If that is what the table shows — a real INT3 cut that
nonetheless stays an order of magnitude over INT4 — it would say compensation attacks the accumulation but
not the underlying coarseness, and the next rung has to attack the grid itself. At INT4-g128 the grid is
already fat enough that RTN was only at 0.2271 degradation, so I expect a real but smaller improvement —
into the low-5.0s, comfortably under RTN's 5.1343. At INT4-g64 RTN was already strong (5.0890);
compensation should still help, but this is the setting where I expect the *smallest* gap, because fine
grouping has already done much of the work error compensation would otherwise do — the two mechanisms
partly overlap, both of them protecting the bulk against outlier-inflated steps. The cost I accept: where
RTN quantized a block in ~22s, GPTQ pays for an inverse and a column sweep per layer — the inverse is a
$4096^3$-scale Cholesky and the sweep is inherently sequential across $d_{\text{col}}/128$ blocks — so I
expect quant-time to rise by roughly an order of magnitude, into the low hundreds of seconds. The bet is
that the INT3 recovery is large enough to justify it — and if INT3 does *not* move much, that would tell me
the bottleneck at 8 levels is the grid coarseness itself, not the rounding objective, which would push the
next rung toward protecting weights *before* rounding rather than compensating *after*.
