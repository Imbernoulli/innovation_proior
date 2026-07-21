RTN's numbers came back and they point straight at its successor. At INT4-g128 it lands 5.1343 — a
degradation of 0.2271 over the 4.9071 FP16 reference, survivable but visible. At INT4-g64 it improves to
5.0890 (degradation 0.1819), exactly the direction I predicted: a smaller group means the scale is set by
the local max over 64 columns instead of 128, fewer typical weights polluted per outlier, so finer
granularity buys a small recovery — the gap is only $0.045$ of perplexity, confirming that at INT4 the
level count, not the group extent, is the bottleneck. The standard INT4 setting is not where RTN breaks.
INT3-g128 is: there it jumps to 6.7341, degradation 1.8270, and the ratio tells the story —
$1.8270/0.2271\approx8$, so the INT3 hit is eight times the INT4 hit at the same group size. That is even
worse than my $\Delta^2/12$ estimate of $\approx5.44\times$ the per-weight noise power, which is the
compounding across 32 blocks doing exactly what I warned it would: the $5.44\times$ is the single-layer
noise, and re-feeding each block's degraded output into the next inflates it into an $8\times$ end-metric
blow-up. So the diagnosis is confirmed — RTN is fine where the grid is fat and falls apart where it is thin.
The reason is structural, the thing I flagged when I built the floor: RTN minimizes per-weight error,
$\lVert\mathbf W-\widehat{\mathbf W}\rVert^2$, element by element, when the quantity that propagates to the
next block is the layer output $\mathbf W\mathbf X$, and it throws away the calibration data the harness
is streaming through `add_batch`. Both of those are what I now spend.

So state the honest objective RTN was approximating badly. For one linear layer with weights $\mathbf W$
and calibration inputs stacked as columns of $\mathbf X$, I want quantized $\widehat{\mathbf W}$ that keeps
the layer's output close: $\arg\min_{\widehat{\mathbf W}}\lVert\mathbf W\mathbf X-\widehat{\mathbf W}\mathbf X\rVert_2^2$,
grid fixed up front, each weight free to land on any grid value. This decomposes by row of $\mathbf W$:
output channel $r$ contributes $\lVert\mathbf w_r^\top\mathbf X-\widehat{\mathbf w}_r^\top\mathbf X\rVert^2$
and the rows do not interact, because the output channels are independent linear functionals of the same
input. So per row it is a quadratic in the weight vector, and the curvature is identical for every row:
differentiate twice and the Hessian is $\mathbf H=2\mathbf X\mathbf X^\top$, which depends only on the
inputs, not on any weights. That the curvature is weight-independent and shared across rows is the
structural fact I build on. It is why the calibration stream matters: $\mathbf X\mathbf X^\top$ is precisely
the input second moment, telling me which input directions real text excites hard — where a weight error
costs the output a lot — and which are nearly dead, where RTN was wasting its worry. I accumulate exactly
this in `add_batch`: reshape the layer input to $(\text{tokens},\text{in\_features})$ and add
$\mathbf X^\top\mathbf X$ into `self.H`, the buffer RTN kept only for interface compatibility.

The accurate small-model idea is the brain-surgeon logic specialized to quantization. Given the row
quadratic, fix one coordinate $q$ to its chosen grid value and move all the *other* still-free coordinates
to compensate optimally. Let $\boldsymbol\delta=\widehat{\mathbf w}-\mathbf w$ and let the quantized target
be $\mathrm{quant}(w_q)$. The row's quadratic increase is
$\tfrac12\boldsymbol\delta^\top\mathbf H\boldsymbol\delta$ subject to
$\mathbf e_q^\top\boldsymbol\delta=\mathrm{quant}(w_q)-w_q$. The Lagrangian gives
$\mathbf H\boldsymbol\delta+\lambda\mathbf e_q=0$, so $\boldsymbol\delta=-\lambda\mathbf H^{-1}_{:,q}$;
enforcing the constraint gives $\lambda=(w_q-\mathrm{quant}(w_q))/[\mathbf H^{-1}]_{qq}$, and the optimal
compensating update to the free set is
$\boldsymbol\delta_F=-\tfrac{w_q-\mathrm{quant}(w_q)}{[\mathbf H^{-1}]_{qq}}(\mathbf H^{-1})_{:,q}$, with a
loss increase $\tfrac12(\mathrm{quant}(w_q)-w_q)^2/[\mathbf H^{-1}]_{qq}$. The $q$-th component of
$\boldsymbol\delta$ is exactly $-(w_q-\mathrm{quant}(w_q))$ — it snaps $w_q$ onto the grid — and the other
entries spread the compensation along the input correlations in $\mathbf H^{-1}$. This is the move RTN never
makes: it lets a quantized weight drift the *remaining* weights to keep the output right, rather than
rounding each one in isolation.

One thing governs how I normalize $\mathbf H$. The harness accumulates $\mathbf X^\top\mathbf X$ raw,
summed over 128 sequences of 2048 tokens, so the diagonal magnitudes depend on how many tokens flowed
through, which varies with layer position. But $\boldsymbol\delta_F$ depends on $\mathbf H$ only through the
ratio $\mathbf H^{-1}_{:,q}/[\mathbf H^{-1}]_{qq}$, and $\mathrm{quant}(w_q)$ does not depend on $\mathbf H$
at all: under $\mathbf H\to c\mathbf H$ both numerator and denominator pick up $1/c$ and the ratio is
unchanged. So a global rescale of $\mathbf H$ moves no rounding decision — `H /= nsamples`, and the
factor-of-2 I dropped from $2\mathbf X\mathbf X^\top$, are both free. What normalizing *does* buy is keeping
the diagonal in a stable absolute range across layers, so the dampening `damp = PERCDAMP * mean(diag(H))`
(which also scales with $\mathbf H$, leaving `damp/H` invariant) stays numerically sane instead of drifting
with token count.

Now the cost, because the naive version explodes. The original brain-surgeon recipe picks, per row, the
cheapest-to-quantize free weight next — a greedy order minimizing
$(\mathrm{quant}(w_q)-w_q)^2/[\mathbf H^{-1}]_{qq}$ — and so maintains a *separate* evolving
$\mathbf H^{-1}$ per row. Each row does $d_{\text{col}}$ steps, each step's rank-one downdate touches all
$O(d_{\text{col}}^2)$ entries of the inverse, so it is $O(d_{\text{row}}\cdot d_{\text{col}}^3)$ for the
layer. On $\mathbf q\_proj$ at $4096\times4096$ that is $\approx2.8\times10^{14}$, and the MLP
$\mathbf{gate}/\mathbf{up}$ at $14336\times4096$ are worse — hopeless per layer, times seven linears times
32 blocks. The rows are paying for the privilege of each choosing its own order; what does that privilege
buy? Greedy ordering barely beats a fixed arbitrary order on large, over-parameterized layers: greedy gains
come from rounding the few high-error weights early while many others are still free to absorb the
compensation, but on a huge layer the "saved" early weights are a vanishing fraction, and the
greedily-deferred ones end up quantized near the end when almost no free weights remain, so the deferral
costs about as much as it saved. So I drop the greedy order and quantize *all rows in the same fixed
left-to-right column order*. Then $\mathbf H$ and $\mathbf H^{-1}$ depend only on $\mathbf X$, identical
across rows, and I downdate one shared inverse exactly $d_{\text{col}}$ times instead of per row. The cost
drops to $O(\max\{d_{\text{row}}\cdot d_{\text{col}}^2,\,d_{\text{col}}^3\})$ — for $4096\times4096$,
$\approx6.9\times10^{10}$, a factor of $\min\{d_{\text{row}},d_{\text{col}}\}=4096$ off. That is the
collapse that makes this run at 7B.

Concretely the per-column update becomes: round column $i$, form each row's scaled error
$\mathbf E_{:,i}=(\mathbf W_{:,i}-\mathrm{quant}(\mathbf W_{:,i}))/[\mathbf H^{-1}]_{ii}$, and push it into
the not-yet-quantized columns $j>i$ via $\mathbf W_{:,j}\mathrel{-}=\mathbf E_{:,i}[\mathbf H^{-1}]_{ij}$ —
the OBS $\boldsymbol\delta$ written for the shared inverse and applied to every row at once, with
$\mathbf E_{:,i}$ playing the per-row role of $(w_q-\mathrm{quant})/[\mathbf H^{-1}]_{qq}$ and row $i$ of
the inverse spreading it. The algorithm is now a column sweep.

Two practical problems remain. First, throughput. The per-column rank-one operation touches a huge matrix
with a couple of FLOPs per entry — all bandwidth, tensor cores idle, repeated $d_{\text{col}}$ times. The
fix is blocking, and it is exact rather than an approximation: the rounding decision for column $i$ depends
only on updates from columns *before* $i$, so updates to far-right columns can be deferred without changing
any decision. I process a block of `BLOCK_SIZE=128` consecutive columns using only the $128\times128$
corner of $\mathbf H^{-1}$ and the 128-wide weight slab, doing the column-by-column compensation *within*
the block; then, once the block is finished, I apply its accumulated error to *all* columns to the right in
one matrix-matrix multiply
$\mathbf W_{:,\text{rest}}\mathrel{-}=\mathbf E_{\text{block}}\,\mathbf H^{-1}_{\text{block},\text{rest}}$.
Same total arithmetic, but the heavy global update is a GEMM that saturates the hardware while the
bandwidth-bound fiddling is confined to the block — an outer loop over `col_start` in steps of 128, an
inner loop over the block's columns computing per-row `err` and updating `W_block[:, j+1:]`, and after the
inner loop a single `W[:, col_end:] -= Err[:, block] @ Hinv[block, col_end:]`.

Second, numerics — and here I have to choose carefully, because I am consuming the inverse Hessian and at
7B scale some layers' $\mathbf H$ are near-singular. Input directions calibration never excited leave
$\mathbf H$ rank-deficient, and the inverse blows up, or $[\mathbf H^{-1}]_{ii}$ goes tiny and the error
$(w-q)/[\mathbf H^{-1}]_{ii}$ shoves the remaining weights off in a wrong direction, poisoning every
downstream block. Two defenses. A mild dampening: add $\lambda$ equal to `PERCDAMP=0.01` of the mean
diagonal of $\mathbf H$ before inverting,
$\mathbf H\mathrel{+}=0.01\cdot\overline{\operatorname{diag}\mathbf H}\cdot\mathbf I$. The diagonal
$\mathbf H_{ii}=\sum_t x_{t,i}^2$ is the energy calibration text puts through input channel $i$, and it
splits the channels into two populations: a well-excited channel has $\mathbf H_{ii}$ of order the mean
$m=\overline{\operatorname{diag}\mathbf H}$ or larger, while a channel real text never lights up has
$\mathbf H_{ii}\approx0$, and those near-dead directions are what make $\mathbf H$ singular. Adding
$\lambda=0.01\,m$ lifts the smallest eigenvalue to at least $\lambda$, so $\lVert\mathbf H^{-1}\rVert_2\le1/\lambda$
is bounded. And the constant is *direction-selective*: for an active channel with $\mathbf H_{ii}=5m$ the
damp changes it to $5.01m$, a $0.2\%$ perturbation its inverse barely feels; for a dead channel with
$\mathbf H_{ii}\approx0$ the damp is the entire diagonal, turning an unbounded inverse entry into a bounded
$\approx100/m$. So $0.01$ is small enough to be transparent to the directions that carry the output, large
enough to be decisive for the ones that would otherwise wreck the sweep. The second defense is a stable
inverse: a Cholesky factorization of the dampened $\mathbf H$, inverted through it — `L = cholesky(H); Hinv
= cholesky_inverse(L)` — with a `pinv` fallback if even the dampened matrix fails to factor. I could avoid
forming $\mathbf H^{-1}$ at all — the OBS downdate is one step of Gaussian elimination, so the scaled
row-tails it needs live in the rows of a Cholesky factor and could be swept directly — but forming the
dense inverse once and reading $[\mathbf H^{-1}]_{ii}$ for the per-column error and rows
$[\mathbf H^{-1}]_{i,\cdot}$ for the propagation is simpler and, with the dampening, stable enough: there is
no iterated inverse to drift the way an iterated rank-one downdate would.

One more piece: grouping. The derivation only assumed a fixed grid and a $\mathrm{quant}(\cdot)$ that rounds
onto it — never one scale per row — so I can use a per-group scale, an independent step for every
`group_size` consecutive columns, and the error-compensation machinery is unchanged. When the column sweep
reaches a group boundary (`col % group_size == 0`), I recompute the symmetric per-group scale from that
group's columns of the *current* weight matrix: $g_{\max}=\max|\mathbf W_{:,\text{group}}|$ per row, scale
$=g_{\max}/q_{\max}$, zero point 0. Because the sweep updates columns as it goes, each group's scale is fit
to the weights *as compensation has already moved them* — the boundaries to the left have been rounded and
their error already pushed rightward into this group before I measure its max — so grouping and the
second-order correction reinforce rather than fight. The same code path serves group 128, group 64, and
per-channel, which is why one class must run across all three settings. Rounding a column is
$\operatorname{clamp}(\operatorname{round}(w_{\text{col}}/\text{scale}),q_{\min},q_{\max})\cdot\text{scale}$,
written into $\mathbf Q$, whose final cast back to the layer dtype is the returned weight. The full module
is in the answer.

Against RTN's numbers, here is what I expect. The point of error compensation is to recover resolution where
the grid is thinnest, so INT3-g128 is the setting I am watching: RTN's 6.7341 (degradation 1.8270) is the
blow-up, and if column-by-column compensation works at all it should cut that substantially, into the low-6
range, because the deferred error now has somewhere to go instead of accumulating. But the destination is
itself quantized, which sets a floor compensation cannot cross: the last column in each sweep has no free
columns to its right to absorb its residual — pure RTN error, uncompensated — and every compensating move
pushes still-free weights onto a grid that re-rounds them, injecting fresh residual whenever the optimal
compensation lands off a grid point. At 16 levels that recycling is cheap, there is slack in the grid, so I
expect INT4 to improve only modestly — into the low-5.0s at INT4-g128, comfortably under RTN's 5.1343, and
the *smallest* gap at INT4-g64 (5.0890), where fine grouping has already done much of the work
compensation would otherwise do. At 8 levels the grid has almost no slack, so compensation should soften
INT3 substantially yet leave it *still* well above INT4 — a real cut whose INT3-to-INT4 ratio has barely
moved would say compensation attacks the accumulation but not the underlying coarseness. The cost I accept:
where RTN quantized a block in ~22s, GPTQ pays for a $4096^3$-scale Cholesky and a sweep that is inherently
sequential across $d_{\text{col}}/128$ blocks, so I expect quant-time to rise by roughly an order of
magnitude, into the low hundreds of seconds. The bet is that the INT3 recovery justifies it — and if INT3
does *not* move much, that would say the bottleneck at 8 levels is the grid coarseness itself, not the
rounding objective.
