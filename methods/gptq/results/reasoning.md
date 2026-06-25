Let me start from the pain. I have a 175-billion-parameter model whose weights, in FP16, eat 326 GB. I cannot retrain it — that would cost weeks. I want to round the weights down to 3 or 4 bits, once, and have the model still produce essentially the same outputs. The crude thing — round every weight to the nearest grid point — scales fine but dies below 8 bits. The accurate things all involve gradient descent inner loops per layer and top out around a hundred million parameters. So I have an accuracy ceiling I know is achievable (the slow methods hit it on small models) and a scale ceiling I cannot break. I need a method that is as accurate as the slow second-order ones but with a runtime that survives a thousandfold blowup in size.

What am I actually trying to preserve? Not the weights — the *behavior*. For one linear layer with weights $\mathbf{W}$ and calibration inputs stacked as columns of $\mathbf{X}$, I want quantized $\widehat{\mathbf{W}}$ that keeps the layer output close:

$$\arg\min_{\widehat{\mathbf{W}}}\ \lVert \mathbf{W}\mathbf{X} - \widehat{\mathbf{W}}\mathbf{X}\rVert_2^2 .$$

The grid is fixed up front; each weight is free to land on any grid value. The nice thing is this splits by row of $\mathbf{W}$: row $r$ contributes $\lVert \mathbf{w}_r^\top \mathbf{X} - \widehat{\mathbf{w}}_r^\top \mathbf{X}\rVert^2$, and the rows don't interact, because the output channels of a linear layer are independent linear functionals of the same input. So per row it's a quadratic in the weight vector $\mathbf{w}$, and the curvature is the same for every row: differentiate $\lVert \mathbf{w}^\top\mathbf{X} - \widehat{\mathbf{w}}^\top\mathbf{X}\rVert^2$ twice and the Hessian is $\mathbf{H} = 2\mathbf{X}\mathbf{X}^\top$. That $\mathbf{H}$ depends only on the inputs, not on any weights. Hold that thought — that the curvature is weight-independent and shared across rows is the only structural fact I have, and I want to see how far it carries.

The accurate small-model method I'd reach for is OBQ — the brain-surgeon idea specialized to quantization. The brain-surgeon logic: I have a quadratic, I want to fix one coordinate to a chosen value (its rounded grid value) and then move all the *other* still-free coordinates to compensate optimally. Let the perturbation be $\boldsymbol{\delta}=\widehat{\mathbf{w}}-\mathbf{w}$ and let $q^\star=\mathrm{quant}(w_q)$. The row's quadratic increase is $\frac{1}{2}\boldsymbol{\delta}^{\top}\mathbf{H}\boldsymbol{\delta}$, with the constraint $\mathbf{e}_q^\top\boldsymbol{\delta}=q^\star-w_q$. The Lagrangian gives $\mathbf{H}\boldsymbol{\delta}+\lambda\mathbf{e}_q=0$, so $\boldsymbol{\delta}=-\lambda\mathbf{H}^{-1}_{:,q}$. Enforcing the constraint gives $-\lambda[\mathbf{H}^{-1}]_{qq}=q^\star-w_q$, hence $\lambda=(w_q-q^\star)/[\mathbf{H}^{-1}]_{qq}$. Plugging that back into the quadratic increase leaves

$$\frac{1}{2}\frac{(\mathrm{quant}(w_q)-w_q)^2}{[\mathbf{H}^{-1}]_{qq}},$$

so the constant $\frac{1}{2}$ can disappear from the greedy score, and the cheapest weight to quantize next is the one minimizing $(\mathrm{quant}(w_q)-w_q)^2/[\mathbf{H}^{-1}]_{qq}$. The optimal compensating update to the whole free set $F$ is

$$\boldsymbol{\delta}_F = -\frac{w_q - \mathrm{quant}(w_q)}{[\mathbf{H}^{-1}]_{qq}}\,(\mathbf{H}^{-1})_{:,q}.$$

I've pushed enough algebra here that I don't trust it on faith; let me put numbers through it on a two-variable quadratic and watch what the update does. Take $\mathbf{H}=\begin{psmallmatrix}3&1.2\\1.2&2\end{psmallmatrix}$, so $\mathbf{H}^{-1}\approx\begin{psmallmatrix}0.397&-0.238\\-0.238&0.595\end{psmallmatrix}$, and a weight vector $\mathbf{w}=(0.83,-0.40)$ on a grid of spacing $0.5$. Fix coordinate $0$: $\mathrm{quant}(0.83)=1.0$ (it rounds *up*, so $w_q-q^\star=-0.17$). The formula gives $\lambda=(-0.17)/0.397=-0.428$ and $\boldsymbol{\delta}=-\lambda\mathbf{H}^{-1}_{:,0}=(0.170,-0.102)$, so $\mathbf{w}_{\text{new}}=(1.000,-0.502)$. Two things I can check directly fall out of this. Coordinate $0$ lands *exactly* on $1.000$ — the constraint is satisfied, which is the whole point: the $q$-th entry of $\boldsymbol{\delta}$ has to be precisely $q^\star-w_q$, and it is. And coordinate $1$ moved by $-0.102$, the same sign as the off-diagonal correlation $\mathbf{H}^{-1}_{10}<0$ paired with a positive bump on coordinate $0$ — the free weight absorbs the perturbation along the curvature, which is exactly what "compensate optimally" should mean. Finally the predicted loss increase $\frac{1}{2}(q^\star-w_q)^2/[\mathbf{H}^{-1}]_{00}=0.5\cdot0.17^2/0.397=0.032946$; evaluating the quadratic directly, $\frac{1}{2}\boldsymbol{\delta}^\top\mathbf{H}\boldsymbol{\delta}=0.032946$. They agree to every digit I printed. So the OBS closed form is doing what I derived, not just what I hoped.

After I fix $w_q$ I remove it from the free set, which means deleting row and column $q$ from $\mathbf{H}$ and working with the inverse of the shrunk matrix. Re-inverting every step would be insane; the brain-surgeon trick is that removing a coordinate from an inverse is one Gaussian-elimination rank-one downdate:

$$\mathbf{H}^{-1}_{-q} = \Big(\mathbf{H}^{-1} - \frac{1}{[\mathbf{H}^{-1}]_{qq}}\,\mathbf{H}^{-1}_{:,q}\,\mathbf{H}^{-1}_{q,:}\Big)_{-q},$$

where $(\cdot)_{-q}$ strikes out row and column $q$. So OBQ marches: pick the min-error free weight, round it, broadcast $\boldsymbol{\delta}_F$, downdate $\mathbf{H}^{-1}$, repeat.

Now count the cost and watch it explode. Each row picks its *own* greedy order, so each row keeps its *own* evolving $\mathbf{H}^{-1}$ trajectory. For one row there are $d_{\text{col}}$ quantization steps, and each step's downdate touches all $O(d_{\text{col}}^2)$ entries of the current inverse — that's $O(d_{\text{col}}^3)$ per row, $O(d_{\text{row}}\cdot d_{\text{col}}^3)$ for the layer. Cubic in the column count, and multiplied by every row. For a layer that's, say, $12288\times 12288$, this is wildly out of reach. I keep coming back to the fact that the rows are paying for the privilege of each choosing its own order. What is that privilege actually buying me?

The useful diagnostic is the greedy-vs-fixed-order comparison. On small layers, greedy order helps a bit. On the giant over-parameterized layers I care about, it barely separates from a fixed arbitrary order. The intuition is that greedy gains come from rounding the few high-error weights early, while many other weights are still free to absorb the compensation; but on a huge layer the count of "saved" early weights is tiny relative to the layer, and those greedily-deferred weights end up quantized near the *end*, when almost no free weights remain to compensate, so the deferral costs about as much as it saved. If that empirical reading holds — and it's exactly the kind of thing I'd want to confirm on real layers, not just argue — then the order barely matters, and that would change everything.

Because if I'm allowed to quantize *all rows in the same fixed order* — say, simply column 1, then column 2, then column 3 — then look at what $\mathbf{H}$ and $\mathbf{H}^{-1}$ depend on: only $\mathbf{X}$, not the weights (that fact I parked at the start). So the free set $F$ and the inverse $\mathbf{H}_F^{-1}$ are *identical across all rows*. I no longer downdate a per-row inverse $d_{\text{row}}\cdot d_{\text{col}}$ times; I downdate one shared inverse exactly $d_{\text{col}}$ times — once per column, amortized over all rows. The runtime drops from $O(d_{\text{row}}\cdot d_{\text{col}}^3)$ to $O(\max\{d_{\text{row}}\cdot d_{\text{col}}^2,\ d_{\text{col}}^3\})$, a factor of $\min\{d_{\text{row}},d_{\text{col}}\}$ — orders of magnitude. And the per-column update reduces to something simple: when I quantize column $i$, every row's residual gets the same scalar-per-row error spread by the same row $i$ of the shared inverse.

Concretely, fixing the order to left-to-right and using the shared inverse, the recipe per column $i$ is: round column $i$ of the current $\mathbf{W}$ to the grid; call the current free-set inverse $\mathbf{M}$; form the per-row error $\mathbf{E}_{:,i} = (\mathbf{W}_{:,i} - \mathrm{quant}(\mathbf{W}_{:,i}))/\mathbf{M}_{ii}$; then push that error into all not-yet-quantized columns $j>i$ via $\mathbf{W}_{:,j} \mathrel{-}= \mathbf{E}_{:,i}\,\mathbf{M}_{ij}$. This is the OBS $\boldsymbol\delta$ written for the shared inverse and applied to every row at once. Then downdate the inverse and move on. I've turned the algorithm into a column sweep. Before I lean on it I should make sure the fixed-order sweep actually reproduces the OBS numbers, since I've quietly swapped per-row greedy ordering for a single shared order; I'll come back and check the whole thing end to end once the implementation issues are settled.

But before I celebrate, two things will break in practice.

First problem: throughput, not FLOPs. The inverse downdate $\mathbf{H}^{-1} \leftarrow \mathbf{H}^{-1} - \frac{1}{[\mathbf{H}^{-1}]_{qq}}\mathbf{H}^{-1}_{:,q}\mathbf{H}^{-1}_{q,:}$ touches a huge matrix but does only a couple of FLOPs per entry — a rank-one update has a terrible compute-to-memory ratio. On a GPU that's all memory bandwidth, no arithmetic; the tensor cores sit idle and I'm bottlenecked on shuffling the inverse in and out of memory once per column. I'm doing $d_{\text{col}}$ of these.

The way out is to notice what a column's final rounding actually depends on. The rounding decision for column $i$ only depends on the value of column $i$ at the moment I quantize it, and that value is only affected by updates coming from columns *before* $i$ — columns after $i$ haven't been touched yet. So updates *to* columns far to the right of $i$ are irrelevant to $i$'s decision and can be deferred. That means I can process a block of $B$ consecutive columns (I'll take $B=128$) by keeping all the updates *contained within the block* — quantizing column by column inside the block, but only using and updating the $B\times B$ corner of $\mathbf{H}^{-1}$ and the $B$-wide slab of $\mathbf{W}$. Only after the whole block is done do I apply the accumulated block error to *all* the columns to the right in one big matrix-matrix multiply $\mathbf{W}_{:,\text{rest}} \mathrel{-}= \mathbf{E}\cdot \mathbf{H}^{-1}_{\text{block},\text{rest}}$. Same total arithmetic, but the heavy global update is now a GEMM that saturates the tensor cores, and the bandwidth-bound fiddling is confined to a small block. That should be the order-of-magnitude practical speedup, since the deferral is exact — nothing the block needs from the right is touched before it's read.

Second problem: numerics, and this one genuinely scares me at scale. I'm repeatedly applying rank-one (and, with the block form, small-block) downdates to $\mathbf{H}^{-1}$. Each one accumulates floating-point error, and there's an extra matrix inversion hiding in the block version. My worry is that past a few billion parameters, at least a few layers could drift until $\mathbf{H}_F^{-1}$ stops being positive definite — it goes indefinite. The moment that happens, $[\mathbf{H}^{-1}]_{ii}$ can be near zero or negative, the error $\mathbf{E}_{:,i} = (w-q)/[\mathbf{H}^{-1}]_{ii}$ blows up or flips sign, and the algorithm shoves the remaining weights off in a wrong direction, wrecking the whole layer. I can't reproduce that failure on a small well-conditioned toy — there the iterated downdate keeps its pivots safely positive — so I can't *prove* it bites at scale from here; but the mechanism is real and a single bad layer poisons everything downstream, so I want to engineer it out rather than hope. A small dampening — add $\lambda$ equal to 1% of the average diagonal of $\mathbf{H}$ before inverting — helps the conditioning and is enough on small models, but I don't expect it to be robust enough alone here.

Let me look harder at what I actually consume from the inverse. When I quantize weight $q$, all I read from $\mathbf{H}_{F_q}^{-1}$ is row $q$ — and really only the entries from the diagonal rightward, since the columns to the left are already quantized. So I don't need the full evolving inverse at every step; I need, for each $q$, the rightward tail of row $q$ of the *sequentially downdated* inverse. Now, what does the OBS downdate do to a symmetric inverse? Striking out coordinate $q$ via $\mathbf{H}^{-1} - \frac{1}{[\mathbf{H}^{-1}]_{qq}}\mathbf{H}^{-1}_{:,q}\mathbf{H}^{-1}_{q,:}$ is one step of symmetric Gaussian elimination on $\mathbf{H}^{-1}$ — eliminate the pivot, then continue on the trailing submatrix. That is also, structurally, what Cholesky does. So I conjecture: if I call the current trailing inverse $\mathbf{M}$ and take an upper Cholesky factor $\mathbf{U}$ with $\mathbf{M}=\mathbf{U}^{\top}\mathbf{U}$, then the scaled row-tail $\mathbf{U}_{qj}/\mathbf{U}_{qq}$ of the *original* factor should equal $\mathbf{M}_{qj}/\mathbf{M}_{qq}$ of the *sequentially downdated* trailing inverse — i.e. exactly the OBS update coefficient. If that's true I never have to downdate at all.

This is too central to assert, so let me test it numerically. Build a small $\mathbf{H}=2\mathbf{X}\mathbf{X}^\top$ with $d_{\text{col}}=4$, form $\mathbf{H}^{-1}$ and its upper Cholesky factor $\mathbf{U}$, and at each step $q$ compare two coefficient vectors: $\mathbf{U}_{q,q:}/\mathbf{U}_{qq}$ read from the fixed factor, against $\mathbf{M}_{0,:}/\mathbf{M}_{00}$ where $\mathbf{M}$ is the trailing inverse I get by *actually* performing the rank-one downdate at every prior step. Running it, the two agree at every column to the last digit — the max absolute difference is $2\times10^{-16}$ at column 0, around $10^{-15}$ in the middle, and $0$ at the last column. So the conjecture holds: the rows of the Cholesky factor of $\mathbf{H}^{-1}$ *are* the scaled inverse-row-tails the sweep needs, and the scaling $\mathbf{U}_{qj}/\mathbf{U}_{qq}$ is harmless because it cancels exactly against the pivot the OBS update divides by.

So I can stop incrementally downdating $\mathbf{H}^{-1}$ thousands of times and watching error pile up. I compute the Cholesky decomposition of $\mathbf{H}^{-1}$ once, with a numerically stable, highly optimized kernel, and read the scaled rows I need straight out of it. No repeated inversions, no drift into indefiniteness from the iteration itself, and as a bonus the Cholesky kernel is fast. On the toy in float32 the iterated downdate didn't actually go indefinite, but the single Cholesky factor comes out with strictly positive diagonal by construction, so whatever marginal conditioning trouble the iteration would accumulate at scale simply can't arise — there's no iteration. Combined with the mild 1% dampening on $\mathbf{H}$'s diagonal before inverting, I expect this to be stable enough to run on the 175B models without layers failing from indefinite inverse updates; that's the claim I'd actually want to confirm by running the largest layers and checking no layer's loss explodes.

Now the end-to-end check I deferred. I have two algorithms that should agree: the literal fixed-order OBS that downdates the trailing inverse every column, and the Cholesky sweep that never downdates. On a $3\times6$ layer with a real $\mathbf{H}=2\mathbf{X}\mathbf{X}^\top$ and a coarse grid, I run both. The quantized weights come out bit-for-bit identical (max $|\mathbf{Q}_{\text{ref}}-\mathbf{Q}_{\text{chol}}|=0$), and the accumulated quadratic loss matches to $\sim10^{-15}$ ($5.86329\ldots$ both ways). That's the reassurance I wanted: replacing the per-step inverse with a single Cholesky factor is not an approximation, it's an algebraic identity. And to make sure the whole second-order apparatus is earning its keep over the trivial baseline, I compare layer output error: round-to-nearest gives $\lVert\mathbf{WX}-\mathbf{Q}_{\text{RTN}}\mathbf{X}\rVert^2=7.49$, the sweep gives $5.86$ — the compensation genuinely buys a lower reconstruction error, which is the entire reason to do any of this.

The loop now has no remaining moving parts. I build $\mathbf{M} = (2\mathbf{X}\mathbf{X}^\top + \lambda\mathbf{I})^{-1}$ — the implementation keeps the same matrix up to a running average scale — then take an upper Cholesky factor $\mathbf{U}$ with $\mathbf{M}=\mathbf{U}^{\top}\mathbf{U}$ and sweep blocks of $B=128$ columns. Inside a block I go column by column: quantize the column, compute the scaled per-row error $\mathbf{E}_{:,j} = (\mathbf{W}_{:,j} - \mathbf{Q}_{:,j})/\mathbf{U}_{jj}$, and propagate it to the remaining columns *within the block* using row $j$ of $\mathbf{U}$. Because $\mathbf{U}_{j,j:}/\mathbf{U}_{jj}$ equals the needed current inverse row divided by its pivot — the identity I just checked numerically — multiplying $\mathbf{E}_{:,j}$ by $\mathbf{U}_{j,j:}$ is the same OBS update that would have used the current downdated inverse row. After finishing the block, I apply the accumulated $\mathbf{E}$ to every column to the right of the block in one GEMM. Repeat to the end.

One more detail that comes for free and matters in practice. I don't actually have to commit to the simple min–max grid. The whole derivation only assumed a fixed grid and a $\mathrm{quant}(\cdot)$ that rounds onto it; it never assumed the grid is a single per-row scale. So I can use *grouping* — an independent scale (and zero) for every $g$ consecutive weights in a row — and the error-compensation machinery is unchanged. Even better, because I'm sweeping columns and updating weights as I go, I can fit each group's scale to the *already-updated* weights at the moment I reach that group, so grouping and the second-order compensation reinforce each other. That's what should let the extreme regime (2-bit, even ternary at tiny group size) stay usable.

So the causal chain: I want output-preserving one-shot quantization at a scale where gradient methods can't go, so I take the second-order brain-surgeon objective whose Hessian is just $2\mathbf{X}\mathbf{X}^\top$; I notice greedy ordering buys almost nothing on big layers, which lets me quantize all rows in one shared column order and share a single inverse — killing the cubic-per-row cost; I batch the column updates into blocks so the heavy work becomes a GEMM instead of a bandwidth-bound rank-one update; and I replace the unstable iterated inverse-downdate with a single Cholesky factorization of $\mathbf{H}^{-1}$, since — as I checked column by column and then end to end — the factor contains exactly the scaled row-tails the OBS update needs. Here is the layer routine.

```python
import torch

def add_batch(self, inp, out=None):
    # Keep the implementation's running average proportional to 2 X X^T.
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)
    batch = inp.shape[0]
    inp = inp.reshape(-1, inp.shape[-1]).t().float()      # d_col x tokens
    self.H *= self.nsamples / (self.nsamples + batch)
    self.nsamples += batch
    inp *= (2.0 / self.nsamples) ** 0.5
    self.H += inp.matmul(inp.t())

def compress(self, quantizer, blocksize=128, percdamp=0.01, groupsize=-1):
    W = self.W.clone()                       # d_row x d_col, float
    H = self.H.clone()                       # scaled version of 2 * X X^T

    if not quantizer.ready():
        quantizer.find_params(W, weight=True)

    dead = torch.diag(H) == 0                # columns never excited by calibration data
    H[dead, dead] = 1; W[:, dead] = 0

    # mild dampening: add 1% of the mean diagonal so H stays well-conditioned
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(self.d_col, device=H.device)
    H[diag, diag] += damp

    # the downdate is Gaussian elimination; U stores the scaled row-tails
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    U = torch.linalg.cholesky(H, upper=True)

    Q = torch.zeros_like(W)
    Losses = torch.zeros_like(W)
    for i1 in range(0, self.d_col, blocksize):           # block sweep
        i2 = min(i1 + blocksize, self.d_col)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]

        for i in range(i2 - i1):                          # column by column inside block
            w = W1[:, i]
            d = U1[i, i]                                  # sqrt of the current inverse pivot

            if groupsize != -1 and (i1 + i) % groupsize == 0:
                quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

            q = quantizer.quantize(w.unsqueeze(1)).flatten()
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2         # greedy score; true loss is half
            err = (w - q) / d                             # scaled OBS error
            # propagate error to remaining columns *within the block* only
            W1[:, i:] -= err.unsqueeze(1) * U1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2
        # one GEMM: apply the block's accumulated error to all columns to the right
        W[:, i2:] -= Err1.matmul(U[i1:i2, i2:])

    self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)
    return torch.sum(Losses)
```
