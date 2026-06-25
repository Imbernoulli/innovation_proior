Let me start from the wall I keep hitting. The pressure in both directions I care about is toward more capacity: image classifiers had been growing with accuracy improvements, and in language the stronger models are deeper/larger ones rather than shallow sentence encoders. So I want to train networks far larger than what a single accelerator holds. A model that's interesting no longer fits in one device's memory. The instant that's true, I'm forced to split the model across devices. The question is how to split it so it's (a) general across architectures, (b) keeps the accelerators busy, and (c) doesn't change the math the user would have run on one device. Those three together are the hard part — any one alone is easy.

Why does the model not fit in the first place? Let me be precise, because the answer steers everything. It's tempting to blame the parameters, but for deep networks it's usually the *activations*. In backprop, to compute the gradient at layer i I need both the gradient coming back from layer i+1 and the *cached forward activation* of layer i. So a naive forward over L layers with a mini-batch of N caches an activation tensor at every layer — memory grows like O(N × L). For a deep model that's what overflows the device, often well before the weights do. So whatever I do, I have to attack activation memory, not just parameter memory.

First instinct: just cut the network into K consecutive chunks — call them cells — and put cell k on device k. The activations flowing across a cell boundary get sent device-to-device. This is appealing because it's totally architecture-agnostic: any network is a sequence of layers, I group consecutive layers, done. No per-operator distributed-matmul engineering, no architecture-specific surgery. And the communication is light — I only ship the activation tensor at the K−1 cell boundaries, not inside layers. Memory per device drops by roughly K.

But run one mini-batch through it and watch what happens. Cell 1 computes on device 1; device 2 sits idle waiting. Cell 1 hands its output to cell 2; now device 2 computes and device 1 sits idle. The network is a chain — layer i+1 can't start until layer i finishes, and in the backward the dependency reverses, the backward of cell i can't start until cell i+1 sends its gradient back. So at any instant exactly *one* device is computing and the other K−1 are idle. I've solved memory and thrown away (K−1)/K of my compute. Useless as it stands. The partitioning is right; the *idleness* is the enemy.

Stare at the idleness. Device 2 is idle because it's waiting for *this* mini-batch's data to arrive from device 1. But device 1 is now free — what if I feed it a *second* independent piece of data right away? Split the mini-batch of size N into M smaller micro-batches. Push micro-batch 1 through cell 1; the moment it moves on to cell 2, push micro-batch 2 into cell 1. Now cell 1 (device 1) works on micro-batch 2 while cell 2 (device 2) works on micro-batch 1. Keep feeding micro-batches in, and the devices should overlap — each on a different micro-batch at a different stage. That's the assembly-line idea. But I'm hand-waving the "should overlap"; let me actually write the schedule down and step through it, because the index arithmetic is where this kind of thing usually breaks.

Let clock t advance one cell-step at a time, and say at clock t cell k processes the micro-batch m = t − k (it exists only when 0 ≤ m < M). Take a tiny case I can fully enumerate, K = 3 cells and M = 4 micro-batches, and read off who is busy at each clock:

```
t=0: cell0<-mb0
t=1: cell0<-mb1, cell1<-mb0
t=2: cell0<-mb2, cell1<-mb1, cell2<-mb0
t=3: cell0<-mb3, cell1<-mb2, cell2<-mb1
t=4:            cell1<-mb3, cell2<-mb2
t=5:                        cell2<-mb3
```

That settles whether the idea even works: at t=2 all three cells are simultaneously busy, each on a different micro-batch (mb2 in cell0, mb1 in cell1, mb0 in cell2). The chain dependency is respected — micro-batch m reaches cell k exactly one clock after it left cell k−1, since m = t−k means it was at cell k−1 at clock t−1. So batch-splitting really does convert the single-active-device chain into a saturated pipeline, but only in the *steady state*; the table also shows the cost — the ramp at the top (t=0,1) and the drain at the bottom (t=4,5), where fewer than three cells are working. I'll have to come back and pay for those.

Now I have to be careful about (c) — not changing the optimization. If I let each micro-batch immediately update the weights as it finishes, then micro-batch 2 in cell 1 would be computed against weights already nudged by micro-batch 1's partial pass — weight staleness, and the effective update would depend on the pipeline depth. That breaks synchronous mini-batch training and muddies convergence. So the discipline has to be: don't update per micro-batch. Run the forward of all M micro-batches through the pipeline with one fixed set of weights, then compute the backward passes from those same weights, *accumulate* all M micro-batch gradients, and apply a *single* update at the end of the mini-batch.

I want to be sure the accumulated update is genuinely identical to the one I'd have computed on the unsplit mini-batch, not just "morally" the same — this is the whole correctness claim, so let me check it rather than assert it. A mini-batch loss is a sum over examples, L(w) = Σₙ loss(xₙ; w), and the gradient of a sum is the sum of the gradients, so splitting the index set into M contiguous chunks and summing the chunk gradients must reproduce the full gradient exactly. Let me confirm on a concrete tiny model — a scalar linear unit with loss ½(w·x − y)², whose per-example gradient is (w·x − y)·x — over N = 12 examples split into M = 3 micro-batches of 4:

```
g_full  = sum over all 12 examples            = 3.374297993752285
g_accum = sum of the 3 per-micro-batch sums   = 3.374297993752285
equal? True
```

Exactly equal, to the last digit. So as long as the per-example contributions are summed (the natural reduction for a summed loss), accumulating M micro-batch gradients and applying one update is bit-for-bit the single-device step. Scaling out changes throughput, not the optimizer's trajectory. There is one place this reasoning leaks: a layer whose output for an example depends on the *other examples in its batch* is not a plain sum over examples. Batch normalization is exactly that — its mean/variance are computed over whatever batch it sees, so under micro-batching it would silently normalize over N/M instead of N. The fix that keeps evaluation well-defined is to let BN compute statistics per micro-batch during training but track a moving average over the whole mini-batch for inference.

But wait — the activation memory. I split the model across K devices so each device only holds L/K layers, but within its cell each device still caches the forward activations of all M micro-batches across all its L/K layers to use in the backward. Let me count: M micro-batches × (L/K layers per cell) × (N/M activation size per micro-batch) = O((L/K)N). The split helped by the factor K, but micro-batching alone has not made the layerwise activation cache disappear; it just slices the same mini-batch into smaller pieces. If I want enough micro-batches to fill the pipeline and still fit a very deep cell, I need another memory lever.

The lever I have is re-materialization: don't cache a cell's internal activations at all, cache only its boundary, and recompute the internals in the backward when they're needed. During the forward, each device stores only partition-boundary activations — the output of the previous cell, which is the input boundary for the next cell — not every internal layer. In the backward, before it needs the internal activations, the device recomputes the cell's forward from that stored boundary activation. So instead of caching L/K layers' worth of activations per micro-batch, I cache one boundary tensor per micro-batch and recompute the rest. Let me work out the peak. I need O(N) to hold the boundary activations across the mini-batch, plus, while recomputing one cell for one micro-batch, the internal activations of that single cell for a single micro-batch: (L/K) layers × (N/M) micro-batch size. So peak activation memory is O(N + (L/K)(N/M)).

Let me put real numbers on that against the O(N × L) baseline, to see whether the improvement is the order-of-magnitude kind I'm hoping for or just a constant-factor tweak. Take a deep Transformer-scale case, N = 1024, L = 96 layers, K = 8 cells, M = 32 micro-batches, with one unit per layer-per-example of activation:

```
naive  O(N*L)            = 1024 * 96                 = 98304
gpipe  N + (L/K)(N/M)    = 1024 + (96/8)*(1024/32)   = 1024 + 12*32 = 1408
ratio                    = 98304 / 1408              ≈ 69.8x
```

So on these numbers re-materialization plus partitioning is a ~70× reduction in peak activation memory, and the structure of the count shows why: the L factor collapses to L/K and the N factor inside the recompute term collapses to N/M, while only an O(N) boundary term survives at full batch size. That's the order-of-magnitude win, and the price is explicit in the derivation — one extra forward over each cell during the backward, since the internals were thrown away. A worthwhile trade.

Now back to the bill I deferred — the ramp and drain I saw in the schedule table. That fill-and-drain idleness is the *bubble*, and I should size it rather than wave at it, because the whole value of micro-batching rides on the bubble being small. From the K = 3, M = 4 table: there are M + K − 1 = 6 clocks, K = 3 cells, so 18 cell-clock slots, of which the busy ones are 1+2+3+3+2+1 = 12. Idle fraction = 1 − 12/18 = 1/3. And the closed form I'd guess for a length-K pipeline running M items, (K−1)/(M+K−1) = 2/6 = 1/3, matches exactly. Good — the formula is the right one. Now the practical question: how large must M be before the bubble stops mattering? Let me just tabulate it for K = 8:

```
K=8, M=8  (M=1K):  (8-1)/(8+8-1)   = 7/15  ≈ 0.467
K=8, M=16 (M=2K):  7/23            ≈ 0.304
K=8, M=32 (M=4K):  7/39            ≈ 0.179
K=8, M=64 (M=8K):  7/71            ≈ 0.099
```

This corrects an assumption I was about to make. I'd been carrying a rule of thumb that "M a few times K" makes the bubble vanish, but at M = 4K the bubble is still ~18% — real overhead, not negligible. It only drops under ~10% around M = 8K, and toward zero as M ≫ K. So the honest statement is weaker than I wanted: the bubble *amortizes* with more micro-batches and you want M as large as the activation budget allows (the O(N + (L/K)(N/M)) memory term is what stops you from cranking M arbitrarily high), but there's no clean cutoff where it disappears — it's a 1/(M+K−1) tail. There's also a second-order improvement I can claim only weakly: a cell's forward recomputation during the backward can start from its cached boundary activation *before* that cell's incoming backward gradient arrives, so some would-be-idle backward slots can be filled with recompute. I'd want to verify the realized utilization on a profiled run before trusting a number for that; here I'll just note it pushes the effective bubble below the formula.

Let me also sanity-check the communication, because I claimed it's light. From the schedule, the only thing that crosses a device boundary is the activation tensor handed from cell k to cell k+1 — at the K−1 boundaries, once per micro-batch. Never the parameters, never anything inside a layer. So the bytes moved are tiny compared to tensor-slicing schemes that all-reduce inside every layer. This means I should not require a fast interconnect: even on a multi-GPU host without NVLink, where cross-device transfers crawl through device-to-host-to-device PCIe, the only thing crossing devices is the boundary activation for each micro-batch. That's a real flexibility win, and it's a consequence of partitioning vertically (whole layers to a device) rather than horizontally (slicing inside layers).

One more practical point on the partitioning itself. I've been assuming the K cells take equal time, but layers aren't uniform — some are heavier in compute or memory. The pipeline runs at the speed of its slowest stage, so an unbalanced split leaves bubbles even with large M (the slowest cell stalls everything downstream of it every clock). So I let each layer carry an optional cost estimate c_i, and I choose the grouping into cells to *minimize the variance* of per-cell cost — balance the stages so no device is the bottleneck. The composite forward of a cell is F_k = f_j ∘ … ∘ f_i, its backward B_k is built from F_k by autodiff, and its cost is the sum of its layers' costs. The whole interface a user touches is then just: the number of partitions K, the number of micro-batches M, and the ordered list of layers. Everything else — placement, boundary communication, the pipeline schedule, re-materialization — is automatic and architecture-agnostic.

Before I commit the schedule to code, one correctness worry about the backward: the forward wavefront sends micro-batches *down* the cells (small k first), so the backward must run the mirror image, draining from the last cell, and each cell's backward must be able to recompute its forward from the boundary it cached on the way down. Let me trace the reverse wavefront with the symmetric index m = t − (K−1−k), same K = 3, M = 4:

```
t=0:                         cell2<-mb0
t=1:            cell1<-mb0,  cell2<-mb1
t=2: cell0<-mb0, cell1<-mb1, cell2<-mb2
t=3: cell0<-mb1, cell1<-mb2, cell2<-mb3
t=4: cell0<-mb2, cell1<-mb3
t=5: cell0<-mb3
```

Cell2 (which holds the final loss gradient) starts first and the wave drains toward cell0 — the correct dependency order, since cell k's backward needs the incoming gradient from cell k+1. And I checked the bookkeeping both ways: enumerating the forward schedule covers every (cell, micro-batch) pair exactly once (12 of them for 3×4), and so does the backward. Nothing dropped, nothing double-counted, so accumulating one gradient contribution per backward visit sums each micro-batch exactly once — which is the additivity I verified earlier. The boundary cached at forward-clock for cell k, micro-batch m is exactly what the backward needs to recompute that cell, so re-materialization slots in without storing anything extra.

So the schedule writes itself from the two traces: partition layers into K cost-balanced cells, place cell k on device k; split the mini-batch into M micro-batches; run the forward wavefront m = t − k caching only boundary activations; run the reverse wavefront m = t − (K−1−k), recomputing each cell's internals from its cached boundary, accumulating all M micro-batch gradients; apply one synchronous update at the end.

```python
from typing import Callable, List

class Layer:
    def __init__(self, forward: Callable, params, cost_fn: Callable = None):
        self.f, self.w, self.cost = forward, params, cost_fn

class LayerSequenceTrainer:
    """General pipeline parallelism with micro-batching + re-materialization.
    All micro-batch gradients are accumulated before one optimizer update."""
    def __init__(self, layers: List[Layer], devices: List[object], M: int):
        self.layers = layers
        self.devices = devices
        self.K = len(devices)
        self.M = M
        # balance the pipeline: group L layers into K cells minimizing cost variance,
        # so no single stage is the bottleneck and the bubble stays small.
        self.cells = self.build_layer_groups()
        # cell k lives on device k; F_k = f_j o ... o f_i, B_k built by autodiff.
        self.F = [self._compose([l.f for l in cell]) for cell in self.cells]
        self.B = [autodiff_backward(F_k) for F_k in self.F]

    def build_layer_groups(self):
        # choose consecutive groupings that minimize variance of sum(cost) per cell
        ...
    def _compose(self, fns):
        ...
    def _on_device(self, k, fn, *args):
        ...
    def _rematerialize(self, k, boundary_activation):
        # rerun cell k's forward from its cached boundary activation
        ...

    def run_forward(self, mini_batch):
        micro_batches = self._split(mini_batch, self.M)
        boundary = [[None] * self.M for _ in range(self.K)]
        x = [[None] * self.M for _ in range(self.K + 1)]
        x[0] = list(micro_batches)
        for t in range(self.M + self.K - 1):
            for k in range(self.K):
                m = t - k
                if 0 <= m < self.M:
                    boundary[k][m] = x[k][m]         # cache the boundary activation feeding cell k
                    x[k + 1][m] = self._on_device(k, self.F[k], x[k][m])
        self._last_boundary = boundary
        return x[self.K]

    def run_backward(self, d_final):
        grads = self._zero_like_params()
        d = [[None] * self.M for _ in range(self.K + 1)]
        d[self.K] = list(d_final)
        for t in range(self.M + self.K - 1):
            for k in reversed(range(self.K)):
                m = t - (self.K - 1 - k)
                if 0 <= m < self.M:
                    acts = self._rematerialize(k, self._last_boundary[k][m])
                    d[k][m], g = self.B[k](d[k + 1][m], acts)  # recomputed activations
                    grads = self._accumulate(grads, g)
        return grads

    def train_step(self, mini_batch, optimizer):
        out = self.run_forward(mini_batch)
        loss = self._loss(out)
        grads = self.run_backward(self._loss_grad(loss))
        optimizer.apply(grads)        # one synchronous update per mini-batch
```

The causal chain, end to end: models must be split because they don't fit, and the thing that overflows is the O(N·L) activation cache; a vertical split into K cells fixes memory and keeps communication to just the boundary activations, but the chain dependency idles all but one device; splitting the mini-batch into M micro-batches and pipelining them fills the devices (the K=3,M=4 trace showed all cells busy in steady state); accumulating all M micro-batch gradients into a single end-of-mini-batch update keeps the step bit-for-bit identical to single-device (verified on the scalar model, g_full = g_accum), except where a layer mixes examples — BatchNorm needs the per-micro-batch / moving-average split; re-materialization drops peak activation memory to O(N + (L/K)(N/M)), ~70× on the N=1024, L=96, K=8, M=32 numbers, at the cost of one recompute; the residual fill/drain bubble is (K−1)/(M+K−1), which amortizes with M but is still ~18% at M=4K and only nears zero as M ≫ K; and cost-balanced partitioning keeps the slowest stage from stalling the rest.
