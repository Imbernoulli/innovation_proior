Let me start from the thing that's actually stopping me: I run out of GPU memory long before I run out of useful depth. I want to train a very deep network — a thousand-layer residual net, or an LSTM unrolled over a long sequence — and the device memory caps me at a couple hundred layers. So I need to understand precisely *what* is eating the memory, because if it were the parameters I'd be stuck, but I suspect it isn't.

In a convolutional or recurrent net the parameters are comparatively small; what's large is the intermediate feature maps — the activations produced at each layer in the forward pass. And here's why they pile up: backprop needs them. When I differentiate a layer, the backward operator for that layer depends on the forward activation that went into it. So the standard training loop computes the forward pass, and *holds onto every layer's activation* until the backward pass walks back and consumes it. With `n` layers, that's `n` activations kept alive simultaneously — memory `O(n)`, linear in depth. That linear term is the ceiling.

What can I already do about graph memory? The frameworks have two tricks, both from liveness analysis on the computation graph. In-place operation: write an op's output into its input's buffer, when the input isn't needed afterward. And memory sharing: once an intermediate has no pending consumers, recycle its buffer for a later node. Deciding what can share is a lifetime question — two values can share a buffer only if their lifetimes don't overlap — and I can compute that with a cheap `O(n)` sweep: give each node a counter of how many consumers still need it, do the op in-place when its input's last consumer is firing, and free a buffer when its counter hits zero. (The exact alternative is to build the conflict graph and color it, but that's `O(n²)`; the counter heuristic is good enough.)

These help, but I need to be honest about *how much*. For prediction they're huge: at inference, layer `i`'s activation dies the instant layer `i+1` consumes it, so I can run the whole network in nearly `O(1)` memory by recycling one or two buffers. But for *training* the same trick only buys a constant factor — 2 to 3× on a deep ResNet — and the asymptotics don't budge. Why? Because the forward activations *can't* die early during training: they have to stay alive until the backward pass, which runs *after* the entire forward pass, reaches back for them. Their lifetimes all overlap the backward pass, so there's almost nothing to share away. Training memory stays `O(n)`. A 2–3× constant doesn't let me train a 1000-layer net; it lets me train a 400-layer one. I need to break the linear scaling itself.

So the in-place/sharing toolkit is exhausted, and the reason it's exhausted is that I'm *storing* every activation. What if I simply don't? The only reason an activation has to live through the forward pass is to be available in the backward pass — but I don't strictly need to *store* it, I need to be able to *produce* it when the backward pass asks. And producing it is just running the forward computation again. So here's the trade: drop some forward activations the moment they've been used in the forward direction, free their memory, and when the backward pass needs one of them, recompute it by re-running the forward computation from the nearest activation I *did* keep.

Let me make that concrete on a linear chain of `n` layers. I'll cut the chain into segments. I keep only the activation at each segment *boundary* — the input to each segment — and inside a segment I drop everything as soon as the forward pass moves past it. Now the forward pass costs only `O(number of boundaries)` activation memory, because the within-segment activations are thrown away. When backprop comes back to a segment, I reload that segment's stored boundary input, re-run the forward through just that segment to regenerate its internal activations into a small local buffer, then run the backward through the segment using those regenerated activations, and free them again before moving to the previous segment. The gradients I get are *exactly* the gradients I'd have gotten by storing everything — I'm recomputing the same numbers, not approximating them.

Now the question that decides everything: how many segments? Suppose `k` equal segments of `n/k` layers each. The memory has two parts. First, I store the `k` boundary activations: `O(k)`. Second, at any moment during backprop I'm holding the regenerated activations of *one* segment in order to backprop through it: a segment has `n/k` layers, so that's `O(n/k)`. (Only one segment's worth at a time, because I free each segment's recomputed activations before the next.) Total:

`memory(k) = O(n/k) + O(k)`.

These two terms pull in opposite directions: more segments means cheaper per-segment recompute buffers (`n/k` shrinks) but more boundaries to store (`k` grows), and fewer segments means the reverse. Minimize the sum. Treating it as `f(k) = n/k + k`, set `f'(k) = -n/k² + 1 = 0`, so `k² = n`, `k = √n`. At that point `f(√n) = n/√n + √n = √n + √n = 2√n`. So with `k = √n` segments, the activation memory is `O(√n)` — sublinear, exactly the break I wanted. A thousand-layer net needs on the order of `√1000 ≈ 32` segments' worth of memory instead of a thousand.

And the compute price is small, which is the whole reason this is worth it. Each layer's forward is computed once in the original forward pass, and at most once more during its segment's recomputation in backprop — so the *extra* cost is one additional full forward pass over the network, no more. Since the backward pass is itself roughly twice as expensive as a forward pass, an extra forward pass is a modest fraction of the total work; in practice it lands around a 30% slowdown. I'm paying ~30% more time to turn `O(n)` memory into `O(√n)`. That's the trade.

Can I push further? The segment structure is self-similar: a segment is itself a little chain of layers, and the trick I just applied to the whole network can be applied *inside* a segment too — recursively. Let `g(n)` be the memory to do a forward+backward on an `n`-layer net under this scheme. If I store `k` intermediates at the top level and apply the same scheme recursively to each sub-path (each of length `n/(k+1)`, since `k` stored intermediates cut the chain into `k+1` pieces), then the memory is the `k` I store plus the memory to recurse on one sub-path:

`g(n) = k + g(n/(k+1))`.

Unrolling this recursion: each level adds `k` and divides the length by `k+1`, so after `log_{k+1} n` levels the length is down to a constant, giving `g(n) = k · log_{k+1}(n)`. The extreme is `k = 1` — store a single intermediate at each level, split in two, recurse — which gives `g(n) = 1 · log_2 n = O(log n)` memory. The price is that recomputation now nests: regenerating an activation deep in the recursion means re-running multiple forward passes, so the compute overhead grows to `O(n log n)`. Usually too expensive to use in practice, but it shows the trade extends all the way down to logarithmic memory. The sweet spot for normal use is `√n`.

The equal-cost-chain picture is idealized, though: real layers don't all cost the same memory or the same compute, so I can't blindly set every segment to `n/√n` layers. Instead I let the user mark, per node, whether its output is *kept* (a boundary, never dropped) or *dropped and recomputed* — a "mirror" attribute, since a dropped node gets duplicated (mirrored) into the backward region of the graph to be recomputed there; with every node marked "keep," this degenerates to ordinary backprop. Then a greedy planner sweeps the graph in topological order accumulating output sizes into a running tally; whenever it passes a candidate split point and the tally exceeds a memory budget `B`, it marks that node as a kept boundary and resets the tally, otherwise it marks the node as dropped. Sweeping `B` trades the two memory terms against each other when costs are non-uniform — a good starting `B` is `√(x·y)` where `x` is the total boundary-storage cost and `y` is the per-stage cost. Because I can statically allocate and get the *exact* memory of any plan, I can search a small grid of `B` and pick the best.

There's also a cheap heuristic that needs no segment math at all: drop the activations of the *low-cost* operations and keep the *expensive* ones. In a `Conv → BatchNorm → Activation` pipeline, the convolution is the expensive op, while batch-norm, the activation function, and pooling are cheap to recompute. So keep the conv output, drop the rest; recomputing them in backprop costs almost nothing, and the memory saving is immediate. This composes with the segment plan.

One correctness subtlety I have to respect: the recomputed activations must be *bit-identical* to the originals, or the gradients won't match. That's fine for deterministic layers, but stochastic ops — dropout, any noise — would produce different values on the second forward. So when I recompute a segment I have to restore the random-number-generator state that was in effect during the original forward, so the same dropout mask (etc.) is drawn. With that, recompute reproduces the exact forward and the gradients are exact.

Now the implementation, grounded in the framework's checkpointing utility. The primitive is a `checkpoint(function, *inputs)` wrapper: in the forward pass it runs `function` *without* building the autograd graph for its internals (so none of the intermediate activations are stored) and saves only the inputs; in the backward pass it re-runs `function` on the saved inputs to regenerate the activations, then backpropagates through that freshly-built local graph. To checkpoint a whole sequential model I split its layer list into `segments` chunks and wrap each chunk in `checkpoint`, so only the chunk boundaries' activations persist.

```python
import torch
import torch.utils.checkpoint as cp

def run_segment(layers, start, end):
    # closure that applies layers[start..end] in sequence
    def forward(x):
        for i in range(start, end + 1):
            x = layers[i](x)
        return x
    return forward

def checkpoint_sequential(layers, segments, x):
    # split n layers into `segments` chunks; checkpoint all but the last.
    n = len(layers)
    seg = n // segments
    end = -1
    for start in range(0, seg * (segments - 1), seg):
        end = start + seg - 1
        # checkpoint(): forward runs without storing this segment's
        # internal activations; backward recomputes them from the saved input.
        x = cp.checkpoint(run_segment(layers, start, end), x, use_reentrant=False)
    # the final segment runs normally (it's the first one backprop hits,
    # so its activations would be needed immediately anyway)
    return run_segment(layers, end + 1, n - 1)(x)

class CheckpointedDeepNet(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        # k = sqrt(n) segments minimizes O(n/k)+O(k) -> O(sqrt(n)) activation memory
        self.segments = max(1, int(len(layers) ** 0.5))

    def forward(self, x):
        return checkpoint_sequential(list(self.layers), self.segments, x)
```

The causal chain: training memory is `O(n)` because every forward activation must survive until the backward pass consumes it, and in-place/sharing can't help because those lifetimes all overlap the backward pass — so they only win a constant factor. The way out is to *not store* the activations but *recompute* them from kept checkpoints, which gives identical gradients for the price of extra forward work. Cutting an `n`-layer net into `k` segments makes the memory `O(n/k) + O(k)`, minimized at `k = √n` for `O(√n)` memory at the cost of a single extra forward pass (~30% slower); recursing the same trick reaches `O(log n)` at `O(n log n)` compute. A per-node mirror flag plus a budget-driven greedy planner handles non-uniform layer costs, dropping cheap ops and keeping expensive ones, and restoring RNG state on recompute keeps the gradients exact.
