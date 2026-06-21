The control settled the diagnosis: with no balancing term the router collapsed to $L_{imb} = 0.1286$ while the cross-entropy stayed unremarkable at perplexity $41.6$ — a tolerable $L_{CE}$ sitting on a clearly skewed allocation. That confirms collapse barely hurts the LM loss, which is exactly why the LM loss does nothing to prevent it, so the cure has to be a term I add by hand that looks at the routing distribution itself and pushes it toward uniform. The obstacle is that the most natural thing to penalize is something I cannot differentiate: what I actually want to control is $f_i$, the fraction of tokens each expert receives, but $f_i$ comes from the hard top-K selection. Counts have zero gradient almost everywhere — nudging the router's weights does not change which experts are top-two until some token crosses a boundary, and then the count jumps. A penalty written purely on $f_i$ hands the router no gradient to follow.

The fix is to pair the non-differentiable count with the one differentiable quantity the router exposes — its softmax probability vector $P_i$, the continuous mass it places on each expert before the hard selection. I propose the Switch/GShard auxiliary loss, added to the cross-entropy:
$$L_{aux} = \alpha \cdot N \cdot \sum_i f_i \, P_i, \qquad \alpha = 10^{-2}.$$
Here $f_i$ is the detached micro-batch count and $P_i$ is the differentiable mean router mass on expert $i$, and the construction works because of *how* the gradient flows. Treat each $f_i$ as a fixed weight — detach it, its gradient is useless anyway — and let the gradient reach the router only through $P_i$. The $f_i$ are then a fixed set of weights, large on the over-used experts and small on the under-used ones. To shrink $\sum_i f_i P_i$ under those weights the router must move probability mass *off* the experts with large weight (the hot ones) and onto the experts with small weight (the cold ones); the only experts left untouched are those already at their fair share. That is exactly the corrective pressure I want, and the division of labor is clean: the non-differentiable count steers *where* the pressure points, the differentiable probability is *how* the pressure is applied.

Two scaling decisions make the surrogate well-behaved. First, I check that uniform routing really is the optimum and not some other configuration — a penalty whose minimum sits away from uniform would be worse than useless. At the balanced fixed point every $f_i = 1/N$ and every $P_i \approx 1/N$, so $\sum_i f_i P_i \approx N \cdot (1/N)(1/N) = 1/N$; the surrogate is minimized exactly there. Second, to make that optimum scale-free I multiply by $N$, so the balanced value is a constant near one regardless of how many experts there are, and then the single coefficient $\alpha$ in front sets how hard the penalty pushes relative to the cross-entropy. I take $\alpha = 10^{-2}$, the textbook value — strong enough to break collapse, weak enough that it does not start dictating the router's predictions and drag $L_{CE}$ up. The penalty is averaged over the $L = 2$ MoE layers.

The one decision I want to name explicitly, because it is the load-bearing limitation, is the set of tokens over which $f_i$ is computed. The classical Switch and GShard losses, and this rung, compute it over the **micro-batch** — the handful of tokens in one forward pass on one device. It is the cheapest, most local choice and it is what the original loss does, but a micro-batch is a small, noisy sample of the corpus and may be genuinely lopsided in content: a slice that happens to be all code, or all one topic, for which skewed expert usage is the *correct*, specialized behavior. Forcing the $f_i$ of that slice toward uniform punishes the router for doing the right thing on it. The micro-batch penalty cannot distinguish collapse, which I want to stop, from legitimate per-slice specialization, which I want to keep, so it buys balance partly by flattening specialization — a cost that surfaces as a cross-entropy held higher than it needs to be. In this small single-process reproduction I emulate the micro-batch by splitting each training batch into four micro-splits, evaluating the penalty on each, and averaging. I expect a large drop in imbalance against the control — this is the established fix and it does break collapse — with the cross-entropy roughly flat at this scale, where the slices are still fairly representative. The over-constraint is the wall: the fix is correct in *form* but applied at the wrong granularity, and the next rung changes nothing about the penalty except the set of tokens the counts are measured over.

```python
import torch

def layer_f_P(probs, topi, N):
    counts = torch.bincount(topi.reshape(-1), minlength=N).float()
    f = counts / counts.sum()          # hard counts (non-differentiable)
    P = probs.mean(0)                  # mean router prob mass (differentiable)
    return f, P

def balance_loss_switch(probs_list, topi_list, N, alpha=1e-2):
    """Switch/GShard aux loss: alpha * N * sum_i f_i P_i, f over the micro-batch."""
    total = 0.0
    for probs, topi in zip(probs_list, topi_list):
        f, P = layer_f_P(probs, topi, N)
        total = total + N * (f.detach() * P).sum()   # gradient enters via P
    return alpha * total / len(probs_list)

# At training time f is computed per micro-batch: split the batch into 4 micro-splits,
# evaluate balance_loss_switch on each, and average — the micro-batch locality penalty.
```
