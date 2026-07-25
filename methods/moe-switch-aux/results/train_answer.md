Having watched an unregularized MoE router collapse — a tolerable cross-entropy sitting on a clearly
skewed token allocation — I needed a term I could add to the cross-entropy that explicitly looks at
the routing distribution and pushes it toward uniform. The obstacle I hit immediately is that the
quantity I most want to control, the fraction of tokens each expert receives, is a hard count coming
from the top-K selection, and counts have no usable gradient: nudging the router's weights does not
change which expert is top-two until some token crosses a boundary, where the count jumps. A penalty
written purely on those counts gives the router nothing to follow.

The method that resolves this is the Switch/GShard auxiliary loss, and its trick is to pair the
non-differentiable count with the one differentiable thing the router exposes — its softmax
probability mass. For each expert I form the product of its token fraction and its mean assigned
probability, and sum over experts. I treat the count as a fixed weight, detached, and let the
gradient flow only through the probability. Now minimizing this weighted sum does exactly what I
want: the counts are large for the over-used experts and small for the under-used ones, so to shrink
the sum the router must move probability mass off the hot experts and onto the cold ones. The count
steers where the pressure points; the probability is how the pressure is applied. I checked that the
uniform routing is the optimum — at balance every fraction and every probability sit near one over
the number of experts — and I multiply the whole penalty by the number of experts so that the
balanced value is a scale-free constant near one, leaving a single coefficient, about a hundredth,
to set its weight against the cross-entropy.

I am explicit about the one decision I expect to revisit: the counts are computed over the
micro-batch, the small slice one forward pass sees, which is what the original loss does. A
micro-batch can be genuinely lopsided in content — a slice that is all code — for which skewed
expert usage is the correct specialized behavior, and forcing its counts toward uniform punishes the
router for doing the right thing. The micro-batch penalty cannot tell collapse, which I want to stop,
from legitimate per-slice specialization, which I want to keep. So I expect this loss to cut the
imbalance sharply against the control while paying for it with a cross-entropy held slightly higher
than necessary — a fix correct in form but applied at the wrong granularity, which is exactly the
opening for measuring the counts over the global batch instead.

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
