The Switch aux loss did what I expected: it cut the imbalance sharply against the control, proving
the `f·P` surrogate steers the router toward uniform. But the previous rung also named its own
suspicion precisely — the counts `f_i` were measured on the micro-batch, the small noisy slice one
forward pass sees, and demanding uniformity of every slice punishes legitimate per-slice
specialization. So the cross-entropy did not come down as far as it could, because some of the
balancing pressure was spent flattening structure I wanted to keep. The diagnosis points at exactly
one knob, and I want to turn only that knob: the set of tokens over which the frequency `f_i` is
computed.

Let me reason about why the granularity is the whole story. The thing I actually care about is a
corpus-level property: across all the data the model trains on, every expert should pull roughly its
fair share of the work, so that no expert dies and the capacity is used. I do *not* care that one
particular micro-batch of two dozen sequences uses experts evenly. In fact I want the opposite — if
a micro-batch happens to be all code, the right behavior is for the code experts to light up and the
others to stay quiet on that slice. Specialization *is* per-slice skew that averages out to balance
over the corpus. The micro-batch penalty conflates these two: it sees the code slice's skew and
cannot distinguish it from the pathological skew of collapse, so it penalizes both. It is enforcing
the right constraint at the wrong scope.

The fix, then, is to measure `f_i` over the global batch — the union of all the micro-batches,
synchronized — instead of each micro-batch alone. Keep the penalty form exactly as it was:
`α · N · Σ_i f_i P_i`, same coefficient, same multiply-by-`N` to keep the balanced optimum
scale-free, same detached counts and differentiable probabilities. The only change is that `f_i` is
now the fraction of tokens routed to expert `i` across the whole batch. Now the penalty asks: across
all this data, is the usage uniform? — and it is silent about whether any individual slice is
uniform. A micro-batch is free to be as specialized as its content demands, so long as the
specializations of different slices cover all the experts when summed. The router can keep its
learned structure and still satisfy the balance constraint, which is exactly the freedom the
micro-batch version denied it.

I should be honest that in this small single-process reproduction there is only one device and the
training batch already *is* the global batch — so the way I make the distinction real is to compute
the previous rung's penalty on a handful of micro-splits of the batch (emulating the micro-batch
locality), and this rung's penalty on the full batch. The contrast is therefore between
"uniformity demanded of each quarter of the batch" and "uniformity demanded of the whole batch." It
is a faithful small-scale stand-in for the real distinction between per-device and global `f`, even
if the absolute gap is muted at this scale where the slices are already fairly representative of
each other.

There is a complementary lever I want to mention because it sits naturally alongside this rung, even
though it is not a loss at all. DeepSeek's auxiliary-loss-free scheme keeps a per-expert bias used
*only* to break ties in the top-K selection — added to the routing scores for ranking, but excluded
from the gate weights that actually combine the experts — and nudges it once per step by a small
constant in the direction that cools overloaded experts and warms underloaded ones. It carries no
gradient; it is a slow control loop on the counts that runs beside the differentiable penalty.
Pairing it with the global-batch loss is attractive because the two act on different surfaces: the
loss gives the router a gradient about balance through `P`, while the bias adjusts the hard
selection directly, catching imbalance the smooth gradient is slow to fix. I will run the
global-batch loss both alone and with this bias added, to see whether the count-level controller buys
anything on top.

So the rung is: the same `f·P` penalty, but with `f` measured over the global batch rather than the
micro-batch, optionally augmented by the gradient-free selection bias. I expect the imbalance to land
in the same good range as the Switch loss — both are the same penalty form and both break collapse —
while the cross-entropy should be at least as low, because the global scope stops over-constraining
the individual slices. The honest caveat is the one I will carry into the next rung: this loss
equalizes the *average* usage, but it has nothing special to say about the experts that have fallen
*well below* their fair share. To the smooth `f·P` term, an expert at a tenth of its fair share and
an expert at nine-tenths of it are both just contributing a small term; the gradient that would
rescue the nearly-dead expert is weak exactly in the tail where rescue matters most. Balancing the
mean is not the same as resurrecting the dying, and that gap — the under-utilized tail the
global-batch loss leaves on the table — is the opening the next rung has to attack.
