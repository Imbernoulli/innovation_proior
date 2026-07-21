Noisy nets cleared the bar by a hair — 118% against 117% — and the flatness is the message, not a
disappointment. A $+1$ median against the decoupled target's $+38$ is exactly the signature of a change that
fixes a minority of games dramatically and leaves the center where it was: a handful of transformed
hard-exploration titles sit in the tail, not at the $29$th position. Read literally, exploration was not what
capped the median game — the typical game already saw enough reward signal under $\epsilon$-greedy and is
bottlenecked somewhere else. So I want an axis that touches *every* game's learning, not a minority's
discovery, and it is sitting in plain sight: how the replay buffer hands transitions to the update. The floor
draws minibatches *uniformly at random*, a choice I never justified — I adopted it only to decorrelate
samples. But decorrelation does not require *uniformity*; it only requires not training on the live stream.
So uniform sampling is free to be wrong, and I should ask what it costs.

First check I am not about to undo the reason replay existed. Sampling by priority still draws each minibatch
from across the entire $10^6$-transition buffer, from many times and policies, just with a non-uniform
weighting; the transitions in a batch are still scattered through the agent's history, not consecutive on the
live stream. Decorrelation is a property of *drawing from the buffer at all*, and prioritization only
re-weights *which* entries are favored — so I can keep the floor's decorrelation while changing the
uniformity I never justified.

Uniform replay replays each transition at the frequency it happened to be experienced, regardless of how much
the agent can still learn from it. That is the waste. Most of the buffer, most of the time, is transitions
the network already predicts well — the TD error on them is tiny, so the gradient they produce is tiny, so an
update spent on them barely moves the weights. Meanwhile the transitions that would teach the most — the
surprising ones, where the prediction is badly wrong — are sampled at exactly the same rate. Make it sharp:
imagine a buffer of $N$ transitions of which only one carries the reward signal that needs to propagate, and
value information travels one Bellman step per replay of that transition. Uniform sampling draws it with
probability $1/N$, so it waits $\sim N$ draws just to move the reward back a single step; propagating along a
chain of length $L$ takes $\sim N\cdot L$. An oracle that always replays the transition that most reduces the
loss propagates in $\sim L$ — a factor of $N$ faster, and with $N=10^6$ that factor is enormous. Chain
several rare-signal stages and the gap compounds: uniform pays the $1/N$ tax once per stage, the oracle
never. I cannot build the oracle — it needs the future loss reductions — but I can approximate it with a
quantity I already compute.

A cheap, already-available proxy for "how much is left to learn from this transition" is the magnitude of its
TD error, $|\delta|$: large means the current prediction is far from its own bootstrap target — surprising,
and a gradient step will move the weights a lot — small means already consistent. And I compute $\delta$ for
every sampled transition anyway. So give transition $i$ priority $p_i=|\delta_i|+\varepsilon$ (the small
$\varepsilon>0$ keeps a zero-error transition samplable) and sample it with $P(i)=p_i^\alpha/\sum_k p_k^\alpha$.
At $\alpha=0$, $P(i)=1/N$ — uniform recovered as a special case, so this is a strict generalization. At
$\alpha=1$, sampling is proportional to raw error, the most aggressive prioritization. I want $\alpha$
strictly inside $(0,1)$, around $0.5$–$0.6$: full-strength $\alpha=1$ concentrates so hard on the current
high-error transitions that each minibatch's diversity collapses, and correlated batches are exactly the
pathology replay existed to prevent. So $\alpha$ trades data efficiency (higher) against batch diversity
(lower), and the safe interior value gets most of the efficiency without reintroducing correlation. I keep it
*stochastic* rather than always picking the single highest-error transition for two reasons: a greedy argmax
would replay a small set over and over and starve the rest (a transition whose error started low would never
be revisited), and TD errors are noisy, so a one-off spike should not let a transition dominate. Full-support
stochastic prioritization fixes both.

There is a real fork in *how* to turn $|\delta|$ into a probability. *Proportional*:
$P(i)\propto(|\delta_i|+\varepsilon)^\alpha$, in direct proportion to a power of the raw error. *Rank-based*:
$P(i)\propto(1/\text{rank}(i))^\alpha$, so only the ordering matters. Rank-based is more robust — insensitive
to outlier errors, since a freak $|\delta|=1000$ and an adjacent-rank $|\delta|=10$ get nearly equal priority
— and its implied distribution is a fixed power law independent of error scale, one fewer thing to vary
across $57$ games. Proportional is more faithful to the actual surprise (it distinguishes $|\delta|=10$ from
$11$, which rank flattens) but more exposed to outliers and to the error scale drifting. I take proportional
because the additive $\varepsilon$ and $\alpha<1$ already temper outlier sensitivity, because the floor's
reward clipping bounds how large a single $|\delta|$ can get so the outlier worry is smaller here than in
general, and because it keeps the priority a direct, interpretable function of surprise. The clipping is
doing quiet work to make the more faithful option safe.

But prioritized sampling changes *what fixed point* the updates converge to. Uniform SGD drives the weights
to where $\frac1N\sum_i\delta_i\nabla Q_i=0$ — the gradient averaged over the *empirical* data distribution.
Sampling from $P$ instead solves $\sum_i P(i)\,\delta_i\nabla Q_i=0$, a *different* solution, while the value
I want is still defined over the real data distribution. The correction is importance sampling: weight each
sampled update by $w_i=(N\,P(i))^{-\beta}$. At $\beta=1$ it is exact:
$\sum_i P(i)\frac{1}{N P(i)}\delta_i\nabla Q_i=\frac1N\sum_i\delta_i\nabla Q_i$, the uniform expectation
recovered. Normalize the weights by $1/\max_l w_l$: the raw $w_i$ is *largest* for the *least*-sampled
transitions, so dividing by the buffer maximum sends the largest weight to exactly $1$ and every other below
$1$ — update magnitudes only ever scale *down*, never up. That matters because combining a large importance
weight with an already-large prioritized TD error could otherwise produce a huge gradient on the noisiest
transitions; capping at $1$ forecloses that at no cost to the correction's relative structure, since scaling
all weights by a constant leaves the gradient direction unchanged. And anneal $\beta:\beta_0\approx0.4\to1$
over training: early on the network changes fast and the notion of a fixed point is moot, so I would rather
have the raw speed of lightly-corrected prioritization; near the end, where an uncorrected bias would pin the
*wrong* solution, I correct fully. It is the mirror image of the floor's $\epsilon$ anneal —
explore-then-exploit there, bias-for-speed-then-correctness here — and set once for all $57$ games.

A couple of details so this composes cleanly. I build it on the decoupled target — the TD error I prioritize
by is the Double-DQN error
$\delta=R+\gamma\,Q_{\theta^-}(S',\arg\max_a Q_\theta(S',a))-Q_\theta(S,A)$ — because prioritizing a biased
error would just prioritize the bias. Bookkeeping: a brand-new transition enters at *maximal* priority so it
is replayed at least once (I have no error estimate for it yet); after replaying transition $i$ I write the
fresh $p_i\leftarrow|\delta_i|+\varepsilon$ back. There is a staleness subtlety: a stored priority is the
error from the *last* replay, not under the current network, so a transition the agent has since learned to
predict carries its old high priority until drawn again. I cannot remove this without the $O(N)$ cost I am
avoiding, but it is self-correcting — a stale-high priority just means one more replay, at which point its
now-small error writes a small priority back. And because prioritization deliberately over-draws high-error
transitions, a batch's mean $|\delta|$ — and hence the mean gradient magnitude — is several times larger than
uniform's; the importance weights pull some back but not all during the anneal. So I cut the learning rate by
roughly $4\times$ to restore the effective step size to the range the encoder was stable at — the same "keep
the update magnitude in a safe band" logic as the reward and error clipping.

One implementation worry: naively, sampling from $P(i)\propto p_i^\alpha$ and maintaining $\sum_k p_k^\alpha$
over a $10^6$-entry buffer is $O(N)$ per draw and update — a million operations for every sampled transition,
every training step over $50$M steps, which would dwarf the conv forward/backward. Uniform replay never had
this problem because a uniform draw is $O(1)$. A **sum-tree** fixes it: a binary tree whose leaves hold
$p_i^\alpha$ and whose internal nodes hold the sum of their children. Updating one leaf is $O(\log N)$
(walk up fixing sums); sampling is $O(\log N)$ via prefix-sum descent — draw a uniform value in
$[0,\text{total}]$ and walk down, going left or right by comparing against the left child's sum. For a
minibatch of $k$ I stratify: split $[0,\text{total}]$ into $k$ equal segments and draw one per segment. A
parallel **min-tree** gives $\min_i p_i^\alpha$ for the normalizer $\max_l w_l$. With $N=10^6$,
$\log_2(10^6)\approx20$, so each operation is about $20$ steps instead of a million — a $50{,}000\times$
reduction that turns the linear non-starter into a cost negligible against the conv pass. The sum-tree is not
an optional detail; it is what makes non-uniform sampling over a $10^6$ buffer feasible at all.

Now the bar. Unlike noisy nets, this touches the data efficiency of *every* game — every game has redundant
transitions it over-replays and rare informative ones it under-replays, so reallocating the gradient budget
toward what is still learnable should lift the broad middle, which is what moves a median. The risk is the
bias: if the importance-sampling correction is mis-set the agent could converge to a subtly wrong value
function, so the $\beta$-anneal is doing real work and I am watching for instability, not just the headline.
But if the analysis is right, this is the first axis since the decoupled target that should give a *broad*
lift rather than a tail one, so I expect a clear jump above $118\%$ — plausibly the largest single step since
the $79\to117$ climb. A near-flat result would falsify the "every game wastes budget on redundant
transitions" story and tell me the typical game's ceiling lies in some part of the agent I have not yet
re-examined.
