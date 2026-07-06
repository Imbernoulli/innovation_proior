Noisy nets cleared the bar by a hair — 118% against 117%, a single point — and the flatness is the message,
not a disappointment. A $+1$ median against a $+38$ from the decoupled target is exactly the signature I
predicted for a change that fixes a minority of games dramatically and leaves the center of the distribution
where it was: the median is the $29$th of $57$ games, and a handful of transformed hard-exploration titles
sit in the tail, not at the $29$th position, so they move the tail and the mean while the median stands
still. Read literally, the result says exploration was not what capped the median game. The hard-exploration titles where parameter noise shines are a
minority; the typical game already saw enough reward signal under $\epsilon$-greedy and is bottlenecked
somewhere else. So I want an axis that touches *every* game's learning, not a minority's discovery, and the
one I have been deferring is sitting in plain sight: how the replay buffer hands transitions to the update.
The floor draws minibatches *uniformly at random* from the buffer, and that is a choice I never justified —
I adopted it only to decorrelate samples. Decorrelation does not require *uniformity*; it only requires not
training on the live stream. So uniform sampling is free to be wrong, and I should ask what it costs.

Before I change the sampling I should check that I am not about to undo the reason replay existed in the
first place. The floor introduced replay to *decorrelate* — to stop training on consecutive, near-identical
frames — and I do not want prioritization to quietly reintroduce correlation. It does not: sampling by
priority still draws each minibatch from across the entire $10^6$-transition buffer, from many different
times and policies, just with a non-uniform weighting; the transitions in a batch are still scattered
through the agent's history, not consecutive on the live stream. So decorrelation is a property of *drawing
from the buffer at all*, and prioritization only re-weights *which* buffer entries are favored, leaving the
decorrelation intact. That confirms the two goals are separable — I can keep the floor's decorrelation while
changing what uniformity I never justified — which is exactly the opening I need.

Uniform replay replays each transition at the frequency it happened to be experienced, regardless of how
much the agent can still learn from it. That is the waste. Most of the buffer, most of the time, is
transitions the network already predicts well — the TD error on them is tiny, so the gradient they produce
is tiny, so an update spent on them barely moves the weights. Meanwhile the transitions that *would* teach
the network the most — the surprising ones, where the prediction is badly wrong — are sampled at exactly
the same rate as the boring ones, and on a game where informative transitions are rare and buried among
redundant ones, the agent spends almost all of its gradient budget on transitions it has nothing left to
learn from. I can make this sharp with a thought experiment. Imagine a buffer of $N$ transitions of which only one
carries the reward signal that needs to propagate, and value information travels one Bellman step per time
that informative transition is replayed. Uniform sampling draws that one transition with probability $1/N$
per sample, so it waits $\sim N$ draws on average just to move the reward back a single step; to propagate
it along a chain of length $L$ takes $\sim N\cdot L$ draws. An oracle that always replays the transition
that most reduces the loss replays the informative frontier *every* time, so it propagates the reward in
$\sim L$ draws — a factor of $N$ faster, and with $N=10^6$ that factor is enormous. Chain several such
rare-signal stages and the gap compounds into the "exponentially faster than uniform" claim: uniform pays
the $1/N$ tax once per stage, the oracle pays it never. That gap is the prize. I cannot build the
oracle — it needs the future loss reductions — but I can approximate it online with a quantity I already
compute.

What is a cheap, already-available proxy for "how much is left to learn from this transition"? The
magnitude of its TD error, $|\delta|$. A large $|\delta|$ means the current network's prediction is far
from its own bootstrap target on that transition — it is surprising under the current value function, and a
gradient step on it will move the weights a lot. A small $|\delta|$ means the transition is already
consistent with what the net believes. And I compute $\delta$ for every sampled transition anyway, as part
of the loss. So: sample transitions in proportion to their TD-error magnitude. Concretely give transition
$i$ a priority $p_i=|\delta_i|+\varepsilon$ (the small $\varepsilon>0$ keeps a zero-error transition from
becoming unsamplable), and sample $i$ with probability $P(i)=p_i^\alpha/\sum_k p_k^\alpha$. The exponent
$\alpha$ interpolates, and it is worth being precise about the two ends. At $\alpha=0$, every
$p_i^\alpha=1$, so $P(i)=1/N$ for all $i$ — exactly uniform replay, the floor's behavior recovered as a
special case, which reassures me the change is a strict generalization rather than a different algorithm.
At $\alpha=1$, sampling is directly proportional to the raw error, the most aggressive stochastic
prioritization. Between them, $\alpha$ tunes *how much* the surprising transitions are favored, and I want a
value strictly inside $(0,1)$ — moderate, around $\alpha\approx0.5$–$0.6$ — for a concrete reason: the
full-strength $\alpha=1$ concentrates sampling so hard on the current high-error transitions that the
effective diversity of each minibatch collapses, and correlated, low-diversity batches are exactly the
pathology replay existed to prevent. A fractional $\alpha$ keeps a strong tilt toward surprise while leaving
enough spread across the buffer that batches stay diverse. So $\alpha$ is not a free knob I set to taste; it
trades data efficiency (higher $\alpha$) against batch diversity (lower $\alpha$), and the safe interior
value is the one that gets most of the efficiency without reintroducing correlation. I deliberately keep it *stochastic* rather than always picking the single highest-error
transition, for two reasons: a greedy argmax would replay a small set over and over and starve the rest of
the buffer (a transition whose error happened to start low would never be revisited and never corrected),
and TD errors are noisy, so a one-off error spike should not let a transition dominate. Stochastic
prioritization with full support fixes both — every transition keeps a non-zero probability.

There is a real fork in *how* to turn $|\delta|$ into a sampling probability, and it is worth walking
because the two natural choices behave differently under the noise I just worried about. One option is
*proportional*: $P(i)\propto(|\delta_i|+\varepsilon)^\alpha$, sample in direct proportion to a power of the
raw error. The other is *rank-based*: sort the buffer by $|\delta|$ and set
$P(i)\propto(1/\text{rank}(i))^\alpha$, so only the *ordering* of the errors matters, not their magnitudes.
Rank-based is the more robust of the two — it is insensitive to outlier errors, because a transition with a
freak $|\delta|=1000$ and one with $|\delta|=10$ that are adjacent in rank get nearly equal priority, so a
single corrupted TD error cannot seize the sampling distribution — and its implied distribution is a fixed
power law independent of the error scale, which is one fewer thing to vary across $57$ games. Proportional
is more faithful to the actual surprise signal (it distinguishes a $|\delta|=10$ from a $|\delta|=11$, which
rank flattens) but is more exposed to error outliers and to the error *scale* drifting over training. Both
are defensible; I take proportional because the small additive $\varepsilon$ and the $\alpha<1$ exponent
already temper the outlier sensitivity, because the reward clipping inherited from the floor bounds how
large a single $|\delta|$ can get so the outlier worry is smaller here than in general, and because it keeps
the priority a direct, interpretable function of the surprise rather than routing it through a sort. The
choice carries risk, but the floor's clipping is doing quiet work to make the more faithful option safe.

But prioritized sampling introduces a bias I have to face honestly, because it changes *what fixed point*
the updates converge to. Uniform SGD drives the weights to where
$\mathbb{E}_{i\sim U}[\delta_i\nabla Q_i]=\frac1N\sum_i\delta_i\nabla Q_i=0$ — the gradient averaged over
the *empirical distribution of the data*. Sampling from $P$ instead drives convergence to
$\mathbb{E}_{i\sim P}[\delta_i\nabla Q_i]=\sum_i P(i)\,\delta_i\nabla Q_i=0$, which weights the
high-priority transitions more and is a *different* solution. I changed which transitions I see, so I
changed the expectation the optimizer solves — and the value I actually want is still the one defined over
the real data distribution, not over my sampling distribution. The textbook correction is importance
sampling: weight each sampled transition's update by $w_i=\big(\frac1N\cdot\frac1{P(i)}\big)^\beta=
(N\,P(i))^{-\beta}$, the ratio of the target (uniform) probability to the sampling probability, raised to a
$\beta$ that controls how much of the bias I correct. At $\beta=1$ the correction is exact:
$\mathbb{E}_{i\sim P}[w_i\delta_i\nabla Q_i]=\sum_i P(i)\frac{1}{N P(i)}\delta_i\nabla Q_i=
\frac1N\sum_i\delta_i\nabla Q_i$ — the uniform expectation is recovered. Use $w_i\delta_i$ in place of
$\delta_i$ in the update, and normalize the weights by $1/\max_l w_l$ over the buffer so they only ever
scale steps *downward* (never blow a step up — pure stability). Check the direction of that normalization
concretely. The raw weight $w_i=(N P(i))^{-\beta}$ is *largest* for the *least*-sampled (lowest-priority)
transitions and smallest for the most-sampled ones — the correction deliberately down-weights the
over-sampled high-priority transitions, which is what removes the bias. Dividing every $w_i$ by the buffer
maximum $\max_l w_l$ sends the largest weight to exactly $1$ and every other weight below $1$, so the update
magnitudes are only ever reduced relative to the un-weighted step, never amplified. That matters because
combining a large importance weight with an already-large prioritized TD error could otherwise produce a
huge gradient exactly on the noisiest transitions; capping the weights at $1$ forecloses that failure mode
at no cost to the correction's *relative* structure, since scaling all weights by a constant leaves the
gradient direction unchanged and only rescales the effective learning rate. And anneal $\beta$ from a $\beta_0$ around $0.4$ up to
$1$ over training: early on the network is changing fast and the whole notion of a fixed point is moot, so
the unbiasedness barely matters and I would rather have the raw speed of aggressive, lightly-corrected
prioritization; near the end, where the updates are settling toward a fixed point and an uncorrected bias
would actually pin the *wrong* solution, I correct it fully at $\beta=1$. The schedule is a deliberate
trade of correctness for speed exactly when speed is what matters: $\beta$ small buys fast early learning at
the cost of a bias that does not yet matter, $\beta\to1$ buys the correct fixed point once the agent is
close enough that the bias would otherwise bite. It is the mirror image of the $\epsilon$ anneal in the
floor — explore-then-exploit there, bias-for-speed-then-correctness here — and like that one it is set
once, on a schedule, for all $57$ games.

A couple of details so this composes cleanly with what I have. I build it on top of the decoupled target
from rung 2 — the TD error whose magnitude I prioritize by is the Double-DQN error
$\delta=R+\gamma\,Q_{\theta^-}(S',\arg\max_a Q_\theta(S',a))-Q_\theta(S,A)$ — because that is my current
best target and prioritizing a biased error would just prioritize the bias. Bookkeeping: a brand-new
transition enters at *maximal* priority so it is guaranteed to be replayed at least once (I have no error
estimate for it yet); after I replay transition $i$ and compute its fresh $\delta_i$, I write the new
priority $p_i\leftarrow|\delta_i|+\varepsilon$ back into the buffer. There is a staleness subtlety here I
should be honest about: a transition's stored priority is the error it had *the last time it was replayed*,
not its error under the current network, and between replays the network has moved on, so the priority is
out of date — a transition the agent has since learned to predict still carries its old high priority until
it is drawn again and corrected downward. This is not a defect I can fully remove, because recomputing every
transition's error every step is exactly the $O(N)$ cost I am avoiding, but it is self-correcting: a
stale-high priority just means the transition gets replayed once more, at which point its now-small error
writes a small priority back and it drops out of contention. The only transitions with *no* estimate at all
are the brand-new ones, which is precisely why they enter at maximal priority — it guarantees each is
replayed soon after arriving, so a real error attaches before the transition ages out, and it gently biases
sampling toward fresh, on-policy experience. And because prioritization raises the
*typical* gradient magnitude, I cut the learning rate by roughly $4\times$ versus uniform so the effective
step size stays in the same range. The reasoning is direct: uniform replay draws transitions of average
error, so a batch's mean $|\delta|$ is the buffer average; prioritized replay deliberately over-draws the
high-error transitions, so a batch's mean $|\delta|$ — and hence the mean gradient magnitude, which is
proportional to it through the TD gradient — is several times larger. The importance weights pull some of
that back down, but not all of it during the anneal when $\beta<1$ leaves the correction partial. So the
net effect is a larger effective step per update, and if I leave the learning rate where the floor had it,
the agent takes systematically bigger steps and risks the instability the floor worked to avoid. Dropping
the learning rate by about a factor of four restores the effective step size to the range the encoder was
stable at — the same "keep the update magnitude in a safe band" logic as the reward and error clipping, now
applied to the shift in gradient scale that prioritization introduces.

One implementation worry: naively, sampling from $P(i)\propto p_i^\alpha$ and maintaining
$\sum_k p_k^\alpha$ over a $10^6$-entry buffer would be $O(N)$ per draw and per update — far too slow at
this scale. Quantify it: with $N=10^6$, an $O(N)$ scan is a million operations for *every* sampled
transition, and I sample a minibatch every training step over $50$M steps, so linear-time sampling would
multiply the per-step cost by $\sim10^6$ and dwarf the conv forward/backward that is supposed to be the
expensive part. Uniform replay never had this problem because a uniform draw is $O(1)$ — the whole cost of
prioritization is the bookkeeping of a non-uniform distribution over a million changing weights, and if that
bookkeeping is linear the method is a non-starter. A **sum-tree** fixes it: a binary tree whose leaves hold $p_i^\alpha$ and whose internal nodes
hold the sum of their children. Updating one leaf is $O(\log N)$ (walk up, fixing sums); sampling is
$O(\log N)$ via prefix-sum descent — draw a uniform value in $[0,\text{total}]$ and walk down, going left
or right by comparing against the left child's sum. For a minibatch of $k$ I stratify: split
$[0,\text{total}]$ into $k$ equal segments and draw one sample per segment, which spreads the batch across
the priority range. A parallel **min-tree** gives $\min_i p_i^\alpha$ in $O(\log N)$ for the weight
normalizer $\max_l w_l$. So the whole scheme is $O(\log N)$ per transition. Put the same numbers back in:
$\log_2(10^6)\approx20$, so each sample or priority update is about $20$ operations instead of a million — a
$50{,}000\times$ reduction that turns the linear non-starter into a cost genuinely negligible against a conv
forward/backward. The sum-tree is not an optimization detail I could skip; it is the thing that makes
non-uniform sampling over a $10^6$ buffer feasible at all, and it is why prioritization can be a practical
method rather than a theoretical one.

Now the bar. Unlike noisy nets, this touches the data efficiency of *every* game: every game has redundant
transitions the agent over-replays and rare informative ones it under-replays, so reallocating the gradient
budget toward what is still learnable should lift the broad middle of the distribution, not just the tails
— which is exactly what moves a median. The risk is the bias: if the importance-sampling correction is
mis-set the agent could converge to a subtly wrong value function and lose on the games where the fixed
point matters, so the $\beta$-anneal is doing real work and I am watching for instability, not just the
headline. But if the analysis is right, this is the first axis since the decoupled target that should give
a *broad* lift rather than a tail one. The contrast with noisy nets is the whole argument: parameter noise
helped a minority of games with a discovery problem and left the median at $118\%$; prioritized replay
reallocates the gradient budget on *every* game, because every game has redundant transitions it
over-replays and rare informative ones it under-replays, so the $29$th-ranked game benefits just as the
tails do. That is the profile that moves a median rather than a tail, so I expect a clear jump above
$118\%$ — plausibly the largest single step since the $79\to117$ climb of the decoupled target — and I will
read a broad lift as confirmation that data efficiency, not exploration, was the binding constraint on the
typical game. A near-flat result, by contrast, would falsify the "every game wastes budget on redundant
transitions" story and tell me the typical game's ceiling lies in some part of the agent I have not yet
re-examined.
