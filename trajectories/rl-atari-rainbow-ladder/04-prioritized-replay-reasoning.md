Noisy nets cleared the bar by a hair — 118% against 117% — and the flatness is the message: exploration
was not what capped the median game. The hard-exploration titles where parameter noise shines are a
minority; the typical game already saw enough reward signal under $\epsilon$-greedy and is bottlenecked
somewhere else. So I want an axis that touches *every* game's learning, not a minority's discovery, and the
one I have been deferring is sitting in plain sight: how the replay buffer hands transitions to the update.
The floor draws minibatches *uniformly at random* from the buffer, and that is a choice I never justified —
I adopted it only to decorrelate samples. Decorrelation does not require *uniformity*; it only requires not
training on the live stream. So uniform sampling is free to be wrong, and I should ask what it costs.

Uniform replay replays each transition at the frequency it happened to be experienced, regardless of how
much the agent can still learn from it. That is the waste. Most of the buffer, most of the time, is
transitions the network already predicts well — the TD error on them is tiny, so the gradient they produce
is tiny, so an update spent on them barely moves the weights. Meanwhile the transitions that *would* teach
the network the most — the surprising ones, where the prediction is badly wrong — are sampled at exactly
the same rate as the boring ones, and on a game where informative transitions are rare and buried among
redundant ones, the agent spends almost all of its gradient budget on transitions it has nothing left to
learn from. I can make this sharp with a thought experiment: on a controlled needle-in-a-haystack task,
where only a handful of transitions carry the signal, an oracle that replays transitions in the order that
most reduces the loss learns *exponentially* faster than uniform. That gap is the prize. I cannot build the
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
$\alpha$ interpolates: $\alpha=0$ is back to uniform, larger $\alpha$ is greedier toward high-error
transitions. I deliberately keep it *stochastic* rather than always picking the single highest-error
transition, for two reasons: a greedy argmax would replay a small set over and over and starve the rest of
the buffer (a transition whose error happened to start low would never be revisited and never corrected),
and TD errors are noisy, so a one-off error spike should not let a transition dominate. Stochastic
prioritization with full support fixes both — every transition keeps a non-zero probability.

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
scale steps *downward* (never blow a step up — pure stability). And anneal $\beta$ from a $\beta_0<1$ up to
$1$ over training: early on the network is changing fast and the whole notion of a fixed point is moot, so
the unbiasedness barely matters and I would rather have the raw speed of aggressive prioritization;
near the end, where the bias would actually pin the wrong solution, correct it fully.

A couple of details so this composes cleanly with what I have. I build it on top of the decoupled target
from rung 2 — the TD error whose magnitude I prioritize by is the Double-DQN error
$\delta=R+\gamma\,Q_{\theta^-}(S',\arg\max_a Q_\theta(S',a))-Q_\theta(S,A)$ — because that is my current
best target and prioritizing a biased error would just prioritize the bias. Bookkeeping: a brand-new
transition enters at *maximal* priority so it is guaranteed to be replayed at least once (I have no error
estimate for it yet); after I replay transition $i$ and compute its fresh $\delta_i$, I write the new
priority $p_i\leftarrow|\delta_i|+\varepsilon$ back into the buffer. And because prioritization raises the
*typical* gradient magnitude (I am deliberately sampling the big-error transitions), I cut the learning
rate by roughly $4\times$ versus uniform so the effective step size stays in the same range.

One implementation worry: naively, sampling from $P(i)\propto p_i^\alpha$ and maintaining
$\sum_k p_k^\alpha$ over a $10^6$-entry buffer would be $O(N)$ per draw and per update — far too slow at
this scale. A **sum-tree** fixes it: a binary tree whose leaves hold $p_i^\alpha$ and whose internal nodes
hold the sum of their children. Updating one leaf is $O(\log N)$ (walk up, fixing sums); sampling is
$O(\log N)$ via prefix-sum descent — draw a uniform value in $[0,\text{total}]$ and walk down, going left
or right by comparing against the left child's sum. For a minibatch of $k$ I stratify: split
$[0,\text{total}]$ into $k$ equal segments and draw one sample per segment, which spreads the batch across
the priority range. A parallel **min-tree** gives $\min_i p_i^\alpha$ in $O(\log N)$ for the weight
normalizer $\max_l w_l$. So the whole scheme is $O(\log N)$ per transition — negligible against a conv
forward/backward.

Now the bar. Unlike noisy nets, this touches the data efficiency of *every* game: every game has redundant
transitions the agent over-replays and rare informative ones it under-replays, so reallocating the gradient
budget toward what is still learnable should lift the broad middle of the distribution, not just the tails
— which is exactly what moves a median. The risk is the bias: if the importance-sampling correction is
mis-set the agent could converge to a subtly wrong value function and lose on the games where the fixed
point matters, so the $\beta$-anneal is doing real work and I am watching for instability, not just the
headline. But if the analysis is right, this is the first axis since the decoupled target that should give
a *broad* lift rather than a tail one — I expect a clear jump above 118%, the largest single step since
117%.
