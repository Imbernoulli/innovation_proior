Dueling reached 151%, and I have now improved every axis of the recipe except the one I deliberately kept
for last because it is the deepest: the value *object* itself. Through five rungs the thing I learn has
stayed a single scalar per action — the decoupled target made that scalar less biased, the dueling head
learned its state-value part from more data, but it is still, in every case, one number: the *mean* of the
return. Every rung so far has been a better way to estimate a mean. The question I have been postponing is
whether estimating the mean is the right problem at all.

Why am I only learning the mean? Because I train the agent to maximize *expected* return, so I learn
$Q(x,a)=\mathbb{E}[Z(x,a)]$, one number. But $Z(x,a)$ — the actual discounted return from taking $a$ in
$x$ and following the policy — is a *random variable*, and on Atari it is a wildly non-trivial one:
stochastic in the rewards, in the transitions, and in a policy that is itself moving during training, so
$Z$ is typically multimodal, skewed, heavy-tailed. Collapsing all of that into its mean throws away
everything about its shape, and a single scalar target with no notion of its own spread has to absorb all
that stochasticity into one number — which is exactly the wobble I have watched the scalar critic produce.
Here is the concrete reason to care even though my policy is still mean-greedy: a *distribution* is a far
richer training signal than a mean. If I learn the whole law of $Z$, every transition tells the network not
just "the return here is about $v$" but "the return here is *this shape* — this much mass here, this tail
there." That is many more constraints per update, a denser supervisory signal that shapes the
representation better, and it tends to help even when I only ever read off the mean to act. So the move is:
stop learning the mean of the return and learn its whole distribution.

Can I even bootstrap a distribution the way I bootstrap a mean? The mean version rests on the Bellman
equation $Q=\mathbb{E}[R+\gamma Q']$. There is a distributional analogue, an equality *in distribution*:
$$Z(x,a)\overset{D}{=}R(x,a)+\gamma\,Z(X',A'),\qquad X'\sim P(\cdot\mid x,a),\;A'\sim\pi(\cdot\mid X').$$
This says the return random variable equals (in law) the immediate reward plus $\gamma$ times the
next-state return random variable. Write the operator $\mathcal T^\pi Z(x,a)\overset{D}{:=}R+\gamma
Z(X',A')$. Is iterating it stable? For *policy evaluation* it is a contraction — but I have to be careful in
*which metric*. In the maximal Wasserstein metric $\bar d_p(Z_1,Z_2)=\sup_{x,a}d_p(Z_1(x,a),Z_2(x,a))$,
$\mathcal T^\pi$ is a $\gamma$-contraction: the constant shift by $R$ cancels (Wasserstein is
shift-covariant) and the scaling by $\gamma$ pulls a factor $\gamma$ out front (Wasserstein is
scale-covariant by $|\gamma|$), so $\bar d_p(\mathcal T^\pi Z_1,\mathcal T^\pi Z_2)\le\gamma\,\bar
d_p(Z_1,Z_2)$ and Banach gives a unique fixed point $Z^\pi$. The metric subtlety matters and I want it on
the record: $\mathcal T^\pi$ is *not* a contraction in KL, total variation, or Kolmogorov distance — those
are "vertical" overlap distances, blind to the horizontal $\gamma$-shrink of the support; only Wasserstein
has the scaling property. (For *control*, with $\pi$ greedy on the mean, the operator keeps the *mean*
contracting toward $Q^*$ but is not a contraction in any distribution metric and may have no fixed
distribution — which is a reason to use a smooth, averaging, gradient update rather than a hard greedy one,
not a reason to abandon the approach.)

So Wasserstein is the metric the operator actually contracts in. The natural idea would be to *train* by
minimizing Wasserstein distance to the bootstrapped target distribution. Here I hit a wall, and it is a
real one: the Wasserstein loss *cannot be minimized from sampled transitions by SGD*. The bootstrapped
target is a mixture over the (sampled) next states, and for a mixture $P=\mathbb{E}_I[P_I]$ one has
$d_p(P,Q)\le\mathbb{E}_I[d_p(P_I,Q)]$ with the inequality generally *strict*, so the gradient of the
sampled Wasserstein distance is a *biased* estimate of the gradient of the true Wasserstein distance —
$\nabla_Q d_p(P_I,Q)$ averaged over samples is not $\nabla_Q d_p(P,Q)$. I cannot minimize the right metric
with the only thing I have (sampled transitions and SGD). So I need a *representation* of the distribution
and a *loss* that I *can* minimize from samples, accepting that the loss will not be Wasserstein itself.

Pick the representation first. Represent $Z(x,a)$ as a discrete distribution on a *fixed* grid of $N$
"atoms" $z_i=V_{\min}+i\,\Delta z$, $\Delta z=\frac{V_{\max}-V_{\min}}{N-1}$, with the network emitting a
softmax over the atoms per action: $Z_\theta(x,a)=\sum_i p_i(x,a)\,\delta_{z_i}$,
$p_i=\operatorname{softmax}(\theta_i(x,a))$. So the head, instead of one scalar per action, emits $N$
logits per action that softmax into a categorical distribution over a shared value grid. (I have to *set*
$V_{\min},V_{\max}$ up front — that is the cost of a fixed grid; with reward clipping and $\gamma=0.99$ on
Atari, $[-10,10]$ comfortably covers the clipped discounted return.)

Now the loss, and here is the move that makes it trainable. Apply the distributional Bellman update to the
target network's distribution: shift and scale each target atom, $\hat{\mathcal T}z_j=r+\gamma z_j$. The
problem is that the shifted atoms $r+\gamma z_j$ do *not* land on my fixed grid $\{z_i\}$ anymore. So
*project* them back: each shifted atom's probability mass $p_j(x',\pi(x'))$ is split between the two nearest
grid atoms by linear interpolation (and clamped to $[V_{\min},V_{\max}]$ at the ends),
$$\big(\Phi\hat{\mathcal T}Z\big)_i=\sum_j\Big[1-\frac{\big|[r+\gamma z_j]_{V_{\min}}^{V_{\max}}-z_i\big|}{\Delta z}\Big]_0^1 p_j(x',\pi(x')).$$
This $\Phi$ produces a target distribution *on my grid*, which I can now compare to my prediction. And the
comparison that *is* minimizable from samples is the cross-entropy: train with the cross-entropy term of
$D_{\mathrm{KL}}\big(\Phi\hat{\mathcal T}Z_{\tilde\theta}(x,a)\,\big\|\,Z_\theta(x,a)\big)$, i.e.
$\mathcal L=-\sum_i m_i\log p_i(x,a)$ with $m=\Phi\hat{\mathcal T}Z_{\tilde\theta}(x,a)$ the projected
target probabilities. This is exactly *multiclass classification over the atoms* — a well-behaved,
SGD-friendly, sample-unbiased loss. I have traded the metric the operator contracts in (Wasserstein,
unminimizable from samples) for a metric I can actually optimize (KL after projection), and the
projection $\Phi$ is what keeps the targets on the grid so the KL is well-defined. It is a deliberate
mismatch, eyes open, and it works because the projection keeps the representation closed under the Bellman
update.

The acting and the rest stay minimal so this is a clean single-axis change over the dueling rung. I act
mean-greedily as before: read the mean of each action's distribution, $Q(x,a)=\sum_i z_i\,p_i(x,a)$, and
take its $\arg\max$ — I keep the *same risk-neutral policy* so any change in median HNS is attributable to
the richer training signal and not to a changed objective. The bootstrap action $a^\star$ is greedy on the
target net's mean; the categorical head replaces the scalar head but the conv torso, the replay buffer, the
$\epsilon$-noise, and the periodic target sync are exactly as the previous rungs left them. $N=51$ atoms on
$[-10,10]$ is the working resolution — enough to resolve the shape of the return distribution without an
unwieldy head (this is the "C51" of the name). The per-atom projection is $O(N)$ and cheap.

Now the bar. This is the most fundamental change in the ladder — it replaces *what the agent learns*, not
how it learns a mean — and the reason to expect a large, broad gain is the denser supervisory signal:
*every* game's return is a non-trivial distribution, so on *every* game the network now gets many
constraints per update where before it got one, which should shape a better representation across the whole
suite, not a minority. So I expect this to clear 151% decisively and to be the single strongest component —
the best the ladder has found from changing *one* thing about the DQN floor. The honest caveat is the fixed
grid: $V_{\min},V_{\max}$ and the $N=51$ resolution are hand-set, and a game whose returns run outside the
support or whose distribution has fine structure between atoms will be served coarsely — a real limit, but
one that trades against a much richer learning signal everywhere else. If this is the strongest single
component, the obvious last question is whether the six improvements I have found — decoupled target,
learned exploration, prioritized replay, dueling head, distributional object — attack *independent* enough
weaknesses that combining all of them in one agent compounds rather than collides. That is the finale.
