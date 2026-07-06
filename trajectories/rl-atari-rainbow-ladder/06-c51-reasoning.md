Dueling reached 151%, an $11$-point lift over the $140\%$ prioritized-replay baseline, and I have now
improved every axis of the recipe except the one I deliberately kept for last because it is the deepest: the
value *object* itself. Tally the ladder so far: $79\to117\to118\to140\to151$, and read the pattern in the
gaps. The three broad, every-game fixes — decoupled target ($+38$), prioritized replay ($+22$), dueling head
($+11$) — carried almost the entire climb, while the one narrow fix (noisy exploration, $+1$) barely
registered on the median. The broad gains are also *shrinking*: $38$, then $22$, then $11$, roughly halving
each time. That is what I would expect if I have been picking off the biggest structural defects first and
am now into diminishing returns on the recipe. Which is precisely the argument for finally spending the
expensive lever I have been saving — the recipe tweaks are running out of room, so the next real jump has to
come from changing *what* is learned, not *how* it is learned. Through five rungs the thing I learn has
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
$\mathcal T^\pi$ is a $\gamma$-contraction, and I can see why from the two covariance properties of the
$1$-dimensional Wasserstein metric, which via the quantile coupling is
$d_p(X,Y)=\big(\int_0^1|F_X^{-1}(u)-F_Y^{-1}(u)|^p\,du\big)^{1/p}$. Shift: adding a constant $a$ sends
$F_X^{-1}\to a+F_X^{-1}$, so the difference $(a+F_X^{-1})-(a+F_Y^{-1})=F_X^{-1}-F_Y^{-1}$ is unchanged and
$d_p(a+X,a+Y)=d_p(X,Y)$. Scale: multiplying by $\gamma$ sends $F_X^{-1}\to\gamma F_X^{-1}$, so the
difference scales by $\gamma$ and $d_p(\gamma X,\gamma Y)=|\gamma|\,d_p(X,Y)$. Now apply both to
$\mathcal T^\pi Z=R+\gamma Z(X',A')$: the common shift by $R$ cancels (both sides get the same $R$), the
scaling by $\gamma$ pulls a factor $\gamma$ out front, so $\bar d_p(\mathcal T^\pi Z_1,\mathcal T^\pi Z_2)
\le\gamma\,\bar d_p(Z_1,Z_2)$ and Banach gives a unique fixed point $Z^\pi$. The metric subtlety matters and I want it on
the record with a counterexample: $\mathcal T^\pi$ is *not* a contraction in KL, total variation, or
Kolmogorov distance. Take two return distributions with disjoint support — $Z_1$ uniform on $[0,\epsilon]$,
$Z_2$ uniform on $[1,1+\epsilon]$ for tiny $\epsilon$. Their total variation is $1$ (maximal — disjoint
supports never overlap) and their KL is infinite. Apply the scaling half of $\mathcal T^\pi$ with
$\gamma<1$: the supports become $[0,\gamma\epsilon]$ and $[\gamma,\gamma+\gamma\epsilon]$, still disjoint, so
the total variation is *still* $1$ and the KL is *still* infinite — no contraction at all. But the
Wasserstein distance went from about $1$ to about $\gamma$: it felt the supports slide closer under the
$\gamma$-shrink. That is the whole point — TV, KL, and Kolmogorov are "vertical" overlap distances, blind to
how far apart the supports are, so they cannot register the horizontal $\gamma$-contraction; only
Wasserstein, which measures horizontal transport cost, has the scaling property that makes $\mathcal T^\pi$
a contraction. (For *control*, with $\pi$ greedy on the mean, the operator keeps the *mean*
contracting toward $Q^*$ but is not a contraction in any distribution metric and may have no fixed
distribution — which is a reason to use a smooth, averaging, gradient update rather than a hard greedy one,
not a reason to abandon the approach.)

So Wasserstein is the metric the operator actually contracts in. The natural idea would be to *train* by
minimizing Wasserstein distance to the bootstrapped target distribution. Here I hit a wall, and it is a
real one: the Wasserstein loss *cannot be minimized from sampled transitions by SGD*. The bootstrapped
target is a mixture over the (sampled) next states, and for a mixture $P=\mathbb{E}_I[P_I]$ one has
$d_p(P,Q)\le\mathbb{E}_I[d_p(P_I,Q)]$ with the inequality generally *strict*. A tiny example makes the
strictness — and its danger — vivid. Let the true target mixture be $P=\tfrac12\delta_0+\tfrac12\delta_2$
(next state gives return $0$ or $2$, equally likely) and suppose my prediction is already *exactly right*,
$Q=P$, so the true distance $d_1(P,Q)=0$. But the sampled loss averages the distance to each *individual*
sample: when I draw the $0$-sample, $d_1(\delta_0,Q)=\tfrac12\cdot0+\tfrac12\cdot2=1$, and likewise the
$2$-sample gives $1$, so $\mathbb{E}_I[d_1(P_I,Q)]=1>0=d_1(P,Q)$. The sampled loss is minimized not at the
correct $Q=P$ but by chasing each sample, so its gradient at the right answer is *nonzero* and points away
from the truth — the estimator is biased, and following it would pull $Q$ off the correct distribution. So
the gradient of the sampled Wasserstein distance is a biased estimate of the gradient of the true one:
$\nabla_Q d_p(P_I,Q)$ averaged over samples is not $\nabla_Q d_p(P,Q)$. I cannot minimize the right metric
with the only thing I have (sampled transitions and SGD). So I need a *representation* of the distribution
and a *loss* that I *can* minimize from samples, accepting that the loss will not be Wasserstein itself.

What representation should the distribution take? The tempting simple choice is a *parametric* one — have
the head output a mean and a variance and model $Z(x,a)$ as a Gaussian. It is compact (two numbers per
action) and its Bellman update is closed-form. But it contradicts the very reason I am doing this: I argued
that $Z$ on Atari is *multimodal* — a game where an action leads to a big reward with some probability and
nothing otherwise has a two-peaked return, and a Gaussian can only ever represent one bump. Forcing a
unimodal Gaussian onto a bimodal return would smear the two peaks into a single wide blob centered between
them, which is arguably a *worse* training signal than the mean I started with, because it asserts
confidence in a value the return never actually takes. So a parametric unimodal family throws away exactly
the structure I want to capture. The alternative is a *nonparametric* representation flexible enough to be
multimodal, and the simplest such thing is a histogram: fix a grid of return values and let the network put
arbitrary probability mass on each. That can represent any shape up to the grid resolution, at the cost of
having to *choose the grid* — which is the tradeoff I will own explicitly.

Pick the representation, then. Represent $Z(x,a)$ as a discrete distribution on a *fixed* grid of $N$
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
Trace the projection on one atom to see it is just linear interpolation of mass. Suppose an atom carrying
probability $0.6$ gets shifted to $\hat{\mathcal T}z_j=1.5$, and my grid has atoms at $\dots,1.2,1.6,\dots$
with $\Delta z=0.4$. Then $1.5$ falls between the atom at $z_l=1.2$ and the atom at $z_u=1.6$, a fraction
$(1.5-1.2)/0.4=0.75$ of the way up. Linear interpolation assigns the *nearer* atom the larger share: the
lower atom $z_l=1.2$ receives $(1-0.75)\times0.6=0.15$ and the upper atom $z_u=1.6$ receives
$0.75\times0.6=0.45$. The two shares sum to $0.6$, so no probability mass is created or destroyed, and the
*expected value* of the split, $0.15\times1.2+0.45\times1.6=0.18+0.72=0.90=0.6\times1.5$, equals the mass
times the true shifted location — the projection preserves the mean exactly while snapping the support back
onto the grid. That mean-preservation is what keeps the categorical representation closed under the Bellman
update without distorting the quantity I ultimately act on. This $\Phi$ produces a target distribution *on
my grid*, which I can now compare to my prediction. And the
comparison that *is* minimizable from samples is the cross-entropy: train with the cross-entropy term of
$D_{\mathrm{KL}}\big(\Phi\hat{\mathcal T}Z_{\tilde\theta}(x,a)\,\big\|\,Z_\theta(x,a)\big)$, i.e.
$\mathcal L=-\sum_i m_i\log p_i(x,a)$ with $m=\Phi\hat{\mathcal T}Z_{\tilde\theta}(x,a)$ the projected
target probabilities. This is exactly *multiclass classification over the atoms* — a well-behaved,
SGD-friendly, sample-unbiased loss. And I can say precisely *why* it is unbiased where Wasserstein was not,
which is the crux of the whole trade. The cross-entropy $-\sum_i m_i\log p_i$ is *linear* in the target
masses $m$, so when $m$ is itself an average over sampled next states, $m=\mathbb{E}_I[m^{(I)}]$, the
expectation passes straight through:
$\mathbb{E}_I\big[-\sum_i m_i^{(I)}\log p_i\big]=-\sum_i\mathbb{E}_I[m_i^{(I)}]\log p_i=-\sum_i m_i\log p_i$.
The gradient of the sampled cross-entropy is therefore an *unbiased* estimate of the gradient of the true
(mixture) cross-entropy — exactly the property the Wasserstein loss lacked, because Wasserstein is a
nonlinear (convex) function of the distribution and the expectation does *not* pass through it, which is
what my $Q=P$ example exposed. So the linearity of cross-entropy in the target is not incidental; it is the
whole reason I can minimize this loss from samples and could not minimize Wasserstein from samples. I am
paying for that with a metric mismatch — I contract in Wasserstein but descend in KL-after-projection — and
the projection $\Phi$ is what makes the mismatch survivable, because it keeps the representation closed
under the Bellman update so the KL is always taken between two grid distributions. I have traded the metric the operator contracts in (Wasserstein,
unminimizable from samples) for a metric I can actually optimize (KL after projection), and the
projection $\Phi$ is what keeps the targets on the grid so the KL is well-defined. It is a deliberate
mismatch, eyes open, and it works because the projection keeps the representation closed under the Bellman
update.

The acting and the rest stay minimal so this is a clean single-axis change over the dueling rung. I act
mean-greedily as before: read the mean of each action's distribution, $Q(x,a)=\sum_i z_i\,p_i(x,a)$, and
take its $\arg\max$ — I keep the *same risk-neutral policy* so any change in median HNS is attributable to
the richer training signal and not to a changed objective. I have the *option* now of a risk-sensitive
policy — the distribution exposes variance and tails I could act on — but taking it would confound "learning
the distribution helps" with "acting on risk helps," so I deliberately do not, and hold the policy fixed at
the mean. This makes the whole rung a strict generalization of the scalar agent: collapse the distribution
to a single atom ($N=1$) and $\sum_i z_i p_i$ is just that atom's value, the mean, and I recover exactly the
point-estimate head I started from. So I am not replacing the agent, I am giving its value object more
degrees of freedom while reading off the same scalar to act — the cleanest possible single-axis change to
the representation. The bootstrap action $a^\star$ is greedy on the
target net's mean; the categorical head replaces the scalar head but the conv torso, the replay buffer, the
$\epsilon$-noise, and the periodic target sync are exactly as the previous rungs left them. $N=51$ atoms on
$[-10,10]$ is the working resolution — the spacing is $\Delta z=(V_{\max}-V_{\min})/(N-1)=20/50=0.4$, so
each atom is a $0.4$-wide bin of return, fine enough to resolve a bimodal or skewed shape but coarse enough
that the head emits only $51|\mathcal A|$ logits rather than an unwieldy number (this is the "C51" of the
name). The choice of $N$ is a genuine tradeoff: too few atoms and $\Delta z$ grows until the categorical
distribution cannot represent the shape it is meant to learn — in the limit $N=1$ it degenerates to one atom
and I am back to a point estimate — while too many balloons the head and starves each atom of probability
mass so it trains poorly; $51$ sits in the usable middle. The rare return that runs past $[-10,10]$ is
clamped onto the endpoint atom rather than lost, a coarse treatment of the tail that is exactly the
fixed-grid limitation I flag below. The per-atom projection is $O(N)$ and cheap.

Now the bar. This is the most fundamental change in the ladder — it replaces *what the agent learns*, not
how it learns a mean — and the reason to expect a large, broad gain is the denser supervisory signal:
*every* game's return is a non-trivial distribution, so on *every* game the network now gets many
constraints per update where before it got one, which should shape a better representation across the whole
suite, not a minority. So I expect this to clear 151% decisively and to be the single strongest component —
the best the ladder has found from changing *one* thing about the DQN floor. And I can sharpen that against
the trend I noticed at the top: the broad recipe fixes were giving diminishing returns, $38\to22\to11$, so
if distributional learning were merely another recipe tweak I would expect a gain in that shrinking
sequence — around the dueling step or smaller, a modest lift over $151$ and no more. If instead it breaks
the trend and clears $151$ by a wide margin, that is evidence the change is categorically different — a new *object*, not a better estimate of the old
one — because only a change of kind would arrest the diminishing-returns pattern that every recipe tweak
has followed. That is the falsifiable edge: a modest gain in line with the shrinking-steps trend would say
distributional learning is just one more incremental fix; a large gain that bucks the trend would confirm
the denser supervisory signal is doing something the mean-estimators could not. The honest caveat is the fixed
grid: $V_{\min},V_{\max}$ and the $N=51$ resolution are hand-set, and a game whose returns run outside the
support or whose distribution has fine structure between atoms will be served coarsely — a real limit, but
one that trades against a much richer learning signal everywhere else. So this is the deepest single change
I have made to the floor: not a better way to estimate the mean, but a replacement of the estimated object
itself, from a scalar to a whole categorical distribution learned by classification over atoms. That is why
I expect it to be the strongest single component the ladder has found — the most that changing one thing
about the DQN floor can buy on the median game.
