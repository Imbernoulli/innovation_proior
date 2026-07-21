Dueling reached 151%, an $11$-point lift over the $140\%$ prioritized-replay baseline. Tally the climb so
far — $79\to117\to118\to140\to151$ — and read the gaps. The three broad, every-game fixes (decoupled target
$+38$, prioritized replay $+22$, dueling head $+11$) carried almost the entire climb, while the one narrow
fix (noisy exploration $+1$) barely registered. The broad gains are also *shrinking*: $38$, $22$, $11$,
roughly halving each time — what I would expect if I have been picking off the biggest structural defects
first and am now into diminishing returns on the recipe. Which is precisely the argument for finally spending
the expensive lever I have been saving: the recipe tweaks are running out of room, so the next real jump has
to come from changing *what* is learned, not *how*. Through five changes the thing I learn has stayed a
single scalar per action — the decoupled target made it less biased, the dueling head learned its
state-value part from more data, but it is still one number: the *mean* of the return. The question I have
been postponing is whether estimating the mean is the right problem at all.

Why only the mean? Because I train the agent to maximize *expected* return, so I learn $Q(x,a)=\mathbb{E}[Z(x,a)]$,
one number. But $Z(x,a)$ — the actual discounted return from taking $a$ in $x$ and following the policy — is
a *random variable*, and on Atari a wildly non-trivial one: stochastic in the rewards, the transitions, and a
policy that is itself moving during training, so $Z$ is typically multimodal, skewed, heavy-tailed.
Collapsing all of that into its mean throws away everything about its shape, and a single scalar target with
no notion of its own spread has to absorb all that stochasticity into one number. Here is the concrete reason
to care even though my policy stays mean-greedy: a *distribution* is a far richer training signal. If I learn
the whole law of $Z$, every transition tells the network not just "the return here is about $v$" but "the
return here is *this shape* — this much mass here, this tail there" — many more constraints per update, a
denser supervisory signal that shapes the representation better, and it tends to help even when I only read
off the mean to act. So the move is: stop learning the mean of the return and learn its whole distribution.

Can I even bootstrap a distribution the way I bootstrap a mean? The mean version rests on $Q=\mathbb{E}[R+\gamma Q']$.
There is a distributional analogue, an equality *in distribution*:
$$Z(x,a)\overset{D}{=}R(x,a)+\gamma\,Z(X',A'),\qquad X'\sim P(\cdot\mid x,a),\;A'\sim\pi(\cdot\mid X').$$
Write the operator $\mathcal T^\pi Z(x,a)\overset{D}{:=}R+\gamma Z(X',A')$. Is iterating it stable? For
*policy evaluation* it is a $\gamma$-contraction, but in *which* metric matters. In the maximal Wasserstein
metric $\bar d_p(Z_1,Z_2)=\sup_{x,a}d_p(Z_1(x,a),Z_2(x,a))$ it contracts, and the two covariance properties
of the $1$-dimensional Wasserstein metric,
$d_p(X,Y)=\big(\int_0^1|F_X^{-1}(u)-F_Y^{-1}(u)|^p\,du\big)^{1/p}$, show why. Shift: adding a constant $a$
sends $F_X^{-1}\to a+F_X^{-1}$, so the difference is unchanged and $d_p(a+X,a+Y)=d_p(X,Y)$. Scale:
multiplying by $\gamma$ sends $F_X^{-1}\to\gamma F_X^{-1}$, so $d_p(\gamma X,\gamma Y)=|\gamma|\,d_p(X,Y)$.
Apply both to $\mathcal T^\pi Z=R+\gamma Z(X',A')$: the common shift by $R$ cancels, the scaling by $\gamma$
pulls a factor $\gamma$ out front, so $\bar d_p(\mathcal T^\pi Z_1,\mathcal T^\pi Z_2)\le\gamma\,\bar d_p(Z_1,Z_2)$
and Banach gives a unique fixed point. The metric choice is not free, and a counterexample pins it:
$\mathcal T^\pi$ is *not* a contraction in KL, total variation, or Kolmogorov. Take $Z_1$ uniform on
$[0,\epsilon]$ and $Z_2$ uniform on $[1,1+\epsilon]$ — disjoint supports, so TV $=1$ and KL $=\infty$. Apply
the $\gamma$-scaling: the supports become $[0,\gamma\epsilon]$ and $[\gamma,\gamma+\gamma\epsilon]$, still
disjoint, so TV is *still* $1$ and KL *still* $\infty$ — no contraction. But the Wasserstein distance went
from about $1$ to about $\gamma$: it felt the supports slide closer. TV, KL, and Kolmogorov are "vertical"
overlap distances, blind to how far apart the supports are; only Wasserstein measures horizontal transport
cost and so registers the $\gamma$-contraction. (For *control*, with $\pi$ greedy on the mean, the operator
keeps the *mean* contracting toward $Q^*$ but is not a contraction in any distribution metric — a reason to
use a smooth, averaging, gradient update rather than a hard greedy one, not to abandon the approach.)

So Wasserstein is the metric the operator contracts in. The natural idea is to *train* by minimizing
Wasserstein distance to the bootstrapped target — and here I hit a real wall: the Wasserstein loss *cannot be
minimized from sampled transitions by SGD*. The bootstrapped target is a mixture over the sampled next
states, and for a mixture $P=\mathbb{E}_I[P_I]$ one has $d_p(P,Q)\le\mathbb{E}_I[d_p(P_I,Q)]$, generally
*strict*. A tiny example makes the danger vivid. Let the true target be $P=\tfrac12\delta_0+\tfrac12\delta_2$
and suppose my prediction is already *exactly right*, $Q=P$, so $d_1(P,Q)=0$. But the sampled loss averages
the distance to each individual sample: the $0$-sample gives $d_1(\delta_0,Q)=\tfrac12\cdot0+\tfrac12\cdot2=1$,
the $2$-sample likewise $1$, so $\mathbb{E}_I[d_1(P_I,Q)]=1>0$. The sampled loss is minimized not at the
correct $Q=P$ but by chasing each sample, so its gradient at the right answer is *nonzero* and points away
from the truth. I cannot minimize the right metric with the only thing I have. So I need a *representation*
plus a *loss* I *can* minimize from samples, accepting the loss will not be Wasserstein.

What representation? The tempting simple choice is *parametric* — output a mean and variance and model
$Z(x,a)$ as a Gaussian, compact with a closed-form Bellman update. But it contradicts the reason I am doing
this: I argued $Z$ on Atari is *multimodal* — an action leading to a big reward with some probability and
nothing otherwise has a two-peaked return — and a Gaussian can only ever be one bump, smearing two peaks into
a wide blob centered between them, which is arguably a *worse* signal than the mean because it asserts
confidence in a value the return never takes. So a unimodal family throws away exactly the structure I want.
The alternative is *nonparametric* and flexible enough to be multimodal, the simplest being a histogram: fix
a grid of return values and let the network put arbitrary mass on each. That represents any shape up to grid
resolution, at the cost of having to *choose the grid* — the tradeoff I will own explicitly.

Represent $Z(x,a)$ as a discrete distribution on a *fixed* grid of $N$ atoms $z_i=V_{\min}+i\,\Delta z$,
$\Delta z=\frac{V_{\max}-V_{\min}}{N-1}$, with the network emitting a softmax over atoms per action:
$Z_\theta(x,a)=\sum_i p_i(x,a)\,\delta_{z_i}$, $p_i=\operatorname{softmax}(\theta_i(x,a))$. The head emits $N$
logits per action instead of one scalar. I have to *set* $V_{\min},V_{\max}$ up front — the cost of a fixed
grid; with reward clipping and $\gamma=0.99$, $[-10,10]$ comfortably covers the clipped discounted return.

Now the loss, and the move that makes it trainable. Apply the distributional Bellman update to the target
network's distribution: shift and scale each target atom, $\hat{\mathcal T}z_j=r+\gamma z_j$. The shifted
atoms no longer land on my grid, so *project* them back: each shifted atom's mass $p_j(x',\pi(x'))$ is split
between the two nearest grid atoms by linear interpolation (clamped to $[V_{\min},V_{\max}]$ at the ends),
$$\big(\Phi\hat{\mathcal T}Z\big)_i=\sum_j\Big[1-\frac{\big|[r+\gamma z_j]_{V_{\min}}^{V_{\max}}-z_i\big|}{\Delta z}\Big]_0^1 p_j(x',\pi(x')).$$
The projection is just linear interpolation of mass, and it preserves the mean exactly. An atom of mass $0.6$
shifted to $1.5$, with grid atoms at $z_l=1.2$ and $z_u=1.6$ ($\Delta z=0.4$), lands a fraction
$(1.5-1.2)/0.4=0.75$ up; the lower atom receives $0.25\times0.6=0.15$ and the upper $0.75\times0.6=0.45$,
summing to $0.6$ (no mass created or destroyed), with expected value $0.15\times1.2+0.45\times1.6=0.90=0.6\times1.5$
— the mass times its true shifted location. That mean-preservation keeps the categorical representation
closed under the Bellman update without distorting the quantity I ultimately act on. The comparison that *is*
minimizable from samples is the cross-entropy: train with the cross-entropy term of
$D_{\mathrm{KL}}\big(\Phi\hat{\mathcal T}Z_{\tilde\theta}(x,a)\,\big\|\,Z_\theta(x,a)\big)$, i.e.
$\mathcal L=-\sum_i m_i\log p_i(x,a)$ with $m=\Phi\hat{\mathcal T}Z_{\tilde\theta}(x,a)$ — exactly multiclass
classification over the atoms. And I can say precisely *why* it is unbiased where Wasserstein was not, the
crux of the trade: the cross-entropy is *linear* in the target masses $m$, so when $m=\mathbb{E}_I[m^{(I)}]$
is itself an average over sampled next states, the expectation passes straight through,
$\mathbb{E}_I\big[-\sum_i m_i^{(I)}\log p_i\big]=-\sum_i m_i\log p_i$. The gradient of the sampled
cross-entropy is therefore an *unbiased* estimate of the true one — exactly the property Wasserstein lacked,
because it is nonlinear in the distribution and the expectation does not pass through it. So I contract in
Wasserstein but descend in KL-after-projection, a deliberate mismatch made survivable by the projection $\Phi$,
which keeps the targets on the grid so the KL is always taken between two grid distributions.

The acting and the rest stay minimal so this is a clean single-axis change over the dueling head. I act
mean-greedily: read the mean of each action's distribution, $Q(x,a)=\sum_i z_i\,p_i(x,a)$, and take its
$\arg\max$ — the *same risk-neutral policy*, so any change in median HNS is attributable to the richer
training signal and not a changed objective. The distribution now exposes variance and tails I *could* act
on, but taking a risk-sensitive policy would confound "learning the distribution helps" with "acting on risk
helps," so I hold the policy fixed at the mean. This makes the change a strict generalization of the scalar
agent: collapse the distribution to a single atom ($N=1$) and $\sum_i z_i p_i$ is just that atom's value, the
mean, recovering the point-estimate head exactly. The conv torso, replay buffer, noise, and periodic target
sync are exactly as before. $N=51$ atoms on $[-10,10]$ is the working resolution — $\Delta z=20/50=0.4$, fine
enough to resolve a bimodal or skewed shape but coarse enough that the head emits only $51|\mathcal A|$
logits (this is the "C51" of the name). Too few atoms and $\Delta z$ grows until the distribution cannot
represent the shape it is meant to learn; too many balloons the head and starves each atom of mass so it
trains poorly; $51$ sits in the usable middle. A return past $[-10,10]$ is clamped onto the endpoint atom
rather than lost — the fixed-grid limitation I flag below. The per-atom projection is $O(N)$ and cheap.

Now the bar. This is the most fundamental change in the ladder — it replaces *what the agent learns*, not how
it learns a mean — and the reason to expect a large, broad gain is the denser signal: *every* game's return
is a non-trivial distribution, so on every game the network now gets many constraints per update where before
it got one. So I expect it to clear 151% decisively. I can sharpen that against the trend I noticed: the
broad recipe fixes gave diminishing returns, $38\to22\to11$, so if distributional learning were merely
another recipe tweak I would expect a gain in that shrinking sequence — around the dueling step or smaller. If
instead it *breaks* the trend and clears $151$ by a wide margin, that is evidence the change is categorically
different — a new *object*, not a better estimate of the old one — because only a change of kind would arrest
the diminishing-returns pattern. That is the falsifiable edge in both directions. The honest caveat is the
fixed grid: $V_{\min},V_{\max}$ and the $N=51$ resolution are hand-set, and a game whose returns run outside
the support or whose distribution has fine structure between atoms is served coarsely — a real limit, traded
against a much richer learning signal everywhere else.
