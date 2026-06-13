The decoupled-target run did exactly what the bias analysis said it would, and the numbers tell me both
that it worked and where its ceiling is. Double DQN landed Breakout 170.7 (seeds 141.6 / 125.6 / 244.8),
Seaquest 6789 (7178 / 5386 / 7804), Pong 20.7 (20.4 / 20.6 / 21.0). Pong is essentially solved — three
seeds within 0.6 of each other, right at the ceiling — which is the cheap game and tells me nothing
except that the critic is functioning. The information is in the other two. Seaquest is high but
*spread*: 5386 to 7804 across seeds, a swing of ~2400, the widest of any game. Breakout is modest and
also spread (125.6 to 244.8). That pattern is the residual disease, not a new one. Decoupling selection
from evaluation removed the *non-uniform* part of the overestimation, but $\theta^-$ is a stale copy of
$\theta$, so right after each `target_network_frequency = 1000` sync the two nets coincide and the
target reverts to a plain max for that interval — the correlated-error floor I admitted I couldn't fully
clear. And on Seaquest's eighteen actions, exactly where the typical-case inflation
$\epsilon\frac{m-1}{m+1}$ is largest, that residual bias and the relative-ordering errors it leaves have
the most room to move the greedy policy from seed to seed. So the variance is the tell: the critic is
still a single scalar per action, a point estimate with no notion of its own uncertainty, and it
absorbs all the stochasticity of returns and bootstraps into one number that wobbles. Tightening the
target-sync gap would shave a little more bias, but it cannot change *what* the critic represents. The
ceiling here is representational, and to break it I have to stop collapsing the return to its mean.

Why am I only learning the mean at all? I train an agent to maximize expected return, so I learn
$Q(x,a)=\mathbb{E}[Z(x,a)]$, one number per state-action. But $Z(x,a)$ — the actual return from
$(x,a)$ — is a random variable: in Seaquest it might be "surface in time, refill oxygen, score
thousands" with some probability and "suffocate now, score little" with the rest, a genuinely bimodal
distribution. The mean collapses it to a single value the agent will rarely actually receive, and worse,
the *spread* of that distribution — exactly what makes one Seaquest seed land at 5386 and another at
7804 — is information the scalar critic throws away on every transition. If I could learn the whole
distribution of $Z$ I would carry strictly more information, the way I would never hesitate to model a
full conditional distribution in supervised learning. The catch in RL is that there are no given
targets; I bootstrap, learning a guess from a guess. So the real question is whether the *machinery*
survives in distribution-space: a Bellman recursion the distribution obeys, a contraction so iterating
it converges, and a loss I can train from sampled transitions by SGD. Let me check the three and let the
representation and loss fall out of what the math allows.

The recursion exists. Peel the first reward off the return: $Z(x,a)=R(x,a)+\gamma\,Z(X',A')$ as an
equality *in distribution*, $X'\sim P(\cdot|x,a)$, $A'\sim\pi(\cdot|X')$. So there is a distributional
Bellman operator $\mathcal{T}^\pi Z(x,a)\overset{D}{=}R(x,a)+\gamma\,Z(X',A')$ that scales the
next-state distribution by $\gamma$, convolves with the transition randomness, and shifts by the reward.
The contraction story, though, lives or dies on the metric. My deep-learning instinct is KL, but the
$\gamma$-scaling alone kills it: scaling the *locations* of a distribution toward zero doesn't change
the *probabilities*, so KL is unchanged under the very operation that should make two distributions
closer; for disjoint supports KL stays infinite as both collapse toward a point mass. KL, total
variation, Kolmogorov — all vertical distances, blind to horizontal movement. The metric that sees the
$\gamma$-shrink is Wasserstein: $d_p(F,G)=(\int_0^1|F^{-1}(u)-G^{-1}(u)|^p\,du)^{1/p}$, a transport
distance that is finite for disjoint supports and scales linearly under multiplication. Under the
maximal Wasserstein metric $\bar d_p$, the operator $\mathcal{T}^\pi$ is a $\gamma$-contraction (the
common reward shift cancels, the scalar $\gamma$ comes out, the mixture over successors is bounded by
the worst case), with unique fixed point $Z^\pi$. So Wasserstein is the *right* metric — and that is
precisely the problem.

Because there is a sampling obstruction: I cannot minimize Wasserstein from samples by SGD. If I form an
empirical target from sampled transitions and minimize the sample-Wasserstein loss, the minimizer of the
expected sample loss is not the minimizer of the true $W_p$ — the gradient is biased. A concrete
one-line check: target $\frac12\delta_0+\frac12\delta_1$ sampled as $\delta_0$ or $\delta_1$, against
$p\delta_0+(1-p)\delta_1$; the true distance $|p-\frac12|$ is minimized at $p=\frac12$, but the expected
*sampled* distance is constant $\frac12$ in $p$, so its gradient points nowhere. The metric the theory
loves is the metric the learner can't follow. This is exactly the wall the prior distributional approach
(fixed atoms on $[V_{\min},V_{\max}]$, probabilities learned, KL after a projection) dodged — and it
dodged it at a cost: it optimizes KL-after-projection rather than Wasserstein, it must be handed the
return range $[V_{\min},V_{\max}]$ as prior knowledge, and the projection exists only because fixed
atoms force disjoint-support collisions. I want something genuinely Wasserstein-aware, trainable online,
with no projection and no support bounds — partly because the scaffold gives me no clean way to know
Seaquest's return range a priori (its returns run into the thousands while Pong's sit near $\pm21$), and
guessing $[V_{\min},V_{\max}]$ wrong is its own failure mode.

So find where the bias comes from. Wasserstein is built from the quantile function $F^{-1}$, and a
single sample is a draw, not an observation of a quantile; the optimal transport reshuffles which sample
pairs with which atom, and that matching's gradient doesn't average to the population gradient. The
fixed-atom agent's free variables are the *probabilities* on fixed locations — it learns the vertical
axis. Turn the parametrization on its side: fix the probabilities to uniform $1/N$ and make the
*locations* $\theta_i$ the learnable thing, so $Z_\theta(x,a)=\frac1N\sum_i\delta_{\theta_i(x,a)}$. Now
I am learning *where* $N$ equal lumps of mass sit — and "where the $i$-th of $N$ equal lumps sits" is
exactly a *quantile* of the return. Three wins fall out immediately: the support is no longer pinned to
any $[V_{\min},V_{\max}]$ (the locations slide to wherever Seaquest's thousands and Pong's tens actually
live, per-state adaptive resolution); there is no projection (the Bellman target's shifted locations are
just numbers I compare to mine, disjoint supports a non-issue); and estimating quantiles is something I
*can* do from samples without a biased gradient.

Which quantiles? Minimize $W_1(Y,U)$ between an arbitrary target $Y$ and a uniform-$N$-Dirac $U$ on
ordered locations $\theta_1\le\cdots\le\theta_N$. Since $F_U^{-1}$ is the staircase equal to $\theta_i$
on the level-cell $(\tau_{i-1},\tau_i]$ with $\tau_i=i/N$,
$W_1(Y,U)=\sum_i\int_{\tau_{i-1}}^{\tau_i}|F_Y^{-1}(\omega)-\theta_i|\,d\omega$, and the cells decouple
— each $\theta_i$ in its own integral. The subgradient of one cell in $\theta$ is
$2F(\theta)-(\tau_{i-1}+\tau_i)$, zero at $F(\theta)=\frac{\tau_{i-1}+\tau_i}{2}$, so the $W_1$-optimal
location is the quantile at the cell *midpoint*: $\hat\tau_i=\frac{2i-1}{2N}$,
$\theta_i=F_Y^{-1}(\hat\tau_i)$. Not the cell edges $i/N$ — the centers $(2i-1)/(2N)$.

Now hit those midpoint quantiles from samples without bias. Quantile parametrization alone does not
unbias Wasserstein — the obstruction above still applies to $W_p$ directly. The unbiasedness has to come
from the *loss*. Quantile regression supplies it: the $\tau$-quantile minimizes
$\mathbb{E}[\rho_\tau(\hat Z-\theta)]$ with $\rho_\tau(u)=u(\tau-\mathbb{1}_{u<0})$, whose subgradient in
$\theta$ is $\Pr(\hat Z<\theta)-\tau$, zero exactly at $\theta=F^{-1}(\tau)$. Crucially its gradient
depends only on the *sign* of $u=\hat Z-\theta$, so a single sample gives an unbiased stochastic
gradient. That is the escape: I cannot descend $W_p$, but I can descend the quantile-regression loss
whose minimizers are the very locations that minimize $W_1$. End-to-end Wasserstein, by way of quantile
regression on the midpoint quantiles.

One wrinkle before a deep net. $\rho_\tau$ has a kink at $u=0$ — its gradient magnitude stays constant
($\tau$ or $1-\tau$) right down to zero error, so the locations jitter and never settle. Round it off
with a Huber: quadratic inside $[-\kappa,\kappa]$, linear outside, weighted by the asymmetric quantile
factor $|\tau-\mathbb{1}_{u<0}|$. At $\kappa=1$ the Huber piece is $\frac12u^2$ inside $[-1,1]$ and
$|u|-\frac12$ outside — the gradient-clipped squared error, but made asymmetric per quantile level. This
is also the right answer for the variance I saw on Seaquest: the squared loss the previous rung used put
all its weight on the single mean and let a few large-return transitions dominate the gradient; the
quantile Huber spreads the supervision across $N$ levels and clips the tail influence, so a high-return
Seaquest outlier shapes the high quantiles instead of yanking the one scalar around.

Control. The objective is still to maximize expected return — I am enriching the representation, not
redefining optimal — so the greedy action maximizes the *mean* of the next-state distribution,
$a^\star=\arg\max_{a'}\frac1N\sum_j\theta_j(x',a')$, a drop-in for the previous rung's $\arg\max_a Q$.
The per-transition target uses the target network: compute $a^\star$ from its next-state
location-averages, form $\mathcal{T}\theta_j=r+\gamma\,\theta_j(x',a^\star)$ for all $j$ ($\gamma$ zeroed
at terminals), and regress each predicted location $\theta_i(x,a)$ against the whole *set* of target
locations with the quantile Huber loss summed over predicted quantiles $i$ and averaged over target
samples $j$ — the all-pairs version of the tabular update. There is no projection and no support range;
the only extra knob over the scalar critic is $N$.

Now fit this to *this* edit surface, because it is a same-named method only in spirit and I must land
the harness's implementation, not the paper's. The fixed `NatureDQNEncoder` gives me 512 features; the
head becomes `Linear(512, n_actions * N)`, reshaped to $(B, n_{\text{actions}}, N)$, with $Q$-values as
the per-action mean over the $N$ — so `q_network.forward` returns means and the loop's eval argmax still
works unchanged. I take $N = 200$, the standard Atari resolution and exactly the width the scaffold's
budget check is sized around ($1.05\times$ the $|\mathcal A|\times200$ head), so I am at the budget, not
over it. The midpoint levels are the fixed buffer $\hat\tau_i=(2i-1)/(2N)$. $\kappa=1$ is hard-coded —
the harness does not expose the $\kappa=0$ hard-loss branch as a runtime option, so I commit to the
Huberized form. One optimizer detail I keep from the distributional recipe even though it differs from
the previous rung: Adam with $\epsilon_{\text{Adam}}=0.01/\text{batch\_size}$ (a larger-than-default
$\epsilon$ that stabilizes the all-pairs asymmetric regression whose per-quantile gradient scales
differ), while the learning rate stays the scaffold's `args.learning_rate = 1e-4` rather than the
$5\times10^{-5}$ a from-scratch distributional run might pick — I am filling the harness's contract, and
its LR is fixed. The target sync stays the hard copy at `target_network_frequency`. The full scaffold
module is in the answer.

So the delta from the decoupled-target rung is concrete: same fixed encoder, same replay,
$\epsilon$-greedy, and periodic copy, but the head emits $N=200$ quantile locations per action instead
of one scalar, the loss is the all-pairs quantile Huber instead of `mse_loss`, and the policy acts on
the mean of the locations. Here is what I expect against the measured numbers. Pong is already at the
ceiling (20.7) and a richer critic can't do much there — I expect it to stay $\approx21$, no worse. The
two informative games should move on *both* axes I diagnosed: the mean should rise because the critic now
models the spread instead of letting it corrupt one scalar, and — the falsifiable part — the
seed-to-seed variance should *not blow up* relative to Double DQN even as the mean climbs, because the
quantile Huber's tail-clipping and distributional averaging are exactly the stabilizers the bare squared
loss lacked. So Breakout should clear Double DQN's 170 (I expect the mid-200s), and Seaquest should
clear its 6789 (I expect into the 9000s) with the high-return transitions now shaping the upper
quantiles rather than yanking the mean from seed to seed. If instead the distributional critic merely
matched the scalar one, that would say the Seaquest variance was never representational and I misread the
diagnosis; if it raised the mean but the spread widened, that would say $N=200$ on a frozen encoder is
under-resolving the tails and the next move is to stop fixing the quantile *levels* at all.
