The decoupled-target run did exactly what the bias analysis said it would, and the numbers tell me both
that it worked and where its ceiling is. Double DQN landed Breakout 170.7 (seeds 141.6 / 125.6 / 244.8),
Seaquest 6789 (7178 / 5386 / 7804), Pong 20.7 (20.4 / 20.6 / 21.0). Pong is essentially solved — three
seeds within 0.6 of each other, right at the ceiling — which is the cheap game and tells me nothing
except that the critic is functioning. The information is in the other two, and it is worth reading the
spread quantitatively rather than eyeballing it. Take the coefficient of variation, std over mean, so I
can compare games on different scales. Breakout: mean 170.7, the three seeds deviate by $-29.1,-45.1,+74.1$,
a standard deviation of $\approx52.8$, so a CoV of $52.8/170.7\approx0.31$. Seaquest: mean 6789, deviations
$+389,-1403,+1015$, std $\approx1025$, CoV $1025/6789\approx0.15$. Pong: std $\approx0.25$, CoV $\approx0.01$.
Two facts sit side by side. In *absolute* terms Seaquest swings widest — a $5386$-to-$7804$ range of
$\sim2400$ points, dwarfing Breakout's $119$-point range — but in *relative* terms Breakout wobbles more,
a third of its mean against Seaquest's one-sixth. One disease through two lenses: a scalar critic
absorbing the return's stochasticity into a single number wobbles by an amount scaling with return
magnitude on the long-horizon game and by a large *fraction* on the small-return, fragile-policy game.
Either way the wobble is the tell.

And it is the *residual* of the disease I already named. Decoupling removed the *non-uniform* part of the
overestimation, but $\theta^-$ is a stale copy of $\theta$, so right after each
`target_network_frequency = 1000` sync the target reverts to a plain max for that interval — the
correlated-error floor I couldn't fully clear — and on Seaquest's eighteen actions, where the inflation
$\epsilon\frac{m-1}{m+1}$ is largest, that residual has the most room to move the greedy policy seed to
seed. The critic is still a single scalar per action, a point estimate with no notion of its own
uncertainty, so it absorbs all the stochasticity and bootstraps into one wobbling number. Tightening the
sync gap shaves a little more bias but cannot change *what* the critic represents. The ceiling is
representational, and to break it I have to stop collapsing the return to its mean.

Three cheaper moves each fail for the same reason: they operate on the scalar target and never give the
critic a notion of its own spread. Widening the sync gap shaves residual *bias* but does nothing for the
discarded spread. $n$-step returns trade target bias for variance while still re-weighting one number
(and the harness feeds single transitions anyway). Swapping `mse_loss` for a Huber calms the
outlier-driven wobble but treats the five-figure Seaquest return as a nuisance to clip rather than as
*signal* — the real upper mass in the return law that makes one seed score 7804 and another 5386 is
exactly what a robust scalar loss discards. All three tame a symptom; only the representational move
attacks the diagnosis.

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
the worst case), with unique fixed point $Z^\pi$.

The disjoint-support case shows why the metric choice bites. Deterministic transition, zero reward, so
$\mathcal{T}Z\overset{D}{=}\gamma Z'$; take $Z_1'=\delta_0$, $Z_2'=\delta_1$, at $W_1$ distance $1$. The
images $\delta_0$ and $\delta_\gamma$ sit at $W_1$ distance $\gamma$ — the contraction factor — but have
disjoint support, so their KL is still infinite, reporting nothing contracted. Wasserstein is the *right*
metric — and that is precisely the problem.

Because there is a sampling obstruction: I cannot minimize Wasserstein from samples by SGD. If I form an
empirical target from sampled transitions and minimize the sample-Wasserstein loss, the minimizer of the
expected sample loss is not the minimizer of the true $W_p$ — the gradient is biased. A concrete
one-line check: target $\frac12\delta_0+\frac12\delta_1$ sampled as $\delta_0$ or $\delta_1$, against
$p\delta_0+(1-p)\delta_1$; the true distance $|p-\frac12|$ is minimized at $p=\frac12$, but the expected
*sampled* distance is constant $\frac12$ in $p$, so its gradient points nowhere. The metric the theory
loves is the metric the learner can't follow. This is exactly the wall the prior distributional approach
(fixed atoms on $[V_{\min},V_{\max}]$, probabilities learned, KL after a projection) dodged — and it
dodged it at a cost. Let me make that cost concrete on these three games, because it is decisive for the
"improve across games" requirement. A fixed grid needs a single $[V_{\min},V_{\max}]$ and a fixed atom
count shared by every game. Pong's returns live in $[-21,21]$; even Double DQN's Seaquest already runs
into the thousands ($5386$–$7804$). If I size the grid for Pong — say $[-25,25]$ with 51 atoms — the
spacing is $\sim1$ point, fine for Pong and hopelessly clipped for Seaquest. If I size it for Seaquest —
$[0,8000]$ with 51 atoms — the spacing is $\sim160$ points, so the *entire* Pong return range of $42$
points falls inside a single atom and the critic is blind. One shared grid cannot serve both without a
huge atom count or per-game tuning, and per-game tuning is exactly what the "a strong method should
improve across games" instruction forbids. So the fixed-atom route optimizes KL-after-projection rather
than Wasserstein, must be handed the return range as prior knowledge I don't cleanly have, and needs a
projection that exists only because fixed atoms force disjoint-support collisions. I want something
genuinely Wasserstein-aware, trainable online, with no projection and no support bounds.

So find where the bias comes from. Wasserstein is built from the quantile function $F^{-1}$, and a
single sample is a draw, not an observation of a quantile; the optimal transport reshuffles which sample
pairs with which atom, and that matching's gradient doesn't average to the population gradient. The
fixed-atom agent's free variables are the *probabilities* on fixed locations — it learns the vertical
axis. Turn the parametrization on its side: fix the probabilities to uniform $1/N$ and make the
*locations* $\theta_i$ the learnable thing, so $Z_\theta(x,a)=\frac1N\sum_i\delta_{\theta_i(x,a)}$. Now
I am learning *where* $N$ equal lumps of mass sit — and "where the $i$-th of $N$ equal lumps sits" is
exactly a *quantile* of the return. Three wins fall out immediately: the support is no longer pinned to
any $[V_{\min},V_{\max}]$ (the locations slide to wherever Seaquest's thousands and Pong's tens actually
live, per-state adaptive resolution — the very thing the fixed grid could not do for both games at
once); there is no projection (the Bellman target's shifted locations are just numbers I compare to
mine, disjoint supports a non-issue); and estimating quantiles is something I *can* do from samples
without a biased gradient.

Which quantiles? Minimize $W_1(Y,U)$ between an arbitrary target $Y$ and a uniform-$N$-Dirac $U$ on
ordered locations $\theta_1\le\cdots\le\theta_N$. Since $F_U^{-1}$ is the staircase equal to $\theta_i$
on the level-cell $(\tau_{i-1},\tau_i]$ with $\tau_i=i/N$,
$W_1(Y,U)=\sum_i\int_{\tau_{i-1}}^{\tau_i}|F_Y^{-1}(\omega)-\theta_i|\,d\omega$, and the cells decouple
— each $\theta_i$ in its own integral. The subgradient of one cell in $\theta$ is
$2F(\theta)-(\tau_{i-1}+\tau_i)$, zero at $F(\theta)=\frac{\tau_{i-1}+\tau_i}{2}$, so the $W_1$-optimal
location is the quantile at the cell *midpoint*: $\hat\tau_i=\frac{2i-1}{2N}$,
$\theta_i=F_Y^{-1}(\hat\tau_i)$ — the centers $(2i-1)/(2N)$, not the cell edges $i/N$. That distinction is
easy to fumble and not free: on the smallest case $N=2$ against a uniform target ($F_Y^{-1}(\omega)=\omega$),
the midpoints $(0.25,0.75)$ give $W_1=1/16+1/16=1/8$, while the edges $(0.5,1.0)$ give $1/8+1/8=1/4$ —
exactly double. So the $(2i-1)/(2N)$ levels are the minimizer, and getting them wrong doubles the error.

Now hit those midpoint quantiles from samples without bias. Quantile parametrization alone does not
unbias Wasserstein — the obstruction above still applies to $W_p$ directly. The unbiasedness has to come
from the *loss*. Quantile regression supplies it: the $\tau$-quantile minimizes
$\mathbb{E}[\rho_\tau(\hat Z-\theta)]$ with $\rho_\tau(u)=u(\tau-\mathbb{1}_{u<0})$, whose subgradient in
$\theta$ is $\Pr(\hat Z<\theta)-\tau$, zero exactly at $\theta=F^{-1}(\tau)$. Here is the crucial
difference from Wasserstein, stated as a single-sample fact: the subgradient of $\rho_\tau$ at one
observed $\hat Z$ is $-(\tau-\mathbb{1}_{\hat Z<\theta})$, which depends only on the *sign* of
$u=\hat Z-\theta$ — it is $\tau-1$ when the sample lands below $\theta$ and $\tau$ when it lands above,
nothing else. Averaging that sign over the sampling distribution gives $\Pr(\hat Z<\theta)-\tau$
directly, no transport matching to reshuffle, so a single sampled transition is already an *unbiased*
stochastic gradient of the population objective. That is the escape: I cannot descend $W_p$, but I can
descend the quantile-regression loss whose minimizers are the very locations that minimize $W_1$.
End-to-end Wasserstein, by way of quantile regression on the midpoint quantiles.

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
the only extra knob over the scalar critic is $N$. Keeping the control rule exactly the previous rung's
mean-argmax is deliberate: I am changing the critic's *representation* and its *loss*, and nothing about
how the policy reads the critic, so any movement in the scores is attributable to the richer critic and
not to a new acting rule.

Now fit this to the edit surface. The fixed `NatureDQNEncoder` gives 512 features; the head becomes
`Linear(512, n_actions * N)`, reshaped to $(B, n_{\text{actions}}, N)$, with $Q$-values the per-action
mean over the $N$ — so `q_network.forward` returns means and the loop's eval argmax works unchanged. I
take $N = 200$, the width the scaffold's budget check is sized around ($1.05\times$ the
$|\mathcal A|\times200$ head), so I sit at the budget, not over it. That is a $200\times$-wider output
than the scalar critic — the literal price of pinning resolution to 200 fixed levels — and it costs a
second time in the loss: the all-pairs quantile Huber is $N^2=40{,}000$ pairwise terms per state-action,
$O(N^2)$ in the resolution. Midpoint levels are the fixed buffer $\hat\tau_i=(2i-1)/(2N)$; $\kappa=1$ is
hard-coded (the harness exposes no $\kappa=0$ branch). I keep one optimizer detail from the distributional
recipe — Adam with $\epsilon_{\text{Adam}}=0.01/\text{batch\_size}$, a larger-than-default $\epsilon$ that
stabilizes the all-pairs asymmetric regression whose per-quantile gradient scales differ — while the
learning rate stays the scaffold's fixed `1e-4`. The target sync stays the hard copy at
`target_network_frequency`. The full scaffold module is in the answer.

Pong is at the ceiling (20.7) and a richer critic can't do much there — it should hold. The two
informative games should move on *both* axes I diagnosed: the mean should rise as the critic models the
spread instead of letting it corrupt one scalar, and — the part I can be proven wrong on — the
seed-to-seed variance should *not blow up* even as the mean climbs, because the quantile Huber's
tail-clipping and distributional averaging are the stabilizers the squared loss lacked. So both Breakout
and Seaquest should clear their Double DQN means (170.7 and 6789). Two failure branches would tell me
something: if the distributional critic merely matches the scalar one, the Seaquest variance was never
representational; if the mean rises but the spread *widens*, $N=200$ is under-resolving the tails, and the
next move is to stop fixing the quantile *levels* at all.
