The decoupled-target run did exactly what the bias analysis said it would, and the numbers tell me both
that it worked and where its ceiling is. Double DQN landed Breakout 170.7 (seeds 141.6 / 125.6 / 244.8),
Seaquest 6789 (7178 / 5386 / 7804), Pong 20.7 (20.4 / 20.6 / 21.0). Pong is essentially solved — three
seeds within 0.6 of each other, right at the ceiling — which is the cheap game and tells me nothing
except that the critic is functioning. The information is in the other two, and it is worth reading the
spread quantitatively rather than eyeballing it. Take the coefficient of variation, std over mean, so I
can compare games on different scales. Breakout: mean 170.7, the three seeds deviate by $-29.1,-45.1,+74.1$,
a standard deviation of $\approx52.8$, so a CoV of $52.8/170.7\approx0.31$. Seaquest: mean 6789, deviations
$+389,-1403,+1015$, std $\approx1025$, CoV $1025/6789\approx0.15$. Pong: std $\approx0.25$, CoV $\approx0.01$.
So two facts sit side by side and I have to hold both honestly. In *absolute* terms Seaquest is the widest
swing by far — a $5386$-to-$7804$ range of $\sim2400$ points, dwarfing Breakout's $119$-point range — but
in *relative* terms Breakout actually wobbles more, a third of its mean against Seaquest's one-sixth.
These are not two diseases; they are one, seen through two lenses. A scalar critic that has to absorb the
stochasticity of the return into a single number will wobble by an amount that scales roughly with the
return magnitude on the long-horizon game (hence Seaquest's huge absolute swing) and by a large *fraction*
on the game where the return is small and the policy is fragile (hence Breakout's large relative swing).
Either way the wobble is the tell.

And it is the *residual* of the disease I already named, not a new one. Decoupling selection from
evaluation removed the *non-uniform* part of the overestimation, but $\theta^-$ is a stale copy of
$\theta$, so right after each `target_network_frequency = 1000` sync the two nets coincide and the
target reverts to a plain max for that interval — the correlated-error floor I admitted I couldn't fully
clear. And on Seaquest's eighteen actions, exactly where the typical-case inflation
$\epsilon\frac{m-1}{m+1}$ is largest, that residual bias and the relative-ordering errors it leaves have
the most room to move the greedy policy from seed to seed. So the variance is the tell: the critic is
still a single scalar per action, a point estimate with no notion of its own uncertainty, and it
absorbs all the stochasticity of returns and bootstraps into one number that wobbles. Tightening the
target-sync gap would shave a little more bias, but it cannot change *what* the critic represents. The
ceiling here is representational, and to break it I have to stop collapsing the return to its mean.

Let me not jump to that conclusion without pricing the cheaper moves, because three of them are sitting
right there and I owe them an honest look. First, I could keep pushing the previous rung's own lever:
widen the sync gap so $\theta$ and $\theta^-$ decorrelate further and the post-sync plain-max intervals
shrink. That shaves the residual bias, but it operates on the same scalar target and leaves the critic a
point estimate — it addresses the part of the wobble that is bias, not the part that is discarded spread,
and the CoV computation says both parts are live. Second, I could switch the bootstrap to $n$-step
returns, trading some bias for lower variance in the target. But that re-weights the same one-number
target; it never gives the critic a notion of its own uncertainty, and the harness feeds me single
transitions, so it is not even a clean edit-surface change. Third — and this is the tempting one, because
it directly targets the Seaquest symptom — I could robustify the *scalar* loss, swap `mse_loss` for a
Huber on the TD error so the occasional five-figure Seaquest return stops dominating the gradient. That
would genuinely calm the outlier-driven wobble. But it treats the large return as a nuisance to clip
rather than as *signal*: the whole reason one Seaquest seed scores 7804 and another 5386 is that the
return distribution has real upper mass, and a robust scalar loss throws that mass away just as
thoroughly as the squared loss did — it stabilizes the mean without ever representing the spread that is
the actual information. All three tame a symptom; none change the object. So the representational move is
not a luxury, it is the only one of the four that attacks the diagnosis instead of the bruise.

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

Let me actually watch the contraction happen on a toy, because "the $\gamma$ comes out" is the kind of
claim I should see rather than assert. Take a deterministic transition and zero reward, so
$\mathcal{T}Z\overset{D}{=}\gamma Z'$, and two candidate successor return laws, $Z_1'=\delta_0$ and
$Z_2'=\delta_1$ — point masses at $0$ and $1$. Their $W_1$ distance is $|0-1|=1$. Apply the operator:
$\mathcal{T}Z_1'=\delta_0$ and $\mathcal{T}Z_2'=\delta_\gamma$, whose $W_1$ distance is $|0-\gamma|=\gamma$.
The gap shrank from $1$ to $\gamma$ — exactly the contraction factor. Add any shared reward $r$ and both
images shift by $r$, leaving the horizontal gap untouched, so the contraction is $\gamma$ regardless of
$r$. And this is the very step KL would misread: $\gamma\delta_1=\delta_\gamma$ has disjoint support from
$\delta_0$, so KL between the two images is still infinite, unchanged from before the operator acted —
the metric would tell me nothing contracted. Wasserstein is the *right* metric — and that is precisely
the problem.

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
$\theta_i=F_Y^{-1}(\hat\tau_i)$. Not the cell edges $i/N$ — the centers $(2i-1)/(2N)$. This is a place I
could easily fumble by reaching for the round number $i/N$, so let me check it costs what I think on the
smallest case, $N=2$ against a target uniform on $[0,1]$ (so $F_Y^{-1}(\omega)=\omega$). The midpoints
are $\hat\tau_1=1/4,\hat\tau_2=3/4$, giving $\theta=(0.25,0.75)$; the cell integrals are
$\int_0^{1/2}|\omega-0.25|\,d\omega=1/16$ and $\int_{1/2}^1|\omega-0.75|\,d\omega=1/16$, total
$W_1=1/8=0.125$. Now the tempting wrong choice, the edges $\theta=(F^{-1}(1/2),F^{-1}(1))=(0.5,1.0)$:
$\int_0^{1/2}|\omega-0.5|\,d\omega=1/8$ and $\int_{1/2}^1|\omega-1|\,d\omega=1/8$, total $W_1=1/4=0.25$.
The midpoints give exactly *half* the transport cost of the edges. So the $(2i-1)/(2N)$ levels are not a
convention, they are the minimizer, and getting them wrong would double my error — worth the care.

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

Now fit this to *this* edit surface, because I must land the harness's implementation. The fixed
`NatureDQNEncoder` gives me 512 features; the head becomes `Linear(512, n_actions * N)`, reshaped to
$(B, n_{\text{actions}}, N)$, with $Q$-values as the per-action mean over the $N$ — so
`q_network.forward` returns means and the loop's eval argmax still works unchanged. I take $N = 200$, the
standard Atari resolution and exactly the width the scaffold's budget check is sized around ($1.05\times$
the $|\mathcal A|\times200$ head), so I am at the budget, not over it. Let me put the parameter arithmetic
down so I know precisely what $N=200$ spends. The head is $512\cdot|\mathcal A|\cdot N + |\mathcal A|\cdot N
= 513\,|\mathcal A|\,N$ parameters: for Breakout ($|\mathcal A|=4$) that is $410{,}400$, for Pong ($6$)
$615{,}600$, for Seaquest ($18$) $1{,}846{,}800$. Compared to the scalar critic's $513\,|\mathcal A|$
head ($9234$ on Seaquest), the quantile head is a full $200\times$ wider — that is the literal price of
pinning the resolution to $200$ fixed levels, and the budget check is sized to allow exactly this and no
more. There is a second cost in the loss: the all-pairs quantile Huber compares every one of $N$
predicted locations against every one of $N$ target locations, so it is $N^2=40{,}000$ pairwise Huber
terms per state-action, times the batch of $32$, over a million per update — an $O(N^2)$ tax that also
scales with the resolution I chose. The midpoint levels are the fixed buffer $\hat\tau_i=(2i-1)/(2N)$.
$\kappa=1$ is hard-coded — the harness does not expose the $\kappa=0$ hard-loss branch as a runtime
option, so I commit to the Huberized form. One optimizer detail I keep from the distributional recipe
even though it differs from the previous rung: Adam with $\epsilon_{\text{Adam}}=0.01/\text{batch\_size}$
(a larger-than-default $\epsilon$ that stabilizes the all-pairs asymmetric regression whose per-quantile
gradient scales differ), while the learning rate stays the scaffold's `args.learning_rate = 1e-4` rather
than the $5\times10^{-5}$ a from-scratch distributional run might pick — I am filling the harness's
contract, and its LR is fixed. The target sync stays the hard copy at `target_network_frequency`. The
full scaffold module is in the answer.

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
