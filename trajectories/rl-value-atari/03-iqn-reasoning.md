The quantile critic did what I bet it would. QR-DQN cleared the decoupled-mean rung on both informative
games — Breakout 252 (seeds 246.9 / 185.7 / 324.7) against Double DQN's 170.7, Seaquest 9027 (7892 /
7038 / 12152) against 6789 — and Pong stayed pinned at the ceiling (20.9). Let me read the size of those
gains rather than just noting the direction: Breakout rose by a factor $252.4/170.7\approx1.48$ (a $48\%$
jump), Seaquest by $9027/6789\approx1.33$ ($33\%$), Pong by a flat $1.3\%$ that is just ceiling noise. So
modeling the spread instead of collapsing it into one scalar paid off exactly where I argued it would, on
the long-horizon, wide-distribution games, and did nothing on the game that was already solved — the
clean signature of a representational fix rather than a tuning artifact.

But I made a *falsifiable* promise last rung — that the seed-to-seed variance would not blow up as the
mean climbed — and I should hold myself to it with the same coefficient of variation I used before. On
Breakout the promise held: CoV fell from Double DQN's $\approx0.31$ to $56.9/252.4\approx0.225$, the
richer critic both raised the mean and steadied it. On Seaquest it did *not*. The three seeds are 7038,
7892, and then a 12152 that is $12152/7038\approx1.73$ times the lowest — and the CoV actually *rose*,
from Double DQN's $\approx0.15$ to $2237/9027\approx0.25$. That is the exact branch I flagged as the tell
last time: "if it raised the mean but the spread widened, that would say $N=200$ is under-resolving the
tails." The mean rose and the spread widened, on Seaquest specifically. So this is not the bias variance I
was chasing two rungs ago; it is the critic telling me something about its *resolution*. One seed found a
much heavier upper tail than the others, and a 200-step staircase is a coarse instrument for a five-figure
return distribution with a long, heavy top.

The thing that keeps bothering me is that I picked the number 200 by hand and then the network is stuck
with it forever. The output layer is $|\mathcal A|\times 200$, those 200 numbers are tied one-to-one to
200 levels $\hat\tau_i=(2i-1)/(2N)$ chosen before training started, and the approximation can never be
finer than 200 no matter how much the frozen encoder has left to give. On Seaquest, whose return
distribution runs into five figures with a long, heavy upper tail, a 200-step staircase is a coarse
description of exactly the tail that the 12152 seed lived in — and every update touches the same fixed
comb of levels, so the agent never gets to put extra resolution where the distribution is most
informative. The resolution is an architectural constant, not something that improves with capacity or
training. That is the wart I want to remove, and it is the natural next move: last rung enriched the value
*object* from a scalar to a distribution; this rung removes the one structural limit left on that
distribution.

So what is the object I am really approximating? The 200 locations are quantiles, and "the location of
the $\tau$-quantile as a function of $\tau$" is just the quantile function $F_Z^{-1}(\tau)$ — the inverse
CDF, a map from a probability $\tau\in[0,1]$ to a return value. The quantile agent samples this function
at 200 fixed grid points and learns those 200 outputs. But $F_Z^{-1}$ is a *continuous* function of
$\tau$. If I could learn the whole function — a map I can query at any $\tau$ — then I am no longer
pinned to a grid: I evaluate it at as many or as few points as I like, and the fidelity is set by how
well the network approximates a curve, which is a matter of capacity and training, not a hardcoded $N$.
Write the target: $Z_\tau(x,a):=F_{Z(x,a)}^{-1}(\tau)$, so that for $\tau\sim U([0,1])$, the value
$Z_\tau(x,a)$ is a genuine sample of the return. Let me make sure that last claim is actually true and not
wishful, because the whole design rests on it. If $\tau\sim U([0,1])$ and I push it through the quantile
function, what is the law of $F^{-1}(\tau)$? Compute the CDF of the output:
$\Pr(F^{-1}(\tau)\le z)=\Pr(\tau\le F(z))=F(z)$, using that $\tau$ is uniform so $\Pr(\tau\le u)=u$. The
pushforward has CDF $F$ exactly — it *is* $Z$. That is inverse-transform sampling, and it says the
quantile function is a complete, lossless encoding of the distribution: knowing how to map a uniform draw
to a return is knowing the whole return law. So instead of learning probabilities on fixed locations, or
locations on fixed probabilities, I learn a deterministic function that *reparameterizes* a base sample
$\tau$ into a return. The distribution is defined implicitly — I never write down its density or CDF, I
only know how to sample it: draw $\tau$, evaluate the network. With enough capacity this approximates any
return distribution, because any distribution is the pushforward of a uniform through its own quantile
function. That is exactly what frees Seaquest's tail: where the 200-comb spent the same resolution
everywhere, an implicit function can bend sharply in the upper quantiles if that is where the data
demands it.

Two questions the grid agent answered by construction I now have to answer by design: how does $\tau$
enter the network, and what trains the function so that $Z_\tau(x,a)$ lands on $F^{-1}_{Z(x,a)}(\tau)$
for *every* $\tau$, not just a chosen few. Take the training first, because it is the part I am sure
about — it is the same quantile-regression machinery the last rung used, and that machinery never cared
that the levels came from a grid. It cares only that I am trying to hit the $\tau$-quantile. For any
single $\tau$, the $\tau$-quantile minimizes $\mathbb{E}_{\hat Z}[\rho_\tau(\hat Z-\theta)]$ with
$\rho_\tau(u)=u(\tau-\mathbb{1}_{u<0})$, whose error-gradient $\tau-\mathbb{1}_{u<0}$ depends only on the
sign of $u$, so one sample gives an unbiased gradient. The Bellman version uses *two independent* base
samples $\tau$ and $\tau'$: build the sampled TD error
$\delta^{\tau,\tau'}=r+\gamma\,Z_{\tau'}(x',\pi(x'))-Z_\tau(x,a)$, where $Z_\tau(x,a)$ is my prediction
at level $\tau$ and $Z_{\tau'}(x',\pi(x'))$ is a bootstrapped target-network sample at level $\tau'$, and
regress the prediction at level $\tau$ toward the target at level $\tau'$ using the quantile loss at
level $\tau$. Crucially, $\tau$ and $\tau'$ are drawn *fresh* from a continuous distribution each update,
not read off a fixed comb. So the function gets trained at a dense, ever-changing set of levels, which is
exactly what forces it to be right *as a function* and not just at 200 pinned points. The kink in
$\rho_\tau$ at $u=0$ hurts a deep net the same way it did last rung — constant-magnitude gradient down to
zero error — so Huberize with threshold $\kappa$, quadratic inside $[-\kappa,\kappa]$ and linear outside,
weighted by the asymmetric quantile factor $|\tau-\mathbb{1}\{u<0\}|$; $\kappa=1$ recovers the
gradient-clipped squared error inside the band but asymmetric per level.

I want to be careful about one thing that is easy to get wrong: why $\tau$ and $\tau'$ must be drawn
*independently* rather than reusing one draw. The quantile-regression objective at level $\tau$ regresses
my prediction $Z_\tau(x,a)$ against *samples of the target return*, and its minimizer is the target's
$\tau$-quantile only when those samples are genuine draws from the whole target law. By the
inverse-transform fact I established, $Z_{\tau'}(x',\pi(x'))$ with $\tau'\sim U([0,1])$ *is* such a draw —
a uniform base sample pushed through the target quantile function is a sample of the bootstrapped return.
So the inner average $\frac1{N'}\sum_j\rho_\tau(r+\gamma Z_{\tau'_j}-Z_\tau)$ is a Monte-Carlo estimate of
$\mathbb{E}_{\hat Z\sim\text{target}}[\rho_\tau(\hat Z-Z_\tau)]$, exactly what I want. Now suppose I tied
them, $\tau'=\tau$. Then the "target sample" would be the target's $\tau$-quantile specifically, not a
uniform draw — I would be regressing the $\tau$-quantile of my prediction toward the $\tau$-quantile of
the target, a quantile-to-quantile pairing. That is no longer quantile regression against a distribution;
it is the fixed-comb matching in disguise, and it throws away precisely the distributional content the
implicit form was supposed to buy — each predicted level would only ever see its own matched target level
and never the *spread* of the target across levels. The independence of $\tau$ and $\tau'$ is what keeps
the target a genuine sample of the whole return law, so decoupling the two draws is not incidental
bookkeeping but the thing that makes the all-pairs loss a distributional loss at all.

Now the sample counts, because they are *knobs* rather than the fixed 200. Sample $N$ values of $\tau$
for predictions and $N'$ values of $\tau'$ for targets, form the all-pairs loss
$\mathcal L=\frac1{N'}\sum_{i=1}^{N}\sum_{j=1}^{N'}\rho^\kappa_{\tau_i}(\delta^{\tau_i,\tau'_j})$ — sum
over predicted levels $i$ (each is regressed at its own level), average over target samples $j$ (a
Monte-Carlo estimate of the bootstrapped distribution, hence $1/N'$). $N$ is how much of my own quantile
function I touch per update; $N'$ is a variance-reduction count on the regression target. Before I set
them, let me be explicit about the two cheaper ways I could have tried to buy Seaquest's tail back and
why neither is the move. I could simply widen the grid — take the last rung's $N$ from 200 to 400. But
the head is $513\,|\mathcal A|\,N$ parameters, so on Seaquest that goes from $1{,}846{,}800$ to
$3{,}693{,}600$, a full $2\times$ the previous head, and the budget check only tolerates
$1.05\times$ the $|\mathcal A|\times200$ reference — so doubling $N$ is not even affordable, and even if
it were it would still be a *fixed, shared* comb spending its extra resolution uniformly rather than on
the tail. Or I could keep $N$ modest but make the levels $\tau_i$ themselves learnable, an adaptive comb.
That is better, but it is still a finite count, and the one comb is shared across every state — Seaquest's
per-state upper tail still cannot get local resolution — and it drags in a small proposal branch and a
nested optimization for the levels. The implicit function dominates both: it is continuous (no finite
count at all), its resolution is per-state and per-$\tau$ because the network can bend the curve wherever
the data pushes, and the "number of quantiles" is demoted to a sampling count I choose per update rather
than an architectural width. Push $N=1$ in my head as a sanity check on that framing: I touch a single
random point of the curve per update, close to a non-distributional agent in loss terms — a clean
diagnostic that the benefit is the implicit representation, not merely many heads. For the working agent I
want both moderate: enough $N$ to shape several parts of the curve, enough $N'$ to denoise the target,
nowhere near a 200-wide output layer. $N=N'=8$ is the balanced setting; the per-update cost is
$8\times8=64$ pairwise terms, against the grid agent's $200\times200=40{,}000$ — a $625\times$ reduction
in the pairwise loss work, and far below its 200-wide output.

Now the architecture: how does $\tau$ get in? The state side should stay as close as possible to the
machinery the scaffold fixes. I take $\psi(x)$ to be the frozen `NatureDQNEncoder` ending in its
512-dim ReLU feature vector — that is just choosing the boundary so the encoder includes the usual hidden
layer, leaving the action head as a single linear map from 512 features to actions. I want to insert
$\tau$-dependence without rebuilding either, so add a function $\phi(\tau)$ that embeds the scalar level
into the same 512-dim space and combine it with $\psi(x)$ before the head. What combination?
Concatenation is the lazy choice, and I should work out exactly what it can and cannot represent before I
reject it, because "obviously multiply" is not a reason. Concatenate $\psi(x)$ and $\phi(\tau)$ into a
$1024$-vector and hit it with a single linear head $W\in\mathbb{R}^{|\mathcal A|\times1024}$. Split $W$
into its two halves $W=[W^\psi\;|\;W^\phi]$; then the output for action $a$ is
$Z_\tau(x,a)=W^\psi_a\!\cdot\!\psi(x)+W^\phi_a\!\cdot\!\phi(\tau)=g_a(x)+h_a(\tau)$ — additively
*separable*, a state term plus a $\tau$ term that share nothing. The consequence is fatal for what I need:
$\partial Z_\tau/\partial\tau=h_a'(\tau)$ is identical for every state, so the *shape* of the quantile
function in $\tau$ is fixed across the whole game and only its *level* $g_a(x)$ slides up and down.
Concretely, take two Seaquest states with the same mean return but different spread — one where I have
just surfaced and the outcome is nearly deterministic (a flat quantile function), one where I am deep with
low oxygen and the return is bimodal, escape-and-score versus suffocate (a quantile function with a sharp
step). Additive $h_a(\tau)$ hands both the *same* curve shape shifted by $g_a$; it cannot give one a flat
and the other a steep $F^{-1}$. But that state-dependent reshaping is the entire point. So I need $\tau$
to *multiply* the state features. Take the Hadamard product $Z_\tau(x,a)\approx f(\psi(x)\odot\phi(\tau))_a$,
so $\phi(\tau)$ gates each feature of $\psi(x)$: now $Z_\tau(x,a)=\sum_j W_{a,j}\,\psi_j(x)\,\phi_j(\tau)$
and $\partial Z_\tau/\partial\tau=\sum_j W_{a,j}\,\psi_j(x)\,\phi_j'(\tau)$ — the $\tau$-slope is reweighted
by the state features $\psi_j(x)$, genuinely state-dependent, so even a single linear head sees a
$\tau$-conditioned input whose *shape* can vary per state. This matters more here than in a from-scratch
network, precisely because the encoder is *frozen*: I cannot adapt $\psi$ to make room for $\tau$, so the
interaction has to be forced through the only thing I control, and multiplication forces it through the
shallow head.

What embedding $\phi(\tau)$? The first thing to rule out is feeding the raw scalar $\tau$ through a
linear layer, and again I want the reason to be a calculation, not a shrug. A linear map of the scalar is
$\phi(\tau)=w\tau+b$, affine in $\tau$; compose it with the single linear head and, up to the one ReLU,
$Z_\tau(x,a)$ is affine in $\tau$, so $F^{-1}(\tau)$ is a straight line in $\tau$ — and a linear quantile
function is the quantile function of one distribution only, the uniform on an interval. It literally
cannot represent a return distribution that bends, let alone the bimodal Seaquest tail. I need a *basis
expansion*: lift the scalar into many features varying at different rates so a linear layer on top can
synthesize an arbitrary smooth function of $\tau$. Cosines of increasing frequency are the natural bounded
basis on an interval — a Fourier-type feature map. Expand $\tau$ into $n$ cosine features
$\cos(\pi i\tau)$, $i=1,\dots,n$, then a linear-then-ReLU into the 512-dim space:
$\phi_j(\tau)=\mathrm{ReLU}(\sum_{i=1}^n\cos(\pi i\tau)\,w_{ij}+b_j)$. (Indexing from 0 only adds a
constant cosine absorbable into the bias.) How much frequency content do I need? The highest term
$\cos(\pi n\tau)$ with $n=64$ completes $32$ full periods across $\tau\in[0,1]$, so it can resolve
features of the quantile function down to a scale of $1/64\approx1.6\%$ of the probability axis — far
finer than any monotone return curve, even Seaquest's stepped one, actually needs. $n=64$ cosines is
plenty and cheap.

Let me make sure I have not quietly defeated the point by adding capacity, because the whole ladder's
budget check is watching. The encoder is untouched and frozen. The head is the same shallow $f$. The only
new parameters are the cosine embedding's `Linear(64, 512)` — that is $64\times512+512=33{,}280$ — and the
action head `Linear(512, |\mathcal A|)`, which is $513\,|\mathcal A|$: $2052$ for Breakout, $3078$ for
Pong, $9234$ for Seaquest. So the total learnable head-plus-embedding is about $35{,}300$ on Breakout,
$36{,}400$ on Pong, $42{,}500$ on Seaquest. Compare that to the grid agent's output layer,
$513\,|\mathcal A|\times200$: $410{,}400$, $615{,}600$, $1{,}846{,}800$ respectively. The ratios are the
decisive part of the accounting — the implicit head is about $11.6\times$ smaller on Breakout,
$16.9\times$ smaller on Pong, and $43.4\times$ smaller on Seaquest, the very game I care most about. I
have *removed* the 200-fold output blowup and replaced it with one small cosine branch, so I am not only
inside the $1.05\times$-QR-DQN budget on all three action-space sizes (4, 6, 18), I am comfortably below
it — the representational gain comes entirely from learning a function of $\tau$, not from width. That is
the property I want: a minimal, single-axis change over the quantile rung that spends *less* capacity, not
more.

Now the policy. The grid agent acts on the mean of its 200 locations. The mean is
$\mathbb{E}_{\tau\sim U([0,1])}[Z_\tau(x,a)]$, so approximate it with $K$ fresh samples and take the
argmax: $\tilde\pi(x)=\arg\max_a\frac1K\sum_k Z_{\tilde\tau_k}(x,a)$, $\tilde\tau_k\sim U([0,1])$, $K=32$
— enough to make the choice a Monte-Carlo estimate of the mean rather than a single noisy draw. The
bootstrapped target uses this greedy action with $\gamma=0$ at terminals; the replay buffer,
$\epsilon$-greedy, and the periodic copy stay as the scaffold fixes them. I deliberately keep the
risk-*neutral* uniform policy here even though the implicit quantile function unlocks more: sampling
$\tau$ from a non-uniform $\mu$ instead of $U([0,1])$ turns the action value into a distorted expectation
$\int_0^1 F^{-1}_Z(u)\,d\mu(u)$ — mass near low quantiles makes the policy risk-averse, mass near high
quantiles risk-seeking, with CVaR$(\eta)$ the lower-tail case $u\sim U([0,\eta])$. That risk knob is a
genuine bonus the representation gives for free, but using it would change *what* is being maximized, and
I want this rung to be a clean comparison: same risk-neutral mean-greedy objective the quantile rung
maximized, so any difference is attributable to the implicit, continuous representation and nothing else.

Fitting this to the edit surface: `q_network.forward(x)` must return per-action Q-values for the loop's
eval-argmax, so I have it return the $K$-sample mean over $\tau$; a separate `get_quantiles(x, taus)`
returns the raw $(B,|\mathcal A|,M)$ samples for the loss. `update` samples $N=8$ prediction $\tau$ and
$N'=8$ target $\tau'$, picks the next action greedily on the target net's $K$-sample mean, forms the
pairwise TD errors and the all-pairs quantile Huber, steps Adam at the scaffold's
`learning_rate = 1e-4` with $\epsilon_{\text{Adam}}=0.01/\text{batch\_size}$, and does the hard target
copy at `target_network_frequency`. The full scaffold module is in the answer.

Now the bar this has to clear, stated against the strongest baseline's real numbers so I can be proven
wrong. QR-DQN set Breakout mean 252.4 (best seed 324.7), Seaquest mean 9027 (best seed 12152), Pong
20.9. Pong is at the ceiling and a richer representation cannot help there — I expect IQN to hold
$\approx21$, and if it dropped that would mean the cosine modulation destabilized the easy case, a clear
failure. Breakout and Seaquest are where the implicit function should pay: I expect IQN to match or beat
the QR-DQN means while removing the fixed-resolution bottleneck, and I expect the *largest* gain on
Seaquest specifically, because that is the game whose return distribution is widest and whose long upper
tail — the one the 12152 seed lived in, the one whose CoV *rose* to $0.25$ under the 200-comb — the fixed
grid resolved most coarsely. The three things I would validate: (1) the $K$-sample mean policy is stable
seed-to-seed, with no resolution-induced variance of the kind the 200-grid showed on Seaquest — concretely
I would want Seaquest's CoV to fall back rather than climb; (2) Seaquest is where the implicit
representation gains most, which is the falsifiable claim — if Breakout improves but Seaquest does not,
then the bottleneck was never resolution and I misread the 12152 seed; and (3) the head stays inside the
parameter budget on all of breakout (4), pong (6), and seaquest (18) actions, which the $11.6/16.9/43.4\times$
accounting above already confirms. If IQN merely matched QR-DQN everywhere, that would say 200 quantiles
already saturated these games and the implicit representation buys nothing at this budget — a clean
negative result, but I expect the Seaquest tail to say otherwise.
