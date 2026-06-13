The quantile critic did what I bet it would. QR-DQN cleared the decoupled-mean rung on both informative
games — Breakout 252 (seeds 246.9 / 185.7 / 324.7) against Double DQN's 170.7, Seaquest 9027 (7892 /
7038 / 12152) against 6789 — and Pong stayed pinned at the ceiling (20.9). So modeling the spread
instead of collapsing it into one scalar paid off exactly where I argued it would, on the long-horizon,
wide-distribution game. But look harder at the Seaquest seeds: 7038, 7892, and then a 12152 that is
nearly double the lowest. The mean rose, but the spread did not collapse — one seed found a much heavier
upper tail than the others. That is not the bias variance I was chasing last rung; it is the critic
telling me something about its *resolution*. The thing that keeps bothering me is that I picked the
number 200 by hand and then the network is stuck with it forever. The output layer is $|\mathcal A|\times
200$, those 200 numbers are tied one-to-one to 200 levels $\hat\tau_i=(2i-1)/(2N)$ chosen before
training started, and the approximation can never be finer than 200 no matter how much the frozen encoder
has left to give. On Seaquest, whose return distribution runs into five figures with a long, heavy upper
tail, a 200-step staircase is a coarse description of exactly the tail that the 12152 seed lived in — and
every update touches the same fixed comb of levels, so the agent never gets to put extra resolution where
the distribution is most informative. The resolution is an architectural constant, not something that
improves with capacity or training. That is the wart I want to remove, and it is the natural next move:
last rung enriched the value *object* from a scalar to a distribution; this rung removes the one
structural limit left on that distribution.

So what is the object I am really approximating? The 200 locations are quantiles, and "the location of
the $\tau$-quantile as a function of $\tau$" is just the quantile function $F_Z^{-1}(\tau)$ — the inverse
CDF, a map from a probability $\tau\in[0,1]$ to a return value. The quantile agent samples this function
at 200 fixed grid points and learns those 200 outputs. But $F_Z^{-1}$ is a *continuous* function of
$\tau$. If I could learn the whole function — a map I can query at any $\tau$ — then I am no longer
pinned to a grid: I evaluate it at as many or as few points as I like, and the fidelity is set by how
well the network approximates a curve, which is a matter of capacity and training, not a hardcoded $N$.
Write the target: $Z_\tau(x,a):=F_{Z(x,a)}^{-1}(\tau)$, so that for $\tau\sim U([0,1])$, the value
$Z_\tau(x,a)$ is a genuine sample of the return. That last sentence is the reparameterization trick
stated for returns: push a uniform random variable through the quantile function and you get a return
sample. So instead of learning probabilities on fixed locations, or locations on fixed probabilities, I
learn a deterministic function that *reparameterizes* a base sample $\tau$ into a return. The
distribution is defined implicitly — I never write down its density or CDF, I only know how to sample it:
draw $\tau$, evaluate the network. With enough capacity this approximates any return distribution,
because any distribution is the pushforward of a uniform through its own quantile function. That is
exactly what frees Seaquest's tail: where the 200-comb spent the same resolution everywhere, an implicit
function can bend sharply in the upper quantiles if that is where the data demands it.

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

Now the sample counts, because they are *knobs* rather than the fixed 200. Sample $N$ values of $\tau$
for predictions and $N'$ values of $\tau'$ for targets, form the all-pairs loss
$\mathcal L=\frac1{N'}\sum_{i=1}^{N}\sum_{j=1}^{N'}\rho^\kappa_{\tau_i}(\delta^{\tau_i,\tau'_j})$ — sum
over predicted levels $i$ (each is regressed at its own level), average over target samples $j$ (a
Monte-Carlo estimate of the bootstrapped distribution, hence $1/N'$). $N$ is how much of my own quantile
function I touch per update; $N'$ is a variance-reduction count on the regression target. Push $N=1$ in
my head: I touch a single random point of the curve per update, close to a non-distributional agent in
loss terms — a clean diagnostic that the benefit is the implicit representation, not merely many heads.
For the working agent I want both moderate: enough $N$ to shape several parts of the curve, enough $N'$
to denoise the target, nowhere near a 200-wide output layer. $N=N'=8$ is the balanced setting; the
per-update cost is $8\times8=64$ pairwise terms, comparable to the grid agent's work and far below its
200-wide output.

Now the architecture: how does $\tau$ get in? The state side should stay as close as possible to the
machinery the scaffold fixes. I take $\psi(x)$ to be the frozen `NatureDQNEncoder` ending in its
512-dim ReLU feature vector — that is just choosing the boundary so the encoder includes the usual hidden
layer, leaving the action head as a single linear map from 512 features to actions. I want to insert
$\tau$-dependence without rebuilding either, so add a function $\phi(\tau)$ that embeds the scalar level
into the same 512-dim space and combine it with $\psi(x)$ before the head. What combination? Concatenation
is the lazy choice, but with a single linear head, if I merely concatenate $\psi(x)$ and $\phi(\tau)$ and
hit them with one linear map, $\tau$ enters only *additively*: the output is (linear in $\psi$) + (linear
in $\phi(\tau)$), and a $\tau$-dependent shift of every action's value can only slide the whole curve up
and down — it cannot reshape the quantile function per state. But the *shape* of $F^{-1}$ has to change
with $\tau$ in a state-dependent way; that is the whole point. So I need $\tau$ to *multiply* the state
features. Take the Hadamard product $Z_\tau(x,a)\approx f(\psi(x)\odot\phi(\tau))_a$, so $\phi(\tau)$
gates each feature of $\psi(x)$ — feature-wise multiplicative modulation, so even a single linear head
sees a genuinely $\tau$-conditioned input. This matters more here than in a from-scratch IQN, precisely
because the encoder is *frozen*: I cannot adapt $\psi$ to make room for $\tau$, so the interaction has to
be forced through the only thing I control, and multiplication forces it through the shallow head.

What embedding $\phi(\tau)$? Feeding the raw scalar through a linear layer is rank-one in $\tau$ and
can't represent rich dependence. I want a basis expansion: lift the scalar into many features varying at
different rates so a linear layer on top can synthesize an arbitrary smooth function of $\tau$. Cosines
of increasing frequency are the natural bounded basis on an interval — a Fourier-type feature map. Expand
$\tau$ into $n$ cosine features $\cos(\pi i\tau)$, $i=1,\dots,n$, then a linear-then-ReLU into the 512-dim
space: $\phi_j(\tau)=\mathrm{ReLU}(\sum_{i=1}^n\cos(\pi i\tau)\,w_{ij}+b_j)$. (Indexing from 0 only adds a
constant cosine absorbable into the bias.) $n=64$ cosines is plenty of frequency content to describe how
the quantile function bends, and it is cheap.

Let me make sure I have not quietly defeated the point by adding capacity. The encoder is untouched and
frozen. The head is the same shallow $f$. The only new parameters are the cosine embedding's
`Linear(64, 512)` and the head maps modulated features to actions. And here the accounting is decisively
in my favor for *this* task: where the grid agent's output layer was $|\mathcal A|\times200$ — large, the
very head the scaffold's budget check is sized around — mine is $|\mathcal A|$ per evaluated $\tau$,
reused across samples. So I have *removed* the 200-fold output blowup and replaced it with one small
embedding branch. Concretely the IQN head plus embedding is a few tens of thousands of parameters over the
shared encoder, well inside the $1.05\times$-QR-DQN budget on all three action-space sizes (4, 6, 18) —
the representational gain comes from learning a function of $\tau$, not from width. That is the property I
want: a minimal, single-axis change over the quantile rung.

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
tail — the one the 12152 seed lived in — the 200-comb resolved most coarsely. The three things I would
validate: (1) the $K$-sample mean policy is stable seed-to-seed, with no resolution-induced variance of
the kind the 200-grid showed on Seaquest; (2) Seaquest is where the implicit representation gains most,
which is the falsifiable claim — if Breakout improves but Seaquest does not, then the bottleneck was
never resolution and I misread the 12152 seed; and (3) the head stays inside the parameter budget on all
of breakout (4), pong (6), and seaquest (18) actions, which the accounting above already confirms. If
IQN merely matched QR-DQN everywhere, that would say 200 quantiles already saturated these games and the
implicit representation buys nothing at this budget — a clean negative result, but I expect the Seaquest
tail to say otherwise.
