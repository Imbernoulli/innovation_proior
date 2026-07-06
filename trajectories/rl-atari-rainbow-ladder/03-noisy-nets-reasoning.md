Decoupling the target carried the median from 79% to 117% — a $38$-point jump, and the typical game is now
above human ($117>100$). That is the broad lift I predicted from fixing a defect that taxed the whole
suite, and its size tells me something useful going forward: the single largest gain available may already
be behind me, since the biggest, most systematic arithmetic error in the floor is now corrected. What
remains are axes whose fixes I expect to be more selective. With the worst arithmetic defect of the floor
fixed, I look down my list of untouched axes and ask which one is quietly costing me games. The target, the loss, and the architecture are now reasonable; the value object
is still a scalar, which I am saving because it is a representational change rather than a recipe tweak.
That leaves exploration and replay sampling. Take exploration first, because $\epsilon$-greedy is the most
obviously crude thing left in the agent, and there is a class of games where I can see exactly why it
fails.

What is $\epsilon$-greedy actually doing? With probability $\epsilon$ it ignores the value function and
samples a uniform action; otherwise it acts greedily. So the exploratory behavior is a coin flip at the
*action output*, re-flipped every single step, completely uncorrelated with the state and with the
previous flip. Think about what that buys on a game where reward only appears after a specific *sequence*
of several deliberate actions — go to a key, then to a door, then through it — with nothing in between.
$\epsilon$-greedy explores such a corridor by, at each step, with small probability replacing the greedy
action with a random one. To traverse a $k$-step exploratory detour by chance it has to win that coin flip
in the right direction roughly $k$ times in a row; the probability decays like $\epsilon^k$, and on the
hard-exploration games $k$ is large enough that it essentially never happens. Put the fixed $\epsilon=0.1$
into that: a modest $k=10$-step corridor has per-attempt probability $0.1^{10}=10^{-10}$, and even over the
entire $200$M-frame budget — about $5\times10^{7}$ agent steps, so at most that many attempts — the expected
number of successful traversals is $5\times10^{7}\times10^{-10}=5\times10^{-3}$, i.e. it happens on the
order of never in a whole training run. And that is the *optimistic* $k=10$; the genuinely hard-exploration
games have longer corridors, where $\epsilon^k$ underflows any realistic budget. So on this family of games
$\epsilon$-greedy does not merely explore slowly, it provably cannot discover the reward at all within the
frames I have — a wall, not a slope. The randomness is *local* in
two senses: it is local in time (each step independent, so no coherent multi-step plan ever forms) and it
is uniform across states (the same $\epsilon$ in a state I understand perfectly and a state I have never
resolved). That is the defect — unstructured, memoryless, state-blind noise — and it is exactly what caps
the agent on the family of games that need committed, structured exploration.

A second, softer complaint: $\epsilon$ is a number I set by hand on a schedule, outside the learning
problem. The agent has no way to *learn* that it should still be exploring in one region and can stop in
another. I would like the amount and the shape of the exploratory noise to be trained by the same loss that
trains the values, so it can be large where behavior is still unsettled and shrink where it has stabilized,
per state, automatically.

There are a few directions I could take to fix this, and I should weigh them against the constraint that
binds this whole benchmark: one agent, one set of hyperparameters, all $57$ games. The best-known
principled cure for hard exploration is a count-based or pseudo-count intrinsic bonus — reward the agent
for visiting novel states, so it is *pulled* toward the unexplored corridor rather than having to stumble
down it by coincidence. It genuinely solves the $\epsilon^k$ problem. But it demands a density model over
$84\times84$ images to estimate visitation counts, which is a second large network with its own training
dynamics and its own hyperparameters (the bonus scale, the density model's capacity), and getting that
scale right on Montezuma without wrecking Pong across a single shared setting is exactly the kind of
per-game tuning the benchmark forbids. It also changes the *reward*, so any median shift would confound
"better exploration" with "a modified objective." A second option is to keep $\epsilon$-greedy but make
$\epsilon$ state-dependent — larger where the agent is uncertain — but "uncertain" needs an uncertainty
estimate I do not have, so this just pushes the problem back a level. A third is fixed-scale parameter
noise: perturb the weights by noise of a hand-set magnitude. That gets the *structure* right (as I am about
to argue) but reintroduces a hand-set scale that cannot be right for every game and every stage of training.
Each alternative either adds an un-shareable hyperparameter or a second model or a changed objective. What I
want is a mechanism that gives structured exploration *and* learns its own scale from the same loss, adding
no new objective and no separate model — and that points somewhere specific.

Both complaints point the same direction: move the stochasticity *upstream*, out of the action output and
into the function that produces the values. If I perturb the network's *parameters* rather than its chosen
action, then a single fixed perturbation induces a *different value function*, and because the perturbation
flows through the conv encoder and the head, its effect on behavior depends on the input state. Hold that
perturbation fixed for a stretch and the agent acts according to one coherent, perturbed value function —
which can prefer a *consistent* off-greedy action in a given state across the whole stretch, rather than a
fresh independent dice roll each step. That is precisely the structured, temporally-extended,
state-dependent exploration $\epsilon$-greedy cannot produce: a perturbed net might decide "in *this* kind
of state, try the door" and stick to it for as long as the sample is held, which is how a multi-step detour
actually gets traversed. Contrast the two on the $k$-step corridor directly. Under $\epsilon$-greedy the
agent needs $k$ independent lucky coin flips, probability $\epsilon^k$, because each step re-rolls. Under a
held parameter sample the *whole trajectory through the corridor* is one draw of one perturbed value
function: if that function happens to prefer the corridor's first off-greedy action, it tends to prefer the
consistent continuation at each subsequent state too, because the same weights evaluate all of them — so a
single lucky draw can carry the agent through many correlated steps at once. The exploration cost drops from
"win $k$ coin flips in a row" to "draw one useful value function," which is the difference between
$\epsilon^k$ and something that can actually happen inside the budget. That is the mechanism by which
structured, temporally-correlated noise defeats a corridor that memoryless dithering cannot.

Now make the perturbation *trainable*, which is what answers the second complaint. Let a noisy parameter be
$\theta=\mu+\sigma\odot\epsilon$, with $\epsilon$ a vector of zero-mean fixed-statistics noise (drawn each
time, not learned) and $\mu,\sigma$ learnable, $\odot$ elementwise. This is not a posterior and I will not
pretend it is — it is a parameterized source of noise whose *scale* $\sigma$ is trained by gradient
descent. The objective becomes the expectation over the noise,
$\bar L(\mu,\sigma)=\mathbb{E}_\epsilon[L(\mu+\sigma\odot\epsilon)]$, and because the noise distribution
does not depend on $\mu,\sigma$, I can pull the gradient inside and estimate it with a single draw
(reparameterization): the gradient w.r.t. $\mu$ is the ordinary weight gradient, and the gradient w.r.t.
$\sigma$ is that same local gradient multiplied by the sampled noise $\epsilon$. Let me check that
componentwise so I trust it. Write $\theta=\mu+\sigma\odot\epsilon$; then $\partial\theta_i/\partial\mu_i=1$
and $\partial\theta_i/\partial\sigma_i=\epsilon_i$, so by the chain rule
$\partial L/\partial\mu_i=\partial L/\partial\theta_i$ and
$\partial L/\partial\sigma_i=(\partial L/\partial\theta_i)\,\epsilon_i$. The $\sigma$-gradient is literally
the weight-gradient scaled by the noise that was injected on that weight — so a parameter whose noise
happened to *reduce* the loss gets its $\sigma$ pushed up (more noise there next time), and one whose noise
*raised* the loss gets its $\sigma$ pulled down. Averaged over draws and steps, $\sigma$ settles wherever
injected variation stops helping the TD loss — the update rule is thus derived, not asserted. So backprop
directly learns, per parameter, whether more or less injected variation lowers the loss — exactly the
self-regulated exploration I wanted. Where the perturbation still helps, $\sigma$ stays up; where the behavior has settled
and noise only hurts the TD loss, $\sigma$ is driven toward zero, automatically and per parameter, with no
external schedule.

Concretely, replace the fully-connected layers of the head with *noisy linear* layers. For a layer with
$p$ inputs and $q$ outputs the map becomes
$y=(\mu^w+\sigma^w\odot\epsilon^w)x+(\mu^b+\sigma^b\odot\epsilon^b)$, with $\mu^w,\sigma^w$ of shape
$q\times p$ and $\mu^b,\sigma^b$ of shape $q$. Drawing a full $q\times p$ noise matrix per layer every step
is the obvious thing but it is too expensive relative to the matmul on a single-threaded value agent, so I
*factor* the noise: draw $p$ input noises and $q$ output noises, pass each through
$f(x)=\operatorname{sign}(x)\sqrt{|x|}$, and set $\epsilon^w_{j,i}=f(\epsilon^{\text{out}}_j)\,
f(\epsilon^{\text{in}}_i)$ and $\epsilon^b_j=f(\epsilon^{\text{out}}_j)$. The weight-noise tensor is an
outer product, so the count of Gaussian draws drops from $pq+q$ to $p+q$. Put numbers on the larger head
layer, $p=3136$ inputs to $q=512$ outputs: unfactorized needs $pq+q=3136\times512+512\approx1.61\times10^6$
fresh Gaussian samples *every step*, while factorized needs $p+q=3648$ — a $440\times$ reduction in draws,
and I resample once per action over $50$M steps, so this is the difference between a trivial cost and one
that rivals the matmul itself. The factorization is not a nicety; it is what makes per-step resampling
affordable at all. The transform keeps each factor
zero-mean and order-one. Check it: $f(x)=\operatorname{sign}(x)\sqrt{|x|}$ is an odd function and $Z$ is
symmetric, so $\mathbb{E}[f(Z)]=0$ by cancellation; and $f(Z)^2=|\operatorname{sign}(Z)|^2|Z|=|Z|$, so
$\mathbb{E}[f(Z)^2]=\mathbb{E}|Z|=\sqrt{2/\pi}\approx0.80$ for a standard normal. A factorized weight entry
is the product of two independent such factors, so its variance is
$\mathbb{E}[f(\epsilon^{\text{out}})^2]\,\mathbb{E}[f(\epsilon^{\text{in}})^2]=\sqrt{2/\pi}\cdot\sqrt{2/\pi}
=2/\pi\approx0.64$ — order one, not exactly one, which is fine because the learnable $\sigma$ absorbs the
constant. The point of the check is that factorizing did not secretly change the *scale* of the injected
noise into something the network cannot calibrate; each entry stays a well-behaved order-one perturbation
whose amplitude $\sigma$ then learns.

I should count what this costs in parameters, because I am doubling something. Each noisy linear layer
carries a $\mu$ and a $\sigma$ of the same shape for both weight and bias, so it has exactly twice the
learnable parameters of the plain layer it replaces. I only make the two *head* layers noisy — the
$3136\to512$ and $512\to|\mathcal A|$ maps — not the conv encoder, so the added parameters are one extra
copy of $3136\times512\approx1.6$M plus one extra $512\times|\mathcal A|$, roughly a $1.6$M-parameter
increase on a $\sim1.7$M-parameter network. That sounds like a near-doubling, but the $\sigma$ parameters
are cheap to train (their gradient is just the weight-gradient times $\epsilon$) and they mostly collapse
toward small values as behavior settles, so the *effective* added capacity is far less than the raw count.
Leaving the conv layers deterministic is deliberate: the exploration I want is over *behavior*, which the
head controls, and perturbing the feature extractor would inject noise into perception itself, which is not
what needs to be stochastic.

Wiring into the value learning: delete the $\epsilon$-greedy schedule entirely and act *greedily* under the
current sampled value network — the exploration now comes from the parameter noise, not from a separate
action-dithering rule. The discipline that matters is when to resample, and the rule follows from what the noise is *for* at each
moment. During acting, the noise is the exploration, so I want it held long enough to produce coherent
multi-step behavior — hold one sample fixed across a stretch rather than re-drawing every step, or I am back
to the memoryless dithering I am trying to escape. During a learning update, the noise is what makes the
loss an expectation over perturbed functions, so I draw fresh noise per optimization step so the
$\sigma$-gradient sees a new $\epsilon$ to correlate against. This value agent updates once per action, so
in practice it resamples the noise before each action and holds it across a replay batch — one draw serving
both the step's behavior and the step's gradient. Online net and target net get *independent*
noise draws ($\epsilon$ vs $\epsilon'$): sharing the same draw between them would correlate the bootstrapped
target with the current estimate, which is exactly the kind of coupling the floor's target network exists
to avoid. So the Double-DQN target I keep from the previous rung becomes: the online sampled net selects
the next action, the target sampled net (independent noise) evaluates it. The composition is clean because
the two decorrelations are orthogonal: the decoupled *target* separates which network selects from which
evaluates (guarding against selection-evaluation bias), while the decoupled *noise* separates the online and
target perturbations (guarding against the estimate being correlated with its own bootstrap). I want both,
and they do not interfere — one is about which weights, the other about which noise draw — so keeping the
decoupled target underneath while adding parameter noise on top loses neither guarantee.

Worth noting how neatly this answers the second complaint — the hand-set schedule — as a free byproduct.
I never write down an exploration schedule at all now; the $\sigma$ values *are* the schedule, and they are
learned. Early in training, when behavior is unsettled and injected noise helps the agent stumble onto
reward, the $\sigma$-gradient keeps $\sigma$ large; late, when the value function has stabilized and any
perturbation only raises the TD loss, the same gradient drives $\sigma$ toward zero and the agent quietly
becomes near-deterministic. That is a per-parameter, per-game, automatically-annealed exploration rate — the
opposite of the single global $\epsilon(t)$ line I used to hand-tune — and it costs nothing beyond the
$\sigma$ parameters I already added. The floor's $\epsilon$-greedy annealed one scalar on a schedule I
guessed; this anneals thousands of scales by gradient descent, each to its own game and its own layer.

I should be sober about what this rung will and will not move, because it tells me where to set the bar.
The benefit is concentrated on the *hard-exploration* games — the ones with long reward-free corridors that
$\epsilon$-greedy could never traverse — and there the gain can be large. But those are a *minority* of the
57. On the bulk of the suite, where Double DQN already explores adequately and the bottleneck is value
estimation rather than discovery, replacing the dithering with learned parameter noise changes little: a
working agent that found enough reward signal under $\epsilon$-greedy will find it under noisy nets too.
The median is a robustness statistic, so a method that lifts a minority of games dramatically while leaving
the majority roughly where they were will move the *median* only slightly — the median game is not a
hard-exploration game. Make the arithmetic explicit: the median is the $29$th of $57$ sorted scores, so it
only moves if the change reaches the *middle* of the ordering. A method that sends five or ten
hard-exploration games from near-zero to spectacular reorders the top of the list but leaves the $29$th
game — a typical, already-explorable title — essentially untouched, so the median barely budges even as the
mean and the tails jump. This is the opposite profile from the decoupled target, whose fix touched every
multi-action game and so moved the center by $38$ points. So I am predicting the two changes have almost
mirror-image signatures: one broad and median-moving, one narrow and tail-moving. So I deliberately expect this to clear the 117% bar but only barely: a near-flat
median, with the real action hidden in the tails on the exploration-bound titles. That modest median is
itself informative — a near-flat result here is not a failure of the method but a measurement: it says
exploration was not what was capping the *typical* game, and it says so precisely because the method
genuinely fixes exploration where exploration was the problem, so a flat median localizes the remaining
bottleneck to the games where exploration was *already* adequate, i.e. almost all of them. That reorients me
away from a minority's discovery problem and toward whatever limits the *typical* game's learning across the
board — the kind of defect that, like the biased max before it, taxes every game rather than a few, and so
is where the next real move on the median has to come from.
