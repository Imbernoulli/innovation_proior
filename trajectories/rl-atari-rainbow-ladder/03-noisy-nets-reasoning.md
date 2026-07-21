Decoupling the target carried the median from 79% to 117% — a $38$-point jump, and the typical game is now
above human. That is the broad lift I predicted from fixing a defect that taxed the whole suite, and its
size suggests the single largest gain may already be behind me, since the biggest, most systematic
arithmetic error in the floor is now corrected. The target, the loss, and the architecture are reasonable;
the value object is still a scalar, which I am saving because it is a representational change rather than a
recipe tweak. That leaves exploration and replay sampling. Take exploration first, because $\epsilon$-greedy
is the most obviously crude thing left, and there is a class of games where I can see exactly why it fails.

What is $\epsilon$-greedy doing? With probability $\epsilon$ it ignores the value function and samples a
uniform action; otherwise it acts greedily. So the exploratory behavior is a coin flip at the *action
output*, re-flipped every step, uncorrelated with the state and with the previous flip. On a game whose
reward appears only after a specific *sequence* of deliberate actions — go to a key, then to a door, then
through it — $\epsilon$-greedy has to win that coin flip in the right direction roughly $k$ times in a row to
traverse a $k$-step detour; the probability decays like $\epsilon^k$. Put $\epsilon=0.1$ and a modest $k=10$
corridor into it: per-attempt probability $0.1^{10}=10^{-10}$, and even over the whole $200$M-frame budget
($\sim5\times10^{7}$ agent steps) the expected number of successful traversals is
$5\times10^{7}\times10^{-10}=5\times10^{-3}$ — on the order of never in a whole run. And that is the
*optimistic* $k=10$; the genuinely hard-exploration games have longer corridors where $\epsilon^k$ underflows
any realistic budget. So on this family $\epsilon$-greedy provably cannot discover the reward at all — a
wall, not a slope. The randomness is *local* in two senses: local in time (each step independent, so no
coherent multi-step plan forms) and uniform across states (the same $\epsilon$ in a state I understand
perfectly and one I have never resolved). A second, softer complaint: $\epsilon$ is a number I set by hand on
a schedule, outside the learning problem — the agent has no way to learn that it should still be exploring in
one region and can stop in another.

There are a few directions, weighed against the binding constraint: one agent, one set of hyperparameters,
all $57$ games. A count-based or pseudo-count intrinsic bonus genuinely solves the $\epsilon^k$ problem by
*pulling* the agent toward novel states, but it demands a density model over $84\times84$ images — a second
large network with its own hyperparameters (bonus scale, model capacity) that cannot be right on Montezuma
and Pong at once under a single shared setting, and it changes the *reward*, confounding any median shift. A
state-dependent $\epsilon$ needs an uncertainty estimate I do not have. Fixed-scale parameter noise gets the
*structure* right but reintroduces a hand-set scale. Each alternative adds an un-shareable hyperparameter, a
second model, or a changed objective. What I want is structured exploration that *learns its own scale* from
the same loss, adding no new objective and no separate model.

Both complaints point upstream: move the stochasticity out of the action output and into the function that
produces the values. Perturb the network's *parameters* and a single fixed perturbation induces a *different
value function*, and because the perturbation flows through the conv encoder and head, its effect on behavior
is state-dependent. Hold that perturbation fixed for a stretch and the agent acts according to one coherent,
perturbed value function that can prefer a *consistent* off-greedy action in a given state — precisely the
structured, temporally-extended exploration $\epsilon$-greedy cannot produce. Contrast the two on the
$k$-step corridor: under $\epsilon$-greedy the agent needs $k$ independent lucky flips, $\epsilon^k$; under a
held parameter sample the *whole trajectory through the corridor* is one draw of one perturbed value
function, so if that function happens to prefer the corridor's first off-greedy action, it tends to prefer
the consistent continuation at each subsequent state too, because the same weights evaluate all of them. The
cost drops from "win $k$ coin flips in a row" to "draw one useful value function" — the difference between
$\epsilon^k$ and something that can happen inside the budget.

Now make the perturbation *trainable*, which answers the second complaint. Let a noisy parameter be
$\theta=\mu+\sigma\odot\epsilon$, with $\epsilon$ zero-mean fixed-statistics noise (drawn each time, not
learned) and $\mu,\sigma$ learnable. This is not a posterior — it is a parameterized noise source whose
*scale* $\sigma$ is trained by gradient descent. The objective becomes the expectation over the noise,
$\bar L(\mu,\sigma)=\mathbb{E}_\epsilon[L(\mu+\sigma\odot\epsilon)]$, and because the noise distribution does
not depend on $\mu,\sigma$, I can pull the gradient inside and estimate it with a single draw
(reparameterization). Componentwise, $\partial\theta_i/\partial\mu_i=1$ and
$\partial\theta_i/\partial\sigma_i=\epsilon_i$, so $\partial L/\partial\mu_i=\partial L/\partial\theta_i$ and
$\partial L/\partial\sigma_i=(\partial L/\partial\theta_i)\,\epsilon_i$: the $\sigma$-gradient is the
weight-gradient scaled by the noise injected on that weight. A parameter whose noise happened to *reduce* the
loss gets its $\sigma$ pushed up (more noise there next time), one whose noise *raised* the loss gets its
$\sigma$ pulled down. Averaged over draws, $\sigma$ settles wherever injected variation stops helping the TD
loss — up where exploration still helps, toward zero where behavior has settled — automatically and per
parameter, with no external schedule. That is the self-regulated exploration I wanted, and it makes the
hand-set schedule vanish for free: I never write an exploration schedule; the $\sigma$ values *are* one, and
they are learned per game and per layer.

Concretely, replace the head's fully-connected layers with *noisy linear* layers. For $p$ inputs and $q$
outputs, $y=(\mu^w+\sigma^w\odot\epsilon^w)x+(\mu^b+\sigma^b\odot\epsilon^b)$, with $\mu^w,\sigma^w$ of shape
$q\times p$. Drawing a full $q\times p$ noise matrix per step is too expensive on a single-threaded value
agent, so *factor* the noise: draw $p$ input noises and $q$ output noises, pass each through
$f(x)=\operatorname{sign}(x)\sqrt{|x|}$, and set
$\epsilon^w_{j,i}=f(\epsilon^{\text{out}}_j)\,f(\epsilon^{\text{in}}_i)$,
$\epsilon^b_j=f(\epsilon^{\text{out}}_j)$. The weight-noise tensor is an outer product, so the count of
Gaussian draws drops from $pq+q$ to $p+q$. On the larger head layer, $p=3136$, $q=512$: unfactorized needs
$\approx1.61\times10^6$ fresh samples every step, factorized needs $3648$ — a $440\times$ reduction, resampled
once per action over $50$M steps, which is the difference between a trivial cost and one rivaling the matmul.
The transform $f$ is odd, so each factor is zero-mean, and a factorized entry stays order-one; the exact
constant does not matter because the learnable $\sigma$ absorbs it.

The parameter cost: each noisy linear layer carries a $\mu$ and a $\sigma$ of the same shape, so it doubles
that layer's learnable parameters. I make only the two *head* layers noisy — the $3136\to512$ and
$512\to|\mathcal A|$ maps, not the conv encoder — so the added parameters are one extra copy of
$3136\times512\approx1.6$M plus one extra $512\times|\mathcal A|$, on a $\sim1.7$M-parameter network. The
$\sigma$ parameters are cheap to train and mostly collapse toward small values as behavior settles, so the
*effective* added capacity is far less than the raw count. Leaving the conv layers deterministic is
deliberate: I want exploration over *behavior*, which the head controls; perturbing the feature extractor
would inject noise into perception itself.

Wiring: delete the $\epsilon$-greedy schedule and act *greedily* under the current sampled value network —
exploration now comes from the parameter noise. The resampling rule follows from what the noise is *for* at
each moment. During acting, the noise *is* the exploration, so hold one sample fixed across a stretch to
produce coherent multi-step behavior rather than re-drawing every step (which would be the memoryless
dithering I am escaping). During a learning update, the noise is what makes the loss an expectation over
perturbed functions, so draw fresh noise per optimization step. This agent updates once per action, so it
resamples before each action and holds across a replay batch. Online net and target net get *independent*
draws ($\epsilon$ vs $\epsilon'$): sharing a draw would correlate the bootstrapped target with the current
estimate, the coupling the floor's target network exists to avoid. So the Double-DQN target carries over —
the online sampled net selects, the target sampled net (independent noise) evaluates. The two decorrelations
are orthogonal: the decoupled *target* separates which network selects from which evaluates; the decoupled
*noise* separates the online and target perturbations. Keeping the decoupled target underneath while adding
parameter noise loses neither guarantee.

I should be sober about what this moves. The benefit is concentrated on the *hard-exploration* games — the
long reward-free corridors $\epsilon$-greedy could never traverse — and there the gain can be large. But
those are a *minority* of the 57. On the bulk of the suite, where the decoupled-target agent already explores
adequately and the bottleneck is value estimation, replacing dithering with learned parameter noise changes
little. A method that lifts a minority dramatically while leaving the median game — a typical, already-
explorable title — untouched reorders the top of the list and moves the mean, but the $29$th score barely
budges. This is the mirror image of the decoupled target, whose fix touched every multi-action game and
moved the center by $38$ points: one broad and median-moving, one narrow and tail-moving. So I expect a
near-flat median here, with the real action hidden in the tails. A near-flat result would itself be a
measurement, not a failure — it would say exploration was not what capped the *typical* game, precisely
because the method genuinely fixes exploration where exploration was the problem. That reorients me toward
whatever limits the typical game's learning across the board — the kind of defect that, like the biased max,
taxes every game rather than a few, which is where the next real move on the median has to come from.
