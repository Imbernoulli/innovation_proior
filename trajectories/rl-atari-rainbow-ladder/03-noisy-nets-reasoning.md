Decoupling the target carried the median from 79% to 117% — the typical game is now above human. With the
worst arithmetic defect of the floor fixed, I look down my list of untouched axes and ask which one is
quietly costing me games. The target, the loss, and the architecture are now reasonable; the value object
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
hard-exploration games $k$ is large enough that it essentially never happens. The randomness is *local* in
two senses: it is local in time (each step independent, so no coherent multi-step plan ever forms) and it
is uniform across states (the same $\epsilon$ in a state I understand perfectly and a state I have never
resolved). That is the defect — unstructured, memoryless, state-blind noise — and it is exactly what caps
the agent on the family of games that need committed, structured exploration.

A second, softer complaint: $\epsilon$ is a number I set by hand on a schedule, outside the learning
problem. The agent has no way to *learn* that it should still be exploring in one region and can stop in
another. I would like the amount and the shape of the exploratory noise to be trained by the same loss that
trains the values, so it can be large where behavior is still unsettled and shrink where it has stabilized,
per state, automatically.

Both complaints point the same direction: move the stochasticity *upstream*, out of the action output and
into the function that produces the values. If I perturb the network's *parameters* rather than its chosen
action, then a single fixed perturbation induces a *different value function*, and because the perturbation
flows through the conv encoder and the head, its effect on behavior depends on the input state. Hold that
perturbation fixed for a stretch and the agent acts according to one coherent, perturbed value function —
which can prefer a *consistent* off-greedy action in a given state across the whole stretch, rather than a
fresh independent dice roll each step. That is precisely the structured, temporally-extended,
state-dependent exploration $\epsilon$-greedy cannot produce: a perturbed net might decide "in *this* kind
of state, try the door" and stick to it for as long as the sample is held, which is how a multi-step detour
actually gets traversed.

Now make the perturbation *trainable*, which is what answers the second complaint. Let a noisy parameter be
$\theta=\mu+\sigma\odot\epsilon$, with $\epsilon$ a vector of zero-mean fixed-statistics noise (drawn each
time, not learned) and $\mu,\sigma$ learnable, $\odot$ elementwise. This is not a posterior and I will not
pretend it is — it is a parameterized source of noise whose *scale* $\sigma$ is trained by gradient
descent. The objective becomes the expectation over the noise,
$\bar L(\mu,\sigma)=\mathbb{E}_\epsilon[L(\mu+\sigma\odot\epsilon)]$, and because the noise distribution
does not depend on $\mu,\sigma$, I can pull the gradient inside and estimate it with a single draw
(reparameterization): the gradient w.r.t. $\mu$ is the ordinary weight gradient, and the gradient w.r.t.
$\sigma$ is that same local gradient multiplied by the sampled noise $\epsilon$. So backprop directly
learns, per parameter, whether more or less injected variation lowers the loss — exactly the self-regulated
exploration I wanted. Where the perturbation still helps, $\sigma$ stays up; where the behavior has settled
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
outer product, so the count of Gaussian draws drops from $pq+q$ to $p+q$; the transform keeps each factor
zero-mean and order-one (for $Z\sim N(0,1)$, $\mathbb{E}[f(Z)]=0$, $\mathbb{E}[f(Z)^2]=\mathbb{E}|Z|=
\sqrt{2/\pi}$, so a factorized weight entry has variance $2/\pi$ — order one, not exactly one).

Wiring into the value learning: delete the $\epsilon$-greedy schedule entirely and act *greedily* under the
current sampled value network — the exploration now comes from the parameter noise, not from a separate
action-dithering rule. The discipline that matters is when to resample. The rule is to hold one sample
fixed between optimization steps; this value agent updates once per action, so in practice it resamples the
noise before each action and holds it across a replay batch. Online net and target net get *independent*
noise draws ($\epsilon$ vs $\epsilon'$): sharing the same draw between them would correlate the bootstrapped
target with the current estimate, which is exactly the kind of coupling the floor's target network exists
to avoid. So the Double-DQN target I keep from the previous rung becomes: the online sampled net selects
the next action, the target sampled net (independent noise) evaluates it.

I should be sober about what this rung will and will not move, because it tells me where to set the bar.
The benefit is concentrated on the *hard-exploration* games — the ones with long reward-free corridors that
$\epsilon$-greedy could never traverse — and there the gain can be large. But those are a *minority* of the
57. On the bulk of the suite, where Double DQN already explores adequately and the bottleneck is value
estimation rather than discovery, replacing the dithering with learned parameter noise changes little: a
working agent that found enough reward signal under $\epsilon$-greedy will find it under noisy nets too.
The median is a robustness statistic, so a method that lifts a minority of games dramatically while leaving
the majority roughly where they were will move the *median* only slightly — the median game is not a
hard-exploration game. So I deliberately expect this to clear the 117% bar but only barely: a near-flat
median, with the real action hidden in the tails on the exploration-bound titles. That modest median is
itself informative — it says exploration was not what was capping the *typical* game, which sends me, next,
to the axis that touches every game's data efficiency rather than a minority's discovery problem: how the
replay buffer samples.
