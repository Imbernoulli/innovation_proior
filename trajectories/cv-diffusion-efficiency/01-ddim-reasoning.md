I have a frozen text-to-image diffusion model and a hard ceiling of twenty denoiser calls per image,
and the only thing I get to write is the update rule inside the sampling loop. There is no prior result
to react to yet, so the job at this rung is just to put down the floor: the simplest sampler that turns
the model's per-step noise prediction into an image inside the budget, so that everything I try later
has a number to beat. The simplest correct choice is a first-order deterministic ODE step — DDIM. But
"DDIM" is not a free-floating recipe I can just transcribe; it is the conclusion of a specific argument
about what the frozen network is actually committed to, and I want to walk that argument all the way
down, because every later rung leans on the same fact and I would rather establish it here cleanly than
gesture at it.

Start with why the original chain is slow, because if the slowness is load-bearing the whole budget
problem is hopeless. The forward process turns a clean latent $z_0$ into Gaussian noise over $T$ levels,
and the generative process is trained to invert it. The reason $T$ has to be large in the ancestral
picture is that each true reverse conditional $q(z_{t-1}|z_t)$ is only close to Gaussian when the step
is tiny — when almost no noise was added between $z_{t-1}$ and $z_t$. Take big steps and the reverse
conditional becomes some multimodal thing a Gaussian cannot model. So large $T$ is not laziness, it is
what keeps the Gaussian generative model valid, and that seems to nail the door shut: the generative
chain approximates the reverse of the forward chain, the forward chain needs many steps to stay
Gaussian-reversible, therefore the generative chain needs many steps. Slow by construction. If that
chain of reasoning held, twenty steps would be a fantasy and I should give up.

But the phrase "approximates the reverse of the forward chain" is doing unexamined work, so let me look
at what training actually pinned down. The objective is the unweighted $\epsilon$-MSE: for a level $t$,
feed the network $z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ and penalize
$\|\epsilon_\theta(z_t,t)-\epsilon\|^2$, summed (or averaged) over $t$. Two structural facts fall out of
staring at this, and together they are the whole game. First, it is a sum of independent per-$t$ terms;
the minimizer of each term does not care what weight multiplies it, so the optimal $\epsilon_\theta$ is
the same function no matter how I reweight across $t$. Second, the input the network ever sees is
$\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ — a draw from the *marginal*
$q(z_t|z_0)=\mathcal N(\sqrt{\bar\alpha_t}\,z_0,(1-\bar\alpha_t)I)$. The loss never references the joint
$q(z_{1:T}|z_0)$; it never asks how $z_t$ and $z_{t-1}$ are correlated. So the trained network is
committed only to those marginals. The Markov forward chain — the particular story that $z_t$ depends
only on $z_{t-1}$ — was one joint that happens to have these marginals, an arbitrary way of threading
the latents together, and the training never saw it. Any other joint with the same marginals is, as far
as the loss is concerned, an equally valid story, and the same network is its solution too. That is the
crack in the "slow by construction" argument: the slowness came from identifying the generative chain
with the reverse of the *Markov* chain, but the network is wedded to the marginals, not to that chain.

So I want to build a different inference process — same marginals, different joint — whose generative
reverse happens to be deterministic, and short. The constraint is exactly one: preserve
$q(z_t|z_0)=\mathcal N(\sqrt{\bar\alpha_t}\,z_0,(1-\bar\alpha_t)I)$ for every $t$. To control marginals
I specify the process backwards, conditioned on $z_0$, writing the reverse conditional
$q_\sigma(z_{t-1}|z_t,z_0)$ directly — which is also the object the generative step will mimic. Make it
Gaussian with a mean affine in $z_0$ and $z_t$ and a free covariance $\sigma_t^2 I$. The natural ansatz
puts a $\sqrt{\bar\alpha_{t-1}}\,z_0$ signal piece plus a term proportional to the residual
$(z_t-\sqrt{\bar\alpha_t}\,z_0)$ — which in distribution is $\sqrt{1-\bar\alpha_t}\,\epsilon$, the very
noise that took $z_0$ to $z_t$ — normalized by $\sqrt{1-\bar\alpha_t}$ and scaled by an unknown $k_t$.
Imposing the marginal constraint by induction downward from $t=T$, the mean comes out
$\sqrt{\bar\alpha_{t-1}}\,z_0$ automatically (at the mean of $z_t$ the residual vanishes), and the
variance forces $k_t^2=1-\bar\alpha_{t-1}-\sigma_t^2$. So the marginal-preserving reverse conditional is
$q_\sigma(z_{t-1}|z_t,z_0)=\mathcal N\big(\sqrt{\bar\alpha_{t-1}}\,z_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,(z_t-\sqrt{\bar\alpha_t}\,z_0)/\sqrt{1-\bar\alpha_t},\,\sigma_t^2 I\big)$,
and the induction closes: every marginal is exactly the fixed Gaussian, with one free $\sigma_t$ per
step subject to $0\le\sigma_t^2\le 1-\bar\alpha_{t-1}$. A whole degree of freedom per step survives the
marginal matching, and it is precisely the stochasticity of the reverse step. Note what this does to the
*forward* direction: by Bayes the implied forward conditional depends on $z_0$ as well as $z_{t-1}$, so
for $\sigma$ below the special value the forward process is no longer Markovian. That is fine — I never
needed Markovian, only the marginals.

Now turn the family into a generative process I can actually run, since at sample time I have $z_t$ but
not $z_0$. The network supplies it: inverting $z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon$
gives the Tweedie clean estimate $z_{0|t}=(z_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(z_t))/\sqrt{\bar\alpha_t}$,
which is exactly the substrate's helper. Plug this predicted clean latent into the reverse conditional in
place of the true $z_0$, and because $(z_t-\sqrt{\bar\alpha_t}\,z_{0|t})/\sqrt{1-\bar\alpha_t}$ is just
$\epsilon_\theta(z_t)$ by construction, the generative update collapses to
$z_{t-1}=\sqrt{\bar\alpha_{t-1}}\,z_{0|t}+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(z_t)+\sigma_t\,\epsilon$.
Three pieces, each readable: jump to where the predicted clean latent sits at level $t-1$; re-inject,
deterministically, exactly the amount of predicted noise the marginal at $t-1$ still wants to carry;
and add fresh randomness. The reason this whole construction is *allowed* on the frozen network is the
check I owe myself — write the variational objective for this generative process and it reduces, term by
term, to a per-$t$ noise-prediction MSE with some positive weight $\gamma_t$. By the first structural
fact (unshared per-$t$ optimum is reweighting-invariant), the minimizer of that weighted loss is the
minimizer of the plain unweighted $\epsilon$-MSE I already trained. One trained network; a continuum of
generative processes indexed by $\sigma$, all sharing its optimum.

Now spend the free knob. The endpoint I want is $\sigma_t=0$ for all $t$: the noise term vanishes and
the update becomes deterministic, $z_{t-1}=\sqrt{\bar\alpha_{t-1}}\,z_{0|t}+\sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta(z_t)$.
This is an implicit generative model — $z_0$ is a fixed deterministic pushforward of $z_T$ — which is
exactly the deterministic DDIM step. Determinism alone does not shorten the chain, though; the update is
still written for all $T$ levels. So I use the marginal-only fact a second time, more aggressively: the
loss constrains the process only through the marginals, so nothing forces the generative process to visit
every $t$. I can define the whole construction on a sub-sequence $\tau=(\tau_1<\dots<\tau_S)$ of
$[1..T]$, reusing the reverse conditional with index pairs $(\tau_i,\tau_{i-1})$ — the
marginal-consistency induction only ever used the marginals at a step's two endpoints, so it does not
care whether the steps are adjacent integers or jumps. Train at $T=1000$, sample on $S=20$, same network,
no retraining. That is the actual cure for the wall clock the original chain imposed, and it is exactly
the setting the harness hands me: it has already chosen a 20-step grid (`self.skip` is the stride), and I
just walk it, one `predict_noise` call per step, twenty steps, twenty NFE — the budget exactly, nothing
left over for a correction.

It is worth seeing why the deterministic member, in particular, is the safe few-step choice and what it
is geometrically, because that tells me precisely how it will fail. Take the $\sigma=0$ update at
adjacent levels, change variables to $\bar z=z/\sqrt{\bar\alpha}$ and $\varsigma=\sqrt{(1-\bar\alpha)/\bar\alpha}$,
and the step becomes $\bar z(t-\Delta t)=\bar z(t)+(\varsigma(t-\Delta t)-\varsigma(t))\,\epsilon_\theta$;
sending $\Delta t\to0$ gives $d\bar z=\epsilon_\theta\,d\varsigma$. The deterministic sampler is an Euler
integration of an ODE in $\varsigma$ — the probability-flow ODE of the variance-exploding diffusion,
reached purely from the variational construction with no Langevin machinery. So the generated image is
that ODE's solution from a fixed initial condition, and the number of sampling steps is just the fineness
of the Euler grid: same $z_T$, different $S$, nearly the same image, with only fine detail moving as the
discretization coarsens. With $\sigma_t>0$ every step injects fresh noise that a short chain has too few
remaining steps to average back down; the $\sigma=0$ process has no injected noise to clean up, so cutting
steps only coarsens an otherwise smooth map rather than leaving residual noise behind. That is the
robustness argument for $\eta=0$ here — and it is also exactly why this is the floor. Twenty Euler steps
each carry an $O(h)$ local error; at this step count $h$ (the spacing in half-log-SNR) is not small, the
curvature of the denoiser's trajectory between adjacent levels is real, and a first-order step ignores it
by holding the clean estimate constant across the interval and moving on a straight chord. So I expect
visibly coarser images than the model could produce with more calls — the budget is the whole difficulty,
and a first-order solver spends it least efficiently.

It also matters that this is *first-order* in the solver sense the later rungs will exploit. DDIM is the
$k=1$ member of the whole family: hold $z_{0|t}$ constant across the step and you get this single line,
with no derivative estimate, no history, no intermediate evaluation — so there is no high-order term that
a large guidance scale could corrupt. That is the same reason DDIM is the robust fallback everywhere: with
nothing to extrapolate, there is nothing to blow up. The cost is the truncation error I just named, and
it is precisely what a higher-order step would buy back by extracting the bend of the trajectory from
calls I already spend.

Two substrate specifics fix the exact form I fill, neither of which the generic derivation above
dictates. First, guidance: the loop forms the prediction in the CFG++ style this codebase fixes — the
guided $\tilde\epsilon=\epsilon_{uc}+s(\epsilon_c-\epsilon_{uc})$ feeds the *clean-image* estimate via
Tweedie, $z_{0|t}=(z_t-\sqrt{1-\bar\alpha_t}\,\tilde\epsilon)/\sqrt{\bar\alpha_t}$, but the
*renoising/direction* term uses the bare *unconditional* $\epsilon_{uc}$, not $\tilde\epsilon$. So my
deterministic step is $z_{t-1}=\sqrt{\bar\alpha_{t-1}}\,z_{0|t}+\sqrt{1-\bar\alpha_{t-1}}\,\epsilon_{uc}$.
This tempers the over-saturation a large $s$ would otherwise bake into the renoised latent, and it is part
of the substrate, not a knob I redesign. Second, I am in VAE latent space, so there is no $[-1,1]$ pixel
bound to threshold against — only the numerical behaviour of the update is in play. The fill is then the
literal DDIM-CFG++ body: text embeds, `initialize_latent`, then the loop over
`self.scheduler.timesteps` with `at = self.alpha(t)`, `at_prev = self.alpha(t - self.skip)`, the guided
prediction, the Tweedie clean estimate, and the deterministic update renoised with `noise_uc`. The final
clean estimate `z0t` is decoded. The full scaffold body is in the answer.

What to watch. FID and CLIP across SD v1.5, SD v2.0, and SDXL at NFE = 20. I expect this to be the worst
of anything sensible — a usable image on every variant, since DDIM is stable and will not diverge, but
the FID should sit well above what a second- or third-order step can reach at the same twenty calls, and
the gap should *widen* on the harder variant. SDXL has the largest latent and the most structure to
resolve in twenty coarse steps, and that is exactly where first-order coarseness costs the most, so I
expect SDXL worst by a wide margin, the easier SD variants lower. That measured gap is exactly what
should force a higher-order update at the next rung: if twenty first-order steps leave quality on the
table, the move is to extract a derivative of the trajectory from the calls I already spend, not to ask
for calls I do not have.
