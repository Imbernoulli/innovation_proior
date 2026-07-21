I have a frozen text-to-image diffusion model and a hard ceiling of twenty denoiser calls per image,
and the only thing I get to write is the update rule inside the sampling loop. There is no prior result
to react to yet, so the job here is just to put down the floor: the simplest sampler that turns
the model's per-step noise prediction into an image inside the budget, so that everything I try later
has a number to beat. The simplest correct choice is a first-order deterministic ODE step — DDIM. But DDIM is the conclusion of
a specific argument about what the frozen network is actually committed to, and I want to walk that
argument down, because everything I try later leans on the same fact. Get the floor wrong — ship
something that diverges, or that assumes the model can do
something it was never trained to do — and every comparison above it measures against a broken baseline.

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
variance $\sigma_t^2+k_t^2$ has to equal $1-\bar\alpha_{t-1}$, which forces $k_t^2=1-\bar\alpha_{t-1}-\sigma_t^2$.
So the marginal-preserving reverse conditional is
$q_\sigma(z_{t-1}|z_t,z_0)=\mathcal N\big(\sqrt{\bar\alpha_{t-1}}\,z_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,(z_t-\sqrt{\bar\alpha_t}\,z_0)/\sqrt{1-\bar\alpha_t},\,\sigma_t^2 I\big)$,
and the induction closes: every marginal is exactly the fixed Gaussian, with one free $\sigma_t$ per
step subject to $0\le\sigma_t^2\le 1-\bar\alpha_{t-1}$. A whole degree of freedom per step survives the
marginal matching, and it is precisely the stochasticity of the reverse step.

Note what this does to the *forward* direction: by Bayes the implied forward conditional depends on $z_0$
as well as $z_{t-1}$, so for $\sigma$ below the special value the forward process is no longer Markovian.
That is fine; I never needed Markovian, only the marginals. And the family is not disconnected from what
I already trust: the ancestral DDPM posterior is the member at
$\sigma_t^2=\tfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}(1-\bar\alpha_t/\bar\alpha_{t-1})$ — the one point
where the forward process becomes Markovian again — so $\sigma$ interpolates continuously from the
stochastic sampler I know to the deterministic one I want.

Now turn the family into a generative process I can actually run, since at sample time I have $z_t$ but
not $z_0$. The network supplies it: inverting $z_t=\sqrt{\bar\alpha_t}\,z_0+\sqrt{1-\bar\alpha_t}\,\epsilon$
gives the Tweedie clean estimate $z_{0|t}=(z_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(z_t))/\sqrt{\bar\alpha_t}$,
which is exactly the substrate's helper. Plug this predicted clean latent into the reverse conditional in
place of the true $z_0$, and because $(z_t-\sqrt{\bar\alpha_t}\,z_{0|t})/\sqrt{1-\bar\alpha_t}$ is just
$\epsilon_\theta(z_t)$ by construction, the generative update collapses to
$z_{t-1}=\sqrt{\bar\alpha_{t-1}}\,z_{0|t}+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(z_t)+\sigma_t\,\epsilon$.
Three pieces, each readable: jump to where the predicted clean latent sits at level $t-1$; re-inject,
deterministically, exactly the amount of predicted noise the marginal at $t-1$ still wants to carry;
and add fresh randomness.

The reason this whole construction is *allowed* on the frozen network is the check I owe myself: write
the variational objective for this generative process and it reduces, term by term, to a per-$t$
noise-prediction MSE with weight $\gamma_t=\lambda_t^2(1-\bar\alpha_t)/(2\bar\alpha_t\sigma_t^2)$, where
the mean-gap scalar is
$\lambda_t=\sqrt{\bar\alpha_{t-1}}-\sqrt{\bar\alpha_t}\,\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}/\sqrt{1-\bar\alpha_t}$.
The reweighting-invariance from the first structural fact only rescues me if every $\gamma_t$ is
*positive* — otherwise "minimize the term" stops meaning "match $z_{0|t}$ to $z_0$" and the equivalence
rots — and since $\gamma_t\propto\lambda_t^2$ it is positive across the whole admissible $\sigma$-range.
So the minimizer of the weighted loss is the minimizer of the plain unweighted $\epsilon$-MSE I already
trained. One trained network; a continuum of generative processes indexed by $\sigma$, all sharing its
optimum.

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
no retraining.

The headache I feared was structural — that few steps must break the sampler — was an artifact of
identifying generative chain length with forward chain length, and that identification was never in the
loss. With a real, imperfect $\epsilon_\theta$ the twenty-step image will not be pixel-identical to the
thousand-step one, but the structure that makes few steps work is intact: every step is a
marginal-preserving move toward the same predicted $z_0$. And it is exactly the setting the harness hands
me: it has already chosen a 20-step grid (`self.skip` is the stride), and I just walk it, one
`predict_noise` call per step, twenty steps, twenty NFE — the budget exactly, nothing left over for a
correction.

Seeing what the deterministic member is geometrically tells me precisely how it will fail. Take the
$\sigma=0$ update at adjacent levels, change variables to $\bar z=z/\sqrt{\bar\alpha}$ and
$\varsigma=\sqrt{(1-\bar\alpha)/\bar\alpha}$, and the step becomes
$\bar z(t-\Delta t)=\bar z(t)+(\varsigma(t-\Delta t)-\varsigma(t))\,\epsilon_\theta$; sending
$\Delta t\to0$ gives $d\bar z=\epsilon_\theta\,d\varsigma$. This is Euler integration of an ODE in
$\varsigma$ — and since the optimal $\epsilon_\theta$ is, up to scale, the score of the noised data
($\nabla_{\bar z}\log p_\varsigma=-\epsilon_\theta/\varsigma$), it is exactly the probability-flow ODE of
the variance-exploding diffusion, reached from the variational construction with no Langevin machinery.
The generated image is that ODE's solution from a fixed initial condition, and the number of sampling
steps is just the fineness of the Euler grid: same $z_T$, different $S$, nearly the same image, with only
fine detail moving as the discretization coarsens. With $\sigma_t>0$ every step injects fresh noise, and a
real (imperfect) $\epsilon_\theta$ only approximately removes it — a twenty-step chain has too few
remaining steps to average the leftover down. The $\sigma=0$ process injects nothing to clean up, so
cutting steps only coarsens an otherwise smooth map. So $\eta=0$ is the safe floor not because
stochasticity is wrong in principle but because at this budget I cannot count on the remaining steps to
clean up anything I add.

I can make "the local error is not small" quantitative.
Twenty Euler steps each carry an $O(h)$ local error where $h$ is the step in
half-log-SNR $\lambda=\tfrac12\log(\bar\alpha/(1-\bar\alpha))$. Tabulate $\lambda$ for a standard
linear-$\beta$ SD schedule ($\beta$ from $10^{-4}$ to $0.02$ over $T=1000$): it runs from about $+4.6$ at
the low-noise end to about $-5.1$ at the high-noise end, a span near $9.7$. A uniform-in-$t$ twenty-step
grid (stride $50$) spaces the $\lambda$-steps unevenly — mean $|\Delta\lambda|\approx0.48$, but the
coarsest step, near the high-noise end where $\lambda$ moves fastest per unit $t$, is $|\Delta\lambda|\approx2.9$.
So $h$ is genuinely order one where it is worst, not small; the curvature of the denoiser's trajectory
between adjacent levels is real, and a first-order step ignores it by holding $z_{0|t}$ constant across the
interval and moving on a straight chord. That is a large first-order truncation error, paid twenty times
over. It also matters that DDIM is *first-order* in the solver sense the later rungs exploit: the $k=1$
member holds $z_{0|t}$ constant across the step, with no derivative estimate, no history, no intermediate
evaluation — so there is no high-order term for a large guidance scale to corrupt, which is why DDIM is the
robust fallback everywhere. The cost is exactly that truncation error, and it is what a higher-order step
would buy back by extracting the bend of the trajectory from calls I already spend.

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
clean estimate `z0t` is decoded. The full `sample` body is in the answer.

So this is the floor: a usable image on every variant since DDIM will not diverge, but FID well above what
a higher-order step reaches at the same twenty calls. By the mechanism, SDXL — largest latent, most
structure to resolve, where the order-one $|\Delta\lambda|\approx2.9$ coarseness costs most — should be the
worst of the three. The next move is forced: extract a derivative of the trajectory from the calls I
already spend, not ask for calls I do not have.
