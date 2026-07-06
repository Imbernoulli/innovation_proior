I have a frozen text-to-image diffusion model and a hard ceiling of twenty denoiser calls per image,
and the only thing I get to write is the update rule inside the sampling loop. There is no prior result
to react to yet, so the job at this rung is just to put down the floor: the simplest sampler that turns
the model's per-step noise prediction into an image inside the budget, so that everything I try later
has a number to beat. The simplest correct choice is a first-order deterministic ODE step — DDIM. But
"DDIM" is not a free-floating recipe I can just transcribe; it is the conclusion of a specific argument
about what the frozen network is actually committed to, and I want to walk that argument all the way
down, because every later rung leans on the same fact and I would rather establish it here cleanly than
gesture at it. If I get the floor wrong — if I ship something that diverges, or that quietly assumes the
model can do something it was never trained to do — then every comparison above it is measuring against
a broken baseline, so this rung is worth doing carefully even though it is "just" DDIM.

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

This is the load-bearing algebra for every rung above, so I do not want to trust a hand-rolled
marginalization I am going to hang everything on — let me run it on numbers. Take a tiny schedule of
cumulative coefficients $\bar\alpha=\{1,\,0.8,\,0.5,\,0.2\}$ at levels $t=0,1,2,3$, fix $z_0=1.7$, and
at each step set the interior $\sigma_t^2$ to $30\%$ of its allowed ceiling $1-\bar\alpha_{t-1}$, which
is an arbitrary point strictly inside the family. Compute $k_t=\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}$ and
the affine slope $b=k_t/\sqrt{1-\bar\alpha_t}$, then propagate the mean and variance of $z_{t-1}$ from
$z_t\sim\mathcal N(\sqrt{\bar\alpha_t}z_0,1-\bar\alpha_t)$ through the conditional. At $t=3$: $\sigma^2=0.150$,
$k=0.5916$, $b=0.6614$, giving out-mean $1.202082$ against target $\sqrt{\bar\alpha_2}\,z_0=1.202082$ and
out-var $0.500000$ against $1-\bar\alpha_2=0.5$. At $t=2$: out-mean $1.520526$ against $1.520526$, out-var
$0.200000$ against $0.2$. At $t=1$ with $\sigma^2=0$: out-mean $1.700000$, out-var $0$. Mean and variance
land on the target marginal to every digit I printed, for a $\sigma$ I chose by whim — so the induction
is not just formally closed, the propagated moments actually match. Note also what this does to the
*forward* direction: by Bayes the implied forward conditional depends on $z_0$ as well as $z_{t-1}$, so
for $\sigma$ below the special value the forward process is no longer Markovian. That is fine; I never
needed Markovian, only the marginals, and those just checked out.

One more confidence check before I lean on this family: does it actually *contain* the sampler everyone
already trusts, or have I built something elegant but disconnected? The original ancestral (DDPM) reverse
posterior $q(z_{t-1}|z_t,z_0)$ is a specific Gaussian, and there is a candidate
$\sigma_t^2=\tfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}(1-\bar\alpha_t/\bar\alpha_{t-1})$ that should
reproduce it if my family is right. On the toy schedule I compare all three moments — variance, the
$z_t$-coefficient, the $z_0$-coefficient — of the DDPM posterior against $q_\sigma$ at that candidate. At
$t=3$: variance $0.375000$ vs $0.375000$, $z_t$-coeff $0.395285$ vs $0.395285$, $z_0$-coeff $0.530330$ vs
$0.530330$. At $t=2$: $0.150000$ vs $0.150000$, $0.316228$ vs $0.316228$, $0.670820$ vs $0.670820$. Every
number matches to six digits, so the ancestral sampler really is one member of this family — the point
$\sigma=\text{that value}$ — and at that $\sigma$ the forward process is Markovian again. The construction
is not off in some parallel universe; it interpolates continuously from the sampler I know to the
deterministic one I want.

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

The reason this whole construction is *allowed* on the frozen network is the check I owe myself, and it
is not enough to wave at it — write the variational objective for this generative process and it reduces,
term by term, to a per-$t$ noise-prediction MSE with some weight $\gamma_t$. Each term is a KL between two
Gaussians of equal covariance $\sigma_t^2 I$, so it is $\|\mu_q-\mu_p\|^2/(2\sigma_t^2)$, and the mean gap
is $\lambda_t\,(z_0-z_{0|t})$ with the scalar
$\lambda_t=\sqrt{\bar\alpha_{t-1}}-\sqrt{\bar\alpha_t}\,\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}/\sqrt{1-\bar\alpha_t}$.
Converting $z_0-z_{0|t}$ to noise space multiplies by $\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$, so
$\gamma_t=\lambda_t^2(1-\bar\alpha_t)/(2\bar\alpha_t\sigma_t^2)$. The reweighting-invariance from the first
structural fact only rescues me if every $\gamma_t$ is *positive* — otherwise "minimize the term" would
stop meaning "match $z_{0|t}$ to $z_0$," and the whole equivalence rots. So $\lambda_t^2>0$ is the thing
to check, not assume. On the same $\bar\alpha=\{1,0.8,0.5,0.2\}$ schedule, sweeping $\sigma_t^2$ from $0$
up to its ceiling: at $t=3$, $\lambda=+0.3536$ at $\sigma^2=0$, $+0.4113$ at $30\%$, $+0.5135$ at $70\%$,
$+0.6959$ approaching the ceiling; at $t=2$, $+0.4472,\,+0.5203,\,+0.6495,\,+0.8803$ across the same
sweep. Strictly positive everywhere and bounded away from zero, so $\gamma_t>0$ across the whole
admissible $\sigma$-range. That closes it: for every positive $\gamma$, the minimizer of the weighted loss
is the minimizer of the plain unweighted $\epsilon$-MSE I already trained. One trained network; a
continuum of generative processes indexed by $\sigma$, all sharing its optimum.

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

That "same network, fewer steps, same image" claim is exactly the kind of thing that sounds too good and
should be run rather than believed, so I trace the actual deterministic sampler on a controlled input.
Build an oracle whose $\epsilon_\theta(z_t,t)$ returns the true noise for a planted $z_0=2.0$ — i.e. a
perfect noise predictor — lay down an eight-level schedule with $\bar\alpha$ marching from $0.98$ down to
$0.02$, and run the $\sigma=0$ sub-sequence sampler on trajectories of very different length from the same
top-level draw. A single jump from the noisiest level, $\text{seq}=[7]$, returns $z_{0}$-estimate
$2.00000000$; four steps $[0,2,4,7]$ return $2.00000000$; the full eight-step grid returns $2.00000000$.
With a perfect predictor the deterministic sampler recovers $z_0$ exactly regardless of how many steps it
takes — even one jump from the top. That is the strongest form of the consistency property, exercised on
the real code path, and it is what makes me believe the sub-sequence cure: the headache I feared was
structural was an artifact of identifying generative chain length with forward chain length, and that
identification was never in the loss. With a real, imperfect $\epsilon_\theta$ the twenty-step image will
not be pixel-identical to the thousand-step one, but the structure that makes few steps work — every step
is a marginal-preserving move toward the same predicted $z_0$ — is what the trace just confirmed. And it is
exactly the setting the harness hands me: it has already chosen a 20-step grid (`self.skip` is the stride),
and I just walk it, one `predict_noise` call per step, twenty steps, twenty NFE — the budget exactly,
nothing left over for a correction.

It is worth seeing why the deterministic member, in particular, is the safe few-step choice and what it
is geometrically, because that tells me precisely how it will fail. Take the $\sigma=0$ update at
adjacent levels, change variables to $\bar z=z/\sqrt{\bar\alpha}$ and $\varsigma=\sqrt{(1-\bar\alpha)/\bar\alpha}$,
and the step becomes $\bar z(t-\Delta t)=\bar z(t)+(\varsigma(t-\Delta t)-\varsigma(t))\,\epsilon_\theta$;
sending $\Delta t\to0$ gives $d\bar z=\epsilon_\theta\,d\varsigma$. The deterministic sampler is an Euler
integration of an ODE in $\varsigma$. I can cross-check that this is the right ODE against the
score-matching picture rather than take it on faith: the optimal $\epsilon_\theta$ predicts, up to scale,
the score of the noised data, $\nabla_{\bar z}\log p_\varsigma=-\epsilon_\theta/\varsigma$, and the
probability-flow ODE of the variance-exploding diffusion is
$d\bar z=-\tfrac12(d\varsigma^2/dt)\nabla_{\bar z}\log p\,dt=\tfrac12(d\varsigma^2/dt)(\epsilon_\theta/\varsigma)\,dt
=(\varsigma\,d\varsigma/dt)(\epsilon_\theta/\varsigma)\,dt=\epsilon_\theta\,d\varsigma$ — identical. So the
deterministic sampler is the probability-flow ODE, reached purely from the variational construction with
no Langevin machinery, and the two views are the same ODE with different discretizations. The generated
image is that ODE's solution from a fixed initial condition, and the number of sampling steps is just the
fineness of the Euler grid: same $z_T$, different $S$, nearly the same image, with only fine detail moving
as the discretization coarsens. With $\sigma_t>0$ every step injects fresh noise that a short chain has too
few remaining steps to average back down; the $\sigma=0$ process has no injected noise to clean up, so
cutting steps only coarsens an otherwise smooth map rather than leaving residual noise behind. That is the
robustness argument for $\eta=0$ here. I should be honest about what that argument does *not* rest on: I
re-ran the four-step oracle trace with $\eta=1$ (the full stochastic member) as well, and it too returns
$z_0=2.0$ with zero variance across two hundred seeds — because a *perfect* denoiser removes whatever noise
the previous step injected, so under an oracle the stochasticity is harmless and both members are exact.
The residual-noise penalty is therefore specifically an *imperfect*-denoiser, short-chain phenomenon: with
a real $\epsilon_\theta$ that only approximately denoises, each step's injected noise is only approximately
removed, and a twenty-step chain has too few remaining steps to average the leftover down. So $\eta=0$ is
the safe floor not because stochasticity is wrong in principle but because at this budget I cannot count on
the remaining steps to clean up anything I add — the toy sharpened that for me rather than proving it.

And it is also exactly why this is the floor, and I can make "the local error is not small" quantitative
rather than rhetorical. Twenty Euler steps each carry an $O(h)$ local error where $h$ is the step in
half-log-SNR $\lambda=\tfrac12\log(\bar\alpha/(1-\bar\alpha))$. Tabulate $\lambda$ for a standard
linear-$\beta$ SD schedule ($\beta$ from $10^{-4}$ to $0.02$ over $T=1000$): it runs from about $+4.6$ at
the low-noise end to about $-5.1$ at the high-noise end, a span near $9.7$. A uniform-in-$t$ twenty-step
grid (stride $50$) spaces the $\lambda$-steps unevenly — mean $|\Delta\lambda|\approx0.48$, but the
coarsest step, near the high-noise end where $\lambda$ moves fastest per unit $t$, is $|\Delta\lambda|\approx2.9$.
So $h$ is genuinely order one where it is worst, not small; the curvature of the denoiser's trajectory
between adjacent levels is real, and a first-order step ignores it by holding $z_{0|t}$ constant across the
interval and moving on a straight chord. That is a large truncation error to pay twenty times over, and it
is the price the floor pays that a higher-order step would buy back.

It also matters that this is *first-order* in the solver sense the later rungs will exploit. DDIM is the
$k=1$ member of the whole family: hold $z_{0|t}$ constant across the step and you get this single line,
with no derivative estimate, no history, no intermediate evaluation — so there is no high-order term that
a large guidance scale could corrupt. That is the same reason DDIM is the robust fallback everywhere: with
nothing to extrapolate, there is nothing to blow up. The cost is the truncation error I just quantified,
and it is precisely what a higher-order step would buy back by extracting the bend of the trajectory from
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
the FID should sit well above what a higher-order step can reach at the same twenty calls, and
the gap should *widen* on the harder variant. SDXL has the largest latent and the most structure to
resolve in twenty coarse steps, and that is exactly where the order-one $|\Delta\lambda|\approx2.9$
coarseness costs the most, so I expect SDXL worst by a wide margin, the easier SD variants lower. I do not
have the numbers yet — the FID table will tell me — but the mechanism gives a sharp, falsifiable shape: if
the failure really is first-order coarseness rather than instability or a guidance pathology, then SDXL's
FID should stand out as the largest of the three, and a higher-order step should later close the most
ground exactly there. That measured gap is exactly what should force a higher-order update at the next
rung: if twenty first-order steps leave quality on the table, the move is to extract a derivative of the
trajectory from the calls I already spend, not to ask for calls I do not have.
