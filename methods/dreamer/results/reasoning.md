OK, let me think this through from scratch. I want an agent that controls something from raw pixels — locomotion, manipulation, the usual continuous-control tasks — and the thing that actually hurts is not final performance, it's how many real environment steps I have to burn to get there. The strong model-free agents on these visual tasks want something like $10^8$ steps. Real steps are slow. So the whole problem, for me, is sample efficiency, and I want to attack it at the root.

Why is model-free so wasteful? Two distinct reasons, and I should keep them separate because they have different fixes. First, every batch of real transitions produces one noisy on-policy gradient and then I throw it away and need fresh data — data is consumed once and locally. Second, the gradient itself is bad. The policy gradient I'd use is the score-function estimator: $g=\mathbb{E}\big[\sum_t (\text{return after }t)\,\nabla_\phi\log\pi_\phi(a_t\mid s_t)\big]$. The multiplier on each action's score is the scalar return that follows it, and that return is contaminated by every *later* action and every bit of environment noise. So the variance of $g$ grows with the horizon. That's why people bolt on value baselines — to subtract a state-dependent constant and shave variance — but baselines only subtract a constant, they don't change the fundamental fact that I'm weighting a log-prob by a noisy scalar.

Here's the lever I keep coming back to. Real environment steps are expensive, but if I had a *model* of the environment — something that predicts the next state and reward from the current state and action — then I could generate as many hypothetical rollouts as I want for free, and reuse my real data many times over. That addresses the first wastefulness. But the second one, the gradient variance, is where it gets interesting: if the model is a *differentiable* function, I don't have to estimate the policy gradient by sampling returns and weighting log-probs at all. I can write the return as an actual differentiable function of the actions and just take $\partial(\text{return})/\partial(\text{action})$ by the chain rule, straight through the model's transition. That's a pathwise gradient, not a likelihood-ratio gradient, and a pathwise gradient uses the *Jacobian* of the return rather than the scalar value times a score — far more information per sample, far lower variance.

So the bet is: learn a differentiable world model, imagine rollouts inside it, and train the policy by backpropagating the gradient of imagined returns through the learned dynamics. Let me see if I can actually build each piece, because there are landmines everywhere.

Start with the model. Observations are images, $64\times64\times3$. I am not going to predict forward in pixel space — it's enormous, slow, and most pixels are irrelevant to control. I want a *latent* state. Encode each image into a compact vector $s_t$, predict forward in $s$-space, and only decode to pixels when I need a training signal. The payoff is concrete: latent states are small, so I can roll thousands of imagined trajectories in parallel and far into the future without ever rendering an image. Good. So I want a latent dynamics model: an encoder $o_t\mapsto s_t$, a transition $s_{t-1},a_{t-1}\mapsto s_t$, and a reward predictor $s_t\mapsto r_t$. This is basically a non-linear Kalman filter / HMM with real-valued states, except conditioned on actions and predicting reward, which is exactly what lets it score action sequences without executing them.

How do I train it? It's a latent-variable sequence model, so the natural thing is a variational objective. I don't observe $s_t$; I infer it. So I need an inference distribution (a posterior) $p(s_t\mid \text{past}, o_t)$ that looks at the current observation, and a generative transition (a prior) $q(s_t\mid \text{past})$ that predicts $s_t$ *without* seeing $o_t$ — because at imagination time I won't have observations, I'll only have the prior. Let me be careful with the roles: the posterior is the thing I filter real sequences with; the prior is the thing I dream with. They had better agree, and the KL between them will be what enforces that.

Let me actually derive the objective rather than just assert "ELBO." I'll frame it as an information bottleneck, which I think is cleaner and tells me *why* the regularizer is there. I want the latent states $s_{1:T}$ to be maximally informative about the observations and rewards $(o_{1:T}, r_{1:T})$ given actions, while extracting as little information as possible from the data at each step:

$$\max\; \mathbb{I}\big(s_{1:T};\,(o_{1:T},r_{1:T})\mid a_{1:T}\big)\;-\;\beta\,\mathbb{I}\big(s_{1:T};\,i_{1:T}\mid a_{1:T}\big),$$

where $i_t$ are the dataset indices that pin down which observation we drew, $p(o_t\mid i_t)=\delta(o_t-\bar o_t)$. The first term wants states that predict the world; the second penalizes how much each state reads off its own data index — i.e. how much fresh information it pulls from the current frame instead of relying on memory. Now bound each term.

First term, the prediction mutual information. By definition it's $\mathbb{E}\big[\ln p(o_{1:T},r_{1:T}\mid s_{1:T},a_{1:T}) - \ln p(o_{1:T},r_{1:T}\mid a_{1:T})\big]$. The second log is the data marginal — it doesn't depend on my representation, so it's an additive constant I can drop. For the first, I don't have the true $p(o,r\mid s,a)$, so I introduce a variational decoder $\prod_t q(o_t\mid s_t)q(r_t\mid s_t)$ and use non-negativity of KL: $\mathbb{E}[\ln p(\cdot\mid s,a)] \ge \mathbb{E}[\ln p(\cdot\mid s,a)] - \mathrm{KL}\big(p(\cdot\mid s,a)\,\|\,\prod_t q(o_t\mid s_t)q(r_t\mid s_t)\big)$, and the right side collapses, term by term, to $\mathbb{E}\big[\sum_t \ln q(o_t\mid s_t)+\ln q(r_t\mid s_t)\big]$. So the first MI is lower-bounded by reconstruction of observations and rewards. Makes sense — to be informative about $o,r$, the state has to let me reconstruct them.

Second term, the information cost. $\mathbb{I}(s_{1:T};i_{1:T}\mid a) = \mathbb{E}\big[\sum_t \ln p(s_t\mid s_{t-1},a_{t-1},i_t) - \ln p(s_t\mid s_{t-1},a_{t-1})\big]$. The index $i_t$ together with $\delta$ just *is* the observation $o_t$, so $p(s_t\mid s_{t-1},a_{t-1},i_t)=p(s_t\mid s_{t-1},a_{t-1},o_t)$ — that's my posterior. The denominator $p(s_t\mid s_{t-1},a_{t-1})$ is the true transition marginal, which I don't have; replace it by my variational prior $q(s_t\mid s_{t-1},a_{t-1})$, and since this term is *subtracted* (we're upper-bounding it to get a lower bound on the whole objective) I again use KL non-negativity in the right direction: $\mathbb{E}[\ln p(\cdot\mid o) - \ln p(\cdot)] \le \mathbb{E}[\ln p(\cdot\mid o) - \ln q(\cdot)] = \mathbb{E}\big[\sum_t \mathrm{KL}\big(p(s_t\mid s_{t-1},a_{t-1},o_t)\,\|\,q(s_t\mid s_{t-1},a_{t-1})\big)\big]$. So the information cost is upper-bounded by the KL from posterior to prior.

Put it together and the model objective is

$$\mathcal{J} = \mathbb{E}\Big[\sum_t \underbrace{\ln q(o_t\mid s_t)}_{\text{reconstruct image}} + \underbrace{\ln q(r_t\mid s_t)}_{\text{reconstruct reward}} - \beta\,\underbrace{\mathrm{KL}\big(p(s_t\mid s_{t-1},a_{t-1},o_t)\,\|\,q(s_t\mid s_{t-1},a_{t-1})\big)}_{\text{posterior toward prior}}\Big].$$

This is exactly an ELBO with a $\beta$ on the KL, and now I understand the KL not just as "regularizer" but as the information toll: it forces the prior to predict the posterior, which is precisely the property I need, because at dream time I sample from the prior and I need it to land where the posterior would have. And the $\beta$/information-bottleneck reading tells me something subtle — penalizing per-step information extraction pushes the model to reconstruct each frame using what it already knew from previous steps, only pulling new bits from the current image when it must, which is what makes it learn long-term dependencies in the state. The expectation is taken under the data and the posterior, and the whole thing is differentiable in the model parameters $\theta$ if I reparameterize the latent sampling. I'll optimize it by stochastic backprop with reparameterized Gaussians.

Now, what *form* should the latent transition take? My first instinct is a fully stochastic latent Markov chain: $s_t\sim q(s_t\mid s_{t-1},a_{t-1})$, sample, repeat. Let me imagine running that forward for fifteen steps. Every step injects fresh noise, and information about something I saw at step 1 has to survive being squeezed through a sampled bottleneck at every subsequent step. That's a wall — purely stochastic transitions are terrible at remembering. Multi-step prediction needs a reliable channel to carry information forward deterministically. The opposite extreme, a fully deterministic recurrent net, carries memory perfectly but can't represent multiple plausible futures and gives the KL nothing to regularize — there's no stochastic state to keep close to a prior. So neither extreme works.

The resolution is to split the state into two parts: a deterministic recurrent component $h_t$ that carries memory cleanly, and a stochastic component $s_t$ that captures what's uncertain. Concretely a recurrent state-space model:

$$h_t = f\big(h_{t-1},\,s_{t-1},\,a_{t-1}\big)\quad[\text{a GRU}],$$

and then the *prior* over the stochastic state reads off the recurrent state, $q(s_t\mid h_t)$, while the *posterior* additionally folds in the current observation's embedding, $p(s_t\mid h_t,\,o_t)$. The full "model state" I'll carry around and feed to everything downstream is the concatenation $\big[s_t;\,h_t\big]$ — stochastic part plus deterministic part. The deterministic $h_t$ is the highway for long-term information; the stochastic $s_t$ is where the KL bites and where multimodality lives. This is the piece I'll reuse as-is for the dynamics; it solves the remember-vs-be-uncertain tension, and it's exactly what makes imagined fifteen-step rollouts coherent.

Both the prior and posterior are diagonal Gaussians whose mean and std come out of small dense nets, and I sample them with the reparameterization trick, $s = \mu + \sigma\odot\epsilon$. That choice isn't cosmetic: I am about to need to backpropagate a behavior gradient *through* these sampled states, so they had better be differentiable functions of their statistics. A non-reparameterized sample would be a dead end for the actor gradient.

So the world model is settled. Now the real contribution: given this model, how do I get a behavior out of it?

The obvious move — and what online planners do — is, from the current state, search over action sequences, predict reward for each over some horizon $H$, pick the best, execute the first action, and re-plan next step. That works but has two problems I can already see. It pays the full planning cost at *every* environment step, which is brutal at action time. And it only sums reward over the fixed horizon $H$, so it's shortsighted: anything good that happens after step $H$ is invisible to the plan. On a task where you have to, say, build up momentum over many steps to get a delayed reward, a horizon-$H$ planner just won't see it.

I want to fix both. For the cost, learn an explicit *policy* $a_\tau\sim q_\phi(a_\tau\mid s_\tau)$ — a network — so acting is one forward pass, no search. For the shortsightedness, I need to account for reward *beyond* the horizon, and the classic way to do that is to learn a *value* function $v_\psi(s_\tau)$ that estimates the expected return from a state, and bootstrap with it at the end of the rollout. So: an actor and a value/critic, both living in latent space, trained on imagined rollouts. This is an actor-critic, but the rollouts and the gradient are going to be unusual.

Let me set up the imagination environment precisely. The latent dynamics define a fully-observed MDP — fully observed because the model state $[s_\tau;h_\tau]$ is Markov by construction. I start imagined trajectories from real states: take a batch of real sequences, filter them with the posterior to get accurate model states $s_t$, and use those as the roots. (Crucial that I start from *posterior* states of real data — if I imagined from scratch the model error would compound from step zero; starting from filtered real states keeps the rollout near the data manifold for as long as the model is any good.) From each root I roll the *prior* forward $H$ steps, at each step sampling an action from the actor $a_\tau\sim q_\phi(a_\tau\mid s_\tau)$, advancing the transition $s_{\tau+1}\sim q_\theta(s_{\tau+1}\mid s_\tau,a_\tau)$, and predicting reward $r_\tau$ and value $v_\psi(s_\tau)$ — all in latent space, never decoding to pixels. The objective in this imagined MDP is to maximize $\mathbb{E}\big[\sum_{\tau} \gamma^{\tau-t} r_\tau\big]$ with respect to $\phi$.

The actor should be a tanh-transformed Gaussian: a dense net outputs a mean and std, sample $\mathcal{N}(\mu_\phi(s),\sigma_\phi(s))$, squash through tanh to bound the action. And again I write the action with the reparameterization trick, $a_\tau=\tanh\big(\mu_\phi(s_\tau)+\sigma_\phi(s_\tau)\,\epsilon\big),\ \epsilon\sim\mathcal{N}(0,I)$, because — same reason as before — I'm going to differentiate the return with respect to $\phi$ *through* the sampled action, and that requires the sample to be a differentiable function of $\mu_\phi,\sigma_\phi$. (For discrete actions I can't reparameterize a categorical, so I'll use straight-through gradients there.)

Now the value target. I have a rollout $\{s_\tau,a_\tau,r_\tau\}_{\tau=t}^{t+H}$ and a value net. How do I estimate the value of each state along it? The crudest is to just sum the rewards to the horizon and stop, $\mathrm{V}_{\mathrm{R}}(s_\tau)=\mathbb{E}\big[\sum_{n=\tau}^{t+H} r_n\big]$ — but that's the shortsighted thing again, it ignores everything past $H$ and doesn't even use the value net (it's a useful ablation: actor with no critic). To see past the horizon, bootstrap: take a $k$-step return that sums $k$ real imagined rewards and then appends the value estimate at the landing state,

$$\mathrm{V}_{\mathrm{N}}^{k}(s_\tau)=\mathbb{E}\Big[\sum_{n=\tau}^{h-1}\gamma^{n-\tau}r_n+\gamma^{h-\tau}v_\psi(s_h)\Big],\qquad h=\min(\tau+k,\,t+H).$$

For $k=1$ this is the one-step TD target $r_\tau+\gamma v_\psi(s_{\tau+1})$ — low variance, but its bias is whatever error $v_\psi$ has. For large $k$ it's almost the full imagined return — low bias, but it piles on the variance and, worse here, the *model error* over many imagined steps. So I have a knob $k$ between bias and variance, just like the return estimators in standard RL. Rather than commit to one $k$, average over all of them with exponential weights — the $\lambda$-return:

$$\mathrm{V}_{\lambda}(s_\tau)=(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}\,\mathrm{V}_{\mathrm{N}}^{n}(s_\tau)\;+\;\lambda^{H-1}\,\mathrm{V}_{\mathrm{N}}^{H}(s_\tau).$$

The $(1-\lambda)\sum\lambda^{n-1}$ weights are a normalized geometric series (they sum to 1 once you fold the last term, which gets the leftover mass $\lambda^{H-1}$ since the rollout is finite). Small $\lambda$ leans on the short, low-variance bootstraps; $\lambda\to1$ leans on the long Monte-Carlo-ish return. This is the bias/variance dial for the imagined target, and the bootstrap by $v_\psi$ at every horizon is exactly what lets a *finite* imagination horizon $H$ still account for reward arbitrarily far in the future — the value soaks up the tail. That directly kills the shortsightedness of a fixed-horizon planner.

Let me make sure I can actually *compute* $\mathrm{V}_\lambda$ cheaply, because writing it as a double sum over $k$ and $n$ would be wasteful. Claim: it satisfies a one-step backward recursion. Define, going from the end of the rollout backward,

$$\mathrm{V}_\lambda(s_\tau) = r_\tau + \gamma\Big[(1-\lambda)\,v_\psi(s_{\tau+1}) + \lambda\,\mathrm{V}_\lambda(s_{\tau+1})\Big],\qquad \mathrm{V}_\lambda(s_{t+H}) = v_\psi(s_{t+H}).$$

Let me verify this reproduces the weighted-average definition by unrolling it. Unroll once: $\mathrm{V}_\lambda(s_\tau)=r_\tau+\gamma(1-\lambda)v_\psi(s_{\tau+1})+\gamma\lambda\,\mathrm{V}_\lambda(s_{\tau+1})$. The term $r_\tau+\gamma v_\psi(s_{\tau+1})$ is the one-step return $\mathrm{V}_{\mathrm N}^1(s_\tau)$; pulling it out, $\mathrm{V}_\lambda(s_\tau)=(1-\lambda)\,\mathrm{V}_{\mathrm N}^1(s_\tau)+\lambda\big(r_\tau+\gamma\,\mathrm{V}_\lambda(s_{\tau+1})\big)$. Now substitute the recursion for $\mathrm{V}_\lambda(s_{\tau+1})$ inside: $r_\tau+\gamma\mathrm{V}_\lambda(s_{\tau+1}) = r_\tau+\gamma\big[r_{\tau+1}+\gamma(1-\lambda)v_\psi(s_{\tau+2})+\gamma\lambda\mathrm{V}_\lambda(s_{\tau+2})\big]$, and $r_\tau+\gamma r_{\tau+1}+\gamma^2 v_\psi(s_{\tau+2})=\mathrm{V}_{\mathrm N}^2(s_\tau)$, so the coefficient of the two-step return comes out as $\lambda(1-\lambda)$. Continue and the coefficient of $\mathrm{V}_{\mathrm N}^k$ is $(1-\lambda)\lambda^{k-1}$, with the residual geometric tail $\lambda^{H-1}$ landing on the final full-horizon return — exactly the closed form above. So the recursion is correct, and it means I can compute all the targets in one backward pass. In code that's a scan from the last step to the first: at each step the carry is updated as $\text{return} \leftarrow \big(r_\tau+\gamma(1-\lambda)v_\psi(s_{\tau+1})\big) + \gamma\lambda\cdot\text{return}$, initialized with the bootstrap $v_\psi(s_{t+H})$.

Now the two losses. The value net should regress these targets:

$$\min_\psi\ \mathbb{E}\Big[\sum_\tau \tfrac12\big\|v_\psi(s_\tau)-\mathrm{V}_\lambda(s_\tau)\big\|^2\Big],$$

and I stop the gradient on the target $\mathrm{V}_\lambda$. Why stop-grad? Because $\mathrm{V}_\lambda$ is itself built from $v_\psi$ at the bootstrap points; if I let the gradient flow into the target I'd be chasing a moving target and the regression would partly optimize by dragging the target toward the prediction, which is not what I want — I want the prediction to move toward a fixed estimate, the usual semi-gradient TD treatment. So target gets a stop-gradient.

And now the centerpiece — the actor. The actor wants states with high value, so

$$\max_\phi\ \mathbb{E}\Big[\sum_{\tau=t}^{t+H}\mathrm{V}_\lambda(s_\tau)\Big].$$

The question is how I take $\nabla_\phi$ of this. The model-free reflex is REINFORCE: treat $\mathrm{V}_\lambda$ as a black-box scalar and write $\nabla_\phi\mathbb{E}\approx\mathbb{E}[\mathrm{V}_\lambda\cdot\nabla_\phi\log q_\phi(a_\tau\mid s_\tau)]$. But that throws away everything I built. The whole point of a differentiable model is that $\mathrm{V}_\lambda$ is *not* a black box — it's a composition of differentiable neural nets. Let me trace the dependency. $\mathrm{V}_\lambda(s_\tau)$ depends on predicted rewards $r_n=r_\theta(s_n)$ and values $v_\psi(s_n)$ along the rollout; those depend on the imagined states $s_n$; each state $s_n$ comes from the transition $s_n\sim q_\theta(s_n\mid s_{n-1},a_{n-1})$, so it depends on the previous state and on the previous *action*; and the action $a_{n-1}=\tanh(\mu_\phi(s_{n-1})+\sigma_\phi(s_{n-1})\epsilon)$ depends on $\phi$. Every single arrow in that chain is a differentiable neural net, and every sampling node is reparameterized. So I can just take the analytic gradient.

Let me write it out for a single rollout to convince myself the chain rule goes where I think. Write the objective for one trajectory as $J=\sum_\tau \mathrm{V}_\lambda(s_\tau)$, and let the state recursion be $s_{n}=g_\theta(s_{n-1},a_{n-1},\epsilon_n)$ with $a_{n-1}=\pi_\phi(s_{n-1},\eta_{n-1})$ (folding the reparameterization noise into $\epsilon,\eta$). Then

$$\frac{\partial J}{\partial \phi}=\sum_\tau \frac{\partial \mathrm{V}_\lambda(s_\tau)}{\partial \phi},\qquad \frac{\partial \mathrm{V}_\lambda(s_\tau)}{\partial\phi}=\sum_{n\le \tau}\frac{\partial \mathrm{V}_\lambda(s_\tau)}{\partial s_n}\,\frac{\partial s_n}{\partial\phi},$$

and the state's own dependence on $\phi$ unrolls through both the action it consumed and the previous state:

$$\frac{\partial s_n}{\partial \phi}=\underbrace{\frac{\partial g_\theta}{\partial a_{n-1}}\frac{\partial \pi_\phi}{\partial\phi}}_{\text{direct: this step's action}}+\underbrace{\Big(\frac{\partial g_\theta}{\partial s_{n-1}}+\frac{\partial g_\theta}{\partial a_{n-1}}\frac{\partial \pi_\phi}{\partial s_{n-1}}\Big)\frac{\partial s_{n-1}}{\partial\phi}}_{\text{recursive: through the past}}.$$

There it is in symbols: the actor parameters influence the value of state $s_\tau$ by perturbing every earlier action, and each such perturbation propagates forward through the transition Jacobian $\partial g_\theta/\partial s$. The gradient I'm using is built from Jacobians $\partial g_\theta/\partial a$, $\partial g_\theta/\partial s$, $\partial r_\theta/\partial s$, $\partial v_\psi/\partial s$ — the *sensitivities* of the model — not from the scalar return times a score. That's the variance win: I'm not Monte-Carlo-estimating $\mathbb{E}[\text{return}\cdot\text{score}]$, I'm reading off how the return changes when I nudge the action, which is exactly the information a gradient should carry. And it costs me nothing extra: the rollout is already a differentiable computation graph, so a single backward pass over it computes $\nabla_\phi J$. This is stochastic backpropagation through the learned dynamics — reparameterization for the continuous latents and actions, straight-through for discrete actions.

Let me sanity-check this against the off-policy actor-critics I know, because the contrast clarifies *why* I get to use a state value here. DDPG/SAC also do a pathwise actor update, $\nabla_\phi Q(s,\pi_\phi(s))$ — but notice they need $Q(s,a)$, an *action*-value, and they only differentiate through *one* step (the action into $Q$). They need $Q$ as a function of the action precisely because they cannot differentiate the environment to find out how the action changes the future; $Q$ is their only window onto "how good is this action." I *can* differentiate the future — through the model — so the action-gradient comes for free by the chain rule through the transition, and a *state* value $v_\psi(s)$ is all I need: it tells me the value of where I land, and the model tells me how my action moves where I land. So state value plus differentiable transition $=$ I never have to learn $Q$. That's a real simplification and it's only available because I have model gradients. SVG had the right gradient idea but only through one step; here I run it through a full $H$-step imagined rollout. And MVE/STEVE use the model only to make better $Q$-*targets* for an otherwise model-free learner — they don't push the policy gradient through the model at all. So the distinguishing move is: the policy gradient itself flows through the learned multi-step dynamics.

One subtlety I need to get right in the rollout, about *which* dependence of the action on the state I keep. When I generate the imagined action at step $\tau$, $a_\tau=\pi_\phi(s_\tau)$, the state $s_\tau$ is itself a differentiable function of $\phi$ (it came from earlier actions). If I let the actor's gradient flow through $s_\tau$ as an *input* to $\pi_\phi$, I'd be conflating two things: the actor changing the action it outputs, versus the actor changing the state it's standing on. The credit I actually want is "how does the value change when the actor changes its action," propagated forward through the dynamics; the value of the future states already captures the consequence of past actions. So when I form the action for the rollout I stop the gradient on the state fed to the policy — the policy's parameter gradient should reach $\phi$ through the *value of future states*, and the state-as-input is just a conditioning value, not a path I want to double-count along. The forward states themselves of course stay differentiable; it's only the state handed to the actor-as-input at each imagined step that I detach.

A couple of remaining details. The discount: I weight each term in both the actor and value objectives by the cumulative product of $\gamma$ (and in tasks that can terminate early, by the cumulative product of a *predicted* discount/continue probability $\hat\gamma_\tau$ from each latent state, so an imagined step that probably ends the episode gets weighted down). And while learning behavior, the world model parameters $\theta$ are held fixed — the representation is trained by its own variational objective on real data; I don't want the behavior gradients reshaping the latent space, that's a separate job. Three losses, three optimizers: model, actor, value.

Let me also pin down the representation flavor, because the model objective above assumed I reconstruct images. Reconstruction is the strong default: decode $o_t$ from $s_t$ with a transposed-conv net, decode reward with a dense net, and the KL closes the loop. Could I avoid predicting pixels? Yes — instead of reconstructing $o_t$ I could predict the *state* from the image and maximize mutual information contrastively: replace the observation decoder with a state model $q(s_t\mid o_t)$ and, since I no longer get to drop a constant observation marginal but now face a *state* marginal, estimate it with a noise-contrastive bound, $\mathcal{J}_{\mathrm S}^t=\ln q(s_t\mid o_t)-\ln\sum_{o'}q(s_t\mid o')$, where the sum over other observations $o'$ in the batch keeps the states diverse and prevents collapse. That avoids the cost and capacity of pixel prediction, at the price of how much information the bound can extract efficiently. Either way the behavior-learning machinery above is untouched — it only needs the latent states and the reward and transition models. Reconstruction is what I'll default to.

Now let me land this on code, mirroring how the pieces actually fit. The model training is one tape: encode the real batch, filter to get posterior and prior states, decode image and reward for the reconstruction terms, take the KL between posterior and prior (clipped at a few free nats so the KL doesn't dominate early and cause posterior collapse), assemble the model loss. The behavior training is two more tapes sharing one imagined rollout. The imagined rollout is a scan that, from the flattened posterior states, repeatedly applies the transition under the (state-detached) policy. Then compute predicted rewards, values, and the $\lambda$-returns by the backward recursion; the actor loss is the negative discounted return (so ascent), the value loss is the squared error to the stop-gradient target.

```python
# --- world model: RSSM (deterministic h + stochastic s), reused as-is -------
class RSSM:
    def img_step(self, prev, prev_action):                 # the PRIOR / transition
        x = concat([prev['stoch'], prev_action])
        x = dense(x); deter = gru(x, prev['deter'])         # h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
        x = dense(deter)
        mean, std = split(dense(x)); std = softplus(std) + 0.1
        stoch = sample(Normal(mean, std))                   # s_t ~ q(s_t | h_t), reparameterized
        return {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}

    def obs_step(self, prev, prev_action, embed):           # the POSTERIOR
        prior = self.img_step(prev, prev_action)            # predict-ahead first
        x = concat([prior['deter'], embed])                 # then fold in the observation
        mean, std = split(dense(x)); std = softplus(std) + 0.1
        stoch = sample(Normal(mean, std))                   # s_t ~ p(s_t | h_t, o_t)
        return {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}, prior

    def get_feat(self, state):
        return concat([state['stoch'], state['deter']])     # model state = [s_t ; h_t]

# --- the lambda-return, as the backward recursion I derived -----------------
def lambda_return(reward, value, pcont, bootstrap, lambda_):
    # next_values[t] = value at t+1 (bootstrap at the end)
    next_values = concat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)   # r_t + gamma(1-lambda) v(s_{t+1})
    # scan backward: return_t = inputs_t + gamma*lambda*return_{t+1}, seeded with bootstrap
    return scan(lambda agg, cur: cur.inputs + cur.pcont * lambda_ * agg,
                (inputs, pcont), start=bootstrap, reverse=True)

# --- imagination: roll the prior forward under the actor, in latent space ---
def imagine_ahead(self, post):
    start = flatten(post)                                   # roots = posterior states of real data
    policy = lambda s: self.actor(stop_grad(get_feat(s))).sample()  # detach state-as-input
    states = scan(lambda prev, _: self.dynamics.img_step(prev, policy(prev)),
                  range(H), start)                          # H steps, no pixels decoded
    return get_feat(states)

# --- the training step: one model loss, two behavior losses -----------------
def train(self, data):
    with tape() as model_tape:                              # 1) fit the world model
        embed = self.encoder(data)
        post, prior = self.dynamics.observe(embed, data['action'])
        feat = self.dynamics.get_feat(post)
        like_img = self.decoder(feat).log_prob(data['image'])
        like_rew = self.reward(feat).log_prob(data['reward'])
        kl = kl_divergence(dist(post), dist(prior))         # KL(posterior || prior)
        kl = maximum(kl, free_nats)                         # clip so KL can't dominate early
        model_loss = kl_scale * kl - (like_img + like_rew)  # = -ELBO

    with tape() as actor_tape:                              # 2) actor: analytic value gradient
        imag_feat = self.imagine_ahead(post)                # differentiable rollout
        reward = self.reward(imag_feat).mode()
        value  = self.value(imag_feat).mode()
        returns = lambda_return(reward[:-1], value[:-1], gamma, value[-1], lambda_)
        discount = stop_grad(cumprod(gamma))                # downweight far / terminating steps
        actor_loss = -mean(discount * returns)              # ascend imagined value; grad flows
                                                            # through r,v -> states -> transition -> actions -> phi

    with tape() as value_tape:                              # 3) value: regress the lambda-return
        value_pred = self.value(imag_feat)[:-1]
        target = stop_grad(returns)                         # semi-gradient: fixed target
        value_loss = -mean(discount * value_pred.log_prob(target))

    model_opt(model_tape, model_loss)                       # theta fixed during 2) and 3)
    actor_opt(actor_tape, actor_loss)
    value_opt(value_tape, value_loss)
```

Tracing the whole thing once more as a causal chain: real environment steps are the scarce resource, so I learn a compact latent world model (the RSSM: deterministic recurrent memory plus a reparameterized stochastic state, trained by a variational/information-bottleneck objective that reconstructs observations and rewards and keeps the prior predicting the posterior). Because that model is differentiable, I can dream — roll it forward in latent space from accurate posterior states, never touching pixels — and I learn behavior entirely inside the dream. To see past the finite imagination horizon I learn a state value and bootstrap with it, mixing $k$-step returns into a $\lambda$-return that dials bias against variance and whose value bootstrap captures reward beyond the horizon. And the move that makes it efficient: I train the actor not by the high-variance score-function gradient but by the analytic, reparameterized gradient of those $\lambda$-returns flowing back through the learned transition into the actions — Jacobians of the model instead of returns-times-scores — using a plain state value rather than an action value because the model itself supplies the action-to-future gradient.
