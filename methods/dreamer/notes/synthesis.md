# Dreamer synthesis (pre-Phase-2)

## Pain point
Model-free RL on visual control (DMC from pixels) is sample-inefficient: every gradient
of the policy ultimately comes from real environment interaction, and the score-function
(REINFORCE) gradient has variance that grows with horizon. Real environment steps are
expensive. If we could learn a differentiable world model, we could (a) imagine rollouts
to amortize data, and (b) — the key — backpropagate the gradient of returns through the
model's transition function directly into the policy, replacing the high-variance
likelihood-ratio gradient with a low-variance pathwise (reparameterized) gradient.

## Load-bearing ancestors
- **Dyna (Sutton 1991)**: interleave model learning, planning-in-model, acting. The
  scaffold of "dynamics learning / behavior learning / environment interaction."
- **PILCO (Deisenroth & Rasmussen 2011)**: analytic policy gradients through a learned
  (GP) dynamics model — proves pathwise model gradients give extreme data efficiency, but
  GP moment-matching doesn't scale to images / long horizons.
- **World Models (Ha & Schmidhuber 2018)**: VAE + MDN-RNN latent dynamics; evolve a tiny
  controller inside the dream. Two-stage, derivative-free (CMA-ES), linear controller.
- **PlaNet (Hafner et al. 2018) + RSSM**: the world model we reuse. Latent dynamics with a
  deterministic recurrent path h_t = GRU(h_{t-1}, s_{t-1}, a_{t-1}) plus a stochastic
  state s_t. Trained by a variational ELBO (reconstruct image + reward, KL posterior‖prior).
  PlaNet then does *online planning* (CEM) at every step — no learned policy or value, so
  planning cost recurs every action and the horizon is finite → shortsighted.
- **Reparameterization / VAE (Kingma & Welling 2014; Rezende et al. 2014)**: pathwise
  gradient through a sampling node, s = μ + σ·ε. This is what makes the stochastic latent
  states and the tanh-Gaussian policy differentiable.
- **DPG/DDPG (Silver 2014; Lillicrap 2015), SAC (Haarnoja 2018)**: reparameterized /
  deterministic actor-critic; the actor ascends ∇_a Q. But they only backprop through a
  *one-step* learned Q, not through transitions — no multi-step model gradient.
- **SVG (Heess et al. 2015)**: value gradients through *one-step* model predictions to
  reduce on-policy gradient variance. Dreamer = SVG taken to a full multi-step imagined
  horizon in latent space.
- **MVE / STEVE (Feinberg 2018; Buckman 2018)**: use a learned model to build better
  multi-step *Q-targets* for a model-free learner; still REINFORCE/Q-learning for the
  policy, model used only for targets, not for the gradient path.
- **TD(λ) / GAE (Sutton 1988; Schulman 2016)**: λ-return — exponentially-weighted average
  of k-step returns; the bias/variance knob. Reused as the imagined value target V_λ.
- **VIB (Tishby 2000; Alemi 2016)**: information bottleneck framing of the ELBO that the
  appendix uses to derive both the reconstruction and contrastive (NCE) objectives.
- **REINFORCE (Williams 1992), A3C/PPO**: the likelihood-ratio baseline being beaten on
  variance.

## The method (final landing)
Three learned pieces, three loops.

1. **World model (RSSM)** with parameters θ:
   - deterministic recurrent state: h_t = f_θ(h_{t-1}, s_{t-1}, a_{t-1})  [GRU]
   - stochastic prior (transition): s_t ~ q_θ(s_t | h_t) = q_θ(s_t | s_{t-1}, a_{t-1})
   - stochastic posterior (representation): s_t ~ p_θ(s_t | h_t, o_t)
   - reward model: q_θ(r_t | s_t, h_t); observation/decoder: q_θ(o_t | s_t, h_t)
   - "model state" / feature = concat(s_t (stoch), h_t (deter)).
   Trained by maximizing the ELBO / VIB lower bound:
   J = E[ Σ_t ln q(o_t|s_t) + ln q(r_t|s_t) − β KL( p(s_t|h_t,o_t) ‖ q(s_t|h_t) ) ].
   KL free-nats clip at 3; β=1 for continuous.

2. **Imagination**: from each posterior state s_t of a real sequence batch, roll the prior
   forward H=15 steps using the actor a_τ ~ q_φ(a_τ|s_τ) and the transition q_θ — purely in
   latent space, NO decoding to pixels. Predict r_τ from reward model and v_ψ(s_τ).

3. **Actor (φ) and value (ψ)**:
   - λ-return target over the imagined trajectory:
     V_N^k(s_τ) = E[ Σ_{n=τ}^{h-1} γ^{n-τ} r_n + γ^{h-τ} v_ψ(s_h) ], h=min(τ+k, t+H)
     V_λ(s_τ) = (1-λ) Σ_{n=1}^{H-1} λ^{n-1} V_N^n(s_τ) + λ^{H-1} V_N^H(s_τ).
     Equivalent recursive (impl) form: V_λ(s_τ) = r_τ + γ[ (1-λ) v(s_{τ+1}) + λ V_λ(s_{τ+1}) ],
     with V_λ(s_{t+H}) = v(s_{t+H}).
   - Value loss: min_ψ E Σ_τ ½‖v_ψ(s_τ) − sg(V_λ(s_τ))‖²  (stop-grad target).
   - Actor objective: max_φ E Σ_τ V_λ(s_τ), optimized by ANALYTIC gradient
     ∇_φ E[Σ_τ V_λ(s_τ)] via reparameterization — value gradients flow back through
     v_ψ and r_θ → through imagined states s_τ (reparam'd) → through the transition q_θ →
     to the action a_{τ-1} = tanh(μ_φ + σ_φ ε) → to φ. World model θ fixed during this.
   - λ-weighted discount weighting (cumprod of γ or predicted pcont) on each term.

## Design decisions → why
- **State value v(s) not action value Q(s,a)**: because we backprop through the dynamics,
  ∂(return)/∂a is obtained by the chain rule through the transition; we don't need Q to
  expose the action-gradient (DDPG needs Q precisely because it can't differentiate the
  env). v over the lower-dim state space is easier to fit.
- **λ-return not raw H-step sum (V_R) and not 1-step (V_N^1)**: V_R ignores reward beyond H
  → shortsighted; raw long sums are high variance and the model compounds error; V_λ is the
  bias/variance knob, and bootstrapping with v(s_H) accounts for reward *beyond* the horizon
  — that's what lets a finite imagination horizon learn long-horizon behavior.
- **Analytic (pathwise) actor gradient not REINFORCE**: REINFORCE multiplies the score by a
  scalar return → variance grows with horizon; pathwise gradient uses the *Jacobian* of the
  return wrt actions, far lower variance, and it's available for free because every step is a
  differentiable neural net. This is the central efficiency claim.
- **Reparameterized stochastic states + tanh-Normal actor**: needed so the sampling nodes in
  the rollout are differentiable; straight-through for discrete actions.
- **stop_gradient on the value target**: it's a regression/Bellman target; letting gradient
  flow into it would chase a moving target / double-count.
- **stop_gradient on the state when computing the policy action inside imagination** (impl
  detail in `_imagine_ahead`): the actor's *own* parameter gradient comes through the value
  of *future* states, not through perturbing the state it conditions on; this keeps the
  credit assignment to the value path.
- **RSSM split deterministic h + stochastic s**: purely stochastic transitions can't reliably
  remember information over many steps (gradient/info bottleneck through sampling); a purely
  deterministic model can't capture multimodal futures / can't be regularized by KL. Split:
  h carries memory, s carries stochasticity. (PlaNet's finding.)
- **KL free nats (3)**: prevent posterior collapse / the KL term from dominating early.
- **β VIB / KL scale**: information bottleneck — limit info extracted per step so the model
  uses memory and learns long-term dependencies.
- **Imagine from posterior states of real sequences**: start imagination from accurate
  filtered states, not from scratch, so model error doesn't compound from t=0.
- **Three separate optimizers / world model fixed during behavior**: clean separation; the
  behavior gradients shouldn't reshape the representation (which is trained by its own ELBO).
