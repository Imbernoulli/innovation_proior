# DDPG — Synthesis (Phase 1.5)

## The pain point (research question)
We want one model-free RL algorithm that learns continuous-action control policies (real-valued, often high-dimensional action vectors — joint torques, steering) directly from low-dimensional state OR raw pixels, with stable training and minimal per-task tuning. DQN cracked high-dimensional *observations* (pixels) but only for *discrete, low-dimensional* action sets, because its policy is the greedy `argmax_a Q(s,a)`. In continuous action spaces that argmax is an inner optimization at every timestep over a non-convex neural net Q — too slow to run online, and naive discretization suffers curse of dimensionality (7-DOF arm at coarse {-k,0,k} = 3^7 = 2187 actions) and throws away action-space structure.

## Load-bearing ancestors (verified against primary text / DPG paper)

1. **Q-learning (Watkins & Dayan 1992).** Off-policy TD control. Greedy target policy μ(s)=argmax_a Q(s,a). Bellman: Q(s,a)=E[r+γ max_a' Q(s',a')]. Loss L(θ)=E[(Q(s,a|θ)−y)²], y=r+γ max_a' Q(s',a'|θ). Gap: the max/argmax over a is the wall in continuous spaces.

2. **DQN (Mnih 2013/2015).** Made nonlinear (deep CNN) Q-function approximation stable, where it was previously believed unstable/divergent. Two innovations: (a) **replay buffer** — store (s,a,r,s'), sample uniform minibatches → breaks temporal correlation, restores approx-iid, enables hardware-efficient minibatching, off-policy reuse; (b) **target network** — a periodically-copied frozen copy Q(·|θ⁻) computes the bootstrap target y, so the regression target isn't chasing the weights being updated → reduces the moving-target feedback loop that causes divergence. Limitation: discrete actions only (output layer = one Q per action; greedy = cheap argmax over finite head).

3. **Deterministic Policy Gradient (Silver et al. 2014).** Key theoretical enabler. Stochastic policy gradient (Sutton 1999): ∇_θ J = E_{s~ρ^π, a~π}[∇_θ log π(a|s) Q^π(s,a)] — must integrate/sample over BOTH states and actions, high variance in high-dim actions. DPG considers a *deterministic* policy a=μ_θ(s). DPG theorem:
   ∇_θ J(μ_θ) = E_{s~ρ^μ}[ ∇_θ μ_θ(s) ∇_a Q^μ(s,a)|_{a=μ_θ(s)} ].
   The gradient integrates over states ONLY (no inner action integral) — it's the chain rule through the critic: move μ in the direction that increases Q. Proof structure mirrors stochastic PG theorem (Leibniz rule to swap ∇ and integral; regularity/compactness conditions). DPG is the zero-variance limit of stochastic PG: as policy variance σ→0, ∇_θ J^stoch → ∇_θ J^det. **Off-policy** form: collect with behavior β, but because no action integral, NO importance-sampling ratio over actions is needed (only the state distribution becomes ρ^β; the actor update is still the chain rule). DPG paper used linear/tile-coding, toy domains; also gave compatible-function-approximation condition for the critic. Gap: never scaled to deep nets / high-dim observations.

4. **NFQCA (Hafner & Riedmiller 2011).** Same actor-critic update rules as DPG but with neural nets — used *batch* learning for stability (intractable for large nets); minibatch NFQCA without policy reset ≈ original DPG, which is the unstable baseline. Naive deep DPG diverges.

5. **Batch normalization (Ioffe & Szegedy 2015).** Normalizes each feature dim across the minibatch to unit mean/variance; keeps running stats for test. Used to handle heterogeneous physical units (positions vs velocities vs different env scales) so one set of hyperparameters generalizes across tasks.

6. **Ornstein–Uhlenbeck process (Uhlenbeck & Ornstein 1930).** OU SDE: dx = θ(μ−x)dt + σ dW; mean-reverting, temporally CORRELATED Gaussian process modeling velocity of a Brownian particle with friction. Used for exploration noise (θ=0.15, σ=0.2) — correlated noise explores better in systems with inertia/momentum than independent per-step Gaussian (a momentary uncorrelated torque deviation barely moves a massive joint; correlated noise produces sustained pushes). (Wawrzyński 2015 introduced similar autocorrelated noise.)

7. **Adam (Kingma & Ba 2014)** for optimization; **ReLU (Glorot 2011)** hidden units; **tanh** output to bound actions.

## The derivation chain (theory → practice)
- Object to maximize: J = E[R_1], expected discounted return from start distribution.
- Q-learning gives a way to *evaluate* via Bellman, but its greedy improvement step needs argmax_a Q — intractable for continuous a.
- DPG theorem replaces argmax with gradient ascent: parameterize policy μ_θ(s); ∇_θ J = E_s[∇_θ μ ∇_a Q|_{a=μ}]. The actor *learns to approximate the argmax* — μ_θ(s) is trained to output the action that maximizes the critic, so Q(s,μ(s)) ≈ max_a Q(s,a). This sidesteps the per-step inner optimization (it's amortized into the actor's weights).
- Critic learns Q via Bellman with the deterministic policy (no inner expectation over a' since policy is deterministic): y = r + γ Q'(s', μ'(s')). Loss = (1/N)Σ(y_i − Q(s_i,a_i))².
- Naive deep version diverges → import DQN's two fixes: replay buffer + target networks. But hard target copies (DQN style) interact badly with the coupled actor-critic, so use **soft target updates**: θ' ← τθ + (1−τ)θ', τ=0.001. Targets move slowly & smoothly → critic regression looks like supervised learning with a near-stationary target → stability. Need target copies of BOTH μ' and Q' to make y_i stable.
- Exploration decoupled from learning (off-policy): behavior = μ(s)+OU noise.

## Design-decision → why table
- **Actor net (μ) to approximate argmax** vs running optimizer per step: amortizes the inner max into weights; one forward pass at act time. Alternative (iterative optimization of a each step) is too slow online.
- **Chain-rule actor update** ∇_θμ·∇_aQ vs likelihood-ratio: deterministic → no action integral → lower variance, no log-prob, no IS ratio off-policy. Requires Q differentiable in a (true for neural Q).
- **Critic trained off-policy with replay** vs on-policy: Bellman for deterministic μ holds regardless of behavior policy → can reuse old data; replay breaks correlation & enables minibatching. Off-policy is *why* replay is sound here.
- **Soft target updates (τ≪1)** vs DQN hard periodic copy: with an interdependent actor+critic, a slowly-moving target is more stable than abrupt copies; turns RL target into ~supervised. Paper found target networks *crucial* (ablation: without them learning is very poor). Trade-off: slows value propagation, but stability wins.
- **Two target nets (μ', Q')** vs one: y depends on both μ' and Q'; both must be stable to give a stable y_i.
- **Batch norm** vs manual feature scaling: auto-normalizes heterogeneous units so one hyperparameter set generalizes across 20+ tasks. (Spinning Up canonical impl drops BN; layernorm/none common later — but in-frame I keep BN as the choice made, and note the canonical clean reimpl is BN-free; for code I follow Spinning Up which is the canonical clean impl, no BN.)
- **OU correlated noise** vs i.i.d. Gaussian: inertia — correlated noise gives sustained, physically meaningful exploratory pushes; uncorrelated noise averages out for massive/inertial actuators. (Later practice shows plain Gaussian often fine; in-frame the chosen design is OU.)
- **tanh output** scaled by act_limit: bounds actions to valid range; smooth & differentiable for chain rule.
- **Actions injected at 2nd layer of Q** (low-dim): state processed first; common Q(s,a) design. (Spinning Up concatenates s,a at input — simpler clean variant.)
- **Final-layer weights init ~U[-3e-3,3e-3]**: keep initial Q and π outputs near zero so early targets are small/stable; other layers init U[-1/√f, 1/√f] (fan-in).
- **lr: actor 1e-4, critic 1e-3; L2 weight decay 1e-2 on Q; γ=0.99; τ=1e-3; replay 1e6; batch 64 (16 pixels).** Critic faster than actor (critic must lead — actor follows a good critic).
- **Adam, ReLU**: standard stable deep-net training.

## Canonical implementation
OpenAI Spinning Up PyTorch DDPG (code/ddpg.py, code/core.py). Clean canonical re-implementation:
- MLPActor: mlp → tanh, scaled by act_limit. MLPQFunction: mlp on cat([s,a]).
- ReplayBuffer FIFO.
- compute_loss_q: q=Q(o,a); backup=r+γ(1−d)Q_targ(o2,μ_targ(o2)) (no_grad); MSE.
- compute_loss_pi: −mean Q(o,μ(o)) (gradient ascent on Q via the actor).
- update: step Q; freeze Q params; step π; unfreeze; polyak target update θ_targ←ρθ_targ+(1−ρ)θ (ρ=polyak=0.995, i.e. τ=1−ρ=0.005).
- get_action: μ(o)+act_noise·N(0,1), clipped to [−act_limit, act_limit] (Spinning Up uses Gaussian act_noise=0.1, not OU).
- Hyperparams: gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch=100, start_steps=10000 (uniform random for exploration warmup), update_after=1000, update_every=50.
- Main loop: act with noise (random first start_steps) → store → every update_every steps do update_every gradient updates.

Note discrepancies to keep honest in answer.md: Spinning Up uses Gaussian noise (not OU), no batch norm, concatenated (s,a) input, polyak=0.995, both lr=1e-3, start_steps uniform-random warmup. The original DDPG used OU noise, batch norm, actor lr 1e-4, action injected at 2nd Q layer, τ=1e-3. The code in answer.md follows the canonical Spinning Up structure (it's the real, runnable canonical impl) while reasoning.md derives the original design choices in-frame.
