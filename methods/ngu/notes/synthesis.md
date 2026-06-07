# NGU synthesis notes (from primary tex + explainers)

## Pain point / research question
Prediction-error and novelty bonuses (RND, ICM, pseudo-counts) all VANISH once a state stops being
novel. After the bonus decays, the agent has no drive to revisit a state, even if revisiting opens
downstream learning. The exploration signal is a one-shot frontier-pusher; it does not maintain
*persistent* exploration. Also: a single value function trained on r^e + beta r^i bakes the
exploratory bias permanently into the policy — you cannot turn it off to exploit. NGU's goal: a
bonus that (i) rapidly discourages revisiting a state *within* an episode, (ii) slowly discourages
visits to states seen many times *across* episodes, (iii) ignores uncontrollable variation; plus a
way to separate the exploratory from the exploitative policy so dense-reward games aren't hurt.

## The two-timescale idea (the heart)
- EPISODIC novelty: per-episode memory M reset every episode; bonus large for states far from
  what's been seen *this episode*. So even a state visited thousands of times across training gives
  full bonus at the start of a fresh episode -> agent re-explores every episode -> "never give up".
- LIFELONG novelty: a slowly-changing global signal (RND error) that down-weights states the agent
  has mastered over the whole of training.
- COMBINE multiplicatively: r^i_t = r^episodic_t * min(max(alpha_t, 1), L), L=5.
  Multiplicative (not additive) because: episodic term is the driver that keeps re-exploring; the
  lifelong factor is a *modulator* in [1, L]. Floor at 1 => lifelong can only AMPLIFY, never kill,
  the episodic drive (so exploration never fully dies); ceiling at L=5 => one anomalous RND spike
  can't blow up the reward. As lifelong novelty vanishes alpha_t -> ~1 and r^i -> r^episodic.

## Episodic module details
- Embedding f: O -> R^p (p=32), the "controllable state". Trained by INVERSE DYNAMICS (Pathak
  2017): Siamese f on (x_t, x_{t+1}), one-hidden-layer MLP h + softmax predicts a_t;
  p(a|x_t,x_{t+1}) = h(f(x_t), f(x_{t+1})), max-likelihood. Forces f to encode only
  action-relevant (controllable) content, ignore uncontrollable distractors. (Reuse of ICM's
  inverse model, but here it ONLY learns the embedding for the memory — no forward model, no
  forward-error reward.)
- Memory M = {f(x_0),...,f(x_{t-1})}, slot-based, reset each episode, ring buffer cap 30000.
- Pseudo-count bonus (Strehl & Littman 2008 count->bonus 1/sqrt(n)):
  r^episodic_t = 1 / sqrt(n(f(x_t))) ~= 1 / ( sqrt( sum_{f_i in N_k} K(f(x_t), f_i) ) + c )
  with n approximated by sum of kernel similarities over k-nearest neighbors in M.
- Kernel (inverse kernel, Neural Episodic Control lineage, Blundell 2016 / Pritzel 2017):
  K(x,y) = eps / ( d^2(x,y)/d_m^2 + eps ), eps=1e-3 in main text (1e-4 in hyperparam table),
  d Euclidean, d_m^2 = running average of squared Euclidean distance of the k-th nearest neighbors
  (per-task normalization so the kernel adapts to typical embedding scale).
- Algorithm (App reward_algorithm): compute k-NN distances d_k; update running d_m^2;
  normalize d_n = d_k / d_m^2; CLUSTER: d_n <- max(d_n - xi, 0) with xi=0.008 (zero out tiny
  distances -> near-identical states treated as one cluster, don't self-inflate count);
  K_v = eps/(d_n+eps); s = sqrt(sum K_v) + c; if s > s_m (=8) then r^episodic=0 else 1/s.
  (s>s_m means already heavily visited this episode -> no bonus.) c=0.001, k=10.

## Lifelong module (RND)
- Random target g: O->R^k frozen; predictor g_hat trained to minimize ||g_hat(x)-g(x)||^2.
- alpha_t = 1 + (err(x_t) - mu_e)/sigma_e, mu_e/sigma_e running mean/std of err. Normalized so
  alpha hovers around 1; clip into [1,L] inside Eq (clipping).

## Agent / UVFA
- Q(x,a,beta_i) one net for family r^{beta_i} = r^e + beta_i r^i; beta_0=0 (exploit),
  beta_{N-1}=beta=0.3 (max explore), N=32 mixtures, sigmoid spacing of beta_i.
- Each beta_i paired with gamma_i, gamma_max=0.997 (exploit, long horizon) down to gamma_min=0.99
  (explore; dense small intrinsic reward -> shorter horizon fine). Log-spaced.
- Base R2D2: recurrent (LSTM) replay distributed DQN, Retrace(lambda=0.95) double-Q loss,
  dueling architecture. Intrinsic reward fed as INPUT (with prev action, prev r^e, prev r^i, beta
  one-hot) so the augmented non-stationary reward stays Markov / handled as observation (avoid POMDP).
- Embedding + RND trained from last 5 frames of each sampled sequence; RL loss on all timesteps.
- Reward transform R2D2: h(x)=sign(x)(sqrt(|x|+1)-1)+1e-3 x. beta(intrinsic weight)=0.3.

## Design-decision -> why table
- episodic memory reset per episode: gives persistent re-exploration; defeats RND's
  vanishing-novelty (the failure flowing from RND).
- multiplicative w/ floor 1, cap L=5: episodic is driver, lifelong is bounded modulator; never kills
  exploration, never explodes.
- inverse-dynamics embedding: controllable-state, noise/distractor-blind (Random Disco Maze motiv).
- kNN pseudo-count + inverse kernel + d_m^2 normalization: generalize counts to never-repeated
  high-dim states; per-task distance scale.
- cluster xi: don't let near-duplicate frames inflate the count (oscillation exploit).
- s_m cap: zero bonus when already saturated this episode.
- UVFA family beta_0=0..beta: separate exploit policy retrievable greedily -> dense games unhurt;
  shared weights => exploratory auxiliary tasks bootstrap representation even with no extrinsic reward.
- gamma_i schedule: exploit needs near-undiscounted (0.997); explore intrinsic dense+small => 0.99.
- R2D2 base + reward-as-input: handle recurrence, off-policy scale, and non-stationary augmented
  reward without breaking Markov assumption.

## Canonical implementation grounding
No single official DeepMind NGU repo is public; widely-referenced faithful reimplementations encode
the same episodic-kNN + RND modulator. The code in the deliverables mirrors Alg 1 (episodic reward)
exactly and the RND modulator, written against the task's IntrinsicBonusModule interface
(compute_bonus / loss / normalize_rollout_rewards / mix_advantages). Kept faithful to the paper's
formulas and constants; the agent base in the task is PPO (not R2D2), so the trace lands the *bonus
module* (the contribution) on the fixed PPO loop, which is exactly the task's editable surface.
