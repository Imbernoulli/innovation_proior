# RND synthesis (arXiv 1810.12894, ICLR 2019; Burda, Edwards, Storkey, Klimov, OpenAI)

## Pain point / research question
Sparse-reward exploration. RL maximizes return; works with dense rewards, fails when rewards are
sparse/hard to find (Montezuma's Revenge: rewards hundreds of steps apart). Need DIRECTED
exploration. Modern RL needs SCALE (many parallel envs, billions of frames), so the exploration
method must scale cheaply. Count/pseudo-count/info-gain/prediction-gain methods are hard to scale.
Want: an exploration bonus that is (a) trivial to implement, (b) works with high-dim image obs,
(c) usable with any policy optimizer, (d) cheap — one forward pass per batch.

## The noisy-TV problem (the obstacle to beat)
Prediction-error bonuses (forward dynamics: predict s_{t+1} from s_t,a_t) attract the agent to
transitions whose answer is a STOCHASTIC function of inputs (TV static, coin flips). The error
never vanishes there → agent gets stuck. Fix in prior work = measure prediction IMPROVEMENT/gain,
but that's expensive to scale.

## Sources of prediction error (the taxonomy)
1. Amount of training data — high error where few similar examples seen (epistemic). <- DESIRABLE,
   this is what makes error a novelty signal.
2. Stochasticity — target function is random (aleatoric). Forward dynamics suffers this (noisy-TV).
3. Model misspecification — info missing / model class too limited.
4. Learning dynamics — optimizer fails to fit.
Want factor 1 only; avoid 2,3.

## Core idea: Random Network Distillation
Make the prediction problem have a DETERMINISTIC answer that lies INSIDE the predictor's model
class. Two networks:
- TARGET f: O → R^k, fixed & randomly initialized (frozen).
- PREDICTOR f̂: O → R^k, trained by SGD to minimize E||f̂(x;θ) − f(x)||^2.
Intrinsic reward i_t = ||f̂(s_{t+1}) − f(s_{t+1})||^2.
- Deterministic target → no factor 2 (aleatoric). Target in predictor's model class → no factor 3
  (misspecification). So error is dominated by factor 1 (epistemic / novelty).
- NNs have lower prediction error on inputs similar to training data → error high on novel states.
- Objection: a powerful optimizer could perfectly mimic f everywhere (e.g. f itself is a perfect
  predictor). MNIST toy experiment shows standard gradient methods DON'T overgeneralize this way —
  test error decreases with number of similar training examples. So it works as a novelty detector.

## Relation to uncertainty quantification (Osband randomized prior functions)
Osband 2018: ensemble of g_θ = f_θ + f_{θ*}, θ* from prior p(θ*), θ minimizes
E||f_θ(x_i)+f_{θ*}(x_i)−y_i||^2 + R(θ); the ensemble approximates the posterior. Specialize
targets y_i = 0: argmin E||f_θ(x)+f_{θ*}(x)||^2 = distilling a random function from the prior. Each
output coordinate = ensemble member (param sharing); MSE = estimate of predictive variance =
uncertainty in predicting the constant-zero function. So RND distillation error ≈ uncertainty.

## Combining intrinsic + extrinsic returns (two value heads)
- NON-EPISODIC intrinsic return explores better: intrinsic return should reflect ALL future novel
  states regardless of episode boundaries. Episodic intrinsic reward can make agent risk-averse
  (game-over → zero future return → avoids risky-but-novel maneuvers; real cost of game over is
  just the opportunity cost of replaying the boring start). Also episodic intrinsic reward can
  leak task info.
- But non-episodic EXTRINSIC reward is exploitable (grab early reward, suicide, repeat). So:
  intrinsic non-episodic, extrinsic episodic.
- Return linear in rewards → R = R_E + R_I. Fit TWO value heads V_E, V_I separately on their own
  returns; V = V_E + V_I. Lets you use DIFFERENT discount factors per stream and combine
  episodic+non-episodic. (γ_E=0.999, γ_I=0.99.)
- Even with same discount, two heads = extra supervisory signal; extrinsic reward is stationary,
  intrinsic is non-stationary → separating helps.
- Advantages combined: A = A_I + A_E. (Coefs: extrinsic 2, intrinsic 1.)

## Normalization (crucial details)
- INTRINSIC REWARD normalization: divide i_t by a running estimate of the std of the intrinsic
  RETURNS (not rewards), so the bonus scale is consistent across envs / over time.
- OBSERVATION normalization: CRUCIAL for the random target (frozen params can't adapt to data
  scale; without it the embedding variance can be tiny, carrying no info). Whiten each dim:
  (x − running_mean)/running_std, then CLIP to [−5,5]. Initialize normalization stats by stepping
  a RANDOM agent for M steps before training. Same obs-normalization for predictor AND target, but
  NOT for the policy network.
- Policy/value net: single-frame x/255, frames stacked 4. Predictor/target: 1 frame stacked,
  normalized+clipped.

## RL algorithm
PPO (clip range [0.9,1.1], λ_GAE=0.95, entropy coef 0.001), Adam lr 1e-4, rollout 128, 128 envs,
4 minibatches, 4 epochs. Predictor trained on only 25% of experience per batch (keep prob 0.25,
to hold effective predictor batch size constant when scaling envs). Sticky actions p=0.25
(non-determinism, prevents memorization). Extrinsic reward clip [−1,1], intrinsic NOT clipped.

## Algorithm (Alg 1)
- Init obs-norm by M random steps.
- For each rollout i: for j=1..K: a_t~π, observe s_{t+1},e_t; i_t = ||f̂(s_{t+1}) − f(s_{t+1})||^2;
  add to batch; update reward-norm stats. Then: normalize intrinsic rewards in batch; compute
  R_I,A_I (non-episodic) and R_E,A_E (episodic); A = A_I + A_E; update obs-norm. For j=1..N_opt:
  optimize θ_π wrt PPO loss; optimize θ_f̂ wrt distillation loss.

## Architecture (canonical jcwleo PyTorch / openai)
- Both target & predictor: conv encoder identical to DQN (Mnih) [openai uses DQN conv;
  jcwleo uses 1→32→64→64 conv with LeakyReLU], then dense.
- TARGET: conv → flatten → single Linear → 512-dim output (k=512). Frozen (requires_grad=False).
- PREDICTOR: conv → flatten → Linear 512 → ReLU → Linear 512 → ReLU → Linear 512. Trainable
  (DEEPER than target — extra FC layers so it can fit the target but doesn't trivially copy it).
- Output dim k = 512.
- Predictor params init (orthogonal); target frozen after random init.

## Design-decision table
| choice | why | rejected alt |
|---|---|---|
| predict a fixed RANDOM network's output | deterministic target (no factor 2) inside predictor's model class (no factor 3) → error = novelty only | forward-dynamics prediction (noisy-TV / aleatoric) |
| MSE distillation error as bonus | NN error low on seen-like inputs, high on novel; one forward pass, cheap, scalable | prediction-improvement/gain (expensive); counts (don't scale) |
| target frozen, predictor trainable & deeper | predictor distills target on visited data; deeper so it can fit but won't trivially copy everywhere | predictor = target (trivial); equal depth |
| two value heads V_E, V_I | combine episodic + non-episodic and different discounts; extra supervision; intrinsic non-stationary | single value head (can't mix streams) |
| intrinsic non-episodic, extrinsic episodic | non-episodic intrinsic explores more (no risk-aversion at game-over); episodic extrinsic avoids suicide-exploit | both episodic (less exploration); both non-episodic (exploitable) |
| γ_E=0.999 > γ_I=0.99 | extrinsic needs long horizon (rewards far apart); high γ_I hurts (empirically) | γ_I=0.999 (hurts), γ_E=0.99 (worse) |
| normalize intrinsic reward by running std of returns | bonus scale consistent across envs/time | raw error (scale varies wildly) |
| observation normalization (whiten+clip[−5,5]) for predictor/target | frozen random target can't adapt to data scale; else embedding variance ~0 | none (degenerate embedding) |
| init obs-norm with random agent for M steps | get sane normalization stats before learning | start from scratch |
| PPO policy optimizer | scalable, little tuning; bonus is optimizer-agnostic | any (the bonus is generic) |
| predictor trained on 25% of batch | hold predictor's effective batch size constant when scaling envs (so intrinsic-reward decay rate constant) | train predictor on all (decays too fast at scale) |
