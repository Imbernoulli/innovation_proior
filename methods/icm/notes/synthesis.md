# ICM synthesis (arXiv 1705.05363, ICML 2017; Pathak, Agrawal, Efros, Darrell, UC Berkeley)

## Pain point / research question
Extrinsic rewards often sparse/absent; random exploration won't stumble into goals. Need an
INTRINSIC reward to drive exploration. Two broad classes of intrinsic reward: (1) novelty of
states (needs density model of states); (2) prediction error / uncertainty about consequences of
actions (needs a dynamics model predicting s_{t+1} from s_t,a_t). Both hard in high-dim image
state spaces.
KEY OBSTACLE: the "noisy-TV" problem. If intrinsic reward = prediction error in raw pixel space,
the agent gets permanently rewarded by inherently-unpredictable but irrelevant stuff (TV static,
moving leaves, shadows, distractors, other agents). Pixel prediction error stays high forever ->
artificial-curiosity trap, exploration stalls. Tabular novelty counts have the same issue.
schmidhuber's "learnable but hard to predict" fix needs measuring learning progress — no feasible
mechanism.

## Three sources of observation change (the taxonomy that defines the right feature space)
(1) things the agent CAN control;
(2) things the agent cannot control but that AFFECT the agent (e.g. another vehicle);
(3) things out of control AND not affecting the agent (moving leaves, TV static).
A good curiosity feature space should model (1) and (2), be UNAFFECTED by (3).

## Core insight
Don't predict in raw pixel space. Predict in a learned FEATURE space φ(s) that encodes only
action-relevant information. Learn φ by self-supervision on an INVERSE DYNAMICS proxy task:
predict the action a_t from φ(s_t),φ(s_{t+1}). Because the net only needs to predict the action,
it has no incentive to encode factors of variation that the agent's actions don't affect (and
that don't help predict the action) — i.e. it naturally drops case (3). Then train a FORWARD
model in this φ-space; its prediction error is the intrinsic reward.

## The two models (ICM)
- State encoder: φ(s) (shared, learned by the inverse task).
- Inverse model g: â_t = g(s_t, s_{t+1}; θ_I) = g(φ(s_t), φ(s_{t+1})). Loss L_I(â_t,a_t).
  Discrete a: g outputs softmax over actions; minimizing L_I = MLE under multinomial =
  cross-entropy.
- Forward model f: φ̂(s_{t+1}) = f(φ(s_t), a_t; θ_F).
  Loss L_F = ½ ||φ̂(s_{t+1}) − φ(s_{t+1})||²_2.
- Intrinsic reward: r_t^i = (η/2) ||φ̂(s_{t+1}) − φ(s_{t+1})||²_2,  η>0.

## Policy and joint objective
- Policy π(s;θ_P) trained (A3C) to maximize E[Σ_t r_t], r_t = r_t^i + r_t^e (r_t^e mostly 0).
- Overall optimization (eq 7):
  min_{θ_P,θ_I,θ_F} [ −λ E_{π}[Σ_t r_t] + (1−β)L_I + β L_F ].
  β∈[0,1] weighs inverse vs forward loss; λ>0 weighs policy gradient vs learning the reward
  signal. NOTE: the −λE[Σr] term is just the policy-gradient objective written as a minimization;
  ICM params θ_I,θ_F are NOT trained by the policy loss, only by L_I,L_F. The reward r^i uses the
  forward error but ICM is trained on its own losses (decoupled from policy reward maximization).

## Why inverse model first (vs. forward-only or pixel)
- Pixel forward model: rewards unpredictable irrelevant pixels (noisy-TV).
- A pure forward model in a LEARNED space could collapse φ to a constant (trivially predictable,
  zero error) — degenerate. The inverse task ANCHORS φ: φ must retain enough info to recover the
  action, so it can't collapse, and it only keeps action-relevant content.
- Prior work (Agrawal 2016) used a joint inverse-forward model but used the forward model only as
  a REGULARIZER for inverse-model features; here the forward ERROR is the reward signal — new use.

## Robustness property
φ has no incentive to encode case-(3) variation -> agent gets no reward for inherently
unpredictable states -> robust to distractors, illumination, noise. (Validated: pixel-curiosity
fails in "very sparse"/noisy settings, ICM doesn't — but that's evaluation, not for context.)

## Architecture (exact)
- Input: RGB -> grayscale, 42×42, concat current + 3 previous frames (temporal). Action repeat 4
  (Doom) / 6 (Mario) at train, none at inference.
- A3C net: 4 conv layers, 32 filters each, 3×3, stride 2, pad 1, ELU after each. Then LSTM 256.
  Two FC heads: value, policy.
- ICM encoder φ: 4 conv layers, 32 filters, 3×3, stride 2, pad 1, ELU. Output dim of φ = 288
  (flattened 4th conv output: 42->21->11->6->3, 3×3×32=288).
- Inverse model: concat φ(s_t),φ(s_{t+1}) -> FC 256 (ReLU) -> FC n_actions (logits) -> softmax CE.
- Forward model: concat φ(s_t), a_t (one-hot) -> FC 256 (ReLU) -> FC 288 (=dim φ). MSE ×0.5.
- β = 0.2, λ = 0.1, lr = 1e-3 (Adam). η = scaling (PREDICTION_BETA in code).
- 20 async workers (A3C), Adam not shared across workers.

## Canonical implementation (pathak22/noreward-rl, model.py StateActionPredictor)
- universeHead: input [N,42,42,1] -> 4 conv (32, 3×3, stride 2) -> flatten [N,288].
- inverse: g=concat(phi1,phi2); g=relu(linear(g,256)); logits=linear(g,n_actions);
  invloss = mean(sparse_softmax_cross_entropy(logits, action_index)).
- forward: f=concat(phi1, asample[one-hot]); f=relu(linear(f,256)); f=linear(f,288);
  forwardloss = 0.5*mean((f - phi2)^2).
- bonus (intrinsic reward) = forwardloss * PREDICTION_BETA  (= η factor).

## Design-decision table
| choice | why | rejected alt |
|---|---|---|
| predict in learned feature space φ, not pixels | avoids noisy-TV; only action-relevant info | pixel prediction error (rewards irrelevant unpredictable pixels) |
| learn φ via inverse dynamics (predict action) | φ keeps only what's needed to explain the agent's effect; drops case (3); can't collapse | random/fixed features (encode irrelevant stuff); forward-only learned φ (collapses to constant) |
| forward model in φ-space; error = reward | measures genuine learnable surprise about controllable dynamics | forward in pixels (noisy-TV) |
| L_I cross-entropy (discrete actions) | MLE under multinomial; standard classification | regression on actions |
| L_F = ½||·||² and r^i=(η/2)||·||² | simple MSE; η scales reward magnitude | other distances |
| joint min (1−β)L_I + βL_F, β=0.2 | balance: inverse must shape φ (dominant), forward provides reward | β=0/1 (one model only) |
| λ=0.1 weight on policy gradient | balance policy learning vs representation learning | larger λ (overweights sparse policy signal) |
| ICM trained on own losses, not policy reward | forward error must reflect true model error, not be gamed by policy | training ICM to max reward (degenerate) |
| A3C policy | scalable on-policy actor-critic; works with intrinsic reward | off-policy (replay complicates on-policy curiosity) |
