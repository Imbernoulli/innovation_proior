# IQL synthesis (Offline RL with Implicit Q-Learning, 2110.06169, title verified)

## Pain point / research question
Offline RL: learn a policy purely from a fixed dataset D collected by behavior policy π_β, no
environment interaction. To improve over π_β you must estimate values of actions NOT in the data —
but Q on out-of-distribution (OOD) actions is unreliable, and standard Q-learning's
target max_{a'} Q(s',a') queries exactly those OOD actions → extrapolation error → overestimation →
policy chases the error. Prior fix: constrain policy near π_β OR regularize Q to be low on OOD
actions. Both impose a tradeoff between improvement and robustness-to-shift. CAN WE never query OOD
actions at all during value training, yet still do multi-step dynamic programming (so we can
"stitch" suboptimal trajectories) and improve substantially over π_β?

## Load-bearing ancestors
- **Standard TD / Q-learning (offline)**: L_TD = E_D[(r + γ max_{a'} Q_θ̂(s',a') − Q_θ(s,a))²],
  π(s)=argmax_a Q. The max over a' queries OOD actions → erroneous Q_θ̂(s',a'), overestimation.
- **Policy-constraint offline methods**: BCQ (Fujimoto 2019, generative model μ(·|s), sample N
  candidate actions, argmax Q over them — but generative model can still emit OOD actions; N plays a
  role like τ does for IQL), BEAR (Kumar 2019, MMD support constraint), BRAC (Wu 2019), TD3+BC
  (Fujimoto 2021, add BC term to policy loss), AWAC (Nair 2020, implicit KL constraint via advantage
  weighting). All still bootstrap a learned Q at OOD actions or query it during improvement.
- **Value-regularization**: CQL (Kumar 2020, push down Q on OOD actions, push up on data),
  Fisher-BRC (Kostrikov 2021). Adds a regularizer; still a tradeoff knob.
- **One-step / single-step methods**: Onestep RL (Brandfonbrener 2021), AWR (Peng 2019). Fit
  Q^{π_β} or V^{π_β} with SARSA (only dataset actions, no OOD), then extract greedy/advantage-weighted
  policy with ONE step of improvement. Safe (never queries OOD), but does NOT iterate dynamic
  programming → cannot stitch → fails on antmaze medium/large where no near-optimal trajectory exists.
  Decision Transformer (Chen 2021): BC-style return-conditioned sequence model, also one-step-ish,
  also can't stitch.
- **SARSA-style fitted Q evaluation**: L = E_{(s,a,s',a')~D}[(r + γ Q_θ̂(s',a') − Q_θ(s,a))²]. Only
  dataset actions a, a'. MSE → fits Q to the MEAN of TD targets → Q ≈ Q^{π_β} (value of behavior
  policy). Never OOD. But it's just policy EVALUATION of π_β, no improvement.
- **Expectile regression** (Newey & Powell 1987; Aigner 1976): τ-expectile m_τ solves
  argmin_m E[L_2^τ(x−m)], L_2^τ(u) = |τ − 1(u<0)|·u². τ=0.5 → mean (MSE). τ→1 upweights positive
  residuals → estimates an upper tail / approaches sup of support. Conditional version
  argmin_{m(x)} E[L_2^τ(y − m(x))] gives a network. Related to quantile regression (Koenker 2001,
  asymmetric L1) used in distributional RL (Dabney 2018 QR-DQN/IQN) but expectile = asymmetric L2,
  trivially a tweak of MSE.
- **Advantage-Weighted Regression (AWR/Peng 2019; REPS Peters 2010; AWAC Nair 2020)**: KL-constrained
  policy improvement max_π E_{a~π}[A(s,a)] s.t. KL(π‖π_β)≤ε has closed-form π*(a|s) ∝
  π_β(a|s)·exp(A(s,a)/λ); project onto parametric policy by weighted max-likelihood:
  L_π = E_D[exp(β·A(s,a))·log π_φ(a|s)], A = Q − V. β = inverse temperature. Only uses dataset
  actions (the exp weight reweights observed (s,a) pairs). β→0 → BC; β→∞ → greedy.
- **Clipped double Q-learning** (Fujimoto 2018, TD3): two Q nets, take min to fight overestimation.
- **Target network / Polyak** (Mnih 2015; soft update α).

## The method (IQL)
Goal value function (Eqn 4): learn Q estimating the IN-SUPPORT max,
  r(s,a) + γ max_{a' : π_β(a'|s')>0} Q_θ̂(s',a') − Q_θ(s,a), evaluated only on dataset pairs.
Achieve it WITHOUT querying a' by expectile regression + a separate value net.

If we just did expectile regression on the SARSA TD residual
  L(θ) = E_{(s,a,s',a')~D}[L_2^τ(r + γ Q_θ̂(s',a') − Q_θ(s,a))]
the upper expectile would mix TWO sources of randomness: the action a' (what we want — the best
in-support action) AND the stochastic transition s'~p(·|s,a) (what we DON'T want — a "lucky" transition
would inflate the target without a real better action). So SEPARATE them with a value net:

  V update (Eqn 5):  L_V(ψ) = E_{(s,a)~D}[ L_2^τ( Q_θ̂(s,a) − V_ψ(s) ) ]
     → V_ψ(s) is the τ-expectile of Q over the ACTION distribution only (the s, a are both from D;
       Q_θ̂ uses the TARGET critic; no a' needed). As τ→1, V_ψ(s) → max_{a in support} Q(s,a).
  Q update (Eqn 6):  L_Q(θ) = E_{(s,a,s')~D}[ ( r(s,a) + γ V_ψ(s') − Q_θ(s,a) )² ]
     → plain MSE; V_ψ(s') already isolated the action-expectile, and the outer expectation over s'
       averages the dynamics (no lucky-sample problem). This is the dynamic-programming backup, and it
       only ever uses dataset (s,a,s').
  Target: θ̂ ← (1−α)θ̂ + αθ (Polyak on the critic). Use clipped double-Q: Q = min(Q1,Q2) in V & policy.

Policy extraction (Eqn 7, AWR): L_π(φ) = E_{(s,a)~D}[ exp(β·(Q_θ̂(s,a) − V_ψ(s)))·log π_φ(a|s) ],
clip exp weight to ≤ 100. Decoupled — policy does NOT influence value training, can run concurrently
or after.

## Theory (must re-derive in reasoning)
Define recursively V_τ(s)=E^τ_{a~μ(·|s)}[Q_τ(s,a)], Q_τ(s,a)=r+γE_{s'}[V_τ(s')], μ = behavior π_β.
- Lemma 1: for bounded-support X with sup x*, lim_{τ→1} m_τ = x*. (expectiles share the same
  supremum; m_τ monotone non-decreasing in τ → limit is sup.)
- Lemma 2: τ1<τ2 ⟹ V_{τ1}(s) ≤ V_{τ2}(s) for all s. Proof = policy-improvement-style telescoping:
  V_{τ1}(s) = E^{τ1}[r+γE_{s'}V_{τ1}(s')] ≤ E^{τ2}[r+γE_{s'}V_{τ1}(s')]  (since m_{τ1}≤m_{τ2})
  = E^{τ2}[r+γE_{s'}E^{τ1}[r'+γE V_{τ1}]] ≤ E^{τ2}[r+γE_{s'}E^{τ2}[...]] ≤ ... ≤ V_{τ2}(s).
- Corollary: V_τ(s) ≤ max_{a:π_β>0} Q*(s,a) for all τ (expectile is a convex combination ≤ max).
- Theorem: lim_{τ→1} V_τ(s) = max_{a:π_β(a|s)>0} Q*(s,a). (Lemma 1 gives V_τ→ in-support max of Q_τ;
  combined with the monotone bound, V_τ → constrained optimal.) So τ interpolates SARSA (τ=0.5,
  evaluates π_β) ↔ in-support Q-learning (τ→1). Larger τ = closer to max but harder optimization.

## Design-decision → why
- expectile (asymmetric L2) NOT quantile (asymmetric L1): we only need ONE statistic (the upper
  expectile), and expectile is a trivial reweighting of the MSE that RL already uses; empirically
  worked somewhat better than quantile's L1.
- SEPARATE V net (not expectile directly on TD residual): otherwise the upper expectile rewards
  lucky stochastic transitions, not better actions. V isolates the action-randomness; Q-MSE re-averages
  dynamics. This is THE central trick.
- target critic Q_θ̂ in V-loss, online V in Q-loss: standard bootstrap stabilization; V chases a slow
  target Q, Q regresses to the just-updated V.
- clipped double-Q (min): overestimation control, carried from TD3/SAC.
- AWR policy extraction (not argmax / not DDPG-style ∇_a Q): argmax/∇_a Q would query Q at policy
  actions = OOD again. AWR reweights ONLY dataset actions by exp(βA), so extraction is also in-sample;
  also gives an implicit KL-to-π_β constraint (good for offline + for online finetuning, per AWAC).
- β inverse temperature: β→0 BC (safe), β→∞ greedy (aggressive). antmaze needs larger τ AND β=10
  (sharp, stitching); locomotion τ=0.7 β=3.
- clip exp weight ≤ 100: prevent a few huge advantages from dominating the BC loss (numerical, from
  Brandfonbrener 2021).
- decoupled policy: value training is policy-free → no actor-critic coupling instability; extraction
  is pure supervised; enables concurrent or post-hoc extraction and online finetuning.

## Canonical hyperparameters (ikostrikov/implicit_q_learning, JAX/Flax — CONFIRMED in code)
- Adam lr 3e-4 for actor/critic/value; 2-layer MLP, 256 hidden, ReLU.
- discount γ=0.99; Polyak τ_polyak (code `tau`) = 0.005.
- expectile τ: code default 0.8; paper appendix τ=0.7 (locomotion) / 0.9 (antmaze) / 0.7 (kitchen/adroit).
- temperature β: code default 0.1 (=multiplier); paper β=3.0 (locomotion) / 10.0 (antmaze) /
  0.5 (kitchen/adroit). NOTE code calls it `temperature` and multiplies: exp((q−v)*temperature). β is
  the inverse temperature in the paper notation. (Default config differs from paper per-domain values.)
- exp advantage clip to 100.0.
- policy = Gaussian with STATE-INDEPENDENT std (state_dependent_std=False), no tanh squash in dist
  (tanh applied at sampling via clip to [-1,1]); log_std_min=-5, log_std_scale=1e-3.
- COSINE decay schedule on actor lr only (optax.cosine_decay_schedule(-actor_lr, max_steps)); dropout
  optional (0.1 for kitchen/adroit).
- 1M gradient steps, batch 256. Reward preprocessing: locomotion rewards /(max−min traj return)*1000;
  antmaze rewards −1.0.
- Update order per step: update V (from target critic), update actor (AWR, from target critic + new V),
  update Q (regress to new V), Polyak target critic.

## Scaffold ↔ code correspondence
Final code fills: Critic/DoubleCritic, ValueCritic, Gaussian policy, expectile loss, update_v
(expectile reg of Q_target−V), update_q (MSE to r+γV(s')), update_actor (AWR exp(βA)·logπ),
Polyak target_update, the per-step update orchestration. Scaffold = generic offline-RL actor-critic
harness with Q net, V net, policy net stubs + a generic "value loss" stub + "policy extraction" stub.

## Self-derivation checks
- L_2^τ(u)=|τ−1(u<0)|u². u>0 (residual positive, target above estimate): weight |τ−0|=τ. u<0: weight
  |τ−1|=1−τ. For τ>0.5, positive residuals weighted MORE → estimate pulled UP → upper expectile. ✓
  Code: weight = where(diff>0, τ, 1−τ), diff = Q−V; if Q>V (under-estimate) weight τ (large), pulls V
  up toward high Q. ✓ matches.
- V-loss uses Q_θ̂(s,a) (target critic, min of two), expectile of (Q−V) over a~D|s. ✓
- Q-loss target r+γ·mask·V_ψ(s'); mask = (1−done). MSE on both q1,q2. ✓
- actor: A=min(Q1,Q2)−V, weight=min(exp(βA),100), loss=−mean(weight·logπ(a|s)). ✓
- Lemma 2 telescoping: each E^{τ1}→E^{τ2} step uses m_{τ1}≤m_{τ2} (Lemma: expectile monotone in τ) and
  monotonicity of the operator (r+γE[·] preserves order). Repeated → V_{τ1}≤V_{τ2}. ✓
- as τ→0.5 V is mean over actions = V^{π_β} (SARSA); as τ→1 V→ in-support max (Q-learning). ✓
