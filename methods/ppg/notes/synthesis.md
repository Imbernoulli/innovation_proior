# PPG synthesis (Phasic Policy Gradient, 2009.04416, title verified)

## Pain point / research question
On-policy actor-critic (PPO etc.) has two key quantities: policy π and value function V. Implementation
decision: SHARE network parameters between π and V, or use SEPARATE networks?
- SHARED: features trained by one objective help the other (esp. high-dim vision inputs like Procgen);
  but (a) must pick a relative weight vfcoef balancing the two losses, and one objective's optimization
  can INTERFERE with the other; (b) forces π and V to train on the SAME data with the SAME sample reuse
  (number of epochs) — an artificial restriction.
- SEPARATE: no interference, can train each with its own sample reuse; but loses the feature sharing.
Want: best of both — keep feature sharing AND decouple the two trainings (independent objectives,
independent sample reuse). Two empirical observations to establish/exploit:
  1. policy↔value interference HURTS performance when params shared (Procgen: but sharing is also
     critical — separate-net PPO does worse, see appendix B).
  2. value optimization TOLERATES much higher sample reuse than policy optimization (PPO's gain from
     more epochs is really the VALUE being under-trained, not the policy — single policy epoch is
     near-optimal once isolated).

## Load-bearing ancestors
- **PPO** (Schulman 2017): clipped surrogate L^clip = Ê[min(r_t Â_t, clip(r_t,1−ε,1+ε)Â_t)],
  r_t=π_θ(a|s)/π_old(a|s); optimize L^clip + β_S·S[π] (entropy bonus); value loss
  L^value = Ê[½(V_θV(s)−V̂targ)²]. Â and V̂targ from GAE. Also proposed an adaptive KL-penalty variant.
- **GAE** (Schulman 2016): Â_t = Σ (γλ)^l δ_{t+l}, δ_t = r_t + γV(s_{t+1})−V(s_t); V̂targ = V + Â.
- **A3C/IMPALA/ACKTR/TRPO/V-MPO/AWR**: the actor-critic family; PPG's policy objective is swappable for
  any of these (TRPO trust region, AWR/V-MPO exp-advantage-weighted MLE). PPO chosen as the default.
- **Auxiliary tasks / value as representation learner**: Bellemare 2019 (adversarial value functions
  as auxiliary), Lyle 2019 (distributional RL's benefit attributed to richer value-as-auxiliary
  signal), Jaderberg 2017 UNREAL (auxiliary tasks). Idea: a value-prediction head trains good features.
- **ITER** (Igl 2020): alternates an RL phase with a DISTILLATION phase (distill teacher π,V into fresh
  student nets) to fight non-stationarity / improve generalization. PPG shares the
  alternate-RL-with-distillation structure but distills V's features INTO the policy net for sample
  efficiency, not into a fresh net for generalization.
- **Behavioral cloning / KL distillation**: KL(π_old ‖ π_θ) preserves a policy while changing its
  representation. PPG uses it to protect the policy during the auxiliary phase.
- **Off-policy replay** (SAC/DDPG/ACER): use a buffer for sample efficiency via off-policy updates.
  SAC also uses SEPARATE π and V nets to avoid interference. PPG uses a buffer too — but ONLY to refit
  value targets and train features, NOT to do off-policy policy improvement.

## The method (PPG)
Two alternating phases.
- POLICY PHASE (N_π iterations of PPO, DUAL networks: separate policy net and value net):
  - rollouts under current π; compute V̂targ via GAE.
  - E_π policy epochs: optimize L^clip + β_S·S[π] wrt θ_π (policy net only).
  - E_V value epochs: optimize L^value wrt θ_V (the TRUE value net).
  - store all (s_t, V̂targ_t) into buffer B.
  - The policy network ALSO has an auxiliary value head V_θπ (shares all params with π except the final
    linear layer). It does nothing during the policy phase except exist.
- AUXILIARY PHASE (every N_π policy updates, E_aux epochs over ALL data in B):
  - first snapshot π_old(·|s) for all s in B (the policy right before the aux phase).
  - optimize L^joint = L^aux + β_clone·Ê[KL(π_old(·|s), π_θ(·|s))] wrt θ_π, where
    L^aux = ½·Ê[(V_θπ(s) − V̂targ)²] is the auxiliary value head's value loss. → distills value
    FEATURES into the policy net (via the shared trunk) while the KL clone term keeps the policy itself
    nearly unchanged.
  - ALSO optimize L^value wrt θ_V on all data in B → extra training of the true value net.
  - V̂targ are the SAME targets from the policy phase, FIXED throughout the aux phase.
  - L^joint and L^value share no params → optimized separately.

So: the policy net learns value-predictive features (representation sharing) WITHOUT the value loss
ever interfering with the policy during the policy phase (they're in different phases, and the policy
is protected by the KL clone). And value gets extra sample reuse via E_aux without forcing the policy
to be re-trained on the same data.

## Hyperparameter roles & why
- N_π (=32): policy updates per aux phase. Frequent aux phases HURT (each interferes with policy
  optimization); infrequent aux phases are critical → N_π large.
- E_π (=1): policy sample reuse. Single policy epoch near-optimal when value training is isolated.
- E_V (=1): value sample reuse DURING policy phase (true value net).
- E_aux (=6): aux-phase sample reuse — the main knob for value sample reuse. Benefit tapers ~6 epochs;
  too many → overfit recent data. Usually raise E_aux (not E_V) to give value more reuse.
- β_clone (=1): weight of KL clone term — trades feature distillation vs policy preservation.
- β_S (=.01): entropy bonus; ε=.2 PPO clip; vfcoef=.5; γ=.999; λ=.95; nstep=256; nminibatch=8 (policy
  phase); aux minibatches per epoch per N_π = 16 (aux_mbsize=4); lr=aux_lr=5e-4 Adam; 100M steps;
  4 workers × 64 envs; reward normalization yes; no LSTM, no frame stack.
- arch: "dual" (default, separate nets ~2× params), "detach" (single net, detach value grad at last
  shared layer during policy phase, full grad in aux phase → recovers most benefit at 1× params),
  "shared" (plain shared net = PPO-like baseline).
- KL-penalty variant: replace L^clip with L^KL = Ê[−Â_t·r_t + β_π·KL(π_old,π_θ)], β_π=1 fixed; performs
  ~same as clipping in PPG (rewards normalized so returns ~unit variance, so clipping less critical).

## Design-decision → why
- Two phases instead of one weighted loss: a single shared net must weight L^clip vs L^value (vfcoef),
  and however it's chosen one objective can interfere with the other; and it ties π & V to the same
  data/sample-reuse. Phases remove BOTH constraints — π trained alone in the policy phase (no value
  interference), value features distilled separately in the aux phase, each with its own sample reuse.
- Auxiliary VALUE head on the policy net (not reuse the true value head): the policy net needs to
  LEARN value-predictive features in its own trunk to benefit from sharing; an aux value head V_θπ
  bolted onto the policy trunk does exactly that, and it's discardable — purely a representation tool.
- KL(π_old, π_θ) clone term in L^joint: distilling value features into the policy trunk would distort
  the policy; the behavioral-cloning KL pins the policy outputs to the pre-aux policy so the trunk
  changes but the policy doesn't drift. Forward direction KL(old ‖ new).
- targets V̂targ FIXED during aux phase: they were computed on-policy in the policy phase; the aux
  phase is pure supervised regression onto fixed targets → stable, no moving target.
- single policy epoch E_π=1: isolating value training reveals the policy gains ~nothing from extra
  epochs; PPO's apparent need for 3 epochs is really under-trained value.
- detach variant: if memory matters, one net + detach value grad at the last shared layer during the
  policy phase gives the no-interference property at 1× params (full grad flows in the aux phase).
- replay buffer NOT for off-policy policy improvement: only to refit value targets and train features;
  PPG stays on-policy for the policy.

## Canonical implementation (openai/phasic-policy-gradient — CONFIRMED)
ppg.py PhasicValueModel (arch dual/detach/shared; pi_enc + vf_enc; pi_head + aux_vf_head;
compute_aux_loss = {vf_aux: ½(vpredaux−vtarg)², vf_true: ½(vpredtrue−vtarg)²}; aux_train adds
pol_distance = KL(oldpd, pd)); ppo.py compute_gae + compute_losses (clip pg + negent + vf); train.py
name2coef={"pol_distance": beta_clone=1, "vf_true": vf_true_weight=1}, n_pi=32, n_aux_epochs=6, lr=5e-4,
ImpalaEncoder. impala_cnn.py = the encoder.

## Scaffold ↔ code correspondence
Final code fills: dual encoder model with policy head + true value head + auxiliary value head; GAE;
PPO clip+entropy policy loss and value MSE (policy phase, E_π / E_V epochs); aux-phase joint loss
(aux value MSE + β_clone·KL clone) + true value MSE over buffer B (E_aux epochs); the
policy-phase/aux-phase outer alternation every N_π. Scaffold = generic on-policy actor-critic with a
shared/dual encoder, GAE, a PPO update stub, and an empty "auxiliary phase" stub + the outer loop stub.

## Self-derivation checks
- L^clip min(rÂ, clip(r,1−ε,1+ε)Â): pessimistic (max of the two negated losses in code:
  pg=max(−Âr, −Â·clip)). ✓ code: pg_losses=max(−adv·ratio, −adv·clamp).
- GAE: δ_t=r+γV(s')−V(s); Â=Σ(γλ)^l δ; V̂targ=V+Â. ✓ code compute_gae.
- aux L^joint = ½Ê[(V_θπ−V̂targ)²] + β_clone·KL(π_old,π_θ); L^aux is the aux head's value MSE. ✓
- aux phase also optimizes L^value (true value head) on B. ✓ (vf_true in compute_aux_loss)
- N_π large (infrequent aux) good; E_π=1; E_aux≈6; β_clone=1. ✓ all in train.py defaults.
