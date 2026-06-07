# DPO synthesis notes (Phase 1.5)

## Pain point (the problem at the time, ~2022-2023)
- RLHF pipeline (Ziegler 2019; Stiennon 2020; Ouyang/InstructGPT 2022; Bai 2022) is the dominant way to align LMs to human preferences. It is 3-stage: (1) SFT; (2) fit a reward model r_phi via Bradley-Terry MLE on preference pairs; (3) RL (PPO) to maximize learned reward minus a KL penalty to the SFT/reference policy.
- Stage 3 is the headache: PPO is complex, unstable, has many moving parts (separate reward model, value function/critic, on-policy sampling from the LM in the training loop, reward normalization/baselines, advantage estimation, clipping, KL coefficient tuning). Sampling from a multi-billion-param LM each step is expensive. High variance, sensitive to hyperparameters.
- Goal: get the SAME thing RLHF optimizes (KL-constrained reward maximization) but without training a separate reward model and without RL.

## Load-bearing ancestors (and their gap)
1. **RLHF / InstructGPT pipeline** (Ziegler 2019, Stiennon 2020, Ouyang 2022, Bai 2022). Defines the objective:
   max_pi E_{x~D, y~pi}[r_phi(x,y)] - beta * KL[pi(y|x) || pi_ref(y|x)].
   Gap: needs explicit reward model + PPO; unstable, expensive.
2. **Bradley-Terry model** (Bradley & Terry 1952). Pairwise preference: p(y1>y2|x) = exp(r(x,y1)) / (exp(r(x,y1))+exp(r(x,y2))) = sigma(r(x,y1)-r(x,y2)). Depends only on the reward DIFFERENCE. Plackett-Luce (Plackett 1975, Luce 1959) generalizes to K-way rankings. Reward MLE loss: -E[log sigma(r(x,yw)-r(x,yl))].
3. **Closed-form optimum of KL-regularized reward max** (Peters & Schaal 2007 reward-weighted regression; Peng 2019 AWR / advantage-weighted regression; Korbak 2022, Go 2023 distributional control; also the control-as-inference / max-ent RL line, Levine 2018). Known result: the optimal policy is pi*(y|x) = (1/Z(x)) pi_ref(y|x) exp(r(x,y)/beta). It's an exponential tilting of the reference by the reward. Gap: Z(x) = sum_y pi_ref(y|x) exp(r(x,y)/beta) is intractable for language (combinatorial over sequences), so this closed form was treated as a target you approximate (via weighted regression or importance sampling), not as something exactly usable.
4. **REINFORCE / PPO** (Williams 1992; Schulman 2017) — the RL machinery used in stage 3. Gap: variance, instability.
5. **Control-as-inference** (Levine 2018) — used in the theory section to diagnose why actor-critic is unstable: the soft value function (= beta log Z = beta log of the normalizer) is the natural baseline; PPO must learn/estimate it, and a bad estimate gives high-variance gradients. DPO's reparam makes that term cancel.

## The key insight chain (the heart)
1. Start from the SAME RLHF objective (Eq. RL), general reward r.
2. The closed-form optimum is pi_r(y|x) = (1/Z(x)) pi_ref(y|x) exp(r/beta). Derivation: rewrite the objective as min_pi E[ KL(pi || pi*) - log Z(x) ], where pi* is that exp-tilted distribution; KL >= 0 with equality iff pi=pi*. So optimum is pi*. (Appendix A.1 — Gibbs' inequality.)
3. The forward direction is useless because Z(x) is intractable. **Invert it.** Solve for r:
   r(x,y) = beta log( pi_r(y|x)/pi_ref(y|x) ) + beta log Z(x).
   Every reward function can be written this way in terms of ITS optimal policy plus an x-only term beta log Z(x).
4. **Substitute into Bradley-Terry.** BT depends only on r(x,y1)-r(x,y2). The beta log Z(x) term is the SAME for both completions (function of x only) -> it cancels:
   p*(y1>y2|x) = sigma( beta log(pi*(y1)/pi_ref(y1)) - beta log(pi*(y2)/pi_ref(y2)) ).
   The intractable Z is gone. The ground-truth-reward-optimal policy pi* satisfies a BT model whose "reward" is the implicit reward r_hat(x,y) = beta log(pi(y|x)/pi_ref(y|x)).
5. **Turn the MLE around.** Instead of MLE on r_phi then RL, do MLE directly on pi_theta using the BT likelihood expressed via the policy. The reward-model NLL loss becomes the DPO loss:
   L_DPO = -E_{(x,yw,yl)~D}[ log sigma( beta log(pi_theta(yw|x)/pi_ref(yw|x)) - beta log(pi_theta(yl|x)/pi_ref(yl|x)) ) ].
   One stage, supervised classification loss, no reward model, no sampling, no RL.

## Derivations to live out inline in reasoning.md
- **A.1 optimal policy**: the KL-completion-of-the-square trick (add/subtract log Z to turn exp-tilt into a normalized distribution), Gibbs inequality. ALL steps.
- **invert** to get r in terms of pi (log both sides, multiply by beta).
- **A.2 substitute into BT**: show the Z cancels explicitly via the fraction; land on sigma form. Note sigma argument is (yw - yl), i.e. preferred minus dispreferred.
- **gradient (A.4)**: u = beta(log ratio_l - log ratio_w) [note repo writes it with l-w inside], chain rule sigma'/sigma = (1-sigma) = sigma(-u); land on
  grad L = -beta E[ sigma(r_hat(x,yl) - r_hat(x,yw)) * (grad log pi(yw) - grad log pi(yl)) ].
  Interpretation: pushes up yw, down yl, weighted by sigma(r_hat_l - r_hat_w) = how WRONG the implicit reward currently orders the pair (weight ~1 when model is wrong, ~0 when already correct). This dynamic weight is what stops degeneration that a naive (unweighted) log-ratio objective (unlikelihood) causes.
- **Plackett-Luce (A.3)**: same cancellation over K-way ranking product.
- **Lemma 1**: reward shift by f(x) leaves BT/PL preference distribution invariant (exp(f) cancels in num/denom).
- **Lemma 2**: reward shift by f(x) leaves the optimal policy invariant (exp(f/beta) cancels in num/Z).
- **Theorem 1**: every reward equivalence class has a representative of the form r=beta log(pi/pi_ref); the projection f(r;pi_ref,beta)(x,y)=r - beta log Z(x) is that representative. So the reparam loses NO generality. Proposition: that representative is UNIQUE in each class (proof by contradiction: pi exp(f/beta)=pi', sum over y => exp(f/beta)=1 => f=0).
- **A.5 / instability**: minimizing KL(pi_theta || pi*) with pi* induced by r_phi gives objective max E_pi[ f(r_phi,pi_ref,beta) - beta log(pi/pi_ref) ], where f = r_phi - beta log Z (Z = soft value fn of pi_ref). Without subtracting that soft value baseline, the policy gradient has high variance -> AC instability. DPO's implicit reward needs no learned baseline because the normalizer cancels structurally.

## Design decisions -> why
- **beta log(pi/pi_ref) as implicit reward**: it's exactly the inverted optimal-policy relation; the log-ratio (not raw log-prob) is what makes Z cancel and keeps it tied to KL strength beta.
- **sigma / logsigmoid (BT) loss**: this is just the reward-model MLE loss with r replaced by the implicit reward; binary cross-entropy on which completion is preferred.
- **subtract ref_logratios (vs reference_free)**: keeps the implicit reward = beta*KL-style log ratio; anchors to pi_ref so the optimum stays the KL-constrained one. Reference-free drops the anchor (an ablation/variant).
- **beta value**: 0.1 default (0.5 for TL;DR). Larger beta = stronger KL constraint = smaller gap allowed.
- **pi_ref = pi_SFT**; if no SFT model, set pi_ref = argmax E[log pi(yw|x)] (Preferred-FT on chosen) to reduce distribution shift between the unknown true sampling policy and pi_ref.
- **sum (not average) token logprobs** over the completion mask (default average_log_prob=False): the sequence logprob is the sum; this is the quantity that the BT/derivation uses.
- **detach() on rewards**: rewards are a logging/monitoring metric, not part of the gradient.
- **RMSprop, lr 1e-6, 150-step warmup, batch 64**: standard finetuning stability defaults.
- **naive unweighted objective degenerates** (unlikelihood): App table shows "when when when..." collapse; the sigma weight is essential.

## Canonical code grounding
- Original repo `preference_loss` + `_get_batch_logps` (sum logsoftmax-gather over loss mask), concatenated chosen+rejected forward pass for policy and (no-grad) reference.
- Paper App B minimal `dpo_loss`.
- TRL DPOTrainer mirrors this: concatenated_forward -> logps -> preference_loss with beta, logsigmoid.
- Saved at code/canonical_dpo.py.

## Sign care (verify in reasoning)
- Loss sigma argument: preferred MINUS dispreferred: beta(logratio_w - logratio_l). logsigmoid of that; minimize negative.
- Gradient weight: sigma(r_hat_l - r_hat_w) (dispreferred minus preferred) = high when wrongly ordered.
- Optimal policy: exp(+r/beta), tilt UP toward high reward.
