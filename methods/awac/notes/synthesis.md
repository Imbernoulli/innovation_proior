# AWAC synthesis (Accelerating Online RL with Offline Datasets, 2006.09359, title verified)

## Pain point / research question
Want: pretrain from a fixed OFFLINE dataset D (arbitrary quality — demos, suboptimal, random) AND
then CONTINUE to improve quickly with a small amount of ONLINE interaction. Real-world robotics:
online-from-scratch is too sample-expensive; pure offline RL is too conservative to finetune. No
prior method does both: must not assume data is optimal, must reuse offline data during online
finetuning (data-efficient), must not destabilize from distribution shift, must not be so
conservative it can't improve online.

## Diagnostic findings (pre-method, on HalfCheetah-v2, 15 expert demos + 100 BC-suboptimal trajs)
1. DATA EFFICIENCY: on-policy finetuning (DAPG) and Monte-Carlo/TD(λ) return methods (AWR, MARWIL)
   are ~an order of magnitude SLOWER than off-policy actor-critic (which bootstraps Q via Bellman
   backups and reuses data). → need off-policy critic.
2. BOOTSTRAP ERROR (offline AC): naively applying SAC offline FAILS — "SAC-scratch" (no prior data)
   ≈ "SACfD-prior" (with prior data in buffer): the off-policy algo can't actually USE the offline
   data. Even BC-pretraining the SAC policy ("SACfD-pretrain") shows an initial DROP then learns
   like scratch. Cause: target Q(s',a'), a'~π, queries OOD actions where Q is wrong → error
   accumulates on a static dataset. → need a policy constraint.
3. EXCESSIVE CONSERVATISM (offline RL w/ explicit behavior model): BEAR/BCQ/BRAC do well OFFLINE but
   barely improve ONLINE (BEAR curve nearly flat during finetune). Cause: they fit an explicit
   parametric behavior model π̂_β by MLE and constrain to it. Offline, π̂_β is fit once. ONLINE, π̂_β
   must track a streaming, multi-modal mixture of offline+online data — density estimation in the
   streaming setting is hard; the behavior model's log-likelihood on data DROPS during finetuning →
   the constraint becomes inaccurate/too conservative → finetuning stalls. → need a constraint that
   needs NO explicit behavior model.

So requirements: off-policy critic (efficiency) + policy constraint (offline stability) + NO behavior
model (online finetuning). AWAC satisfies all three with an IMPLICIT constraint.

## Load-bearing ancestors
- **Actor-critic / policy iteration** (Konda & Tsitsiklis 2000): alternate policy evaluation (fit Q^π
  by Bellman backups) and improvement (max E_π[Q]). Off-policy with replay buffer.
- **SAC** (Haarnoja 2018) + **TD3 twin critics** (Fujimoto 2018): the base. Twin Q, min for target,
  Polyak target nets. AWAC built on top of "twin SAC". (AWAC's clean form drops the entropy term in
  the actor objective but keeps twin-Q TD critic.)
- **Standard improvement step**: θ = argmax_θ E_{s~D}[E_{a~π_θ}[Q_φ(s,a)]] — samples actions FROM the
  policy → OOD when offline → bootstrap error.
- **Offline-RL explicit-constraint methods**: BCQ (Fujimoto 2019), BEAR (Kumar 2019), BRAC (Wu 2019),
  ABM (Siegel 2020). Add constraint D(π_θ, π_β) ≤ ε via a fit π̂_β (MLE) used as penalty or sampler.
  Gap: behavior model must be re-fit online → hard, too conservative.
- **AWR** (Peng 2019), **RWR** (Peters 2007), **REPS** (Peters 2010), **MARWIL** (Wang 2018), **MPO**
  (Abdolmaleki 2018): EM/KL-constrained-improvement → advantage-weighted maximum likelihood.
  KL-constrained problem max_π E_{a~π}[A] s.t. KL(π‖π_β)≤ε → closed form
  π*(a|s) ∝ π_β(a|s) exp(A(s,a)/λ). AWR estimates V^{π_β} by MONTE-CARLO / TD(λ) (value of the
  BEHAVIOR policy) → sample-inefficient and asymptotically capped. ABM fits an explicit behavior
  model. AWAC's key differences: estimate Q^π (the CURRENT policy) by off-policy bootstrapping (not
  V^{π_β} by MC), and enforce the constraint IMPLICITLY (no behavior model).
- **CRR** (Wang 2020, concurrent): critic-regularized regression, equivalent policy update, but
  offline-only (doesn't study finetuning).

## The method (AWAC)
Two steps, alternating (actor-critic):
- **Policy evaluation** (critic): off-policy TD, twin Q, minimize Bellman error
  L(φ) = E_D[(Q_φ(s,a) − y)²], y = r + γ E_{s',a'~π}[Q_{φ̄}(s',a')] (target net, min of twin). This
  estimates Q^π of the CURRENT policy by bootstrapping → efficient, reuses off-policy data.
- **Policy improvement** (actor): constrained
  π_{k+1} = argmax_π E_{a~π}[A^{π_k}(s,a)] s.t. KL(π‖π_β) ≤ ε.
  Note max_a Q = max_a A (V is constant in a), so use advantage A = Q^{π_k}(s,a) − V^{π_k}(s),
  V(s)=E_{a~π}[Q(s,a)].

### Derivation (full appendix, must re-derive in reasoning)
Constrained problem with normalization ∫π=1. Lagrangian:
  L = E_{a~π}[A(s,a)] + λ(ε − KL(π‖π_β)) + α(1 − ∫π da).
∂L/∂π = A(s,a) − λ(log π(a|s) − log π_β(a|s) + 1) − α  (since KL = E_π[log π − log π_β], its
derivative wrt π(a|s) is log π − log π_β + 1). The appendix's printed derivative flips the log signs,
but the displayed closed form and direct differentiation agree with this corrected derivative.
Set = 0, solve for log π: log π = (1/λ)A + log π_β − 1 − α/λ. Exponentiate:
  π*(a|s) = (1/Z(s)) π_β(a|s) exp(A(s,a)/λ),  Z(s) the partition function.
Project onto parametric π_θ by minimizing FORWARD KL under data state dist ρ_{π_β}(s):
  argmin_θ E_{ρ}[ KL(π* ‖ π_θ) ] = argmin_θ E_{ρ} E_{π*}[ −log π_θ ].
FORWARD KL (not reverse) is the crucial choice: KL(π*‖π_θ) = E_{a~π*}[log π* − log π_θ], the only
θ-dependent term is −E_{a~π*}[log π_θ]; we can sample a~π* by IMPORTANCE-reweighting samples from the
BUFFER β: E_{a~π*}[−log π_θ] = E_{a~β}[ (π*/π_β)(−log π_θ) ] = E_{a~β}[ (1/Z) exp(A/λ) (−log π_θ) ].
The π_β factor CANCELS — no behavior model needed. So:
  θ_{k+1} = argmax_θ E_{(s,a)~β}[ log π_θ(a|s) · exp( A^{π_k}(s,a)/λ ) ]   (drop Z(s), see below).
Reverse KL KL(π_θ‖π*) would instead require evaluating log π_β (a density model) and sampling actions
from π_θ (possibly OOD for Q) → the very things that break finetuning. Forward KL avoids both.
Pinsker bound (appendix): for discrete π_θ ≥ α_θ, KL(π*‖π_θ) ≤ (2/α_θ)D_TV² ≤ (1/α_θ)KL(π_θ‖π*) — so
minimizing reverse KL also bounds forward KL (justifies the equivalence loosely).

### Z(s) omission (appendix, must mention)
Z(s) = ∫ π_β(a|s) exp(A/λ) da = E_{a~π_β}[exp(A/λ)] is a per-STATE normalizer. Dropped because:
- empirically estimating it (K=10 per-batch samples) made performance WORSE (Table: use Z(s) gives
  pen 84%/door 0%/relocate 0% vs omit Z(s) pen 98%/door 95%/relocate 54%).
- it only reweights STATES not actions; the buffer state dist already differs from π_θ's, so
  preserving it is low-value; bad estimates add variance like degenerate importance weights.
- can be bounded C2 ≤ Z(s) ≤ C1 via Cauchy-Schwarz (upper) and Polya-Szego reverse-CS (lower) with
  f=π_β, g=exp(A/λ): Z ≤ √(∫f² ∫g²)=C1; Z ≥ 2(√(M_fM_g/m_fm_g + m_fm_g/M_fM_g))^{-1} C1 = C2. (bounds
  loose.) In practice normalize over the BATCH instead.

### Practical actor weight
weight = exp( A(s,a)/λ ), A = Q(s,a_data) − V(s), V(s) = Q(s, a~π(·|s)) (one sample; or min twin).
λ (called beta in rlkit, the exponent denominator / KL multiplier) is a fixed hyperparameter:
paper λ=0.3 manipulation, λ=1.0 MuJoCo benchmark. Lower λ → sharper/greedier; higher λ → closer
to BC. rlkit normalizes weights over the batch (softmax(score/beta)); paper Eqn 9 is the per-state
exp form.

## Design-decision → why
- Off-policy Q^π via bootstrapping (not MC V^{π_β} like AWR): bootstrapping reuses off-policy data and
  estimates the CURRENT policy's value → far more sample-efficient and even higher asymptotic perf;
  MC/TD(λ) of the behavior policy is slow and caps improvement at one-step-from-π_β.
- IMPLICIT constraint via forward-KL projection (not explicit π̂_β model): explicit behavior models
  are hard to fit online (streaming, multimodal) → over-conservative finetuning. Forward KL lets the
  π_β factor cancel, so the actor update is weighted max-likelihood on buffer samples with NO model.
- forward KL not reverse KL: reverse KL needs log π_β density + sampling π_θ actions (OOD for Q);
  forward KL needs neither → the constraint is enforced while staying entirely in-sample.
- advantage weighting not argmax / not ∇_a Q ascent: argmax/∇_aQ query Q at policy actions → OOD when
  offline; reweighting buffer actions by exp(A/λ) is in-sample and is the analytic constrained optimum.
- drop Z(s): empirically better, only reweights states, normalize over batch instead.
- twin Q + min + target net + Polyak: standard overestimation control from TD3/SAC.
- clamp Q ≤ 0 for the dexterous tasks (rewards are non-positive) — small stabilizer.

## Canonical hyperparameters (rlkit AWAC, CONFIRMED in awac_trainer.py + Table 2)
- built on twin SAC; qf/policy lr 3e-4, Adam; discount 0.99; target τ 5e-3 (Polyak).
- policy 4×256 ReLU, Q 4×256 ReLU; policy weight decay 1e-4, Q weight decay 0.
- batch 1024; replay 1e6; 25000 pretraining (offline) steps; 1 train batch per env step; reward scale 1.
- paper λ (lagrange) = 0.3 manipulation / 1.0 MuJoCo; rlkit examples call this beta and use/sweep
  nearby values (hand beta=0.5 with clip_score=0.5; MuJoCo beta=2 in example script). exploration
  noise = none (stochastic policy).
- critic target uses min of twin target Q at a'~π; rlkit code can subtract α·logπ, but AWAC example
  configs disable automatic entropy tuning and set alpha=0, matching the clean paper objective.
- weights normalized over batch (default normalize_over_batch=True → softmax(score/β)); score = q_adv −
  v_pi; v_pi = min(qf1,qf2)(s, a~π). clip_score optional.
  (FLAG: rlkit default uses softmax-over-batch weighting and a SAC-entropy critic term; the paper's
  clean Eqn 9 uses exp(A/λ) per-state and no entropy. I present the clean paper form, noting the twin
  critic + min, which is the load-bearing structure both share.)

## Scaffold ↔ code correspondence
Final code fills: twin Critic, tanh-bounded-mean Gaussian policy, critic TD update (twin, min target), advantage
computation A=Q(s,a)−V(s) with V≈Q(s,π-sample), advantage-weighted actor loss
−E_β[exp(A/λ)·logπ(a|s)] batch-normalized, Polyak, offline-pretrain-then-online loop. Scaffold =
generic off-policy actor-critic harness with twin Q, policy, TD critic stub, "policy improvement"
stub, replay buffer shared by offline+online.

## Self-derivation checks
- ∂/∂π of KL(π‖π_β)=∫π(log π − log π_β): derivative wrt π(a) = log π − log π_β + 1. So ∂L/∂π =
  A − λ(log π − log π_β + 1)·(−1)... careful: L has +λ(ε − KL), so ∂(λ(ε−KL))/∂π = −λ(log π − log π_β
  + 1). Plus ∂(α(1−∫π))/∂π = −α. Plus ∂E_π[A]/∂π = A. Total: A − λ(log π − log π_β + 1) − α = 0.
  → λ log π = A − λ log π_β·(−1)... solve: A − λ log π + λ log π_β − λ − α = 0 → λ log π = A + λ log π_β
  − λ − α → log π = (1/λ)A + log π_β − 1 − α/λ → π = π_β exp(A/λ) exp(−1−α/λ) = (1/Z) π_β exp(A/λ). ✓
  (paper appendix prints a sign-flipped derivative line, but the closed form uses the corrected result.)
- forward KL projection E_{π*}[−log π_θ], importance from β: E_{π*}[f] = E_β[(π*/π_β)f] =
  E_β[(1/Z)exp(A/λ) f]; π_β cancels. ✓
- A = Q^{π_k}(s,a) − V^{π_k}(s); maximizing E_π[Q] ⟺ E_π[A] since V indep of a. ✓
