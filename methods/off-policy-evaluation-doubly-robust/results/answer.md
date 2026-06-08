# Doubly Robust Off-policy Value Evaluation

## Problem

Estimate the value of a **target policy** from data logged by a different **behavior policy**, observing only the reward of the action actually taken. Two classical estimators fail in complementary ways: a fitted reward/value **model** (direct method / regression) has low variance but an un-auditable bias from its approximation error; **importance sampling** (inverse propensity weighting) is unbiased given the behavior policy but has variance that explodes as the policies diverge and, over a horizon H, grows exponentially because the cumulative importance ratio is a product of H per-step ratios.

## Key idea

Combine them: use the model as a **baseline** and importance-weight only the model's **residual** (the part it got wrong), not the raw reward. The model serves as a **control variate** whose expectation under the target policy is known. This yields a single estimator that is

- **doubly robust** — unbiased if *either* the model *or* the importance weights are correct, because its bias is the *product* of the two model errors; and
- **low variance** — the importance-weighting variance penalty scales with the squared *model error* Δ² instead of the squared *reward magnitude*, so a decent model sharply reduces variance.

## Contextual-bandit estimator

For the deterministic target-policy theorem, with target policy π, behavior estimate p̂, and reward model ρ̂_a(x):

V̂_DR = (1/|S|) Σ_{(x,a,r_a)} [ (r_a − ρ̂_a(x))·I(π(x)=a)/p̂(a|x) + ρ̂_{π(x)}(x) ].

For stochastic target policies, the same per-sample contribution is q̂(x,π) + iw·(r − q̂(x,a)), with iw = π(a|x)/p̂(a|x) and q̂(x,π) = Σ_a π(a|x)q̂(x,a). Importance sampling is the special case q̂ ≡ 0; the direct method drops the correction term.

**Bias theorem.** With Δ(a,x) = ρ̂_a(x) − ρ_a(x) (model error) and δ(a,x) = 1 − p(a|x)/p̂(a|x) (propensity error), and ρ̂, p̂ fixed independently of S:

|E[V̂_DR] − V^π| = |E_x[ Δ δ ]|.

The bias is the product Δδ, so it vanishes if Δ = 0 (model correct) **or** δ = 0 (weights correct). Compare: direct method |E_x[Δ]|, IPS |E_x[ρ_{π(x)} δ]|. For stochastic π, replace Δδ by Σ_a π(a|x)Δ(a,x)δ(a,x).

**Variance theorem** (stationary policy; ε = (r_a−ρ_a)I/p̂ is reward noise):

Var[V̂_DR] = (1/|S|)( E[ε²] + Var_x[ρ_{π(x)} + Δδ] + E_x[ ((1−p)/p)·Δ²(1−δ)² ] ).

Three terms: reward noise; target-value variance; and the importance-weighting penalty, which carries **Δ²**. The IPS variance is identical but with **ρ_{π(x)}²** in place of Δ² in the penalty — so whenever |Δ| < |ρ_{π(x)}|, DR's penalty is strictly smaller, dramatically so when p ≪ 1. (Direct-method variance (1/|S|)Var_x[ρ_{π(x)}+Δ] is lowest but pays it back in bias.)

## Sequential (RL) estimator

Writing step-wise IS recursively (V_step-IS^{H+1-t} = ρ_t(r_t + γ V_step-IS^{H-t})) exposes each step as a one-step bandit whose stochastic return r_t + γ·(future) has mean Q(s_t,a_t). Applying the bandit DR per step, with a fitted Q̂ and V̂(s) = Σ_a π_1(a|s)Q̂(s,a), define V_DR^0 = 0 and

V_DR^{H+1-t} = V̂(s_t) + ρ_t ( r_t + γ V_DR^{H-t} − Q̂(s_t, a_t) ),  ρ_t = π_1(a_t|s_t)/π_0(a_t|s_t),

with the trajectory estimate V_DR = V_DR^H. It is unbiased because E_{a~π_0}[ρ_t Q̂(s_t,a)] = V̂(s_t), so the control variate cancels in the mean; Q̂ ≡ 0 recovers step-wise IS.

**Variance theorem** (per trajectory; Δ(s,a) = Q̂(s,a) − Q(s,a)):

V_t[V_DR^{H+1-t}] = V_t[V(s_t)] + E_t[V_t[ρ_t Δ(s_t,a_t)|s_t]] + E_t[ρ_t² V_{t+1}[r_t]] + E_t[γ²ρ_t² V_{t+1}[V_DR^{H-t}]].

Four sources of randomness — transition, action, reward, future. **Only the action term depends on the model**, through Δ; a good Q̂ shrinks it, and step-wise IS (Q̂≡0, so Δ = −Q) has Var_t[ρ_t Q(s_t,a_t)|s_t] in that slot instead.

**DR-v2 (transition control variate).** Replacing the Q̂ subtraction by a model reward plus a transition control variate,
−R̂(s_t,a_t) − γV̂(s_{t+1})·P̂(s_{t+1}|s_t,a_t)/P(s_{t+1}|s_t,a_t), reduces transition variance when the reward and transition model are good. Since the true P in the denominator is unavailable, the practical approximation P̂/P ≡ 1 makes the last term γV̂(s_{t+1}) and introduces bias bounded by ε V_max Σ_{t=1}^H γ^t, where ε = max_{s,a}‖P̂(·|s,a) − P(·|s,a)‖_1.

**Optimality.** For discrete tree MDPs, the variance of any unbiased off-policy estimator is lower bounded (Cramér–Rao) by Σ_{t=1}^{H+1} E[ρ_{1:(t-1)}² Var_t[V(s_t)]]. DR with a perfect model (Q̂ = Q, Δ ≡ 0) attains this bound exactly. So the transition-stochasticity variance DR cannot remove is intrinsic to the problem; a better Q̂ moves DR toward that floor by shrinking the action-stochasticity term. (For DAG MDPs ρ_{1:t-1} is replaced by the occupancy ratio P_1(s_{t-1},a_{t-1})/P_0(s_{t-1},a_{t-1}).)

**Independence requirement.** Unbiasedness needs ρ̂/Q̂/p̂ fit on data held out from the samples the estimator averages. The k-fold trick (fit on the rest, apply on each fold, average) recovers data efficiency while keeping each fold unbiased.

## Code

```python
import numpy as np

# ---------- contextual bandit ----------
def direct_method(q_hat, contexts, pi_target):
    return np.mean([np.dot(pi_target(x), q_hat(x)) for x in contexts])

def importance_sampling(rewards, actions, pscore, contexts, pi_target):
    return np.mean([r * pi_target(x)[a] / p
                    for r, a, p, x in zip(rewards, actions, pscore, contexts)])

def doubly_robust(rewards, actions, pscore, contexts, pi_target, q_hat, lam=np.inf):
    """V_DR = E_{pi_target}[q_hat] + iw * (r - q_hat(x, a)).

    With lam=np.inf, bias = E[Delta * delta] (zero if model OR weights correct);
    the variance penalty scales with Delta^2, not reward^2. Finite lam clips weights
    and trades that exact unbiasedness for bounded contributions.
    q_hat must be fit on data independent of (rewards, actions, contexts).
    """
    est = []
    for r, a, p, x in zip(rewards, actions, pscore, contexts):
        pi_x = pi_target(x)
        q_hat_x = q_hat(x)
        iw = pi_x[a] / p
        if lam < np.inf:
            iw = min(iw, lam)
        baseline = np.dot(pi_x, q_hat_x)
        q_hat_factual = q_hat_x[a]
        est.append(baseline + iw * (r - q_hat_factual))
    return np.mean(est)

# ---------- sequential / RL ----------
def step_wise_is(trajectory, gamma, pi_target, pi_behavior):
    v = 0.0
    for s, a, r in reversed(trajectory):
        rho = pi_target(s)[a] / pi_behavior(s)[a]
        v = rho * (r + gamma * v)                      # = DR with q_hat == 0
    return v

def doubly_robust_sequential(trajectory, gamma, pi_target, pi_behavior, q_hat, v_hat):
    """Recursive DR: V_DR^{H+1-t} = v_hat(s_t) + rho_t (r_t + gamma V_DR^{H-t} - q_hat(s_t, a_t)).

    v_hat(s) = sum_a pi_target(a|s) q_hat(s, a). q_hat / v_hat fit on independent data.
    """
    v = 0.0                                            # V_DR^0 = 0
    for s, a, r in reversed(trajectory):
        rho = pi_target(s)[a] / pi_behavior(s)[a]
        v = v_hat(s) + rho * (r + gamma * v - q_hat(s, a))
    return v

def k_fold_dr(dataset, gamma, pi_target, pi_behavior, fit_models, k=2):
    """Rotate folds: fit on the rest, apply DR on each fold, average (stays unbiased)."""
    folds = np.array_split(np.arange(len(dataset)), k)
    estimates = []
    for j in range(k):
        rest = np.concatenate([folds[m] for m in range(k) if m != j])
        q_hat, v_hat = fit_models([dataset[i] for i in rest])
        for i in folds[j]:
            estimates.append(
                doubly_robust_sequential(dataset[i], gamma, pi_target,
                                         pi_behavior, q_hat, v_hat))
    return np.mean(estimates)
```
