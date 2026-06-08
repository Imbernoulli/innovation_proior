# Q-learning

## Problem

Learn to act optimally in a finite controlled Markov process — maximise expected discounted reward, discount 0 < γ < 1 — from a stream of (state, action, reward) transitions, **without a model** of the rewards ρ(x,a) or transitions P_xy[a], and **without** following any particular policy while learning.

## Key idea

Store **action-values** Q(x,a) (the return from doing a at x then acting optimally), not state-values. The optimal policy is then implicit: act greedily, f*(x) = argmax_a Q*(x,a). Learn Q directly from each experienced transition by bootstrapping toward the one-step **optimal** backup r + γ max_{a'} Q(x',a'). Because the target maxes over the next state's actions rather than using the action actually taken, it samples the Bellman optimality backup whose fixed point is Q*, regardless of the behaviour policy generating the data — Q-learning is **off-policy**. With a decreasing learning rate satisfying the Robbins–Monro conditions and every state–action pair tried infinitely often, the iterates converge to Q* with probability 1.

## The algorithm

On the transition (x_t, a_t, r_t, x_{t+1}), with learning factor α_t and discount γ:

    Q_{t+1}(x_t, a_t) = (1 − α_t) Q_t(x_t, a_t) + α_t ( r_t + γ max_a Q_t(x_{t+1}, a) )        (1)

and Q is unchanged at every other (x,a). Equivalently, in temporal-difference form,

    Q(x_t,a_t) ← Q(x_t,a_t) + α_t [ r_t + γ max_a Q(x_{t+1},a) − Q(x_t,a_t) ].

Define V_t(x) = max_a Q_t(x,a); the implicit greedy policy is f^Q_t(x) = argmax_a Q_t(x,a). The behaviour used to choose a_t is unrestricted (e.g. ε-greedy) provided it visits every (x,a) infinitely often.

## Convergence theorem (proved)

**Theorem.** Let rewards be bounded, |r_n| ≤ R, and learning rates 0 ≤ α_n < 1. For each (x,a) let n^i(x,a) be the index of the i-th time a is tried in x, and suppose
  Σ_{i=1}^∞ α_{n^i(x,a)} = ∞   and   Σ_{i=1}^∞ [α_{n^i(x,a)}]² < ∞   for all x,a.
Then Q_n(x,a) → Q*(x,a) as n → ∞, for all x,a, with probability 1.

**Proof.** Construct the **action-replay process (ARP)**, an artificial controlled Markov process built from the observed episodes and learning rates. Each episode (x_t,a_t,y_t,r_t,α_t) is a "card"; the deck is infinite with card 0 holding the initial values Q_0(x,a). An ARP state is (x,n): a real state x and a level n. From (x,n) under a: discard cards above n; deal down for the first (x,a)-match at episode t; with probability α_t **replay** it — emit r_t, move to (y_t, t−1); with probability 1−α_t discard and keep dealing; reaching card 0 absorbs with reward Q_0(x,a). Every action descends a level, so the ARP terminates; discount it by γ.

*Lemma A — the iterates are the ARP's optimal action-values:* Q_n(x,a) = Q*_ARP((x,n),a). By induction on n. Base: at level 0 the ARP can only absorb and pay Q_0(x,a), so Q_0(x,a) = Q*_ARP((x,0),a). Step: assume Q_{n−1}(x,a) = Q*_ARP((x,n−1),a), so V*((x,n−1)) = max_a Q_{n−1}(x,a) = V_{n−1}(x). For (x,a) ≠ (x_n,a_n) the ARP at level n equals level n−1, giving Q_n(x,a) = Q_{n−1}(x,a). For (x,a) = (x_n,a_n), dealing from level n gives the mixture
  Q*_ARP((x_n,n),a_n) = (1−α_n) Q*_ARP((x_n,n−1),a_n) + α_n ( r_n + γ V*((y_n,n−1)) )
                      = (1−α_n) Q_{n−1}(x_n,a_n) + α_n ( r_n + γ V_{n−1}(y_n) ) = Q_n(x_n,a_n),
which is exactly update (1). ∎

*Lemma B — the ARP converges to the real process.*
- **B.1 (truncation):** ignoring the (s+1)-th state on costs |δ| < γ^s · R/(1−γ) → 0, since |V| < R/(1−γ); so finite s-step comparisons suffice. (Non-discounted absorbing case: (1−p*)^j → 0 plays γ^s's role, with V* ≤ u*R/p*.)
- **B.2 (no fall-back):** from a high enough level, the ARP stays above any fixed l over s actions with probability ≥ 1−ε, because the all-tails product Π(1−α_{n^i}) < exp(−Σα_{n^i}) → 0 as the start level grows (uses Σα = ∞).
- **B.3 (model convergence):** by the stochastic-approximation theorem (Kushner & Clark thm 2.3.1; Robbins–Monro): X_{n+1} = X_n + β_n(ξ_n − X_n) with Σβ=∞, Σβ²<∞, ξ bounded mean Ξ ⇒ X_n → Ξ w.p.1. The ARP's reward R_{(x,n)}(a) and transition P^{(n)}_{xy}[a] are exactly such α-weighted running averages of the unbiased real samples, so R_{(x,n)}(a) → ρ_x(a) and P^{(n)}_{xy}[a] → P_xy[a] w.p.1, uniformly (finite S, A).
- **B.4 (close models, close values):** if rewards are within η and transitions within η/R, the s-step action-values differ by at most ηs(s+1)/2 (two-step difference < 3η; per-step errors accumulate as 1+2+···+s).

*Assembly:* given ε, choose s (B.1) so γ^s R/(1−γ) < ε/6, giving < ε/6 truncation cost on each side. Choose l (B.3) so for n>l the one-step ARP model has |P^{(n)}_{xy}[a]−P_{xy}[a]| < ε/[3s(s+1)R] and |R^{(n)}_x(a)−R_x(a)| < ε/[3s(s+1)]. Choose h (B.2) so the probability of straying below l in s actions is < min{ε(1−γ)/6sR, ε/[3s(s+1)R]}. Conditioning on staying above l gives |P'^{(n)}_{xy}[a]−P_{xy}[a]| < 2ε/[3s(s+1)R] and |R'^{(n)}_x(a)−R_x(a)| < 2ε/[3s(s+1)]. Then B.4 with η = 2ε/[3s(s+1)] contributes ε/3; straying contributes at most (2sR/(1−γ))·prob < ε/3; the two truncation tails add ε/6 + ε/6. Applied to the optimal sequence of either process, |Q*_ARP((x,n),a) − Q*(x,a)| < ε. By Lemma A this is |Q_n(x,a) − Q*(x,a)| < ε. Hence Q_n → Q* w.p.1. ∎

**Why off-policy works.** The target r + γ max_a Q(x',a) is the Bellman *optimality* backup: the max performs the policy-improvement step (V*(x) = max_a Q*(x,a)) inside every update, so the target does not depend on the action a_t actually taken or on the behaviour policy. The behaviour need only be exploratory enough to visit every (x,a) infinitely often. Replacing max_a Q(x',a) by Q(x', a') for the actually taken next action a' would instead learn the value of the behaviour being followed.

## Tabular code

```python
import numpy as np
from collections import defaultdict

def q_learning(env, n_actions, gamma, alpha_schedule, epsilon, n_steps):
    # Tabular Q(x,a); convergence is proved for the look-up-table representation.
    Q = defaultdict(lambda: np.zeros(n_actions))

    def behaviour(state):                      # off-policy: any exploratory rule
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)
        return int(np.argmax(Q[state]))        # greedy w.r.t. current Q

    state = env.reset()
    for t in range(n_steps):
        action = behaviour(state)
        next_state, reward, done = env.step(action)

        # one-step optimal backup: max over next-state actions (off-policy target)
        target = reward + (0.0 if done else gamma * np.max(Q[next_state]))

        alpha = alpha_schedule(state, action)  # Sum(a)=inf, Sum(a^2)<inf per (x,a)
        Q[state][action] += alpha * (target - Q[state][action])   # update (1)

        state = env.reset() if done else next_state
    return Q
```

## Worked sanity check

A two-state deterministic chain x0 →(a)→ x1 →(b)→ x1 with reward 0 then 1 each step, γ=0.9. The fixed point of (1) sets Q(x1,b) = 1 + 0.9·max_a Q(x1,a) = 1 + 0.9·Q(x1,b), so Q*(x1,b) = 10, and Q*(x0,a) = 0 + 0.9·max_a Q(x1,a) = 9. With α decreasing and a_t chosen by any exploratory rule, repeated application of (1) drives Q toward (Q(x0,a)=9, Q(x1,b)=10) — the discounted-return optimum — independently of how often non-greedy actions were tried, illustrating the off-policy max.
