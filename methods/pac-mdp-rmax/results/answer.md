# R-MAX and the PAC-MDP guarantee: optimism under uncertainty

## Problem

An agent acts online in an unknown finite MDP with N states, A actions, and slot payoffs in [0, R_max]. The stochastic-game version uses N states and k actions per player, so a learnable slot is a state plus a joint action and there are Nk² slots; in a single-agent MDP there are NA slots. The goal is a high-probability, finite-time guarantee: after a number of actions and amount of computation polynomial in the problem size, 1/ε, 1/δ, and the relevant time-scale T, the agent's return is near the best achievable by policies with ε-return mixing time T. For discounted return, T is the horizon after which the discounted tail is negligible.

## Key idea

Maintain a complete optimistic model. Every insufficiently sampled slot is assumed to yield reward R_max until its payoff is observed, and to transition with probability 1 to a fictitious absorbing high-reward state until its transition model is known. Known slots use their empirical transition estimates and learned payoffs. Planning once in this model automatically chooses between exploration and exploitation: if a valuable unknown slot is reachable, the optimistic value pulls the policy toward it; if the policy rarely reaches unknown slots, then it remains in the accurately modeled region and its real value is close to the optimistic value.

## Algorithm

```python
import numpy as np

def rmax_known_threshold(n_states, n_actions, R_max, eps, delta, T, *, joint_action=False):
    """R-MAX K1 threshold.

    joint_action=False gives the single-agent MDP slot count N*A.
    joint_action=True gives the stochastic-game slot count N*k^2.
    """
    slot_count = n_states * (n_actions**2 if joint_action else n_actions)
    log_arg = delta / (6.0 * slot_count)
    transition_term = (4.0 * n_states * T * R_max / eps) ** 3
    confidence_term = -6.0 * (np.log(log_arg) ** 3)
    return int(max(np.ceil(transition_term), np.ceil(confidence_term))) + 1

def value_iteration_optimistic(known, Phat, Rmodel, R_max, T, gamma, n_states, n_actions):
    """T-step planning in the optimistic induced model."""
    def absorber_tail(steps):
        if steps <= 0:
            return 0.0
        if np.isclose(gamma, 1.0):
            return R_max * steps
        return R_max * (1.0 - gamma**steps) / (1.0 - gamma)

    U = np.zeros(n_states)
    pi = [[0] * n_states for _ in range(T)]
    for t in range(T - 1, -1, -1):
        tail = absorber_tail(T - t - 1)
        Q = np.empty((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                if known[s][a]:
                    Q[s, a] = Rmodel[s, a] + gamma * Phat[s][a].dot(U)
                else:
                    Q[s, a] = Rmodel[s, a] + gamma * tail
            pi[t][s] = int(np.argmax(Q[s]))
        U = Q.max(axis=1)
    return pi

def r_max(env, n_states, n_actions, R_max, eps, delta, T, gamma):
    K1 = rmax_known_threshold(n_states, n_actions, R_max, eps, delta, T)

    counts = np.zeros((n_states, n_actions, n_states))
    n_sa = np.zeros((n_states, n_actions), dtype=int)
    reward_seen = np.zeros((n_states, n_actions), dtype=bool)
    known = [[False] * n_actions for _ in range(n_states)]
    Phat = [[np.zeros(n_states) for _ in range(n_actions)] for _ in range(n_states)]
    Rmodel = np.full((n_states, n_actions), R_max, dtype=float)

    s = env.current_state()
    while True:
        pi = value_iteration_optimistic(known, Phat, Rmodel, R_max, T, gamma,
                                        n_states, n_actions)
        for t in range(T):
            a = pi[t][s]
            r, s_next = env.step(a)
            if not reward_seen[s, a]:
                Rmodel[s, a] = r
                reward_seen[s, a] = True
            if n_sa[s, a] < K1:
                counts[s, a, s_next] += 1
                n_sa[s, a] += 1
                if n_sa[s, a] == K1:
                    known[s][a] = True
                    Phat[s][a] = counts[s, a] / K1
                    s = s_next
                    break
            s = s_next
```

## Guarantee

With probability at least 1−δ, R-MAX attains expected T-step average return at least Opt(Π(ε,T))−2ε within a number of steps polynomial in N, k or A, T, 1/ε, 1/δ, and R_max. This statement uses the slot-payoff setting above; stochastic reward means require adding a reward-estimation concentration term. Rescaling the internal tolerance gives any desired final ε slack.

The proof has four moving parts. The simulation lemma says that if every known transition probability is within η of the truth, then any fixed T-step policy has average-return error at most R_max·N·T·η; choosing η = ε/(2NTR_max) spends only half the value budget on model error. Hoeffding gives the known threshold

K_1 = max(⌈(4NTR_max/ε)³⌉, ⌈−6 ln³(δ/(6M_slots))⌉) + 1,

where M_slots is Nk² for the stochastic-game version and NA for a single-agent MDP. The implicit explore-or-exploit lemma compares the real MDP M with the optimistic induced model M_L: either the R-MAX policy hits an unknown slot within T steps with probability at least α/R_max, or its real T-step average return is within α of optimal. Finally, pigeonhole bounds the total useful unknown-slot visits by M_slotsK_1, and concentration bounds both the number of exploration attempts needed and the realized exploitation return; for K_2 exploration attempts the deviation term is K_2^{2/3}, yielding exponentially small failure in K_2^{1/3}.

## Design choices

- **R_max optimism:** zero or pessimistic initialization can make a greedy planner ignore unknown actions forever; R_max gives untried payoffs and unknown continuations the largest admissible value until their estimates are trustworthy.
- **Fictitious absorbing state:** the unknown slot has a stable maximal continuation value, so the planner values reaching it correctly for the optimism argument.
- **Hard known threshold:** the simulation lemma needs fixed empirical accuracy; freezing a slot at K_1 samples makes the counting and union bound finite.
- **One optimistic plan:** explicit explore-or-exploit requires a branch; the optimistic induced model folds both cases into a single planning call.
