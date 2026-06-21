The problem is how to make a sequence of decisions when each decision changes the state of the system and the return depends on the whole sequence. The naive approach is to enumerate every possible sequence of actions, but that grows exponentially with the horizon and becomes impossible even for modest problems. In stochastic settings it is worse, because a fixed sequence ignores the random states that will be observed before future actions are chosen. The right object is not a precommitted plan but a rule that maps each observed state to an action. The difficulty is then to find such a rule without searching over all possible futures.

The key insight is that the future can be summarized by a single number: the best return obtainable from each state. If the tail of a policy were not optimal from the state it reaches, replacing that tail with a better one would improve the whole policy, contradicting optimality. Therefore the tail of an optimal policy must itself be optimal. This lets us write a local equation in which the value of a state equals the best one-step reward plus the value of the resulting next state. Dynamic programming, founded on Bellman's principle of optimality, turns global sequence optimization into recursive state-value computation.

The method works by defining a value function and deriving a backward recursion or fixed-point equation from the principle of optimality. For a finite-horizon problem with immediate reward r and deterministic transition T, the value at time t is V_t(p) = max_k { r_t(p,k) + V_{t+1}(T_t(p,k)) }, with a terminal condition at the final stage. For stochastic transitions the continuation value becomes an expectation over the next-state distribution. For infinite-horizon discounted problems the value satisfies V = TV, where T applies one-step maximization plus discounted continuation. Because T is a contraction when rewards are bounded and the discount factor lies in [0,1), value iteration converges to the unique fixed point. Policy iteration alternates between evaluating a fixed policy and greedily improving it, also converging under the same conditions.

```python
import numpy as np
from typing import Callable, Tuple


def deterministic_finite_horizon(
    states: list,
    actions: list,
    terminal_value: Callable,
    reward: Callable,
    transition: Callable,
    T: int,
):
    """Backward induction for a deterministic finite-horizon problem."""
    V = {s: terminal_value(s) for s in states}
    policy = {}

    for t in range(T - 1, -1, -1):
        V_prev = V.copy()
        for s in states:
            best_val = float("-inf")
            best_a = None
            for a in actions:
                s_next = transition(t, s, a)
                val = reward(t, s, a) + V_prev[s_next]
                if val > best_val:
                    best_val = val
                    best_a = a
            V[s] = best_val
            policy[(t, s)] = best_a
    return V, policy


def stochastic_finite_horizon(
    states: list,
    actions: list,
    terminal_value: Callable,
    reward: Callable,
    transitions: Callable,  # returns list of (prob, next_state)
    T: int,
):
    """Backward induction with stochastic transitions."""
    V = {s: terminal_value(s) for s in states}
    policy = {}

    for t in range(T - 1, -1, -1):
        V_prev = V.copy()
        for s in states:
            best_val = float("-inf")
            best_a = None
            for a in actions:
                exp_cont = sum(p * V_prev[z] for p, z in transitions(t, s, a))
                val = reward(t, s, a) + exp_cont
                if val > best_val:
                    best_val = val
                    best_a = a
            V[s] = best_val
            policy[(t, s)] = best_a
    return V, policy


def value_iteration(
    states: list,
    actions: list,
    reward: np.ndarray,  # shape (n_states, n_actions)
    transition_probs: np.ndarray,  # shape (n_states, n_actions, n_states)
    gamma: float,
    theta: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discounted infinite-horizon value iteration."""
    n_s, n_a = reward.shape
    V = np.zeros(n_s)

    while True:
        Q = reward + gamma * np.einsum("saz,z->sa", transition_probs, V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new

    policy = Q.argmax(axis=1)
    return V, policy


def policy_iteration(
    states: list,
    actions: list,
    reward: np.ndarray,
    transition_probs: np.ndarray,
    gamma: float,
    theta: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discounted infinite-horizon policy iteration."""
    n_s, n_a = reward.shape
    policy = np.zeros(n_s, dtype=int)

    while True:
        # Policy evaluation: solve linear system for current policy.
        P_pi = transition_probs[np.arange(n_s), policy, :]
        r_pi = reward[np.arange(n_s), policy]
        V = np.linalg.solve(np.eye(n_s) - gamma * P_pi, r_pi)

        # Policy improvement.
        Q = reward + gamma * np.einsum("saz,z->sa", transition_probs, V)
        new_policy = Q.argmax(axis=1)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    return V, policy


if __name__ == "__main__":
    # Small inventory example: choose order quantity to maximize profit.
    states = [0, 1, 2]
    actions = [0, 1, 2]

    def terminal_value(s):
        return 0.0

    def reward(t, s, a):
        # revenue from sold units minus holding cost for leftover inventory
        demand = 1
        sold = min(s + a, demand)
        leftover = max(s + a - demand, 0)
        return 5.0 * sold - 1.0 * leftover - 0.5 * a

    def transition(t, s, a):
        demand = 1
        next_stock = max(s + a - demand, 0)
        return min(next_stock, 2)

    V, pol = deterministic_finite_horizon(
        states, actions, terminal_value, reward, transition, T=3
    )
    print("Values:", V)
    print("Policy:", pol)

    # Discounted MDP value iteration on a tiny random MDP.
    n_s, n_a = 3, 2
    np.random.seed(0)
    r = np.random.randn(n_s, n_a)
    P = np.random.dirichlet(np.ones(n_s), size=(n_s, n_a))
    V, pi = value_iteration(states, actions, r, P, gamma=0.9)
    print("Value iteration values:", V)
    print("Value iteration policy:", pi)

    V2, pi2 = policy_iteration(states, actions, r, P, gamma=0.9)
    print("Policy iteration values:", V2)
    print("Policy iteration policy:", pi2)
```
