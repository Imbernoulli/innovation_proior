# AlphaZero

## Problem it solves

The strongest game-playing programs reach superhuman strength by encoding enormous amounts of game-specific human knowledge: handcrafted evaluation functions plus heuristic alpha-beta search for chess and shogi, or networks bootstrapped from millions of human games plus rollout-based MCTS for Go. None of it transfers between games. AlphaZero is a single, generic reinforcement-learning algorithm that, given only the rules of a game, starts from random weights and reaches superhuman strength purely by self-play — no human data, no handcrafted features, no game-specific search heuristics, no rollouts — and the same procedure works for chess, shogi, and Go.

## Key idea

One deep network with two heads, `(p, v) = f_θ(s)`: a policy prior `p` (move probabilities `p_a = Pr(a|s)`) and a value `v ≈ E[z|s]` (expected game outcome). The network drives a Monte-Carlo tree search; the search produces, at the root, a visit-count distribution `π` that is a *policy improvement* over `p` (lookahead reallocates visits according to deeper value estimates). Self-play with the improved policy generates games scored by the rules to an outcome `z`, and the network is trained to chase its own search — `p → π` and `v → z`. This is policy iteration: the search is the improvement operator, the network fit is the evaluation/projection. MCTS (which averages leaf values) is chosen over minimax (which takes min/max and amplifies the worst approximation error) precisely so a single imperfect non-linear evaluator is usable inside the search. The only domain knowledge is the rules, used to step states, list legal moves, and score terminals.

## The search (MCTS, no rollouts)

Each edge `(s,a)` stores `N(s,a)`, a running mean `Q(s,a)` of backed-up values, and `P(s,a)`. Equivalently, `Q` is `W/N` without storing a separate total `W`. A simulation descends from the root by the predictor-UCB rule:

```
a* = argmax_a [ Q(s,a) + U(s,a) ],   U(s,a) = c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)).
```

`Q` exploits; `U` explores, weighted by the prior `P` and shrinking as a child is visited (`1/(1+N)`), while `√(Σ_b N)` keeps exploration alive as the parent grows. At a leaf, one network call gives `(p, v) = f_θ(s_L)`; children are initialised with zero visits and prior `P=p_a`, and `v` (the value head) replaces the rollout. Backup walks the path by incrementing `N` and updating `Q ← (N_old Q + v)/(N_old + 1)`, flipping the sign of `v` each ply (two-player, value from the side-to-move's perspective). Root exploration noise prevents starving a good move the prior dismissed:

```
P(s_root, a) = (1 − ε) p_a + ε · η_a,   η ~ Dir(α),   ε = 0.25,
```

with `α` scaled inversely to the typical number of legal moves (`≈ 0.3` chess, `0.15` shogi, `0.03` Go). The play policy reads out the visit counts with a temperature:

```
π(a | s_root) = N(s_root, a)^{1/τ} / Σ_b N(s_root, b)^{1/τ}.
```

`τ = 1` (proportional, exploratory) during self-play; `τ → 0` (greedy argmax) during evaluation.

## Training

Self-play plays full games with both sides moving by MCTS, `a_t ~ π_t`. The rules API keeps terminal detection separate from the numeric outcome, so a draw can be the target value `0`; terminal positions are scored as `z ∈ {−1, 0, +1}` (loss/draw/win), written from each position's player's perspective. Each position stores `(s_t, π_t, z)`. The network is trained by gradient descent on

```
l = (z − v)^2 − π^T log p + c‖θ‖^2,
```

mean-squared error pulling `v` toward the outcome (policy evaluation), cross-entropy pulling `p` toward the searched policy (policy improvement), and `L2` weight decay. A single network is maintained and updated continually; self-play always uses the latest weights, with no separate best-player and no gating match. No symmetry augmentation is used (chess and shogi are not symmetric), and one hyperparameter set is shared across all three games except the Dirichlet `α`.

## Working code

```python
import math
import numpy as np

EPS = 1e-8


class MCTS:
    """One search tree. Edges (s,a) store N, running Q, and prior P."""

    def __init__(self, game, model, args):
        self.game, self.model, self.args = game, model, args
        self.Qsa, self.Nsa, self.Ns, self.Ps = {}, {}, {}, {}
        self.Es, self.Vs = {}, {}
        self.noised_roots = set()

    def _normalise_policy(self, policy, legal_actions):
        policy = np.asarray(policy, dtype=np.float64) * legal_actions
        total = float(np.sum(policy))
        if total <= 0:
            policy = np.asarray(legal_actions, dtype=np.float64)
            total = float(np.sum(policy))
        return policy / total

    def _adjust_root_prior(self, state_key):
        legal = self.Vs[state_key]
        legal_actions = np.flatnonzero(legal)
        if len(legal_actions) == 0:
            return
        alpha = self.args.dirichlet_alpha
        epsilon = getattr(self.args, "root_noise_fraction", 0.25)
        noise = np.random.dirichlet([alpha] * len(legal_actions))
        prior = np.array(self.Ps[state_key], dtype=np.float64)
        prior[legal_actions] = (1 - epsilon) * prior[legal_actions] + epsilon * noise
        self.Ps[state_key] = self._normalise_policy(prior, legal)
        self.noised_roots.add(state_key)

    def action_prob(self, canonical_state, temperature=1.0, explore=True):
        root = self.game.string_representation(canonical_state)
        sims = self.args.num_mcts_sims
        if explore and root not in self.Ps:
            self.search(canonical_state)
            if root in self.Ps:
                self._adjust_root_prior(root)
            sims -= 1
        elif explore and root in self.Ps and root not in self.noised_roots:
            self._adjust_root_prior(root)

        for _ in range(max(0, sims)):
            self.search(canonical_state)

        counts = np.array(
            [self.Nsa.get((root, a), 0) for a in range(self.game.action_size())],
            dtype=np.float64,
        )
        if np.sum(counts) <= 0:
            counts = np.asarray(self.Vs[root], dtype=np.float64)
        if temperature == 0:
            best = np.random.choice(np.flatnonzero(counts == np.max(counts)))
            probs = np.zeros_like(counts)
            probs[best] = 1.0
            return probs
        counts = counts ** (1.0 / temperature)
        return counts / float(np.sum(counts))

    def search(self, canonical_state):
        state_key = self.game.string_representation(canonical_state)
        if state_key not in self.Es:
            self.Es[state_key] = self.game.terminal_value(canonical_state, 1)
        if self.Es[state_key] is not None:
            return -self.Es[state_key]

        if state_key not in self.Ps:
            prior, value = self.model.predict(canonical_state)
            legal = np.asarray(self.game.legal_actions(canonical_state), dtype=np.float64)
            self.Ps[state_key] = self._normalise_policy(prior, legal)
            self.Vs[state_key], self.Ns[state_key] = legal, 0
            return -value

        legal, best_score, best_action = self.Vs[state_key], -float("inf"), -1
        for action in range(self.game.action_size()):
            if not legal[action]:
                continue
            if (state_key, action) in self.Qsa:
                score = self.Qsa[(state_key, action)] + self.args.cpuct * self.Ps[state_key][action] \
                    * math.sqrt(self.Ns[state_key]) / (1 + self.Nsa[(state_key, action)])
            else:
                # Break the first all-zero tie by the prior, matching the PUCT intent.
                score = self.args.cpuct * self.Ps[state_key][action] * math.sqrt(self.Ns[state_key] + EPS)
            if score > best_score:
                best_score, best_action = score, action

        next_state, next_player = self.game.next_state(canonical_state, 1, best_action)
        next_state = self.game.canonical_form(next_state, next_player)
        value = self.search(next_state)

        if (state_key, best_action) in self.Qsa:
            old_n = self.Nsa[(state_key, best_action)]
            self.Qsa[(state_key, best_action)] = (old_n * self.Qsa[(state_key, best_action)] + value) / (old_n + 1)
            self.Nsa[(state_key, best_action)] = old_n + 1
        else:
            self.Qsa[(state_key, best_action)] = value
            self.Nsa[(state_key, best_action)] = 1
        self.Ns[state_key] += 1
        return -value


class SelfPlayTrainer:
    """Self-play produces (state, searched policy, final outcome) examples."""

    def __init__(self, game, model, args):
        self.game, self.model, self.args = game, model, args
        self.examples = []

    def play_game(self):
        examples = []
        state, player = self.game.initial_state(), 1
        while True:
            canonical = self.game.canonical_form(state, player)
            mcts = MCTS(self.game, self.model, self.args)
            pi = mcts.action_prob(canonical, temperature=1.0, explore=True)
            examples.append((canonical, player, pi))
            action = np.random.choice(len(pi), p=pi)
            state, player = self.game.next_state(state, player, action)
            outcome = self.game.terminal_value(state, player)
            if outcome is not None:
                return [
                    (past_state, past_pi, outcome if past_player == player else -outcome)
                    for past_state, past_player, past_pi in examples
                ]

    def learn(self):
        for _ in range(self.args.num_iterations):
            iteration_examples = []
            for _ in range(self.args.self_play_games):
                iteration_examples.extend(self.play_game())
            self.examples.extend(iteration_examples)
            if len(self.examples) > self.args.max_examples:
                self.examples = self.examples[-self.args.max_examples:]
            self.model.train(self.examples)


def position_loss(policy_logits, value, target_policy, target_value, parameters, weight_decay=1e-4):
    logits = np.asarray(policy_logits, dtype=np.float64)
    target_policy = np.asarray(target_policy, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[None, :]
        target_policy = target_policy[None, :]

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_policy = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    policy_loss = -np.mean(np.sum(target_policy * log_policy, axis=1))
    pred_value = np.asarray(value, dtype=np.float64).reshape(-1)
    target_value = np.asarray(target_value, dtype=np.float64).reshape(-1)
    value_loss = np.mean((target_value - pred_value) ** 2)
    l2 = weight_decay * sum(np.sum(np.asarray(param) ** 2) for param in parameters)
    return value_loss + policy_loss + l2
```

The network body is a residual convolutional stack shared by both heads; the board is encoded as spatial planes from the rules (piece locations over a short history, side to move, castling, repetition, no-progress) and the policy head emits a distribution over the move encoding (for chess, an `8×8×73` plane stack over pick-up square and move type), with illegal moves masked out and renormalised.
