# AlphaZero

## Problem it solves

The strongest game-playing programs reach superhuman strength by encoding enormous amounts of game-specific human knowledge: handcrafted evaluation functions plus heuristic alpha-beta search for chess and shogi, or networks bootstrapped from millions of human games plus rollout-based MCTS for Go. None of it transfers between games. AlphaZero is a single, generic reinforcement-learning algorithm that, given only the rules of a game, starts from random weights and reaches superhuman strength purely by self-play — no human data, no handcrafted features, no game-specific search heuristics, no rollouts — and the same procedure works for chess, shogi, and Go.

## Key idea

One deep network with two heads, `(p, v) = f_θ(s)`: a policy prior `p` (move probabilities `p_a = Pr(a|s)`) and a value `v ≈ E[z|s]` (expected game outcome). The network drives a Monte-Carlo tree search; the search produces, at the root, a visit-count distribution `π` that is a *policy improvement* over `p` (lookahead reallocates visits according to deeper value estimates). Self-play with the improved policy generates games scored by the rules to an outcome `z`, and the network is trained to chase its own search — `p → π` and `v → z`. This is policy iteration: the search is the improvement operator, the network fit is the evaluation/projection. MCTS (which averages leaf values) is chosen over minimax (which takes min/max and amplifies the worst approximation error) precisely so a single imperfect non-linear evaluator is usable inside the search. The only domain knowledge is the rules, used to step states, list legal moves, and score terminals.

## The search (MCTS, no rollouts)

Each edge `(s,a)` stores `N(s,a)`, `W(s,a)`, `Q(s,a) = W(s,a)/N(s,a)`, `P(s,a)`. A simulation descends from the root by the predictor-UCB rule:

```
a* = argmax_a [ Q(s,a) + U(s,a) ],   U(s,a) = c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)).
```

`Q` exploits; `U` explores, weighted by the prior `P` and shrinking as a child is visited (`1/(1+N)`), while `√(Σ_b N)` keeps exploration alive as the parent grows. At a leaf, one network call gives `(p, v) = f_θ(s_L)`; children are initialised `N=W=Q=0, P=p_a`, and `v` (the value head) replaces the rollout. Backup walks the path: `N += 1`, `W += v`, `Q = W/N`, flipping the sign of `v` each ply (two-player, value from the side-to-move's perspective). Root exploration noise prevents starving a good move the prior dismissed:

```
P(s_root, a) = (1 − ε) p_a + ε · η_a,   η ~ Dir(α),   ε = 0.25,
```

with `α` scaled inversely to the typical number of legal moves (`≈ 0.3` chess, `0.15` shogi, `0.03` Go). The play policy reads out the visit counts with a temperature:

```
π(a | s_root) = N(s_root, a)^{1/τ} / Σ_b N(s_root, b)^{1/τ}.
```

`τ = 1` (proportional, exploratory) for the opening moves of self-play; `τ → 0` (greedy argmax) thereafter and at evaluation.

## Training

Self-play plays full games with both sides moving by MCTS, `a_t ~ π_t`. The terminal is scored by the rules to `z ∈ {−1, 0, +1}` (loss/draw/win), written from each position's player's perspective. Each position stores `(s_t, π_t, z)`. The network is trained by gradient descent on

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
    """One search tree. Edges (s,a) store N, running Q, and prior P. The network
    supplies leaf value v and prior p; the rules step states, list legal moves,
    and score terminals."""

    def __init__(self, game, net, args):
        self.game, self.net, self.args = game, net, args
        self.Qsa, self.Nsa, self.Ns, self.Ps = {}, {}, {}, {}
        self.Es, self.Vs = {}, {}

    def get_action_prob(self, board, temp=1):
        for _ in range(self.args.numMCTSSims):
            self.search(board)
        s = self.game.stringRepresentation(board)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]
        if temp == 0:                                          # greedy (tau -> 0)
            best = np.random.choice(np.flatnonzero(counts == np.max(counts)))
            probs = [0] * len(counts); probs[best] = 1
            return probs
        counts = [c ** (1.0 / temp) for c in counts]           # pi(a) = N(a)^{1/tau} / sum_b N(b)^{1/tau}
        total = float(sum(counts))
        return [c / total for c in counts]

    def search(self, board):
        s = self.game.stringRepresentation(board)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(board, 1)      # rules: terminal score
        if self.Es[s] != 0:
            return -self.Es[s]
        if s not in self.Ps:                                   # leaf: one net call replaces a rollout
            self.Ps[s], v = self.net.predict(board)            # (p, v) = f_theta(s)
            valids = self.game.getValidMoves(board, 1)
            self.Ps[s] = self.Ps[s] * valids                   # mask illegal moves
            ssum = np.sum(self.Ps[s])
            self.Ps[s] = self.Ps[s] / ssum if ssum > 0 else (valids / np.sum(valids))
            self.Vs[s], self.Ns[s] = valids, 0
            return -v
        valids, best, best_a = self.Vs[s], -float("inf"), -1
        for a in range(self.game.getActionSize()):             # predictor-UCB
            if not valids[a]:
                continue
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] \
                    * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
            if u > best:
                best, best_a = u, a
        a = best_a
        next_board, next_player = self.game.getNextState(board, 1, a)   # rules: step state
        next_board = self.game.getCanonicalForm(next_board, next_player)
        v = self.search(next_board)
        if (s, a) in self.Qsa:                                 # backup: Q = W/N, N += 1
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)], self.Nsa[(s, a)] = v, 1
        self.Ns[s] += 1
        return -v                                              # sign flip up the tree (two-player)


class Coach:
    """Self-play -> train. Both sides move by MCTS; the search policy pi is the
    training target, the final outcome z is the value target."""

    def __init__(self, game, net, args):
        self.game, self.net, self.args = game, net, args
        self.mcts = MCTS(game, net, args)

    def execute_episode(self):
        examples, board, player, step = [], self.game.getInitBoard(), 1, 0
        while True:
            step += 1
            canonical = self.game.getCanonicalForm(board, player)
            temp = int(step < self.args.tempThreshold)         # tau=1 early, tau->0 later
            pi = self.mcts.get_action_prob(canonical, temp=temp)
            examples.append([canonical, player, pi, None])
            a = np.random.choice(len(pi), p=pi)                # a ~ pi
            board, player = self.game.getNextState(board, player, a)
            r = self.game.getGameEnded(board, player)
            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != player))) for x in examples]

    def learn(self):
        for _ in range(self.args.numIters):
            examples = []
            for _ in range(self.args.numEps):
                self.mcts = MCTS(self.game, self.net, self.args)   # fresh tree per game
                examples += self.execute_episode()
            self.net.train(examples)                               # fit toward (pi, z)
            # single continually-updated network; always self-play with latest weights


def loss(p_logits, v, target_pi, target_z, theta, c=1e-4):
    # l = (z - v)^2 - pi^T log p + c||theta||^2
    log_p = p_logits - np.log(np.sum(np.exp(p_logits)))
    value_loss = np.mean((target_z - v) ** 2)                  # policy evaluation: v -> z
    policy_loss = -np.mean(np.sum(target_pi * log_p, axis=1))  # policy improvement: p -> pi
    return value_loss + policy_loss + c * np.sum(theta ** 2)
```

The network body is a residual convolutional stack shared by both heads; the board is encoded as spatial planes from the rules (piece locations over a short history, side to move, castling, repetition, no-progress) and the policy head emits a distribution over the move encoding (for chess, an `8×8×73` plane stack over pick-up square and move type), with illegal moves masked out and renormalised.
