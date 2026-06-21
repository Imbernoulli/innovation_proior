## Research question

For most of the history of artificial intelligence, the strongest game-playing programs have been built the same way: take a hard combinatorial game, hand the machine as much human expertise about that specific game as can be encoded, and search. In chess and shogi this means a position evaluation function whose features were designed by grandmasters and whose weights were tuned over decades, driving an alpha-beta search studded with game-specific heuristics. In Go it means a neural network initialised by imitating millions of recorded human moves, combined with a search that estimates positions by random playouts. Each of these systems is built for its own domain, with the knowledge specific to that game.

The question is whether a *single* algorithm, given nothing about a game but its rules — no human games to imitate, no handcrafted evaluation features, no game-specific search heuristics — can start from random play and reach superhuman strength purely by playing itself. "Tabula rasa" is the setting: random initial weights, self-play only, the same procedure applied to several different games. Such a system would have to produce, from zero, both a way to *evaluate* positions and a way to *choose where to search*, and it would have to generate its own training signal, since there is no external teacher. It begins with random weights, so the only data available at the outset is the play of a weak agent.

## Background

The field at this time is split cleanly by domain, and the split is along the axis of how much human knowledge the system encodes.

In chess and shogi, the dominant paradigm is **alpha-beta minimax search over a handcrafted linear evaluation**. A position `s` is described by a sparse vector of handcrafted features `φ(s)` — material values, material-imbalance tables, piece-square tables, mobility, king safety, pawn structure, bishop pair, and dozens more, each designed by human experts. The features are combined linearly, `v(s, w) = φ(s)^T w`, with weights `w` set by a mix of manual and automatic tuning. This raw evaluation is trusted only for "quiet" positions, so a domain-specialised *quiescence search* first resolves pending captures and checks. The value of a position is then the minimax over a deep tree, made tractable by alpha-beta pruning (cut any branch provably dominated by another) plus a large pile of heuristics: aspiration windows, principal variation search, null-move pruning, futility pruning, search extensions for forcing moves and reductions for unpromising ones, move ordering by iterative deepening and by killer / history / counter-move heuristics, transposition tables, opening books, and endgame tablebases. Decades of refinement have made these engines (Deep Blue historically; Stockfish and Elmo currently) play far above human level. The cost is total domain-specificity: every component encodes knowledge about *this* game, and the system cannot be moved to another problem without being substantially rebuilt by hand.

A property of minimax search worth recording. Alpha-beta computes an *explicit* minimax value, which means whatever the largest evaluation error anywhere in the searched subtree is, that error tends to propagate to the root — the min and max operators latch onto extreme values rather than averaging them away.

In Go, the structure of the game (translational invariance, local liberties, rotational/reflectional symmetry, a simple "place a stone" action space, binary outcomes) suits convolutional networks, and the strongest programs combine **deep policy and value networks with Monte-Carlo tree search**. The most prominent such system first trained a policy network by supervised learning on millions of expert human moves, refined it by policy-gradient self-play, trained a value network on the resulting games, and then searched with MCTS whose leaf evaluations *mix a learned value with the return of a random rollout* — a fast playout policy plays the position out to the end, and the rollout result is averaged with the value-net estimate. This reaches superhuman Go strength but still leans on a large corpus of human games to bootstrap the policy, and on rollouts (a Monte-Carlo estimate that is unbiased but noisy) inside the search.

The load-bearing concepts underneath all of this:

- **Minimax / alpha-beta.** The game-theoretic value of a position is the minimax over the game tree; alpha-beta prunes provably irrelevant branches. Efficiency depends critically on move ordering, hence the heuristics above. The error-propagation property noted above is intrinsic to the explicit min/max.

- **Monte-Carlo tree search (MCTS)** (Coulom, 2006; Kocsis & Szepesvári's UCT, 2006). Repeatedly simulate trajectories from the root: descend the tree by an upper-confidence rule balancing an exploitation term (the running mean value of an action) against an exploration bonus, expand a leaf, evaluate it (classically by a random rollout to the end of the game), and back the evaluation up the path, updating each edge's visit count and mean value. The visit counts at the root concentrate on strong moves. Unlike minimax, MCTS *averages* values over a subtree rather than taking explicit min/max.

- **Predictor + UCB (PUCT)** (Rosin, 2011). A variant of the UCB selection rule that weights the exploration bonus of each action by a prior probability supplied by a "predictor", so the search can be biased toward moves a predictor considers promising rather than exploring uniformly.

- **Policy iteration.** Alternate *policy evaluation* (estimate the value of the current policy) with *policy improvement* (make the policy greedier with respect to those values). Under mild conditions this converges to the optimum.

- **Self-play temporal-difference learning in games.** Earlier work showed that a value function can be learned by self-play with no human data: TD-Gammon (Tesauro, 1995) reached expert backgammon by TD learning of a neural value function from self-play; in chess and shogi a line of work (NeuroChess; Beal & Smith; KnightCap and Giraffe using TD-leaf, which updates the leaf value of the principal variation; Meep using TreeStrap, which updates all nodes of an alpha-beta search) learned evaluation functions by self-play, in some cases from random weights, reaching master-level play. These establish that self-play *can* learn an evaluator from scratch — but each still wraps the learned evaluator in a handcrafted alpha-beta search and, mostly, in handcrafted input features.

## Baselines

**Alpha-beta engines with handcrafted evaluation** (Deep Blue, Campbell et al. 2002; Stockfish; Elmo). Core idea and math as above: `v(s,w) = φ(s)^T w` over expert features, embedded in a deeply pruned, heavily heuristic minimax search, plus quiescence search, transposition tables, opening books, endgame tablebases. These define superhuman strength in chess and shogi and search tens of millions of positions per second. Gap: every component is hand-built for the specific game; the system encodes human knowledge rather than discovering it.

**Policy/value networks + MCTS bootstrapped from human data** (the strong Go programs). A policy net `p = f(s)` and value net `v = g(s)`, the policy initialised by supervised learning on human expert moves and refined by self-play, driving an MCTS whose leaf value is a mixture of the value net and a random rollout. Gap: it requires a human-game corpus to bootstrap the policy, uses two separate networks, and relies on noisy rollouts inside the search; it is also tuned to a symmetric, binary-outcome game.

**Self-play TD learners for board games** (TD-Gammon, Tesauro 1995; KnightCap, Baxter et al.; Giraffe, Lai 2015; Meep, Veness et al.). Learn an evaluation function by temporal-difference learning from self-play, sometimes from random initial weights, by regressing the current value toward a bootstrapped target derived from search (TD-leaf updates the principal-variation leaf; TreeStrap updates all searched nodes). Gap: they learn only the *evaluator*; the search itself remains a handcrafted alpha-beta engine with handcrafted move ordering, and most still use handcrafted input features. None learns *where to search* from scratch.

**MCTS with rollouts** (Coulom 2006; UCT, Kocsis & Szepesvári 2006). Estimate leaf value by averaging the outcomes of random playouts, select by UCB. Gap: rollout estimates are high-variance, and in tactical games like chess pure-rollout MCTS has historically been far weaker than alpha-beta; rollouts also embed a hand-tuned playout policy.

## Evaluation settings

The natural yardsticks are three games: chess, shogi (a larger, harder variant of chess in which captured pieces change sides and may be dropped back onto the board), and Go (19×19). The rules supply legal moves, state transitions, terminal detection, and terminal scoring, including draws in chess and shogi. Strength is measured by **Elo rating**, estimated by Bayesian logistic regression from head-to-head games at matched thinking time (e.g. one second or one minute per move), with the baseline player's rating anchored to its standard value; the opponents are the strongest available program in each game (a top alpha-beta engine for chess and shogi, a strong network-plus-MCTS program for Go). Search effort is reported as simulations per move and as positions evaluated per second.

## Code framework

The primitives that already exist: a deep-net library with convolutional and residual blocks and a stochastic-gradient optimiser with momentum and weight decay; an experience buffer; a self-play harness that lets a program play complete games against itself or an opponent; and a generic Monte-Carlo tree search given (a) a way to step a position forward under a move, (b) the legal moves at a position, (c) a way to detect and score a terminal position, and (d) some way to evaluate and to bias the expansion of a leaf. The rules of each game supply (a), (b), (c) — they are the only domain knowledge admitted. The game object supplies `initial_state`, `next_state`, `legal_actions`, `terminal_value`, `canonical_form`, `action_size`, and `string_representation`; `terminal_value` returns `None` while play is ongoing and otherwise returns `-1`, `0`, or `+1` from the queried player's view. The empty model slot is expected to expose `predict` and `train` once it is filled.

What does *not* yet exist is the object that supplies the leaf evaluation and the expansion bias, and the procedure that trains it from self-play. The scaffold leaves those as empty slots.

```python
import math
import numpy as np


class MCTS:
    def __init__(self, game, model, args):
        # TODO: initialise edge statistics, cached legal moves, cached terminal
        #       values, and any root-level bookkeeping needed by the search.
        pass

    def _normalise_policy(self, policy, legal_actions):
        # TODO: mask illegal actions and renormalise the remaining prior.
        pass

    def _adjust_root_prior(self, state_key):
        # TODO: define any root-level exploration adjustment.
        pass

    def action_prob(self, canonical_state, temperature=1.0, explore=True):
        # TODO: run a generic tree search and return a distribution over actions.
        pass

    def search(self, canonical_state):
        # TODO: descend, expand, evaluate, and back up through the tree.
        pass


class SelfPlayTrainer:
    def __init__(self, game, model, args):
        # TODO: keep the game, model, search arguments, and replay examples.
        pass

    def play_game(self):
        # TODO: play one complete self-play game and return training examples.
        pass

    def learn(self):
        # TODO: alternate self-play data generation with model training.
        pass


def position_loss(policy_logits, value, target_policy, target_value, parameters, weight_decay=1e-4):
    # TODO: define the objective that fits the model to self-play targets.
    pass
```
