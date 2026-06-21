The strongest game programs have historically been built by encoding vast amounts of human knowledge about a single game: handcrafted evaluation features, move-ordering heuristics, opening books, endgame tables, and deep alpha-beta search for chess and shogi, or policy and value networks bootstrapped from millions of human games combined with rollout-based Monte-Carlo tree search for Go. These systems are superhuman in their domain but almost useless outside it, because the knowledge is welded to the game. The challenge is to find one algorithm that can learn any of these games from scratch, given only the rules, with random initial weights and no external teacher. The central difficulty is the bootstrap: a randomly initialized player is weak, so the data it generates about itself is weak, and there is no stronger signal to climb toward unless the algorithm can manufacture one.

The key observation is that search itself can provide that stronger signal. A Monte-Carlo tree search guided by a learned network does not merely execute the network's first guess; it corrects that guess by looking ahead. The visit counts at the root therefore represent a policy that is better than the raw network prior, because moves that look good to the prior but fail on deeper inspection lose visits, while moves the prior undervalued gain visits. If the network is then trained to imitate those visit counts and to predict the outcomes of the games played with them, the network improves, which makes the search stronger, which makes the training signal stronger. This closed loop is what allows tabula-rasa learning to bootstrap from random play.

The method is AlphaZero. It uses a single deep neural network with a shared trunk and two heads: a policy head that outputs move probabilities, and a value head that outputs an estimate of the expected game outcome. The network takes as input a canonical encoding of the board supplied by the rules and produces both outputs at once, because the representations useful for choosing moves and for judging positions are largely the same, and training both tasks jointly improves generalization compared with two separate networks. The network is the only learned component; there are no handcrafted evaluation features, no opening books, no endgame tables, and no fast rollout policy.

The search is a variant of Monte-Carlo tree search that uses the network in place of random rollouts. At each position the tree stores, for every legal action, a visit count, a running mean action value, and a prior probability from the network policy. The search proceeds by repeatedly selecting a path from the root to a leaf, expanding the leaf with one network evaluation, and backing up the resulting value up the path. Selection uses a predictor-UCB rule that balances exploitation through the running mean value against exploration weighted by the network prior: an action with a high prior and few visits receives a large exploration bonus, while an action that has been searched extensively sees its bonus shrink. Once a leaf is reached, the network is called once to obtain a policy vector and a scalar value; the policy is masked to legal moves and renormalized, and the value is used as the leaf evaluation. During backup the value is accumulated into each edge on the path, with the sign flipped at each ply because what is good for one player is bad for the other.

A subtle but important choice is using MCTS, which averages leaf values, rather than alpha-beta minimax, which propagates explicit min and max values. Because the neural value function is imperfect, a single large error anywhere in a minimax subtree can propagate straight to the root and corrupt the result. Averaging, by contrast, tends to cancel scattered errors as the subtree grows, making a powerful but approximate neural evaluator usable inside the search. To prevent the prior from starving moves it initially undervalues, Dirichlet noise is added to the root prior on the first visit, giving every legal move a nonzero chance of being explored. The root prior is also the only place where a game-specific scalar enters the system, in the form of the Dirichlet concentration parameter scaled roughly inversely to the typical number of legal moves.

After search, the root visit counts are converted into a policy by raising them to a temperature and normalizing. During self-play the temperature is one, producing a soft distribution that samples diverse moves and keeps the training data varied. During evaluation the temperature is driven toward zero, selecting the most-visited move greedily. Self-play then proceeds by sampling moves from this improved policy, playing complete games, and recording every position together with the improved policy and the final game outcome. The outcome is encoded as a scalar in {-1, 0, +1} representing loss, draw, or win from the perspective of the player to move at that position, with draws distinguished from unfinished positions so the value head can learn the true expected outcome including draws.

The network is trained by gradient descent on a combined loss. The value head is regressed toward the final outcome with squared error, which performs policy evaluation: it estimates how good the current improved policy is. The policy head is trained toward the search visit distribution with cross-entropy, which performs policy improvement: it bakes the lookahead-corrected policy back into the network. L2 weight decay is added to keep the weights small. A single network is maintained and updated continually; self-play always uses the latest weights, without a separate best-player gate or evaluation matches. The whole procedure is the same across chess, shogi, and Go, with only the board encoding, action encoding, and the Dirichlet concentration parameter differing.

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
