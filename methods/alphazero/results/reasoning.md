OK, let me think this through from scratch. I want a program that plays a hard game — chess, say, or shogi, or Go — at superhuman strength, but I want to give it nothing about the game except the rules. No handcrafted evaluation function. No move-ordering heuristics. No corpus of human games to imitate. Random initial weights, and the only thing it ever does to get better is play itself. Can that even work? The honest worry is the bootstrap. At the start the thing is a random player. Everything it knows about the game it has to extract from games played by a random player, then a slightly-less-random player, and so on. If the only signal I have is "this player's move," I'm chasing my own incompetence. So before anything else I need to find a source of signal that is *better* than the current player, so there's something to climb toward.

Let me look at what the strong programs in these games actually do, because the answer to "what's better than the current policy" might be hiding in plain sight. In chess and shogi the recipe is a handcrafted linear evaluation `v(s,w) = φ(s)^T w` — features designed by grandmasters, weights tuned over decades — wrapped in a deep alpha-beta search with a quiescence search, transposition tables, and a tower of pruning and move-ordering heuristics. The thing searches tens of millions of positions a second and plays beautifully. But it's a monument to human knowledge about *one game*; nothing in it transfers, and I've sworn off all of it — both the features and the heuristics are exactly what I'm refusing to write. In Go the recipe is different: train a policy network by imitating millions of human moves, train a value network, then run Monte-Carlo tree search whose leaf value mixes the value net with the result of a random rollout to the end of the game. Closer to what I want — there are networks doing the evaluating — but it's bootstrapped from human games, it uses two separate networks, and it leans on rollouts inside the search.

The second recipe has one piece I need to stare at, because it is the only thing in either system that smells like the better-than-current signal I need: the *search* itself. Forget how the leaf is evaluated for a second. MCTS takes a position, simulates many trajectories, and the visit counts at the root are not just the network's first guess; they are that guess corrected by looking ahead. If the lookahead has any value, the search output can be a stronger policy than the raw network that drives it. Then I have my better-than-current signal: run the search, and train the network toward the search. Then the search, now driven by a stronger network, is stronger still. That's the loop. The bootstrap problem dissolves if the search is genuinely a policy-improvement operator — let me hold that thought, because I'll have to actually believe it, not just hope it.

So the plan has the shape of policy iteration: improvement (the search produces a better policy than the network), then evaluation/projection (fit the network back onto what the search did), repeat. Now I have to design the three pieces concretely: what the network predicts, how the search uses it, and what exactly the network is trained toward.

Start with the network. The search needs two things at a node that I'd otherwise hand-build. It needs to *evaluate* a position — how good is this for the side to move — which is the job the handcrafted `φ(s)^T w` used to do. And it needs to know *where to look first* — which is the job the move-ordering heuristics used to do. So let the network output both: a scalar value `v ≈ E[z | s]`, the expected game outcome from this position, and a vector of move probabilities `p` with `p_a = Pr(a | s)`, a prior over which move to play. One network, `(p, v) = f_θ(s)`, takes the board (encoded as planes from the rules — piece locations, side to move, castling, repetitions, and so on) and emits both. Why one network with two heads rather than two? Because the two tasks share almost all of the work — both need to understand the position — so a shared trunk learns a better representation than either task alone would, and the value target and the policy target each act as a regulariser on the other. Two separate networks would duplicate the representation and, in practice, generalise worse. So: one body, a policy head, a value head.

Now the search. I want MCTS, but I have to settle the leaf evaluation, because that's where the Go programs used rollouts and I'm suspicious of rollouts. A rollout plays the position out to the end with a fast policy and uses the result; it's an unbiased estimate of the outcome but it's *noisy* — one random playout tells you very little, and you need many to average down the variance, and the fast playout policy is itself a hand-tuned thing I don't want. But I already have a value head `v` that's supposed to estimate `E[z|s]` directly. So replace the rollout with `v`: when the search reaches a leaf, evaluate it once with the network, read off `(p, v)`, and use `v` as the leaf's value. No playout, no fast policy, one network call. The value is biased (the network is imperfect) but low-variance, and as the network trains it gets better — whereas a rollout stays noisy forever.

But wait — if my leaf evaluations are biased by an imperfect network, why is MCTS the right search to wrap around them, rather than the alpha-beta minimax the chess engines use? This is the thing that decides whether the whole approach can work, so let me reason about it carefully rather than assume. Alpha-beta computes an explicit minimax: at each node it takes the min or the max of its children. The trouble with min and max is that they *select* the extreme value. If my non-linear evaluator has a spurious large error anywhere in the searched subtree, a max will happily grab it, and that single bad estimate propagates straight to the root. That's exactly why people who tried to drop neural evaluations into alpha-beta engines couldn't beat the fast handcrafted ones — the search amplifies the approximation error. MCTS does the opposite: it *averages* the leaf evaluations over the subtree below an action. Average, not max. So the spurious errors, scattered in sign, tend to cancel as the subtree grows, instead of being latched onto. That's the property that makes a powerful-but-imperfect evaluator usable: pair the non-linear network with the averaging search, not the min/max search. Good — MCTS it is, and now I believe the pairing for a reason, not by fiat.

Now run the search and define exactly what each node stores and how I descend. The tree has edges `(s, a)`, one per legal move from a position. At each edge I keep a visit count `N(s,a)`, a running mean value `Q(s,a)` of the values backed up through that edge, and the prior `P(s,a)` the network assigned to that move; if I wanted a separate total value `W`, this `Q` would just be `W/N`, but the running mean is enough. A simulation starts at the root and descends, and at each node I have to pick which child to go to. This is the exploit-versus-explore choice, and I want to bias it by the network's prior — that's the whole point of having a policy head, to tell the search where to look first, replacing the move-ordering heuristics. So the selection rule needs an exploitation term that prefers high-value moves and an exploration term that prefers moves I haven't tried much *and* that the prior likes. Let me build the exploration bonus from those requirements. It should grow with the prior `P(s,a)` (look where the network points). It should shrink as I visit a move more — once I've explored a child a lot, the bonus for it should fade, so divide by something increasing in `N(s,a)`; `1/(1 + N(s,a))` does it and stays finite at `N=0`. And it should not vanish too fast relative to the parent — as the parent gets visited more, I should keep being willing to try its less-visited children, so scale by `√(Σ_b N(s,b))`, the square root of the parent's total visits, which is the usual UCB-style growth that keeps exploration alive. Put them together with a constant `c_puct` to set the overall exploration weight:

```
a* = argmax_a [ Q(s,a) + U(s,a) ],   U(s,a) = c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)).
```

Early in the search `N` is small everywhere, so `U` dominates and the search fans out along the prior; in code I only need a tiny epsilon to let the prior break the all-zero first tie. As visits accumulate, `Q` takes over and the search exploits what it's learned. That's the predictor-UCB rule — the "predictor" being the network's prior `P`. It's the piece that lets the search be *selective* the way a strong human is selective, concentrating on a handful of promising moves rather than expanding everything uniformly, which is why it can be devastating at a thousand simulations where uniform search would need millions.

Descend by that rule until I hit a position not yet in the tree — a leaf. Expand it: call the network once, `(p, v) = f_θ(s_L)`, mask out illegal moves, renormalise the remaining prior, and let every legal action have an edge whose count starts at zero. The value `v` is the evaluation of this leaf. Now back it up. Walk back up the path I descended, and for every edge `(s, a)` on it, increment the visit count and fold the new value into the running mean, `Q ← (N_old Q + v)/(N_old + 1)`. One subtlety in a two-player game: value is from the perspective of the side to move, and the side to move alternates every ply, so the same leaf value `v` is good for one player and bad for the other — I flip the sign of `v` at each level as I propagate it up. (Concretely, the search evaluates everything in a canonical "side-to-move" frame, and the backed-up value negates each step.) After many simulations, the visit counts `N(s_root, a)` at the root encode how much the search favored each move.

Now the claim I parked earlier: that the search output is a better policy than the network's prior `p`. Why should the visit counts be an improvement? Because the search *spends* its visits according to value. The selection rule keeps sending simulations down the high-`Q` actions and stops sending them down actions that the deeper evaluations reveal to be bad — even if the prior `p` initially liked them. So the visit distribution is the prior corrected by lookahead: moves the network overrated get their visits choked off once the search sees they lead nowhere, and moves the network underrated get more visits once the search confirms them. The lookahead is information the raw `p` didn't have. So the visit-count distribution is a policy-improvement step over `p` — it is `p` plus the correction that searching provides. That's precisely the better-than-current signal I needed for the bootstrap, and now I see *why* it's better, not just that it is.

So turn the visit counts into a policy. The obvious thing for self-play is proportional to visits, because I want the training games to keep sampling the moves that the search still considers plausible rather than collapse instantly to one line. I also want the same readout to cover greedy play when I evaluate the player. A temperature does exactly that:

```
π(a | s_root) = N(s_root, a)^{1/τ} / Σ_b N(s_root, b)^{1/τ}.
```

With `τ = 1` this is just proportional to visit counts — a soft distribution I can sample from during self-play, which keeps the data diverse. As `τ → 0` it concentrates all mass on the most-visited move — greedy play for evaluation. The `π` I get is the improved policy, the thing I'll train toward.

One gap in the exploration story. If I always descend by the rule above, and the prior `p` happens to assign near-zero probability to a move that's actually good, the search may never try it — `U` is proportional to `P`, so a move the network ignores gets almost no exploration bonus, and it stays unexplored, and the network never learns it was good. The self-play loop could get stuck never discovering certain moves. I need to force some exploration at the root, independent of the prior, so every root move gets at least a chance. Mix Dirichlet noise into the root prior:

```
P(s_root, a) = (1 − ε) p_a + ε · η_a,   η ~ Dir(α),   ε = 0.25.
```

The Dirichlet draw spreads probability across all root moves, so even a move the network dismissed gets a nonzero prior at the root and thus a real chance of being searched. Only at the root — deeper in the tree I trust the prior, because the point of the prior is to be selective. The concentration `α` controls how spread the noise is, and the right `α` depends on how many moves there typically are: in a game with many legal moves I want the noise spread thin, in a game with few I want it more concentrated, so I scale `α` inversely with the typical branching — small for Go (many moves), larger for chess and shogi. (Roughly `α ≈ 0.03` for Go, `0.3` for chess, `0.15` for shogi.) This is the one place I let the typical move count leak in, and it's a single scalar, not knowledge about the game.

Now the training. I play complete games of self-play: at every position `s_t`, run the search, get `π_t`, sample the move `a_t ~ π_t`, step the real game by the rules. Both players move this way. The rules need to tell me two separate things at the end: that the game is terminal, and what the outcome is. A draw is a real terminal outcome but its value is `0`, so I cannot use `0` to mean both "not ended" and "draw" in the learning target. When the game ends, the terminal value is `z ∈ {−1, 0, +1}` for loss / draw / win. (Note the draw — these games can be drawn, and I want the value to be the *expected outcome* including draws, not the probability of winning, which is why I let `z` take three values and `v` estimate `E[z]`.) For each position in the game I store `(s_t, π_t, z)`, where `z` is the final outcome written from the perspective of the player who was to move at `s_t`. Now I have a dataset of (position, improved-policy, eventual-outcome) triples, and I fit the network to it.

What should the loss be? Two jobs. The value head should predict the outcome, so regress `v` onto `z`; outcomes are bounded and the value is a single scalar, so squared error `(z − v)^2` is the natural choice. The policy head should imitate the improved policy `π` — that's the whole loop, the network chasing its own search. The natural loss for matching a probability distribution is cross-entropy, `−π^T log p`, which drives `p` toward `π`. Add `L2` weight decay `c‖θ‖^2` to keep the weights from blowing up. So:

```
l = (z − v)^2 − π^T log p + c‖θ‖^2.
```

Gradient descent on this, mini-batched over positions sampled from recent self-play. Step back and look at what this loss *is*, because the unification is the satisfying part. The cross-entropy term pulls the network's prior toward the search's improved policy — that's the policy-improvement step being baked into the network. The squared-error term pulls the network's value toward the outcomes that this very policy produced — that's the policy-evaluation step. So one gradient step on this loss is simultaneously the improvement and the evaluation half of policy iteration, and the only "teacher" anywhere is the network's own search. There is no human data, no handcrafted evaluation, no rollout — the network is trained to imitate the lookahead-corrected version of itself and to predict the games that version plays. That's the entire mechanism. Random weights produce a weak search, the weak search is still a bit stronger than the raw weak network, the network is pulled toward that slightly-stronger search, which makes the next search stronger, and the ratchet turns.

There are two loop choices I have to settle. During self-play I keep sampling from the proportional visit-count distribution; the greedy `τ → 0` readout is for evaluation, where I want the strongest move rather than diverse training data. Do I keep a "best player so far" and only generate self-play games from it, swapping in a new network only when it proves stronger in a match? That's one way to keep the data-generating policy from regressing. But it adds a whole evaluation-and-gating machine, and it stalls the loop while the match runs. Simpler: just maintain a single network, update it continually, and always generate self-play with the latest weights — no separate best-player, no gating match. The continual update is a little less stable in principle but far simpler and keeps the data fresh, and the loss is self-correcting enough that it holds together. And do I exploit board symmetries — generate the eight rotations/reflections of each position as extra training data, or average the network over symmetric transforms during search? That's free strength when the rules are symmetric, as in Go. But in chess and shogi the rules are *not* symmetric — pawns move only forward, castling differs by side, promotion happens on a specific rank — so I cannot assume any symmetry. To keep one algorithm across all three games, I drop the symmetry augmentation entirely rather than special-case it. Same reasoning for hyperparameters: reuse one set across games, with the sole exception of the Dirichlet `α` scaled by move count, because anything else would be game-specific tuning of the kind I'm trying to avoid.

I can write the loop now.

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

I end up with one closed loop: I wanted one network to learn a game from zero by self-play with no human data and no rollouts, and the bootstrap needs a signal stronger than the current player; the search supplies it, because MCTS guided by the network's prior produces a visit distribution that is a policy improvement over that prior, and MCTS (averaging) rather than minimax (min/max) is what makes a single dual-head network usable as the leaf evaluator despite its approximation errors; so I select by predictor-UCB, expand a leaf with one network call for `(p, v)`, back up a running `Q` with a per-ply sign flip, add Dirichlet noise at the root so no move is starved, read out `π ∝ N^{1/τ}`, and train the network by `(z − v)^2 − π^T log p + c‖θ‖^2` — squared error doing policy evaluation, cross-entropy doing policy improvement — so the network forever chases the lookahead-corrected version of itself, and the only domain knowledge anywhere is the rules.
