OK, let me think this through from scratch. I want a program that plays a hard game — chess, say, or shogi, or Go — at superhuman strength, but I want to give it nothing about the game except the rules. No handcrafted evaluation function. No move-ordering heuristics. No corpus of human games to imitate. Random initial weights, and the only thing it ever does to get better is play itself. Can that even work? The honest worry is the bootstrap. At the start the thing is a random player. Everything it knows about the game it has to extract from games played by a random player, then a slightly-less-random player, and so on. If the only signal I have is "this player's move," I'm chasing my own incompetence. So before anything else I need to find a source of signal that is *better* than the current player, so there's something to climb toward.

Let me look at what the strong programs in these games actually do, because the answer to "what's better than the current policy" might be hiding in plain sight. In chess and shogi the recipe is a handcrafted linear evaluation `v(s,w) = φ(s)^T w` — features designed by grandmasters, weights tuned over decades — wrapped in a deep alpha-beta search with a quiescence search, transposition tables, and a tower of pruning and move-ordering heuristics. The thing searches tens of millions of positions a second and plays beautifully. But it's a monument to human knowledge about *one game*; nothing in it transfers, and I've sworn off all of it — both the features and the heuristics are exactly what I'm refusing to write. In Go the recipe is different: train a policy network by imitating millions of human moves, train a value network, then run Monte-Carlo tree search whose leaf value mixes the value net with the result of a random rollout to the end of the game. Closer to what I want — there are networks doing the evaluating — but it's bootstrapped from human games, it uses two separate networks, and it leans on rollouts inside the search.

The second recipe has one piece I need to stare at, because it is the only thing in either system that smells like the better-than-current signal I need: the *search* itself. Forget how the leaf is evaluated for a second. MCTS takes a position, simulates many trajectories, and the visit counts at the root concentrate on the strong moves. Empirically, the move the search recommends is much stronger than the move the raw network would pick on its own. So the search is doing *something* to the policy — taking a network's first guess and improving on it by looking ahead. If that's true, then I have my better-than-current signal: run the search, and its output is a stronger player than the network that drives it. Train the network toward the search. Then the search, now driven by a stronger network, is stronger still. That's the loop. The bootstrap problem dissolves if the search is genuinely a policy-improvement operator — let me hold that thought, because I'll have to actually believe it, not just hope it.

So the plan has the shape of policy iteration: improvement (the search produces a better policy than the network), then evaluation/projection (fit the network back onto what the search did), repeat. Now I have to design the three pieces concretely: what the network predicts, how the search uses it, and what exactly the network is trained toward.

Start with the network. The search needs two things at a node that I'd otherwise hand-build. It needs to *evaluate* a position — how good is this for the side to move — which is the job the handcrafted `φ(s)^T w` used to do. And it needs to know *where to look first* — which is the job the move-ordering heuristics used to do. So let the network output both: a scalar value `v ≈ E[z | s]`, the expected game outcome from this position, and a vector of move probabilities `p` with `p_a = Pr(a | s)`, a prior over which move to play. One network, `(p, v) = f_θ(s)`, takes the board (encoded as planes from the rules — piece locations, side to move, castling, repetitions, and so on) and emits both. Why one network with two heads rather than two? Because the two tasks share almost all of the work — both need to understand the position — so a shared trunk learns a better representation than either task alone would, and the value target and the policy target each act as a regulariser on the other. Two separate networks would duplicate the representation and, in practice, generalise worse. So: one body, a policy head, a value head.

Now the search. I want MCTS, but I have to settle the leaf evaluation, because that's where the Go programs used rollouts and I'm suspicious of rollouts. A rollout plays the position out to the end with a fast policy and uses the result; it's an unbiased estimate of the outcome but it's *noisy* — one random playout tells you very little, and you need many to average down the variance, and the fast playout policy is itself a hand-tuned thing I don't want. But I already have a value head `v` that's supposed to estimate `E[z|s]` directly. So replace the rollout with `v`: when the search reaches a leaf, evaluate it once with the network, read off `(p, v)`, and use `v` as the leaf's value. No playout, no fast policy, one network call. The value is biased (the network is imperfect) but low-variance, and as the network trains it gets better — whereas a rollout stays noisy forever.

But wait — if my leaf evaluations are biased by an imperfect network, why is MCTS the right search to wrap around them, rather than the alpha-beta minimax the chess engines use? This is the thing that decides whether the whole approach can work, so let me reason about it carefully rather than assume. Alpha-beta computes an explicit minimax: at each node it takes the min or the max of its children. The trouble with min and max is that they *select* the extreme value. If my non-linear evaluator has a spurious large error anywhere in the searched subtree, a max will happily grab it, and that single bad estimate propagates straight to the root. That's exactly why people who tried to drop neural evaluations into alpha-beta engines couldn't beat the fast handcrafted ones — the search amplifies the approximation error. MCTS does the opposite: it *averages* the leaf evaluations over the subtree below an action. Average, not max. So the spurious errors, scattered in sign, tend to cancel as the subtree grows, instead of being latched onto. That's the property that makes a powerful-but-imperfect evaluator usable: pair the non-linear network with the averaging search, not the min/max search. Good — MCTS it is, and now I believe the pairing for a reason, not by fiat.

Now run the search and define exactly what each node stores and how I descend. The tree has edges `(s, a)`, one per legal move from a position. At each edge I keep four numbers: a visit count `N(s,a)`, the total value accumulated through that edge `W(s,a)`, the mean value `Q(s,a) = W(s,a)/N(s,a)`, and the prior `P(s,a)` the network assigned to that move. A simulation starts at the root and descends, and at each node I have to pick which child to go to. This is the exploit-versus-explore choice, and I want to bias it by the network's prior — that's the whole point of having a policy head, to tell the search where to look first, replacing the move-ordering heuristics. So the selection rule needs an exploitation term that prefers high-value moves and an exploration term that prefers moves I haven't tried much *and* that the prior likes. Let me build the exploration bonus from those requirements. It should grow with the prior `P(s,a)` (look where the network points). It should shrink as I visit a move more — once I've explored a child a lot, the bonus for it should fade, so divide by something increasing in `N(s,a)`; `1/(1 + N(s,a))` does it and stays finite at `N=0`. And it should not vanish too fast relative to the parent — as the parent gets visited more, I should keep being willing to try its less-visited children, so scale by `√(Σ_b N(s,b))`, the square root of the parent's total visits, which is the usual UCB-style growth that keeps exploration alive. Put them together with a constant `c_puct` to set the overall exploration weight:

```
a* = argmax_a [ Q(s,a) + U(s,a) ],   U(s,a) = c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)).
```

Early in the search `N` is small everywhere, so `U` dominates and the search fans out along the prior; as visits accumulate, `Q` takes over and the search exploits what it's learned. That's the predictor-UCB rule — the "predictor" being the network's prior `P`. It's the piece that lets the search be *selective* the way a strong human is selective, concentrating on a handful of promising moves rather than expanding everything uniformly, which is why it can be devastating at a thousand simulations where uniform search would need millions.

Descend by that rule until I hit a position not yet in the tree — a leaf. Expand it: call the network once, `(p, v) = f_θ(s_L)`, mask out illegal moves, renormalise the remaining prior, and let every legal action have an edge whose count starts at zero. The value `v` is the evaluation of this leaf. Now back it up. Walk back up the path I descended, and for every edge `(s, a)` on it: `N(s,a) += 1`, `W(s,a) += v`, `Q(s,a) = W(s,a)/N(s,a)`. One subtlety in a two-player game: value is from the perspective of the side to move, and the side to move alternates every ply, so the same leaf value `v` is good for one player and bad for the other — I flip the sign of `v` at each level as I propagate it up. (Concretely, the search evaluates everything in a canonical "side-to-move" frame, and the backed-up value negates each step.) After many simulations, the visit counts `N(s_root, a)` at the root encode how much the search favored each move.

Now the claim I parked earlier: that the search output is a better policy than the network's prior `p`. Why should the visit counts be an improvement? Because the search *spends* its visits according to value. The selection rule keeps sending simulations down the high-`Q` actions and stops sending them down actions that the deeper evaluations reveal to be bad — even if the prior `p` initially liked them. So the visit distribution is the prior corrected by lookahead: moves the network overrated get their visits choked off once the search sees they lead nowhere, and moves the network underrated get more visits once the search confirms them. The lookahead is information the raw `p` didn't have. So the visit-count distribution is a policy-improvement step over `p` — it is `p` plus the correction that searching provides. That's precisely the better-than-current signal I needed for the bootstrap, and now I see *why* it's better, not just that it is.

So turn the visit counts into a policy. The obvious thing is proportional to visits, but I also want the same readout to cover greedy play when I evaluate the player. A temperature does exactly that:

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

A couple of choices about *how* the loop runs that I should pin down rather than leave vague. Do I keep a "best player so far" and only generate self-play games from it, swapping in a new network only when it proves stronger in a match? That's one way to keep the data-generating policy from regressing. But it adds a whole evaluation-and-gating machine, and it stalls the loop while the match runs. Simpler: just maintain a single network, update it continually, and always generate self-play with the latest weights — no separate best-player, no gating match. The continual update is a little less stable in principle but far simpler and keeps the data fresh, and the loss is self-correcting enough that it holds together. And do I exploit board symmetries — generate the eight rotations/reflections of each position as extra training data, or average the network over symmetric transforms during search? That's free strength when the rules are symmetric, as in Go. But in chess and shogi the rules are *not* symmetric — pawns move only forward, castling differs by side, promotion happens on a specific rank — so I cannot assume any symmetry. To keep one algorithm across all three games, I drop the symmetry augmentation entirely rather than special-case it. Same reasoning for hyperparameters: reuse one set across games, with the sole exception of the Dirichlet `α` scaled by move count, because anything else would be game-specific tuning of the kind I'm trying to avoid.

The pieces now close into one loop. I want a single network to learn a game from zero by self-play, with no human data and no rollouts, and the bootstrap requires a source of signal stronger than the current player. That source is the search: MCTS guided by the network's prior produces, at the root, a visit distribution that is a policy improvement over the prior, because the search reallocates visits according to deeper value estimates. I use MCTS specifically (not minimax) because it averages leaf evaluations and so tolerates the network's approximation errors instead of amplifying them, which lets a single dual-head network `(p, v) = f_θ(s)` supply both the leaf value (replacing rollouts and the handcrafted evaluation) and the expansion prior (replacing move-ordering heuristics). The search selects children by the predictor-UCB rule, exploits `Q` and explores in proportion to `P` and `√(Σ N)/(1+N)`, expands a leaf with one network call, backs up `v` with a per-ply sign flip and `Q = W/N`, injects Dirichlet noise at the root so no good move is starved, and reads out `π ∝ N^{1/τ}`. Self-play with `a ~ π` generates games scored by the rules to an outcome `z`, and the network is fit by `(z−v)^2 − π^T log p + c‖θ‖^2` — the squared-error term doing policy evaluation, the cross-entropy term doing policy improvement, one network chasing its own lookahead-corrected self. A single continually-updated network, the same procedure for chess, shogi, and Go, and the only domain knowledge anywhere is the rules, used by the search to step states, list legal moves, and score terminals.

Here is the core in code.

```python
import math
import numpy as np

EPS = 1e-8


class MCTS:
    """One search tree. Edges (s,a) store N, W (via running Q), and the prior P.
    The network supplies the leaf value v and the prior p; the rules step states,
    list legal moves, and score terminals."""

    def __init__(self, game, net, args):
        self.game, self.net, self.args = game, net, args
        self.Qsa, self.Nsa, self.Ns, self.Ps = {}, {}, {}, {}   # Q(s,a), N(s,a), N(s), P(s,·)
        self.Es, self.Vs = {}, {}                                # cached terminal value, legal mask

    def get_action_prob(self, board, temp=1):
        # run the simulations, then read out pi proportional to N^{1/temp}
        for _ in range(self.args.numMCTSSims):
            self.search(board)
        s = self.game.stringRepresentation(board)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]
        if temp == 0:                                            # greedy: argmax visits (tau -> 0)
            best = np.random.choice(np.flatnonzero(counts == np.max(counts)))
            probs = [0] * len(counts); probs[best] = 1
            return probs
        counts = [c ** (1.0 / temp) for c in counts]             # pi(a) = N(a)^{1/tau} / sum_b N(b)^{1/tau}
        total = float(sum(counts))
        return [c / total for c in counts]

    def search(self, board):
        s = self.game.stringRepresentation(board)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(board, 1)        # rules: terminal detection + score
        if self.Es[s] != 0:
            return -self.Es[s]                                   # terminal outcome, flipped for the parent

        if s not in self.Ps:                                     # leaf: one network call replaces a rollout
            self.Ps[s], v = self.net.predict(board)              # (p, v) = f_theta(s)
            valids = self.game.getValidMoves(board, 1)           # rules: legal move mask
            self.Ps[s] = self.Ps[s] * valids                     # zero out illegal moves in the prior
            ssum = np.sum(self.Ps[s])
            self.Ps[s] = self.Ps[s] / ssum if ssum > 0 else (valids / np.sum(valids))
            self.Vs[s], self.Ns[s] = valids, 0
            return -v                                            # back up v, flipped for the parent

        valids, best, best_a = self.Vs[s], -float("inf"), -1
        for a in range(self.game.getActionSize()):               # predictor-UCB selection
            if not valids[a]:
                continue
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] \
                    * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 at first visit
            if u > best:
                best, best_a = u, a

        a = best_a
        next_board, next_player = self.game.getNextState(board, 1, a)   # rules: step the state
        next_board = self.game.getCanonicalForm(next_board, next_player)
        v = self.search(next_board)                              # recurse to the leaf, get value (already flipped)

        if (s, a) in self.Qsa:                                   # backup: running mean Q = W/N, N += 1
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)], self.Nsa[(s, a)] = v, 1
        self.Ns[s] += 1
        return -v                                                # flip sign on the way up (two-player)
```

```python
import numpy as np


class Coach:
    """Self-play -> train. Each game is played by MCTS for both sides; the
    search policy pi is the training target, the final outcome z is the value."""

    def __init__(self, game, net, args):
        self.game, self.net, self.args = game, net, args
        self.mcts = MCTS(game, net, args)

    def execute_episode(self):
        examples, board, player, step = [], self.game.getInitBoard(), 1, 0
        while True:
            step += 1
            canonical = self.game.getCanonicalForm(board, player)
            temp = int(step < self.args.tempThreshold)           # tau=1 early (explore), tau->0 later (greedy)
            pi = self.mcts.get_action_prob(canonical, temp=temp)  # improved policy from the search
            examples.append([canonical, player, pi, None])        # store (state, player, pi); z filled in at the end
            a = np.random.choice(len(pi), p=pi)                   # sample a ~ pi
            board, player = self.game.getNextState(board, player, a)
            r = self.game.getGameEnded(board, player)             # rules: 0 if ongoing, else outcome
            if r != 0:
                # z written from each position's player's perspective
                return [(x[0], x[2], r * ((-1) ** (x[1] != player))) for x in examples]

    def learn(self):
        for _ in range(self.args.numIters):
            examples = []
            for _ in range(self.args.numEps):
                self.mcts = MCTS(self.game, self.net, self.args)  # fresh tree per game
                examples += self.execute_episode()
            self.net.train(examples)                              # fit toward (pi, z); see the loss below
            # single continually-updated network: always self-play with the latest weights, no gating match


def loss(p_logits, v, target_pi, target_z, theta, c=1e-4):
    # l = (z - v)^2 - pi^T log p + c||theta||^2
    log_p = p_logits - np.log(np.sum(np.exp(p_logits)))           # log-softmax of the policy head
    value_loss = np.mean((target_z - v) ** 2)                     # policy evaluation: v -> z
    policy_loss = -np.mean(np.sum(target_pi * log_p, axis=1))     # policy improvement: p -> pi
    l2 = c * np.sum(theta ** 2)
    return value_loss + policy_loss + l2
```

The causal chain, once more: I wanted one network to learn a game from zero by self-play with no human data and no rollouts, and the bootstrap needs a signal stronger than the current player; the search supplies it, because MCTS guided by the network's prior produces a visit distribution that is a policy improvement over that prior, and MCTS (averaging) rather than minimax (min/max) is what makes a single dual-head network usable as the leaf evaluator despite its approximation errors; so I select by predictor-UCB, expand a leaf with one network call for `(p, v)`, back up with `Q = W/N` and a per-ply sign flip, add Dirichlet noise at the root so no move is starved, read out `π ∝ N^{1/τ}`, and train the network by `(z − v)^2 − π^T log p + c‖θ‖^2` — squared error doing policy evaluation, cross-entropy doing policy improvement — so the network forever chases the lookahead-corrected version of itself, and the only domain knowledge anywhere is the rules.
