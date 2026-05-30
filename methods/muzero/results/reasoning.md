OK, let me think this through from scratch. I have an agent that, in chess and Go, plans superbly: it runs Monte-Carlo tree search, descends a tree of future positions, evaluates leaves with a network that predicts a policy prior and a value, backs the evaluations up, and the visit counts at the root give a move that is much stronger than the raw network. That search is a policy-improvement operator — the visit distribution beats the prior — and self-play against it bootstraps to superhuman strength. But every step of that search leans on a perfect simulator. To descend the tree I need to know which state follows an action; to expand a node I need its legal actions; to evaluate a leaf correctly I need to know whether it is terminal and what the outcome is. All three are the *rules of the game*. Take the rules away — hand me Atari pixels, or a robot — and the search has nothing to step through. So the question that actually matters: how do I keep this search when I don't have a simulator?

The obvious move is model-based RL: learn the dynamics, then plan against the learned model exactly as before. So let me try the textbook version and see where it breaks. I learn a model that predicts the next observation from the current observation and action — at the pixel level, an image predictor. Then I unroll that predictor inside the tree. Two things go wrong, and they're worth being precise about. First, predicting and planning over full images is computationally hopeless at the scale of a deep search — every node is a generated frame. Second, and this is the deeper one: I'm spending the model's entire capacity learning to render pixels. A faithful renderer has to get the background texture, the score font, every sprite exactly right. But almost none of that detail matters for deciding what to do. I'm forcing the network to be good at a task — reconstruction — that is not the task I care about, and the planning-relevant signal is a tiny fraction of what it's modeling. Empirically this is exactly what's observed: pixel-reconstructing model-based agents trail well-tuned model-free agents on Atari, even on data efficiency, the one axis where a model should help most. So reconstruction is the wrong objective.

Let me back up and ask what the search actually consumes at a node. Walk through one simulation. At an internal node I need: a value, to know how good the position is; a policy prior, to bias which child to explore; and, if there are intermediate rewards, the reward of the transition that got me here. To descend I need a next state — but here's the thing — I never *look* at that state directly. I only ever feed it back into the same three predictors at the next node. The state is purely an intermediate object that carries information from one node's predictions to the next. The search never decodes it into an observation. It never checks it against the real environment. So why am I demanding the state reconstruct anything?

That reframing is the whole game. If the only things the search reads off a state are (value, policy, reward), then the only thing I need the model to get right is: when I roll it forward under a sequence of hypothetical actions, the (value, policy, reward) it produces at each step should match what would actually be true after those actions in the real environment. Nothing else. The internal state need not equal the true environment state. It need not let me reconstruct the observation. It need not "mean" anything at all. It is just the hidden activations of a recurrent network, free to encode whatever representation makes the next prediction accurate. The agent can invent, internally, whatever dynamics make its planning come out right.

This is the value-equivalence idea, and there's prior work circling it. The Predictron builds an abstract MDP as a hidden layer of a network and trains it so its rolled-out cumulative reward matches the real value by TD — but it only predicts value, with no actions, so I can't search with it. Value prediction networks go further: an MDP model grounded in real actions, unrolled, trained so the cumulative reward conditioned on a simple lookahead's actions matches the environment. Closer. But it predicts no policy. And without a policy prediction I cannot run the AlphaZero-style search whose visit counts give me policy improvement — the very engine I'm trying to keep. So the gap is clear: I need a value-equivalent learned model that *also* predicts a policy, so the strong search survives.

So let me design the model around the three quantities the node needs. I'll factor it into three learned functions. A representation function turns the past observations into an initial internal state: `s^0 = h_θ(o_1, …, o_t)`. A dynamics function is the recurrence — given the current internal state and a hypothetical action it produces the immediate reward and the next internal state: `r^k, s^k = g_θ(s^{k-1}, a^k)`. And a prediction function reads a policy and value off an internal state: `p^k, v^k = f_θ(s^k)`. The dynamics function deliberately mirrors the shape of an MDP — state, action in; reward, next state out — so that any MDP planner, including MCTS, slots straight onto it. I'll keep it deterministic for now; stochastic transitions are a complication I don't need to plan well in these domains. Composed, the model is `μ_θ(o_1,…,o_t, a^1,…,a^k) → (p^k, v^k, r^k)`: feed in the real past and a hypothetical action sequence, get out the predicted policy, value, and reward after that sequence. That's everything the search reads.

Now run the search in this latent space. It's the same MCTS shape as before, but every place that used to call the simulator now calls the model. Each edge `(s, a)` of the tree stores statistics `{N, Q, P, R, S}` — visit count, mean value, prior, reward, and the cached next state. A simulation starts at the root state `s^0` and descends. To pick the action at each node I use the predictor-UCB rule, the same one the rules-based search used:

```
a^k = argmax_a [ Q(s,a) + P(s,a) · (√Σ_b N(s,b) / (1 + N(s,a))) · (c1 + log((Σ_b N(s,b) + c2 + 1)/c2)) ]
```

with `c1 = 1.25`, `c2 = 19652`. The first term exploits the value estimate; the second is the exploration bonus, weighted by the prior and shrinking as a child is visited more — the log term lets the prior keep mattering even at very high visit counts. While I'm descending through already-expanded nodes I just look up the cached transition and reward, `s^k = S(s^{k-1}, a^k)`, `r^k = R(s^{k-1}, a^k)`. When I reach a leaf I expand it: call the dynamics function for the reward and next state, `r^l, s^l = g_θ(s^{l-1}, a^l)`, cache them, then call the prediction function `p^l, v^l = f_θ(s^l)` to get the policy prior and value at the new node, and initialize each child edge with `N=0, Q=0, P=p^l`. Exactly one call to `g` and one to `f` per simulation — same order of compute per node as the rules-based search, so I haven't made the search more expensive.

Three places where the missing simulator forces a change, and I should handle each honestly rather than pretend the rules are still there. State transitions: there's no perfect transition anymore, so I use `g_θ`'s learned transition — that's the whole point. Legal actions: the old search masked the prior to legal moves at every node using the rules. Inside the tree I can't query the environment for legality, so I can only mask at the *root*, where the real environment is actually present. Everywhere deeper I let the network propose over all actions and trust it to learn near-zero prior on actions that never occur in real trajectories — it sees only legal play during training, so illegal actions simply get no support. Terminal states: the old search stopped at terminals and used the simulator's exact outcome. I have no terminal detector inside the tree, so I give terminals no special treatment — the search may even step *past* the end of an episode. To make that benign I train terminal states as absorbing: the network learns to predict the same value forever once the episode is over, so passing through a terminal node doesn't inject garbage.

Now the backup, and here I have to generalize past the two-player, undiscounted, ±1-terminal setting the rules-based search was built for. I want intermediate rewards, a discount `γ` that may not be 1, and value estimates that are *unbounded* (Atari scores, not a win/draw/loss). Backing up from a leaf at depth `l`, for each node at depth `k` along the path I form the `(l−k)`-step discounted return that bootstraps off the leaf value:

```
G^k = Σ_{τ=0}^{l-1-k} γ^τ · r_{k+1+τ}  +  γ^{l-k} · v^l
```

and fold it into the running mean at the edge above:

```
Q(s^{k-1}, a^k) := (N · Q + G^k) / (N + 1),    N := N + 1.
```

In a two-player game the value flips sign each ply, as usual. There's a snag in the UCB rule once values are unbounded: the rule combines `Q` with a prior-weighted bonus, which only makes sense if `Q` lives on a known scale — in the old two-player setting `Q ∈ [0,1]` and that's fine. With arbitrary-magnitude returns I have no fixed scale, and I refuse to hand the algorithm the environment's max score as prior knowledge, because that would be smuggling domain knowledge back in. So I normalize `Q` on the fly using the minimum and maximum values seen anywhere in the tree so far:

```
Q̄(s,a) = (Q(s,a) − min_{tree} Q) / (max_{tree} Q − min_{tree} Q)
```

and feed `Q̄ ∈ [0,1]` into the UCB rule. Self-calibrating, no prior knowledge.

The search outputs what it always did: a visit-count distribution `π_t` over root actions and a root value `ν_t`. I sample the real action `a_{t+1} ~ π_t` and step the actual environment, which hands me a real observation and a real reward. That's how I act.

Now the part that makes or breaks everything: how do I train `h`, `g`, `f` so that this latent search is actually good? The model has no reconstruction loss to anchor it, so the *only* thing that shapes the internal states is whether the three predictions match real targets at every unrolled step. Let me figure out each target by asking what the prediction should have been, in hindsight, after `k` real steps actually elapsed.

The policy prediction `p^k_t` at hypothetical step `k` should match the policy that real play used at time `t+k`. And what is the best policy I have at any time step? Not the raw network prior — the *search* output. The search is a policy-improvement operator: its visit-count distribution is provably an improvement over the prior it started from. So the policy target is `π_{t+k}`, the visit-count distribution produced by the search at time `t+k`. Training the prior toward the searched policy is what closes the self-improvement loop — the network chases its own improved behavior, and the improved behavior gets better as the network does.

The reward prediction `r^k_t` is the easy one: it should match the reward that was actually observed at that step, `u_{t+k}`. Observed scalar, nothing to estimate.

The value prediction `v^k_t` is the subtle one. In the rules-based, board-game world the value target was just the final game outcome `z ∈ {−1,0,+1}` — you play to the end and learn toward who won. But Atari episodes are tens of thousands of steps long, discounted, with reward all along the way. Waiting for the final return as the target would be absurdly high-variance over that horizon. So I bootstrap: take the actual rewards for `n` steps and then bootstrap with a value estimate `n` steps out. Which value estimate? I have two candidates — the raw network value `v`, or the *search* value `ν` from running MCTS at that future state. The search value is the improved estimate, the same way the visit counts are an improved policy; bootstrapping off `ν` gives a stronger, lower-bias target than bootstrapping off the raw `v`, and far lower variance than the full return. So the value target is

```
z_t = u_{t+1} + γ u_{t+2} + … + γ^{n-1} u_{t+n} + γ^n ν_{t+n}.
```

For board games this collapses cleanly: `γ = 1`, no intermediate rewards, and I let `n` run to the end of the game, so `z_t` is exactly the final outcome — the old target falls out as a special case. For Atari I use `n = 10`. (And there's a nice contrast hiding here, which I want to check rather than assume. The model-free alternative would be to train a Q-function directly on a bootstrapped return and skip the search entirely. But that target is both high-bias and high-variance — it inherits whatever the behavior policy happened to do. Bootstrapping off the *search* value instead routes the learning signal through the policy-improvement operator. I'd expect that to be a substantially stronger signal; it's the difference between learning toward "what I did" and learning toward "what searching says I should do.")

So I have three targets at every hypothetical step `k`: policy `π_{t+k}`, value `z_{t+k}`, reward `u_{t+k}`. The loss just sums the per-quantity errors over the `K` unrolled steps, plus L2:

```
l_t(θ) = Σ_{k=0}^{K} [ l^r(u_{t+k}, r^k_t) + l^v(z_{t+k}, v^k_t) + l^p(π_{t+k}, p^k_t) ] + c‖θ‖².
```

For the per-term losses, policy is always cross-entropy, `l^p(π,p) = π^T log p`. For value and reward I have a choice, and the scale of the quantity decides it. In board games value sits in `[−1,1]` and there's no intermediate reward, so squared error `(z−q)^2` for value and zero reward loss is fine. In Atari, rewards and values span many orders of magnitude across the 57 games and within a single game, and a squared-error regression onto a target that ranges over thousands is unstable — the gradients are dominated by the largest-scale games. So I borrow the trick of representing each scalar categorically. First squash with an invertible transform `h(x) = sign(x)(√(|x|+1) − 1) + εx`, `ε = 0.001`, which compresses large magnitudes while staying invertible. Then represent the squashed scalar as a distribution over a fixed discrete support — integers from `−300` to `300`, size 601 — by putting the weight on the two nearest integers (a target of `3.7` becomes `0.3` on `3` and `0.7` on `4`). Train it with cross-entropy, `l^v(z,q) = φ(z)^T log q` and likewise for reward. At inference I take the expectation under the softmax to get a scalar back and invert the squash. Cross-entropy over a bounded support is stable regardless of the underlying scale; that's why it beats MSE here.

Now I have to be careful about how gradients flow, because the model is unrolled `K` steps and trained by backprop through time, and BPTT through a recurrence is where magnitudes silently blow up or vanish. There are `K+1` predictions all feeding the same shared parameters, so the total gradient grows with how far I unroll; scaling every head's loss by `1/K` keeps the gradient magnitude roughly constant in `K`. And the dynamics function is applied repeatedly, so gradient from every downstream step piles onto it through the chain — halving the gradient *entering* the dynamics function at each recurrent step keeps the total gradient delivered to `g` roughly constant across the unroll rather than compounding. One more stabilizer: I min-max scale the internal state into `[0,1]` after every application of `h` and `g`. That bounds the activations the recurrence carries and puts the state on the same range as the action encoding I concatenate in, which keeps the whole BPTT well-behaved. None of these change what's being optimized; they just keep the recurrent optimization conditioned.

Let me settle the remaining concrete choices, because each is a place I could get it wrong. The networks reuse the residual-convolutional architecture that worked for the rules-based agent — `h` and `g` share that body (16 residual blocks, 256 planes; slightly leaner than the 20 blocks before, because `g` is called many times per search and I want each call cheap), and `f` is the same policy-value head. The dynamics function concatenates the action, encoded as planes at the state's spatial resolution, onto the state. For Atari the representation function takes the last 32 frames *and* the last 32 actions — I include actions because, unlike a board game, an Atari action often has no immediately visible effect on the screen, so the history of what I pressed is genuine information — and downsamples the 96×96 input through strided convolutions and pooling down to a 6×6 latent, so the latent search is tractable. For data sampling I prioritize replay by `|ν_i − z_i|`, the gap between the search value and the observed return at a position — the positions where the search and the eventual return disagree are the ones the value function still has the most to learn from.

Let me re-derive the whole causal chain once, fast, to make sure it closes. I want the search but I have no simulator. The search only ever reads value, policy, and reward off a node and never decodes the state, so I don't need a model of observations — I need a model whose *predictions* are right under unrolling, which is value-equivalence, and reconstruction is the wrong objective because it wastes capacity. So I learn three functions — `h` to enter the latent space, `g` to step it, `f` to read off `(p, v)` — and run the same MCTS in that latent space, with learned transitions instead of the rules, masking only at the root and treating terminals as absorbing because I can't query the environment inside the tree, and normalizing `Q` by the tree's min/max because values are now unbounded. The search emits an improved policy (visit counts) and an improved value, and those *are* my training targets: the policy target is the visit distribution, the value target is an `n`-step return bootstrapped off the search value (collapsing to the final outcome for board games), the reward target is the observed reward. Summing those three losses over `K` unrolled steps, with `1/K` and `1/2` gradient scalings and a bounded-support categorical value/reward loss to survive Atari's scales, trains `h`, `g`, `f` jointly end-to-end — and because the only pressure on the latent state is to make those three predictions match, the state organizes itself into exactly whatever supports good planning, with no instruction to be anything else.

Here is the model, grounded in a working implementation.

```python
import torch

# h, g, f as one module. Value and reward are categorical over a support of
# size 2*support_size+1; support_to_scalar/scalar_to_support convert and apply
# the invertible squash. State is min-max scaled to [0,1] after h and g.

def support_to_scalar(logits, support_size):
    probs = torch.softmax(logits, dim=1)
    support = torch.arange(-support_size, support_size + 1).float().to(logits.device)
    x = torch.sum(support * probs, dim=1, keepdim=True)
    # invert h(x) = sign(x)(sqrt(|x|+1) - 1) + eps*x
    eps = 0.001
    x = torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)
    return x

def scalar_to_support(x, support_size):
    eps = 0.001
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x  # squash
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor(); prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1, device=x.device)
    logits.scatter_(2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1))
    idx = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < idx, 0.0)
    idx = idx.masked_fill_(2 * support_size < idx, 0.0)
    logits.scatter_(2, idx.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits


def scale_to_01(s):  # bound the latent state; matches action-input range
    smin = s.min(1, keepdim=True)[0]; smax = s.max(1, keepdim=True)[0]
    scale = (smax - smin).clamp_min(1e-5)
    return (s - smin) / scale


class MuZeroNet(torch.nn.Module):
    def __init__(self, obs_dim, n_actions, enc=64, support=300):
        super().__init__()
        self.n_actions = n_actions
        self.full_support = 2 * support + 1
        # h: o_1..o_t -> s^0
        self.h = torch.nn.Sequential(torch.nn.Linear(obs_dim, enc), torch.nn.ELU(),
                                     torch.nn.Linear(enc, enc))
        # g: (s^{k-1}, a^k) -> s^k  (action one-hot concatenated), plus reward head
        self.g_state = torch.nn.Sequential(torch.nn.Linear(enc + n_actions, enc), torch.nn.ELU(),
                                           torch.nn.Linear(enc, enc))
        self.g_reward = torch.nn.Linear(enc, self.full_support)
        # f: s^k -> (policy logits, value)
        self.f_policy = torch.nn.Linear(enc, n_actions)
        self.f_value = torch.nn.Linear(enc, self.full_support)

    def initial_inference(self, observation):     # representation + prediction
        s = scale_to_01(self.h(observation))
        policy, value = self.f_policy(s), self.f_value(s)
        reward = scalar_to_support(torch.zeros(len(observation), 1).to(s.device), (self.full_support - 1)//2).squeeze(1)
        return value, reward, policy, s

    def recurrent_inference(self, s, action):     # dynamics + prediction
        a = torch.zeros(s.shape[0], self.n_actions, device=s.device)
        a.scatter_(1, action.long(), 1.0)
        s2 = scale_to_01(self.g_state(torch.cat([s, a], dim=1)))
        reward = self.g_reward(s2)
        policy, value = self.f_policy(s2), self.f_value(s2)
        return value, reward, policy, s2
```

```python
import math, numpy

class MinMaxStats:                 # self-calibrating Q scale for unbounded values
    def __init__(self): self.lo, self.hi = float("inf"), -float("inf")
    def update(self, v): self.lo, self.hi = min(self.lo, v), max(self.hi, v)
    def normalize(self, v): return (v - self.lo) / (self.hi - self.lo) if self.hi > self.lo else v

class Node:
    def __init__(self, prior):
        self.N, self.value_sum, self.prior = 0, 0.0, prior
        self.reward, self.state, self.children = 0.0, None, {}
    def expanded(self): return len(self.children) > 0
    def value(self): return 0 if self.N == 0 else self.value_sum / self.N
    def expand(self, reward, state, policy_logits, actions):
        self.reward, self.state = reward, state
        ps = torch.softmax(torch.tensor([policy_logits[0][a] for a in actions]), 0).tolist()
        for a, p in zip(actions, ps): self.children[a] = Node(p)


def ucb(parent, child, mm, c1=1.25, c2=19652, discount=0.997):
    pb_c = (math.log((parent.N + c2 + 1) / c2) + c1) * math.sqrt(parent.N) / (child.N + 1)
    prior_score = pb_c * child.prior
    value_score = mm.normalize(child.reward + discount * child.value()) if child.N > 0 else 0
    return prior_score + value_score


def run_mcts(model, observation, legal_actions, num_simulations=50, discount=0.997):
    root = Node(0)
    value, reward, policy_logits, state = model.initial_inference(observation)
    root.expand(support_to_scalar(reward, 300).item(), state, policy_logits, legal_actions)  # legal mask: root only
    mm = MinMaxStats()
    for _ in range(num_simulations):
        node, search_path = root, [root]
        while node.expanded():                                  # descend by pUCT
            action, node = max(node.children.items(), key=lambda kv: ucb(search_path[-1], kv[1], mm))
            search_path.append(node)
        parent = search_path[-2]
        value, reward, policy_logits, state = model.recurrent_inference(   # expand leaf via g + f
            parent.state, torch.tensor([[action]]))
        node.expand(support_to_scalar(reward, 300).item(), state, policy_logits, list(range(model.n_actions)))
        value = support_to_scalar(value, 300).item()
        for n in reversed(search_path):                         # backup with reward + discount
            n.value_sum += value; n.N += 1
            mm.update(n.reward + discount * n.value())
            value = n.reward + discount * value
    visits = numpy.array([c.N for c in root.children.values()], dtype="float32")
    pi = visits / visits.sum()                                  # improved policy = visit counts
    return root, dict(zip(root.children.keys(), pi)), root.value()  # also the search value nu
```

```python
def compute_target_value(root_values, rewards, index, td_steps=10, discount=0.997):
    # n-step return bootstrapped from the SEARCH value, not the raw net value
    b = index + td_steps
    value = root_values[b] * discount ** td_steps if b < len(root_values) else 0.0
    for i, r in enumerate(rewards[index + 1: b + 1]):
        value += r * discount ** i
    return value


def loss_function(value, reward, policy_logits, t_value, t_reward, t_policy):
    # cross-entropy for all three (categorical value/reward survive Atari scales)
    lv = (-t_value * torch.nn.LogSoftmax(1)(value)).sum(1)
    lr = (-t_reward * torch.nn.LogSoftmax(1)(reward)).sum(1)
    lp = (-t_policy * torch.nn.LogSoftmax(1)(policy_logits)).sum(1)
    return lv, lr, lp


def update_weights(model, optimizer, batch, K, support=300):
    obs, actions, target_v, target_r, target_pi = batch
    target_v = scalar_to_support(target_v, support); target_r = scalar_to_support(target_r, support)

    value, reward, policy, state = model.initial_inference(obs)     # k = 0
    preds = [(value, reward, policy)]
    for k in range(1, actions.shape[1]):                            # unroll K steps via g
        value, reward, policy, state = model.recurrent_inference(state, actions[:, k])
        state.register_hook(lambda grad: grad * 0.5)                # half gradient into dynamics
        preds.append((value, reward, policy))

    v_loss = r_loss = p_loss = 0
    for k, (value, reward, policy) in enumerate(preds):
        lv, lr, lp = loss_function(value, reward, policy, target_v[:, k], target_r[:, k], target_pi[:, k])
        if k == 0: lr = torch.zeros_like(lr)                        # no reward target at the root step
        v_loss = v_loss + lv; r_loss = r_loss + lr; p_loss = p_loss + lp
    loss = (v_loss + r_loss + p_loss).mean() / K                    # 1/K keeps gradient ~constant in K
    optimizer.zero_grad(); loss.backward(); optimizer.step()        # weight_decay = L2 term c||theta||^2
    return loss.item()
```

The causal chain, once more: I wanted to keep tree search but had no simulator; the search only reads value, policy, and reward at a node and never decodes the state, so I learned a value-equivalent model — three functions `h`, `g`, `f` with no reconstruction loss — ran MCTS in its latent space with learned transitions, root-only legal masking, absorbing terminals, and min-max-normalized `Q`, then trained `h`, `g`, `f` end-to-end against the search's own improved outputs: visit-count policy, search-bootstrapped `n`-step value, observed reward, summed over `K` unrolled steps. The model is pulled toward nothing except making those three predictions correct, so its internal states become whatever representation makes planning come out right.
