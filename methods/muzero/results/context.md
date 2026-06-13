# Context

## Research question

Two families of decision-making agents dominate, and each owns a different kind of domain. On one side, tree-based lookahead planning — running a search over future move sequences — has produced agents that defeat human world champions in checkers, chess, shogi, Go, and poker, and drives real applications from logistics to chemical synthesis. On the other side, model-free reinforcement learning estimates a value function or policy directly from interaction and is state of the art on visually complex domains such as the Atari arcade games.

Each family has a hard boundary. Planning agents need an exact model of the environment — the rules of the game, or an accurate simulator — to expand the search tree: to know what state follows an action, which actions are legal, and when an episode ends. That requirement bars them from any domain where the dynamics are complex, unknown, or only observed through raw sensors (robotics, industrial control, a screen of pixels). Model-free agents need no such model, but they cannot do the precise, deep lookahead that wins in chess and Go; they react rather than plan.

The question is whether one agent can have both: the sample-efficient, superhuman lookahead of a planner, in domains where no simulator is given and the only inputs are raw observations. A solution would have to *learn* whatever model the search needs, from interaction alone, and learn it well enough that planning inside the learned model is as useful as planning inside the true environment.

## Background

The standard way to add planning when the model is unknown is model-based reinforcement learning: first learn a model of the environment's dynamics, then plan against it. Classically the model is a Markov decision process (MDP) — a state-transition model predicting the next state and a reward model predicting the reward of a transition, conditioned on the chosen action. Once you have such a model, off-the-shelf MDP planners apply: value iteration, or Monte-Carlo tree search.

There is a structural problem with this recipe in large or partially observed environments. The agent must first build a state representation, then learn a model that predicts in that representation, then plan. These three things — representation learning, model learning, planning — are trained separately, by different objectives. The model is fit to be an accurate model, not to be a *useful* one for the search that will consume it. Modeling errors that are harmless for one-step prediction can compound catastrophically as the planner unrolls the model many steps deep.

A second, empirically observed problem concerns *what* the learned model is asked to predict. The most common approach grounds the model in the observation stream: it learns to reconstruct the next image at the pixel level, or to predict a latent state from which the pixels can be reconstructed. The reported fact about these systems is that on visually rich domains they remain far from the best model-free agents — even on data efficiency, where a learned model should help most. Two reasons are visible in the measurements. Planning at pixel granularity is computationally intractable at scale. And forcing the latent state to be sufficient to reconstruct the observation spends the majority of the model's capacity on detail that is irrelevant to deciding what to do — a faithful pixel predictor is not the same thing as a good planner.

A different line of work attacks both problems at once by changing the training objective of the model. Instead of asking the model to reproduce observations, it asks the model to be *value-equivalent*: an abstract MDP is constructed and trained so that planning inside it yields the same cumulative reward, or the same value, as planning in the real environment. The abstract model's transition function is under no obligation to match real states; it is treated as a hidden layer of a deep network, and the network is trained end-to-end so that its predicted values match the real ones, e.g. by temporal-difference learning. The diagnostic appeal is that the model is now optimized for exactly the quantity the planner needs, so there is no separation between model learning and planning, and no capacity wasted on reconstruction.

The load-bearing technical pieces this rests on:

- **Monte-Carlo tree search (MCTS)** (Coulom, 2006; Kocsis & Szepesvári, 2006; Rosin's predictor-UCB, 2011). Repeatedly simulate trajectories from the root, descending the tree by an upper-confidence rule that balances a value estimate against an exploration bonus, expanding a leaf, evaluating it, and backing the evaluation up the path. It converges asymptotically to the optimal action in single-agent domains and to the minimax value in zero-sum games. It needs a generative model to step through.

- **n-step bootstrapping** (Sutton). A value target can interpolate between the Monte-Carlo return and a one-step temporal-difference target: sum the first *n* discounted rewards and then bootstrap with a value estimate, `z = u_{t+1} + γu_{t+2} + … + γ^{n-1}u_{t+n} + γ^n V(s_{t+n})`. Small *n* is low-variance but biased by the value estimate; large *n* is unbiased but high-variance.

- **Categorical value/reward representation with value rescaling** (Pohlen et al., 2018). When the targets a network must regress span many orders of magnitude, regressing the scalar directly is unstable. Apply an invertible squashing transform `h(x) = sign(x)(√(|x|+1) − 1) + εx` to compress scale, and represent the squashed scalar as a probability distribution over a fixed discrete support, training it with cross-entropy. The scalar is recovered as the expectation under the distribution, then unsquashed.

## Baselines

**AlphaZero / AlphaGo Zero** (Silver et al., 2017, 2018) — the planning agent to beat in its own domains. It interleaves MCTS with self-play. A single network `f(s) → (p, v)` predicts, for a board position `s`, a policy prior `p` over moves and a scalar value `v`. The search uses a perfect simulator of the game in three places: to compute state transitions while descending the tree, to obtain the set of legal actions at each node (used to mask the prior), and to detect terminal positions (where the simulator's exact outcome replaces the network value). Selection uses a predictor-UCB rule,

```
a = argmax_a [ Q(s,a) + P(s,a) · (√Σ_b N(s,b) / (1 + N(s,a))) · (c1 + log((Σ_b N(s,b) + c2 + 1)/c2)) ]
```

The search returns a visit-count distribution over root actions and a root value. Training targets: the policy is trained toward the search's visit-count distribution (the search is a policy-improvement operator over the prior), and the value toward the final game outcome `z ∈ {−1, 0, +1}`, with a squared-error value loss and a cross-entropy policy loss, plus L2 regularization. The gap it leaves: every one of its three uses of the simulator is a use of the *rules*. Remove the rules and the search cannot run at all. It is also specialized to two-player, zero-sum, undiscounted games with a single terminal reward — no intermediate rewards, no discount, no single-agent setting.

**Model-free value learning — DQN and its descendants** (Mnih et al., 2015; and Ape-X, R2D2, IMPALA, Rainbow). Learn the optimal action-value function `Q(s,a)` directly from interaction, with no model: bootstrap an n-step return onto a target-network value, store transitions in a replay buffer, sample (with prioritization in later variants), and act greedily/ε-greedily. These hold the state of the art on Atari. The gap: there is no lookahead. The agent's improvement signal is the bootstrapped return, which is simultaneously high-bias (it trusts a possibly-wrong value estimate) and high-variance (it follows whatever the behavior policy did), and in domains demanding precise multi-step calculation the absence of search shows.

**Observation-reconstructing model-based RL** (PlaNet, Hafner et al.; SimPLe, Kaiser et al.; world models, Ha & Schmidhuber; embed-to-control). Learn a latent dynamics model trained to reconstruct or predict the observation, then plan in it. The gap, as measured on Atari: they trail well-tuned model-free agents, even on data efficiency, because pixel-level planning is intractable and reconstruction spends capacity on planning-irrelevant detail.

**Value-equivalent models.** The Predictron (Silver et al., 2017) learns an abstract MDP, trained by TD so its rolled-out cumulative reward matches the real value — but it predicts value only, with no actions. Value iteration networks (Tamar et al., 2016) and TreeQN (Farquhar et al., 2018) learn local/abstract MDPs whose internal planning approximates the optimal value. Value prediction networks (Oh et al., 2017) are the closest: an MDP model grounded in real actions, unrolled, trained so the cumulative reward conditioned on the actions of a simple lookahead matches the environment. The gap each leaves: their lookahead is driven by value alone — none of them predicts a *policy*.

## Evaluation settings

The natural yardsticks predate any new agent. For precision planning: Go (19×19), chess, and shogi, with strength measured by Elo from head-to-head games against a strong baseline player at matched thinking time, ratings fit by Bayesian logistic regression. For visually complex control: the 57 games of the Arcade Learning Environment, observations as raw RGB frames, with the standard 30-minute / 108,000-frame episode cap, action repeat of 4, and two stochasticity-mitigation protocols — random no-op starts (0–30 no-ops before handing control to the agent) and human-expert start positions. The Atari summary metric is the human-normalized score `(s_agent − s_random)/(s_human − s_random)`, aggregated as mean and median across the 57 games. Search budgets are reported as simulations per move.

## Code framework

The primitives that already exist: a deep-net library with convolutional and residual blocks, SGD with momentum and weight decay, an experience replay buffer, and a self-play / training split where actors generate trajectories with the current network and a learner updates the network from sampled trajectories. MCTS exists as a generic tree search given some way to step a state forward and to evaluate a node.

What does *not* exist yet is the model that the search will step through, and the way that model is trained. The scaffold leaves those as empty slots.

```python
# --- the model the agent will plan with: empty slots ---
class Network:
    def initial_inference(self, observation):
        # observation(s) -> (..., internal_state)
        # TODO: turn raw observations into the object the search starts from,
        #       and read off whatever quantities the search needs at a node.
        raise NotImplementedError

    def recurrent_inference(self, internal_state, action):
        # (internal_state, action) -> (..., next_internal_state)
        # TODO: step the internal state forward under a hypothetical action,
        #       and read off whatever quantities the search needs at the new node.
        raise NotImplementedError


# --- planning given the model ---
def run_search(config, root_observation, legal_actions, network):
    # TODO: build a tree whose nodes are internal states; descend by an
    #       upper-confidence rule using the per-node quantities; expand a leaf
    #       by stepping the model; back up the evaluation.
    #       Return a recommended distribution over actions and a root value.
    raise NotImplementedError


# --- targets the model is trained to match ---
def make_targets(trajectory, index, num_unroll_steps):
    # TODO: for each of the next num_unroll_steps+1 positions, produce the
    #       targets the model's predictions are compared against.
    raise NotImplementedError


def loss_function(predictions, targets):
    # TODO: compare each predicted quantity to its target and sum over the
    #       unrolled steps; add regularization.
    raise NotImplementedError


# --- already-standard training loop ---
def train(config, replay_buffer, network, optimizer):
    for _ in range(config.training_steps):
        batch = replay_buffer.sample_batch(config.num_unroll_steps)
        for observation, actions, targets in batch:
            predictions = unroll(network, observation, actions)  # initial + recurrent
            loss = loss_function(predictions, targets)
            optimizer.minimize(loss)
```
