## Research question

Value-based reinforcement learning on the full Atari suite from raw pixels: one agent, one set of
hyperparameters, scored across all 57 Atari 2600 games in the Arcade Learning Environment (ALE) under
the standard **no-ops** evaluation regime, after $200$M frames of training. Performance is reported as the
**median human-normalized score (HNS)** over the 57 games, where a game's HNS is
$100\times\frac{\text{agent}-\text{random}}{\text{human}-\text{random}}$, so $0\%$ is random play and
$100\%$ is the human reference. The median (not the mean) is the headline number precisely because a few
games on which agents reach thousands of percent would otherwise dominate a mean; the median asks instead
*on a typical game, how close to human is this agent.*

The thing being designed is the **deep Q-learning agent itself** — how the action-value (or the return)
is represented, how the bootstrap target is built, how transitions are sampled from replay, and how the
agent explores. Everything coarse-grained around that is held fixed: the ALE games and their action sets,
the $84\times84$ grayscale 4-frame-stack preprocessing, the $200$M-frame budget, the no-ops evaluation
protocol, and the single-agent/single-hyperparameter constraint (no per-game tuning). The free variable
across the rungs of this ladder is exactly one question: **which modification to the deep Q-learning
recipe should I add, and what does it buy on the median game.**

## Prior art before the first rung

The agents this ladder reacts to and builds from, each with the gap it leaves open:

- **Neural fitted Q / online Q-learning from pixels.** Bootstrapped value learning with a function
  approximator on correlated, online samples is unstable: consecutive frames are highly correlated, the
  data distribution shifts as the policy changes, and the regression target moves every step because it
  is computed from the same weights being updated. Training oscillates or diverges. Gap: nothing decouples
  the samples from each other or the target from the online weights, so the optimization that works for
  supervised learning does not transfer.

- **Tabular Q-learning and its overestimation.** Watkins' Q-learning is the off-policy control algorithm
  whose max-target makes learning from replayed (off-policy) data sound. But even in the tabular setting
  the $\max$ over noisy action values is biased upward (van Hasselt, 2010, noted this and proposed keeping
  *two* value tables, one to select and one to evaluate). Gap: the bias is a property of the $\max$
  estimator, and it does not go away — it gets worse — when the values come from a function approximator
  whose errors are large and correlated across states.

- **Exploration by $\epsilon$-greedy dithering.** The standard exploration rule injects randomness at the
  action output: with probability $\epsilon$ act uniformly at random, otherwise greedily. Gap: this is
  unstructured, state-independent noise — a fresh coin flip every step — and on games that need a
  coordinated sequence of exploratory actions before any reward appears, independent action noise is a
  poor way to discover that sequence. $\epsilon$ is also a hand-set schedule, the same in well-known and
  unknown states alike.

- **The scalar value object.** Across all of the above, the learned object is a single number per
  state-action, $Q(s,a)=\mathbb{E}[Z(s,a)]$ — the *mean* of the return. Gap: it discards the shape of the
  return distribution (its spread, skew, multimodality from stochastic rewards/transitions and a moving
  policy), and a single scalar target with no notion of its own uncertainty must absorb all of that
  stochasticity into one wobbly number.

These set up the ladder. Each rung asks which *one* of these open gaps — instability, overestimation,
exploration, the way replay samples, the architecture of the head, or the scalar value object — to close
next, measured by what it does to the median human-normalized score across all 57 games. The final rung
asks whether the gaps are independent enough that closing *all* of them in a single agent compounds.

## The fixed evaluation

- **Benchmark.** All 57 games of the Arcade Learning Environment (the "Atari-57" suite). One agent
  architecture and one set of hyperparameters are used for every game; there is no per-game tuning. This
  single-agent constraint is the whole point of the benchmark — it measures general competence, not a
  collection of specialists.
- **Preprocessing.** Standard DQN Atari front end: RGB$\to$grayscale, downsample to $84\times84$, max-pool
  over the two most recent emulator frames (to remove flicker), frame-skip $4$ (repeat each chosen action
  for $4$ emulator frames), stack the last $4$ processed frames as the state, and clip rewards during
  training. Terminal-on-life-loss may be used during training; evaluation uses true episode termination.
- **Budget and protocol.** $200$M training frames ($50$M agent steps at frame-skip $4$). Evaluation in the
  **no-ops** regime: each evaluation episode begins with a random number (up to $30$) of no-op actions to
  randomize the start state, then the greedy (or noisy-greedy) policy is rolled out to true termination.
- **Metric.** For each game, human-normalized score
  $\text{HNS}=100\times\frac{\text{score}_{\text{agent}}-\text{score}_{\text{random}}}{\text{score}_{\text{human}}-\text{score}_{\text{random}}}$.
  The reported number for an agent is the **median HNS over the 57 games** (higher is better). Each rung's
  feedback is this single median-HNS percentage.

## The editable agent

Everything inside the deep Q-learning agent is on the table, one axis at a time:

- the **representation** of the value (what object the head predicts per action);
- the **bootstrap target** (how the next-state value is selected and evaluated, and over how many steps);
- the **head architecture** (how the encoder features are mapped to per-action outputs);
- the **replay sampling** distribution (how transitions are drawn from the buffer for each update);
- the **exploration** mechanism (how the agent deviates from the greedy action to gather data).

The shared substrate that every rung keeps unless it explicitly changes it: a convolutional encoder over
the $84\times84\times4$ input feeding a value head, a large experience-replay buffer, a periodically
synced target network, the Bellman/TD update, and discount $\gamma=0.99$. The starting point of the
ladder is the bare deep Q-learning loop — replay + a frozen target copy + a scalar $Q$-head trained by the
squared (clipped) TD error — and each rung changes exactly one of the axes above and re-measures the
median HNS over all 57 games.
