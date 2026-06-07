# Initial context: intrinsic exploration for sparse-reward Atari

## Research question

I want an agent that learns on the hard-exploration Atari games — the sparse-reward ones where a
positive reward can be hundreds of steps away and a plain policy almost never stumbles into one. The
three games in front of me span the difficulty range: **Tutankham** (medium, rewards reachable but
not dense), **Frostbite** (hard exploration), and **Private Eye** (the hardest — long-horizon,
deceptive, with large negative rewards available for wrong moves). Across all three I am scored by
the same metrics: `eval_return` (mean evaluation episodic return at the fixed training budget),
`auc` (area under the evaluation-return curve over training, so *how fast* as well as *how high*),
and `nonzero_rate` (the fraction of evaluation episodes that score anything at all — a blunt "did it
ever find the reward" signal). Higher is better on all three, and the bar I hold myself to is that a
method should help across *multiple* games, not buy one game at the cost of another.

The whole experiment is deliberately narrow so the exploration question is isolated. The PPO
training loop is **fixed** — the rollout collection, the clipped-surrogate update, the
policy/value architecture, the optimizer, and the Atari preprocessing (grayscale $84\times84$,
frame-skip, 4-frame stack, terminal-on-life-loss) are all frozen. The *only* thing I get to design
is an **intrinsic-bonus module**: how a per-transition novelty signal is computed, how it is
normalized, how its own networks (if any) are trained, and how the resulting intrinsic advantage is
mixed with the extrinsic advantage. Everything I try plugs into one fixed interface — a module that
computes a bonus, normalizes the rollout's intrinsic rewards, and exposes a training loss, plus a
function that combines the two advantage streams.

## Why start with no bonus at all

Before I add any machinery I should pin down what the bare loop does, because the failure of the
no-bonus agent is what defines the problem. The policy gradient ascends
$\mathbb{E}[\nabla\log\pi\cdot A]$, where the advantage $A$ is built from the environment reward.
When that reward is dense, exploration is free: the stochastic policy and an entropy bonus jiggle
the agent around, it bumps into rewards, and the advantage points somewhere useful. When the reward
is sparse, the advantage is *almost always zero* — there is nothing to ascend, because the agent
has seen no reward to be advantaged over. The agent's only exploration is the undirected noise of
its own action distribution, and on a hard-exploration game that noise will not carry it across
hundreds of reward-free steps to the first payoff. So the natural baseline — vanilla PPO with the
clipped extrinsic reward and nothing else — is the weakest thing I can run *by construction*: it
has no mechanism for directed exploration, and on the games where directed exploration is the entire
problem it should mostly find nothing.

That is the right place to begin: establish the no-bonus agent, read exactly where and how it fails,
and let each subsequent design step be forced by the specific failure of the one before it. The
substrate is the fixed interface above; the first "method" is the trivial one — leave the intrinsic
bonus empty and run PPO on the extrinsic reward alone.
