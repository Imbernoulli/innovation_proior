No leaderboard result for this task yet; ordering by published consensus (see reasoning).

`baseline:fastsac` is ranked **above PPO and below FastTD3**. Basis: as an off-policy
maximum-entropy actor-critic it reuses replay experience and so is materially more sample-efficient
than on-policy PPO under the fixed budget. But the FastTD3 source reports that on HumanoidBench
locomotion, SAC-family methods underperform the deterministic FastTD3 backbone — maximizing entropy
over a high-dimensional humanoid action is hard, and a well-explored deterministic actor (exploration
supplied by 128 parallel envs) exploits the value function more aggressively. No measured per-seed
numbers exist for `rl-offpolicy-sample-efficiency`; none are fabricated here.
