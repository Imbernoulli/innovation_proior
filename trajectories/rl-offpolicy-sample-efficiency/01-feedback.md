No leaderboard result for this task yet; ordering by published consensus (see reasoning).

`baseline:ppo` is ranked **weakest** of the three. Basis: PPO is on-policy and cannot reuse
experience, so under a fixed ~12.8M-frame budget it is the least sample-efficient of the ladder;
on HumanoidBench locomotion the FastTD3 source establishes that off-policy actor-critics (FastSAC,
FastTD3) reach higher return than PPO within the same budget, with PPO weakest on the
harder-to-explore tasks (walk, run). No measured per-seed numbers exist for `rl-offpolicy-sample-efficiency`;
none are fabricated here.
