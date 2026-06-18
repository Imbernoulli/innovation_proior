No leaderboard result for this task yet; ordering by published consensus (see reasoning).

`baseline:fasttd3` is ranked **strongest** of the three. Basis: the FastTD3 source establishes it as
a high-performance off-policy backbone that, on HumanoidBench locomotion under this kind of
parallel-env budget, outperforms both PPO (on-policy, sample-inefficient) and SAC-family methods —
its deterministic actor exploits the distributional value function aggressively while exploration is
supplied by 128 parallel environments, removing the entropy tax that caps FastSAC. It is the
scaffold's default fill. No measured per-seed numbers exist for `rl-offpolicy-sample-efficiency`;
none are fabricated here.
