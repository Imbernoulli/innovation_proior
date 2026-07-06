I treat the input scale as potentially large in `deadline` and `work`, so the per-decision budget must be O(1) time and O(1) memory. I explicitly reject dynamic programming over all remaining times/progress values because that can be O(deadline * work), which can TLE or MLE for large jobs. I also reject policies that simulate possible future availability sequences, because the sequence is hidden and can be adversarial for the guarantee requirement.

The core constraint is feasibility. At time `t`, there are `deadline - t` timesteps left including the current one. If remaining work is `ceil(work - progress)`, then on-demand can complete exactly one unit per remaining step. Therefore, to guarantee completion from the current state, I must use on-demand whenever spending the current step on spot would risk making the remaining work exceed the remaining time.

Since `spot_available` is observed by the API before deciding, I can safely use spot only when it is currently available. If spot is unavailable, choosing it gives no progress and can lose restart overhead, so it is never useful for cost or feasibility. If spot is available, choosing it gives the same progress as on-demand for that step, so using spot is safe whenever the state is currently feasible. If spot is cheaper than on-demand, I should choose it in those safe available moments. If spot is not cheaper, there is no cost benefit, so I choose on-demand.

This is a general algorithm, not a case-specific construction:

1. Compute `remaining = ceil(max(0, work - progress))`.
2. Compute `time_left = deadline - t`.
3. If `remaining <= 0`, return the cheaper available resource, though either choice is irrelevant to completion.
4. If `remaining >= time_left`, there is no slack. Every remaining step must produce progress, so choose `'ondemand'`.
5. Otherwise there is slack. If spot is available and cheaper, choose `'spot'`; otherwise choose `'ondemand'`.

I also include a conservative restart-overhead margin when spot is unavailable: because an unavailable spot step can reduce progress by `restart_overhead`, the policy never chooses spot when `spot_available` is false. That avoids the restart loss entirely, which is the simplest correct way to preserve the guarantee.

Worked example: let `deadline = 5`, `work = 4`, `restart_overhead = 1`, `spot_price = 1`, `ondemand_price = 3`.

At `t = 0`, progress `0`, remaining `4`, time left `5`, slack `1`. If spot is available, choose spot and progress becomes `1`.
At `t = 1`, progress `1`, remaining `3`, time left `4`, slack `1`. If spot is unavailable, choose on-demand and progress becomes `2`.
At `t = 2`, progress `2`, remaining `2`, time left `3`, slack `1`. If spot is available, choose spot and progress becomes `3`.
At `t = 3`, progress `3`, remaining `1`, time left `2`, slack `1`. If spot is unavailable, choose on-demand and progress becomes `4`, completed before the deadline.

A brute-force sanity check on small instances agrees with the feasibility rule: whenever `remaining >= time_left`, any non-progress step makes `remaining > future_steps`, so completion can no longer be guaranteed. Whenever `remaining < time_left` and spot is currently available, using spot produces the same progress as on-demand and preserves feasibility. Therefore the cheapest safe online action is exactly the rule above.