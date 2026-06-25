# Posterior Sampling For Reinforcement Learning

The method is: keep a Bayesian posterior over finite-horizon MDPs with known state/action sets, horizon `tau`, and start-state distribution `rho`. At the start of episode `k`, sample one MDP `M_k` from `f(. | H_{t_k})`, compute an optimal policy `mu_k = mu^{M_k}` for that sampled MDP, execute it for the episode, and update the posterior from the observed rewards and transitions.

Strens (2000) supplied the original RL algorithm under "Bayesian Dynamic Programming": sample a model hypothesis, solve it by dynamic programming, and retain the hypothesis over a trial so exploration stays goal-directed. What this rule was missing was a finite-time guarantee: if the prior `f` is the distribution of the true MDP `M^*`, then

`E[Regret(T, pi_tau^PS)] = O(tau S sqrt(A T log(S A T)))`.

The key proof move is posterior symmetry. At an episode boundary, conditioned on the history, `M^*` and `M_k` have the same distribution. Therefore, for any model function `g` selected using `H_{t_k}`,

`E[g(M^*) | H_{t_k}] = E[g(M_k) | H_{t_k}]`.

Applying this to the optimal-value functional removes the unknown true optimal policy from the regret expression. Summed over episodes, expected regret equals expected surrogate regret:

`tilde Delta_k = sum_s rho(s) (V^{M_k}_{mu_k,1}(s) - V^{M^*}_{mu_k,1}(s))`.

That surrogate depends on the policy actually executed. Bellman telescoping turns it into one-step model error along visited states and actions, with the martingale transition-noise term disappearing in conditional expectation. UCRL2-style confidence radii then bound the sampled-vs-true Bellman error, but those confidence sets live only in the analysis. The algorithm samples one plausible MDP; it never constructs or maximizes over an optimistic set.

The MATLAB repository `iosband/psrl_2013` is faithful to that core loop: `PSRL.m` samples transition and reward parameters from the posterior at each episode boundary, calls finite-horizon Bellman planning on the sampled MDP, executes for `tau` steps, and updates posterior statistics. It is simulation code rather than the full theorem object, so the formal time-indexed policy and regret statement above come from the analysis rather than from the simulation.

That separation is the distinctive contribution. Earlier Bayesian RL either faced intractable Bayes-adaptive planning or added explicit optimism/bonuses. This result shows that the single-sample, episode-committed rule already has efficient exploration because posterior symmetry gives it "implicit optimism" in expectation.
