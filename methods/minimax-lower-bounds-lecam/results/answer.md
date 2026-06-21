# Le Cam Minimax Lower Bounds

Le Cam's two-point method proves minimax lower bounds by embedding a binary testing problem inside an estimation problem.

Let `P_0, P_1 in mathcal P`, write `theta_i = theta(P_i)`, and suppose

`rho(theta_0, theta_1) >= 2s`.

For any estimator `hat theta`, define the induced test

`psi(X) = argmin_{i in {0,1}} rho(hat theta(X), theta_i)`.

If `psi` is wrong under world `i`, then `rho(hat theta, theta_i) >= s`. Hence any estimator with uniformly small error would imply a binary test with small testing error. But the best possible test between `P_0^n` and `P_1^n` still satisfies

`inf_psi [P_0^n(psi=1) + P_1^n(psi=0)] = 1 - TV(P_0^n, P_1^n)`.

Therefore, for a power loss `rho^p`, a standard two-point lower bound is

`inf_{hat theta} sup_{P in mathcal P} E_P[rho(hat theta, theta(P))^p] >= (s^p / 2) [1 - TV(P_0^n, P_1^n)]`.

The practical proof pattern is:

1. Pick two admissible distributions whose target values differ by at least `2s`.
2. Bound `TV(P_0^n, P_1^n)` away from `1`.
3. Use KL or Hellinger when total variation is hard to compute directly.

KL enters through inequalities such as Pinsker, which turn small `KL(P_0^n || P_1^n)` into small total variation. Hellinger enters because Hellinger affinity and distance have clean product behavior, making it easier to keep `P_0^n` and `P_1^n` statistically close after `n` samples.

The unique insight is the separation between parameter geometry and experiment geometry. The two worlds are deliberately chosen to be far in the answer space but close in the observation space. If an estimator were accurate in both worlds, it would solve a testing problem that information theory says is unsolvable. This is why the proof is more basic than algorithm analysis: it lower-bounds all measurable decision rules at once, not a particular construction.

