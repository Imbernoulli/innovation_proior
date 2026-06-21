I start with the ordinary supervised answer because it is the thing that almost works. For a sequence of observations `x_1, ..., x_m` and final outcome `z`, the delta rule trains each prediction `P_t = P(x_t, w)` against `z`: `Delta w_t = alpha (z - P_t) grad_w P_t`. This is simple, but it has the wrong temporal shape. The update for the first prediction cannot be computed when the first prediction is made, because the target `z` is not known. The learner has to remember earlier observations or gradients until the sequence ends.

That delay is not merely inconvenient. The final outcome can be a bad target for an earlier state. If a new game position leads into a position I already know is bad, but the episode ends in a lucky win, the final outcome says to increase the value of the new position. The continuation says the opposite. What I really want is the expected outcome from the current state, not the one sampled outcome that happened to occur after additional random events.

So I look for a learning signal available before the final outcome. The sequence already contains one: predictions change over time. If I define the terminal outcome as one more prediction, `P_{m+1} = z`, then the supervised error telescopes:

`z - P_t = P_{m+1} - P_t = sum_{k=t}^m (P_{k+1} - P_k)`.

This identity is exact. Substitute it into the total supervised weight change for the whole sequence and swap the two sums:

`sum_{t=1}^m alpha (z - P_t) grad_w P_t = sum_{k=1}^m alpha (P_{k+1} - P_k) sum_{t=1}^k grad_w P_t`.

That gives an incremental rule. At temporal step `k`, I only need the new prediction difference `P_{k+1} - P_k` and a running sum of past gradients. With a linear predictor this produces the same per-sequence weight change as Widrow-Hoff, but the work is spread through the sequence instead of being delayed until the final outcome appears.

Equal credit to every past prediction is only one endpoint. A more local temporal-credit rule should weight recent predictions more strongly than remote ones. Use an exponentially decaying trace:

`e_t = sum_{k=1}^t lambda^{t-k} grad_w P_k`, with `0 <= lambda <= 1`.

The exponential form matters because it gives the online recursion:

`e_t = grad_w P_t + lambda e_{t-1}`.

Now the update at step `t` is:

`Delta w_t = alpha (P_{t+1} - P_t) e_t`.

At `lambda = 1`, the trace is the full running sum and I recover the incremental rearrangement of the final-outcome supervised rule. At `lambda = 0`, the trace contains only the current gradient, so the update is:

`Delta w_t = alpha (P_{t+1} - P_t) grad_w P_t`.

That endpoint is the conceptual break. The final outcome has been replaced by the next prediction. This is bootstrapping: the learner improves one estimate using another estimate. The target is moving because it depends on `w`, so this is not ordinary gradient descent on a fixed supervised objective. But it is still principled because the desired target is the expected outcome, and the next prediction can be a lower-variance estimate of that expectation than the final sampled return.

This raises a danger. If predictions are trained only to match other predictions, a constant useless function can satisfy the consistency constraint. The boundary must be anchored. In a terminal episode, the last prediction is pinned to the actual external outcome; in a cumulative-signal problem, the observed immediate signal enters directly. That anchored boundary condition lets prediction differences propagate real information backward through the sequence.

Now I need to know whether the bootstrapped endpoint is stable in a clean mathematical case. Let the data come from an absorbing Markov chain. Let `Q` be the nonterminal transition matrix and let `h_i` be the expected terminal-outcome contribution when state `i` exits to a terminal state. The ideal prediction vector satisfies:

`v = h + Qv`, hence `v = (I - Q)^{-1} h`,

because an absorbing chain has `Q^n -> 0` and therefore `(I - Q)^{-1} = sum_{n>=0} Q^n`.

For linear TD(0), updating after each sequence, write `X` for the matrix whose columns are the state observation vectors and `D = diag(d_i)` for expected visit counts to retained nonterminal states. The expected prediction-vector recursion is:

`X^T wbar_{n+1} = alpha X^T X D h + [I - alpha X^T X D (I - Q)] X^T wbar_n`.

If the powers of `M = I - alpha X^T X D (I - Q)` go to zero, the fixed point is exactly right:

`(alpha X^T X D (I - Q))^{-1} alpha X^T X D h = (I - Q)^{-1} h`.

The inverses used here are legitimate when the observation vectors are linearly independent and never-visited states have been discarded, so `X^T X` and `D` are nonsingular.

So the real task is to make `M^n -> 0`. Let `B = D(I - Q)`. I first need `B` to be positive definite. It is not symmetric, so I look at `S = B + B^T`. Its diagonal entries are `S_ii = 2 d_i(1 - p_ii) > 0`; its off-diagonal entries are `S_ij = -d_i p_ij - d_j p_ji <= 0`. The row sum is `d_i(1 - sum_{j in N} p_ij) + mu_i`, using `d^T(I - Q) = mu^T`, so every row is weakly diagonally dominant and nonnegative. Each connected component has a strict row because every retained state is reachable from some starting distribution mass or from a state that is, and absorption prevents a closed nonterminal component. The diagonal-dominance lemma makes `S` positive definite, hence `D(I - Q)` positive definite.

Now lift this through the feature matrix. If `mu` is an eigenvalue of `A = X^T X D(I - Q)` with eigenvector `y`, set `z = (X^T X)^{-1} y`. Then the real part of `y* D(I - Q) y = mu (Xz)* (Xz)` is positive, because `D(I - Q)` is positive definite and `(Xz)* (Xz) > 0`. Therefore every eigenvalue `mu = a + bi` of `A` has `a > 0`.

The constant on the learning rate follows directly. Each eigenvalue of `M` is `1 - alpha mu`, and

`|1 - alpha mu|^2 = 1 - 2 alpha a + alpha^2(a^2 + b^2)`.

This is less than `1` exactly when `0 < alpha < 2a / (a^2 + b^2)`. Choose `epsilon` as the minimum of that bound over the finitely many eigenvalues. Then every `0 < alpha < epsilon` puts all eigenvalues of `M` inside the unit circle, so the mean predictions converge to `(I - Q)^{-1} h`. The constant-stepsize predictions still fluctuate around their mean; the theorem here is about convergence in expected prediction, not almost-sure freezing of a sample path.

The repeated-presentation case explains why finite data can be used differently. Take a finite training set, retain linearly independent state observations, build the empirical Markov model from transition counts, and present the same set repeatedly. Widrow-Hoff fits the observed returns in the training set. TD(0) converges instead to the certainty-equivalence predictions of the empirical model, `(I - Qhat)^{-1} hhat`. If state `A` is observed only once and that one trajectory ends in `0`, outcome fitting drives `A` toward `0`; if `A` always transitions to state `B` and `B` has many observations indicating value `0.75`, the bootstrapped prediction for `A` inherits `B`'s value. The method uses transition structure, not just terminal labels.

For cumulative prediction, I rewrite the target rather than starting over. If the return is a discounted sum,

`z_t = c_{t+1} + gamma c_{t+2} + gamma^2 c_{t+3} + ...`,

then it satisfies:

`z_t = c_{t+1} + gamma z_{t+1}`.

Therefore correct predictions should satisfy:

`P_t = c_{t+1} + gamma P_{t+1}`.

The learning signal is the violation of this recursive consistency equation:

`delta_t = c_{t+1} + gamma P_{t+1} - P_t`.

The trace must decay by the same discount as the future credit, so the online trace is `e_t = grad_w P_t + gamma lambda e_{t-1}`, and the weight update is `w <- w + alpha delta_t e_t`. For an undiscounted finite final-outcome task, set `gamma = 1`, intermediate signals to zero, and pin the terminal prediction to the observed outcome.

This also clarifies the relationship to Monte Carlo learning and dynamic programming. Monte Carlo learning samples real trajectories but waits for complete returns and does not bootstrap. Dynamic programming bootstraps from successor estimates but needs a model to take expectations over successors. This update samples the actually observed successor and bootstraps from the current prediction at that successor. It combines Monte Carlo sampling with dynamic-programming bootstrapping.

The conditioning story fits the same equation. Rescorla-Wagner changes association from actual reinforcement minus predicted reinforcement, `lambda - Vbar`. In real time, the reinforcement term becomes immediate reinforcement plus discounted next prediction, `lambda_{t+1} + gamma Vbar_{t+1}`. The discrepancy `lambda_{t+1} + gamma Vbar_{t+1} - Vbar_t` is the same signed error: immediate signal plus successor prediction minus current prediction. A prediction can therefore serve as a secondary reinforcer, which explains how credit moves backward before a primary outcome arrives.

The final method is simple. Keep a predictor, keep an eligibility trace, and after each sampled transition compute an error from immediate signal plus discounted successor prediction minus current prediction. Use that error to adjust all currently eligible weights. For linear prediction:

```python
delta = reward + gamma * V(next_state) - V(state)
e = gamma * lam * e + x
w = w + alpha * delta * e
```

With `lambda = 0`, this is the pure one-step bootstrapped update. With `lambda = 1` in an undiscounted finite episode, it recovers the incremental form of the supervised sequence update. Between them is a continuum that trades off lower-variance bootstrapping against longer-range return propagation.
