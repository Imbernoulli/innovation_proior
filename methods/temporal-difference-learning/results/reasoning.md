I start with the ordinary supervised answer because it is the thing that almost works. For a sequence of observations `x_1, ..., x_m` and final outcome `z`, the delta rule trains each prediction `P_t = P(x_t, w)` against `z`: `Delta w_t = alpha (z - P_t) grad_w P_t`. This is simple, but it has the wrong temporal shape. The update for the first prediction cannot be computed when the first prediction is made, because the target `z` is not known. The learner has to remember earlier observations or gradients until the sequence ends.

That delay is not merely inconvenient. The final outcome can be a bad target for an earlier state. If a new game position leads into a position I already know is bad, but the episode ends in a lucky win, the final outcome says to increase the value of the new position. The continuation says the opposite. What I really want is the expected outcome from the current state, not the one sampled outcome that happened to occur after additional random events.

So I look for a learning signal available before the final outcome. The sequence already contains one: predictions change over time. If I define the terminal outcome as one more prediction, `P_{m+1} = z`, then the gap between the current prediction and the outcome is a sum of one-step changes:

`z - P_t = P_{m+1} - P_t = sum_{k=t}^m (P_{k+1} - P_k)`.

That is just the telescoping cancellation, so it should hold term by term. Before trusting it I want to see it with numbers. Take `m = 3`, predictions `P_1 = 0.4, P_2 = 0.6, P_3 = 0.3`, and outcome `z = P_4 = 1.0`. For `t = 1`: `z - P_1 = 0.6`, and the sum of differences is `(0.6 - 0.4) + (0.3 - 0.6) + (1.0 - 0.3) = 0.2 - 0.3 + 0.7 = 0.6`. For `t = 2`: `z - P_2 = 0.4`, sum `= (0.3 - 0.6) + (1.0 - 0.3) = -0.3 + 0.7 = 0.4`. For `t = 3`: `z - P_3 = 0.7`, sum `= 1.0 - 0.3 = 0.7`. All three agree, so the identity is exact, not just for `t = 1`.

Substitute it into the total supervised weight change for the whole sequence and swap the two sums:

`sum_{t=1}^m alpha (z - P_t) grad_w P_t = sum_{k=1}^m alpha (P_{k+1} - P_k) sum_{t=1}^k grad_w P_t`.

That gives an incremental rule. At temporal step `k`, I only need the new prediction difference `P_{k+1} - P_k` and a running sum of past gradients. With a linear predictor this should produce the same per-sequence weight change as Widrow-Hoff, while spreading the work through the sequence instead of delaying it to the final outcome. The swap-of-sums step is exactly where I could have an off-by-one or a transposed index, so I check it directly rather than believe the algebra. With three random feature vectors in three dimensions, a random `w`, `z = 1`, `alpha = 0.05`, I accumulate the supervised change `sum_t alpha(z - w^T x_t) x_t` and, separately, the incremental change using a running gradient sum `e_k = x_k + e_{k-1}` and difference `P_{k+1} - P_k` (predictions held at the same `w` across the sequence). Both come out to `[0.0292, 0.0158, -0.0351]`, identical to four decimals. So the rearrangement is genuinely the same update, only retimed.

Equal credit to every past prediction is only one endpoint. A more local temporal-credit rule should weight recent predictions more strongly than remote ones. Use an exponentially decaying trace:

`e_t = sum_{k=1}^t lambda^{t-k} grad_w P_k`, with `0 <= lambda <= 1`.

The exponential form matters because it gives the online recursion:

`e_t = grad_w P_t + lambda e_{t-1}`.

Now the update at step `t` is:

`Delta w_t = alpha (P_{t+1} - P_t) e_t`.

At `lambda = 1`, the trace is the full running sum, which is the case I just checked numerically, so I recover the incremental rearrangement of the final-outcome supervised rule. At `lambda = 0`, the trace contains only the current gradient, so the update is:

`Delta w_t = alpha (P_{t+1} - P_t) grad_w P_t`.

The final outcome has dropped out of this endpoint entirely; it has been replaced by the next prediction. The learner improves one estimate using another estimate. The target is moving because it depends on `w`, so this is not ordinary gradient descent on a fixed supervised objective. It can still be principled, because the desired target is the expected outcome and the next prediction can be a lower-variance estimate of that expectation than the final sampled return. Whether it actually converges to anything sensible is not obvious at this point, and I do not want to assume it; the moving target is exactly the kind of thing that diverges.

There is also a clear failure mode to worry about first. If predictions are trained only to match other predictions, a constant useless function satisfies the consistency constraint everywhere with zero error. The boundary must be anchored. In a terminal episode the last prediction is pinned to the actual external outcome; in a cumulative-signal problem the observed immediate signal enters directly. That anchored boundary condition is what lets prediction differences carry real information backward through the sequence rather than collapsing to a constant.

So I need to know whether the bootstrapped `lambda = 0` rule is stable in a clean mathematical case. Let the data come from an absorbing Markov chain. Let `Q` be the nonterminal transition matrix and let `h_i` be the expected terminal-outcome contribution when state `i` exits to a terminal state. The ideal prediction vector should satisfy:

`v = h + Qv`, hence `v = (I - Q)^{-1} h`,

because an absorbing chain has `Q^n -> 0` and therefore `(I - Q)^{-1} = sum_{n>=0} Q^n`.

I want to make sure this `v` is the thing I actually believe the predictions should be, so I instantiate the diagnostic walk from the evaluation setting: five nonterminal states in a line, equal-probability left/right moves, left exit to outcome `0`, right exit to outcome `1`. Here `Q` has `0.5` on each off-diagonal, and `h` is zero except for the rightmost state, which exits right to outcome `1` with probability `0.5`, giving `h_5 = 0.5`. Solving `(I - Q)^{-1} h` numerically gives `v = [0.1667, 0.3333, 0.5, 0.6667, 0.8333]`, which is `[1/6, 2/6, 3/6, 4/6, 5/6]`. That is exactly the probability of right-side termination from each state in a symmetric walk, which I can also see directly: a symmetric random walk on `{0,...,6}` absorbed at the ends hits the right end from state `k` with probability `k/6`. The two agree, so `(I - Q)^{-1} h` is the right target.

For linear TD(0), updating after each sequence, write `X` for the matrix whose columns are the state observation vectors and `D = diag(d_i)` for expected visit counts to retained nonterminal states. Following the per-sequence expected update through, the expected prediction-vector recursion is:

`X^T wbar_{n+1} = alpha X^T X D h + [I - alpha X^T X D (I - Q)] X^T wbar_n`.

If the powers of `M = I - alpha X^T X D (I - Q)` go to zero, the iteration has a unique fixed point, and solving for it gives:

`(alpha X^T X D (I - Q))^{-1} alpha X^T X D h = (I - Q)^{-1} h`,

the ideal prediction. The inverses used here are legitimate when the observation vectors are linearly independent and never-visited states have been discarded, so `X^T X` and `D` are nonsingular.

So the real task is to make `M^n -> 0`. Let `B = D(I - Q)`. I first want `B` to be positive definite. It is not symmetric, so I look at `S = B + B^T`. Its diagonal entries are `S_ii = 2 d_i(1 - p_ii) > 0`; its off-diagonal entries are `S_ij = -d_i p_ij - d_j p_ji <= 0`. The row sum is `d_i(1 - sum_{j in N} p_ij) + mu_i`, using `d^T(I - Q) = mu^T`, so every row is weakly diagonally dominant and nonnegative. Each connected component has a strict row because every retained state is reachable from some starting-distribution mass or from a state that is, and absorption prevents a closed nonterminal component. By the diagonal-dominance argument `S` should then be positive definite, hence `D(I - Q)` positive definite. This is the load-bearing step, so I check it on the walk. The expected visit counts starting from the middle state, `d = (I - Q)^{-T} mu` with `mu` a point mass at the center, come out `[1, 2, 3, 2, 1]` — symmetric and largest at the start state, as expected. Forming `S = B + B^T` for `B = D(I - Q)` and taking its symmetric eigenvalues gives `[0.383, 1.197, 2.819, 4.803, 8.798]`, all strictly positive. So `S` is positive definite on this instance, matching the argument.

Now lift this through the feature matrix. If `mu` is an eigenvalue of `A = X^T X D(I - Q)` with eigenvector `y`, set `z = (X^T X)^{-1} y`. Then the real part of `y* D(I - Q) y = mu (Xz)* (Xz)`, and since `D(I - Q)` is positive definite its quadratic form has positive real part and `(Xz)* (Xz) > 0`. Therefore every eigenvalue `mu = a + bi` of `A` has `a > 0`. For the walk with tabular features `X = I` this says the eigenvalues of `B` itself should have positive real part; computing them gives real parts `[4.358, 2.366, 1.395, 0.247, 0.634]`, all positive. (They happen to come out purely real here, so `b = 0`, but the general bound below does not need that.)

The constant on the learning rate follows. Each eigenvalue of `M` is `1 - alpha mu`, and

`|1 - alpha mu|^2 = 1 - 2 alpha a + alpha^2(a^2 + b^2)`.

This is less than `1` exactly when `0 < alpha < 2a / (a^2 + b^2)`. Choose `epsilon` as the minimum of that bound over the finitely many eigenvalues. Then every `0 < alpha < epsilon` puts all eigenvalues of `M` inside the unit circle, so the mean predictions converge to `(I - Q)^{-1} h`. I want to see that this `epsilon` is a real threshold and not a loose sufficient condition, so I evaluate it on the walk. The per-eigenvalue bounds `2a/(a^2+b^2)` are `[0.459, 0.845, 1.433, 8.108, 3.155]`, so `epsilon = 0.459`. Then I scan the spectral radius of `M = I - alpha B`: at `alpha = 0.458` it is `0.996 < 1`, at `alpha = 0.459` it is `1.0003 > 1`, and it keeps growing past that. The predicted threshold and the actual stability boundary coincide to three decimals, so the bound is tight here, not just sufficient. Finally I run the expected recursion itself, `wbar <- wbar + alpha(D h - B wbar)` with `alpha = 0.1`, and after enough iterations it settles on `[0.1667, 0.3333, 0.5, 0.6667, 0.8333]` — exactly `(I - Q)^{-1} h`. So the bootstrapped rule does not merely avoid divergence; in expectation it lands on the ideal predictions. The constant-stepsize predictions still fluctuate around their mean; the statement here is about convergence in expected prediction, not almost-sure freezing of a sample path.

The repeated-presentation case explains why finite data can be used differently. Take a finite training set, retain linearly independent state observations, build the empirical Markov model from transition counts, and present the same set repeatedly. Widrow-Hoff fits the observed returns in the training set. TD(0) converges instead to the certainty-equivalence predictions of the empirical model, `(I - Qhat)^{-1} hhat`. The two can disagree sharply, and I want a concrete case rather than a slogan. Suppose state `A` always transitions to state `B`, `B` terminates, and across the data `B` exits to outcome `1` three quarters of the time, so its empirical exit value is `0.75`. The empirical model is `Qhat = [[0,1],[0,0]]`, `hhat = [0, 0.75]`, and `(I - Qhat)^{-1} hhat = [0.75, 0.75]`: TD assigns `A` the value `0.75`, inherited entirely through `B`. Now suppose `A` itself was observed only once, on a trajectory that happened to terminate in `0`. Outcome fitting has only that single return for `A`, so Widrow-Hoff drives `V(A)` toward `0`. Same data, and the two rules land on `0.75` versus `0` for `A`. The bootstrapped rule is using the transition `A -> B` and `B`'s many observations, not `A`'s one accidental label.

For cumulative prediction, I rewrite the target rather than starting over. If the return is a discounted sum,

`z_t = c_{t+1} + gamma c_{t+2} + gamma^2 c_{t+3} + ...`,

then it satisfies:

`z_t = c_{t+1} + gamma z_{t+1}`.

Therefore correct predictions should satisfy:

`P_t = c_{t+1} + gamma P_{t+1}`.

The learning signal is the violation of this recursive consistency equation:

`delta_t = c_{t+1} + gamma P_{t+1} - P_t`.

The trace must decay by the same discount as the future credit, so the online trace is `e_t = grad_w P_t + gamma lambda e_{t-1}`, and the weight update is `w <- w + alpha delta_t e_t`. For an undiscounted finite final-outcome task, set `gamma = 1`, intermediate signals to zero, and pin the terminal prediction to the observed outcome — which recovers the `P_{m+1} = z` anchoring I started from.

This also clarifies the relationship to Monte Carlo learning and dynamic programming. Monte Carlo learning samples real trajectories but waits for complete returns and does not bootstrap. Dynamic programming bootstraps from successor estimates but needs a model to take expectations over successors. This update samples the actually observed successor and bootstraps from the current prediction at that successor. It combines Monte Carlo sampling with dynamic-programming bootstrapping.

The conditioning story fits the same equation. Rescorla-Wagner changes association from actual reinforcement minus predicted reinforcement, `lambda - Vbar`. In real time, the reinforcement term becomes immediate reinforcement plus discounted next prediction, `lambda_{t+1} + gamma Vbar_{t+1}`. The discrepancy `lambda_{t+1} + gamma Vbar_{t+1} - Vbar_t` is the same signed error: immediate signal plus successor prediction minus current prediction. A prediction can therefore serve as a secondary reinforcer, which is one way credit moves backward before a primary outcome arrives.

The final method is simple. Keep a predictor, keep an eligibility trace, and after each sampled transition compute an error from immediate signal plus discounted successor prediction minus current prediction. Use that error to adjust all currently eligible weights. For linear prediction:

```python
delta = reward + gamma * V(next_state) - V(state)
e = gamma * lam * e + x
w = w + alpha * delta * e
```

With `lambda = 0`, this is the pure one-step bootstrapped update whose expected fixed point I verified to be `(I - Q)^{-1} h`. With `lambda = 1` in an undiscounted finite episode, it recovers the incremental form of the supervised sequence update I checked against Widrow-Hoff. Between them is a continuum that trades off lower-variance bootstrapping against longer-range return propagation.
