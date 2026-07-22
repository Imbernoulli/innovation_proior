I keep seeing the same problem from two directions. Temporal-difference learning gives me a beautiful local update: compare a prediction now with a prediction one step later, and use that difference to improve the current prediction. Dynamic programming gives me the fixed point that those predictions should satisfy. But the learner still has to carry the value function in some representation, and that representation decides whether one experience teaches me about many states or about only the state I just visited.

The first thing I need is not a new backup rule. I need a way for the features to express the right kind of similarity. In a static problem, nearby inputs should have nearby answers. In this temporal problem, that is too crude. The value of a state is not sitting inside the state; it is distributed across what tends to happen after the state. So two states should look alike to the value learner when their futures look alike, not merely when their coordinates are close or their hand-designed features overlap.

The barrier example makes the demand concrete. A coarse code over a maze does exactly what I ask of a generic local approximator: it spreads an update to neighboring locations. Most of the time that is helpful. Across a barrier it is wrong. Two grid cells can be close in the coordinate system and far in the problem, because reaching the same future locations requires a long detour. The representation needs to bend with temporal reachability. I cannot get that from an a priori metric, and I do not want to learn a complete world model just to repair a feature space.

So I ask what a state has to resemble. If state i tends to be followed by state j, then the value at i partly depends on the value at j. If i tends to be followed by a whole distribution of later states, then the feature vector for i should somehow contain that whole distribution. The feature should say: from here, these are the states I expect to occupy later, with near visits counting more than distant visits. Let me try to make that literal and see whether it actually buys anything, because "describe a state by its predicted future" is the kind of slogan that can sound deep and compute to nothing.

In the absorbing-chain notation, let Q be the transition matrix among nonterminal states. Starting from i, the amount of future occupancy of j is the current occupancy term plus the one-step chance of reaching j, plus the two-step chance, and so on:

`I(i,j) + Q(i,j) + Q^2(i,j) + ...`.

If the chain is absorbing, the powers of Q decay and this series should converge. The geometric-series identity for matrices says the sum is `(I - Q)^-1` when it converges. I want to see this actually happen on numbers before I lean on it, because matrix geometric series are exactly where I tend to fool myself. Take a small chain with discount folded in, states 0,1,2, where 2 is a sink that loops to itself:

```
gamma P =
[ 0    0.72  0.18 ]
[ 0    0.45  0.45 ]
[ 0    0     0.9  ]
```

Summing `I + gamma P + (gamma P)^2 + ...` for a couple thousand terms gives

```
[ 1   1.309   7.691 ]
[ 0   1.818   8.182 ]
[ 0   0      10     ]
```

and inverting `(I - gamma P)` directly gives the same matrix to fourteen decimal places. The bottom-right entry is 10, which I can sanity-check by hand: the sink loops with discounted weight 0.9, so its own discounted occupancy is `1/(1-0.9) = 10`. The series is not a formal flourish; it is a finite, computable thing. So a row of `(I - Q)^-1` is a legitimate state description: it describes a state by the discounted set of states that succeed it.

Now the algebra hints at something stronger than "nice feature." The ideal prediction for the absorbing-chain value problem is itself a power series in Q:

`r = h + Qh + Q^2 h + ... = (I - Q)^-1 h`,

where h is immediate expected return. The matrix I just chose as the representation is the same matrix that maps immediate return to long-run return. Call it M. Then `r = M h`. In the usual discounted continuing notation this is

`M = I + gamma P + gamma^2 P^2 + ... = (I - gamma P)^-1`

and the claim is `V = M R`. I should not just believe this from the symbol-matching; the two derivations could agree formally and still mismatch if I have an index or a discount wrong. So I solve the Bellman system `(I - gamma P) V = R` directly on the same chain, with reward only in the sink, `R = (0,0,1)`, and compare to `M R`:

```
V from Bellman solve : [7.691, 8.182, 10]
M R                  : [7.691, 8.182, 10]
```

They match. So value really does factor as predicted-occupancy times reward.

What does that factoring give me operationally? The reward is no longer tangled together with the transition structure inside a single scalar estimate. To pressure-test that, I change the reward and keep the transitions: put the reward in state 1 instead, `R' = (0,1,0)`, and recompute `M R'` without touching M:

```
M R'  : [1.309, 1.818, 0]
Bellman with R' : [1.309, 1.818, 0]
```

Same answer, and I never re-solved any temporal recursion — I only re-weighted the columns of a matrix I already had. That is the concrete content of "reward-side flexibility": a new reward is a matrix-vector product, not a fresh round of backups.

Next I want to know whether using M as features actually makes the residual learning problem easy, or whether I have just relocated the difficulty. Suppose I use each row of M as the feature vector for the corresponding state and fit a linear value approximation, predicting `M w`. The true value vector is `M R`. The optimal weights solve `M w = M R`. Since M came out invertible on the example, the unique solution is `w* = R`. I do not want to trust that just because M happened to be invertible above, so I fit it numerically — least-squares `w` for `M w = M R` on the sink-reward case:

```
w* from lstsq : [0, 0, 1]   (= R)
```

So with these features the optimal weights are exactly the immediate reward vector. The multi-step temporal consequences have moved into the features, and the remaining supervised problem is local — the weights carry no temporal memory at all.

I can see the same cancellation in the update itself rather than only at the optimum. With a general feature matrix X, the expected batch TD update carries the transition factor `(I - Q)` — that factor is the temporal coupling that makes information propagate gradually, one backup at a time. If the feature matrix is M, then `(I - Q) M` should collapse, because M is `(I - Q)^-1`. I check it on the discounted version of the example:

```
(I - gamma P) M =
[ 1  0  0 ]
[ 0  1  0 ]
[ 0  0  1 ]
```

It is the identity. So the transition factor cancels exactly: the update relaxes weights toward immediate reward instead of threading reward backward through the Markov chain again and again. I am not appending dynamic programming to TD; I am exposing the resolvent of the transition process as the representation, so the value learner sees a transformed problem whose optimal weights are memoryless rewards.

Now the objection I have been deferring. If M is `(I - gamma P)^-1`, do I need to know P? If so, I have quietly smuggled a model into the representation and gained nothing over the model-based route I was trying to avoid. Let me look at what an entry of M actually is. `M(s,s')` is the expected discounted number of future visits to `s'` starting from `s` — a discounted sum of future signals. That is precisely the kind of object TD already learns; the only change is that the signal is no longer reward but the occupancy indicator for `s'`. So I should be able to learn M by TD without ever forming P.

The target writes itself by analogy with the value backup. On a transition `s_t -> s_{t+1}`, the row for the current state should equal the one-hot vector for where I am now, plus the discounted predicted row from the next state:

`e(s_t) + gamma M_hat(s_{t+1}, .)`.

The error is

`delta(.) = e(s_t) + gamma M_hat(s_{t+1}, .) - M_hat(s_t, .)`,

and the update is

`M_hat(s_t, .) <- M_hat(s_t, .) + alpha delta(.)`.

This is TD(0) with a vector-valued prediction error: one error component per possible successor state. That symmetry with value TD is suggestive, but suggestive is not the same as correct — bootstrapping a whole matrix could in principle converge to the wrong fixed point. So I run it on the example, sampling transitions from P (with occasional restarts so the walk does not get stuck in the sink), `alpha = 0.02`, a few hundred thousand steps, and compare the learned `M_hat` to the analytic M:

```
analytic M :
[ 1   1.309   7.691 ]
[ 0   1.818   8.182 ]
[ 0   0      10     ]

TD-learned M_hat :
[ 1   1.357   7.643 ]
[ 0   1.889   8.111 ]
[ 0   0      10     ]
```
