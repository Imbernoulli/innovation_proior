I keep seeing the same problem from two directions. Temporal-difference learning gives me a beautiful local update: compare a prediction now with a prediction one step later, and use that difference to improve the current prediction. Dynamic programming gives me the fixed point that those predictions should satisfy. But the learner still has to carry the value function in some representation, and that representation decides whether one experience teaches me about many states or about only the state I just visited.

The first thing I need is not a new backup rule. I need a way for the features to express the right kind of similarity. In a static problem, nearby inputs should have nearby answers. In this temporal problem, that is too crude. The value of a state is not sitting inside the state; it is distributed across what tends to happen after the state. So two states should look alike to the value learner when their futures look alike, not merely when their coordinates are close or their hand-designed features overlap.

The barrier example makes the demand concrete. A coarse code over a maze does exactly what I ask of a generic local approximator: it spreads an update to neighboring locations. Most of the time that is helpful. Across a barrier it is wrong. Two grid cells can be close in the coordinate system and far in the problem, because reaching the same future locations requires a long detour. The representation needs to bend with temporal reachability. I cannot get that from an a priori metric, and I do not want to learn a complete world model just to repair a feature space.

So I ask what a state has to resemble. If state i tends to be followed by state j, then the value at i partly depends on the value at j. If i tends to be followed by a whole distribution of later states, then the feature vector for i should somehow contain that whole distribution. The feature should say: from here, these are the states I expect to occupy later, with near visits counting more than distant visits. That is the insight before the method. The right geometry is a geometry of predicted future occupancy.

Now I write the most literal version of that idea. In the absorbing-chain notation, let Q be the transition matrix among nonterminal states. Starting from i, the amount of future occupancy of j is the current occupancy term plus the one-step chance of reaching j, plus the two-step chance, and so on:

`I(i,j) + Q(i,j) + Q^2(i,j) + ...`.

Because the chain is absorbing, this series converges to `[(I - Q)^-1](i,j)`. A row of `(I - Q)^-1` is therefore a state description: it describes a state by the discounted set of states that succeed it. I call this row the successor representation of the state.

Then the algebra tells me why this is more than a clever feature. Sutton's ideal prediction for the same absorbing-chain value problem is already

`r = h + Qh + Q^2 h + ... = (I - Q)^-1 h`,

where h is immediate expected return. The matrix I just choose as the representation is exactly the matrix that maps immediate return to long-run return. If I call it M, then `r = M h`. In the usual discounted continuing notation this becomes

`M = I + gamma P + gamma^2 P^2 + ... = (I - gamma P)^-1`

and

`V = M R`.

This is the split I am looking for. M is predictive state occupancy structure; R is reward. Value is their product. The reward is no longer tangled together with the temporal transition structure inside a single scalar estimate.

I should check that the split is not only a notation trick. Suppose I use each row of M as the feature vector for the corresponding state, and I learn a linear value approximation. The predicted value vector is `M w`. The true value vector is `M h` in the absorbing-chain notation, or `M R` in the discounted notation. Since M is invertible in the setting I am using, the optimal weights are just the immediate reward vector: `w* = h` or `w* = R`. That means the weights no longer have to encode multi-step temporal consequences. The hard temporal part has moved into the features, and the remaining supervised problem is local.

I check the batch TD algebra and see the same point in the update itself. With a general feature matrix X, the expected TD update contains the transition factor `(I - Q)`. This is exactly the temporal coupling that makes information propagate gradually. If the feature matrix is the successor matrix, then `(I - Q)M = I`, so the transition factor cancels. The update relaxes weights toward immediate reward rather than threading reward backward through the Markov chain again and again. That is the technical force of the matrix insight: I am not appending dynamic programming to TD. I am exposing the resolvent of the transition process as the learned representation, so the value learner sees a transformed problem whose optimal weights are memoryless rewards.

Now I hit the obvious wall. If M is `(I - gamma P)^-1`, do I need to know P? If so, I have simply smuggled a model into the representation. But each entry of M is itself a prediction over time. It asks how often state s' will be occupied in the future when I start from state s. That is the same kind of object TD already learns: a discounted sum of future signals. The signal is no longer reward; it is the occupancy indicator for s'.

So I can learn a whole row of M by TD. On a transition from `s_t` to `s_{t+1}`, the target for the row of the current state is the one-hot vector saying where I am now, plus the discounted row predicted from the next state:

`e(s_t) + gamma M_hat(s_{t+1}, .)`.

The error is

`delta(.) = e(s_t) + gamma M_hat(s_{t+1}, .) - M_hat(s_t, .)`,

and I update

`M_hat(s_t, .) <- M_hat(s_t, .) + alpha delta(.)`.

This is TD(0) with a vector-valued prediction error. There is one error for each possible successor state. That is the extra cost I pay: I learn many predictions rather than one. But I do not need a supplied transition matrix, I do not need to search over action sequences, and I do not need a general hidden representation to discover the temporal metric blindly.

The result sits between the two familiar extremes. A pure model-free value function is cheap, but reward and transition effects are compiled together. Change a reward, and the new value has to be backed up through experience. A full model is flexible, but it stores one-step dynamics and uses planning or repeated updates to turn that model into values. The successor matrix compiles multi-step transition statistics instead. If rewards change and transitions stay fixed, I change R and recompute `M R`. If transitions change, M itself is stale and has to be relearned. The method buys reward-side flexibility at the cost of transition-side flexibility.

That is also why the barrier and latent-learning examples matter. If an agent wanders before reward is present, it can learn the predictive geometry of the environment because M does not depend on the reward. Across a barrier, states that are spatially adjacent but temporally separated acquire different rows because their future occupancies differ. When reward finally appears, the value is read through the already learned predictive map. The distinctive move is the separation: value is no longer the primitive learned object; it is the product of a learned predictive occupancy structure and a reward vector.
