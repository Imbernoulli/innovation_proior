I start with the order of play, because that is where the problem gets its teeth. I have to choose before I see the cost vector. If I try to match the best action on each round, I am asking for clairvoyance. The best fixed action in hindsight is the right target: it is strong enough to mean I have learned something, but weak enough that an adversarial sequence can still allow a theorem.

The majority idea is the first thing I have to discard. In binary prediction, the majority of experts can be wrong every day while a quiet minority expert is always right. Uniform random choice is no better in the long run, because it pays the average expert forever. So I need memory, and the memory has to change the distribution over decisions.

A weight on each decision is the natural memory. The hard question is how the weights should move. If I move them additively, a repeatedly bad decision only falls behind by a difference. It may remain visible for too long, and a decision that has failed a thousand times is not separated sharply enough from one that failed once. What I need is a ratio effect. Repeated failure should compound. That pushes me to make each weight a product of per-round factors.

So I let costly decisions shrink by a factor depending on their cost, and I play in proportion to the resulting weights. The proportional play is not cosmetic. If `Phi = sum_i w_i` and `p_i = w_i / Phi`, then the weighted sum of costs is exactly `sum_i w_i m_i = Phi * (m . p)`. This identity is what I lose if I commit deterministically to the weighted majority side. Randomization turns the algorithm's round cost into a linear quantity that can appear in the potential calculation.

Now I need the proof to tell me whether the idea is real. Let the cost-form update be `w_i <- w_i(1 - eta m_i)` with `m_i in [-1,1]` and `eta <= 1/2`. I track `Phi(t) = sum_i w_i(t)`. One step gives

`Phi(t+1) = sum_i w_i(t)(1 - eta m_i(t)) = Phi(t)(1 - eta m(t) . p(t))`.

Since `1 + x <= exp(x)`, this is at most `Phi(t) exp(-eta m(t) . p(t))`. Multiplying over time gives

`Phi(T+1) <= n exp(-eta sum_t m(t) . p(t))`.

This is the upper handle: if my expected cumulative cost is large, the total weight must have been driven down.

The lower handle has to know about a fixed comparator decision. For any `i`, the total potential is at least that decision's weight:

`Phi(T+1) >= w_i(T+1) = prod_t (1 - eta m_i(t))`.

Here I hit the sign issue. If costs were only zero or one, I would just get `(1 - eta)^(number of mistakes)`. But costs can be negative, and negative cost should increase a weight. I split the rounds. For nonnegative `m_i(t)`, convexity gives `(1 - eta)^x <= 1 - eta x`. For negative `m_i(t)`, the matching inequality is `(1 + eta)^(-x) <= 1 - eta x`. Thus the same product is bounded below by a positive-round factor times a negative-round factor:

`prod_t (1 - eta m_i(t)) >= (1 - eta)^(sum_{m_i>=0} m_i(t)) * (1 + eta)^(-sum_{m_i<0} m_i(t))`.

Now the potential is squeezed. I take logs of the upper and lower bounds, rearrange, and use `ln(1/(1-eta)) <= eta + eta^2` and `ln(1+eta) >= eta - eta^2`. The sign split recombines into the comparator's total cost plus `eta` times the comparator's absolute cost:

`sum_t m(t) . p(t) <= sum_t m_i(t) + eta sum_t |m_i(t)| + ln(n)/eta`.

That is the theorem I wanted. It holds for every fixed decision `i`, so it holds for the best one in hindsight. If all costs have absolute value at most one, the regret is at most `eta T + ln(n)/eta`. When `sqrt(ln(n)/T) <= 1/2`, balancing the two terms with `eta = sqrt(ln(n)/T)` gives `2 sqrt(T ln n)`; otherwise I keep `eta` capped at `1/2` and use the displayed bound directly. The average regret goes to zero.

This also explains why the deterministic warm-up stalls at a factor of two. In the deterministic weighted-majority proof, a mistaken round only tells me that at least half the weight was wrong, so the potential drops by a factor like `1 - eta/2`. The learner's mistake count enters through that half-weight event. In the randomized version, the expected loss itself is `m . p`, so every small shift in the distribution is accounted for exactly. The linearity of expectation removes the factor-two scar.

I should also check the entropy interpretation, because it tells me what the update is doing geometrically. If I compare the current distribution to a comparator distribution `p`, the update reduces `RE(p || p(t))` whenever my current expected cost is too high relative to `p`. In the unrestricted simplex this is the same algorithm written in normalized coordinates. With a convex restriction on distributions, I can update first and then project back using relative entropy; the Bregman projection only helps. This is the same proof idea in a different language: the algorithm is not guessing the best decision, it is repeatedly moving closer to any comparator that is proving cheaper.

The variations now fall into place. If I use exponential factors `exp(-eta m_i)`, I get Hedge. The proof is almost the same, but the second-order term depends on my own distribution's squared costs rather than on the fixed comparator's absolute costs. That is sometimes fine, but for the LP and set-cover reductions the linear factor's comparator-side penalty is the cleaner tool. If I am maximizing gains rather than minimizing costs, I run the rule on `-m`, so the update becomes `w_i <- w_i(1 + eta m_i)`, with `eta <= 1` keeping the factor nonnegative, and the inequality reverses.

The broad applications are no longer mysterious. In a zero-sum game, rows are decisions. Each round I play a distribution over rows, the opponent returns a best-response column, and that column is the cost vector. The regret inequality says my average play is near the game value, so no-regret play constructs approximate minimax strategies.

For a feasibility LP, the decisions are constraints. Given weights over constraints, an oracle only has to satisfy the weighted average constraint `p^T A x >= p^T b`. If it cannot, that weight vector is an infeasibility certificate. If it can, I feed back the satisfaction amount `A_i x - b_i` as the cost of constraint `i`, scaled by the width. A well-satisfied constraint has positive cost and is down-weighted; a violated constraint has negative cost and is up-weighted. In gains-form code the same signal appears with the sign flipped, as reward `b_i - A_i x`, because the generic update multiplies by `1 + eta reward`. Averaging the oracle's returned points then satisfies every constraint up to the error dictated by the regret bound.

Set cover is the extreme case where the learning rate is one. Covered elements drop to weight zero, uncovered elements retain weight, and maximizing the current expected coverage is exactly the greedy set-cover rule. Boosting is the same pressure with examples as weighted objects: examples classified correctly lose relative emphasis, misclassified examples gain relative emphasis, and the final vote succeeds because an example misclassified by the majority has low cumulative correctness while every weak round has correctness above one half under the current distribution.

So the final insight is a proof design as much as an update rule. I make weights multiplicative so history is stored as a product. I randomize proportionally so the potential's one-step change is exactly controlled by expected cost. I sandwich the potential between the algorithm's cumulative cost and any comparator's surviving weight, or equivalently watch relative entropy fall toward the comparator. That is why the same small rule controls regret and why so many algorithmic reductions can be made to look like experts with adversarial costs.
