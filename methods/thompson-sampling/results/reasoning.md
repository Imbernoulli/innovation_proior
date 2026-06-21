I start with the fact that the next individual cannot wait. If I had the luxury of gathering a large sample first, I could estimate two probabilities and argue about error bounds afterward. But here every observation is purchased by assigning someone to one treatment or the other. The data are small exactly because acting and learning are the same stream.

The obvious rule is to look at the current data and choose the treatment whose estimated chance is larger. That feels efficient for a moment, but it is too brittle. A few early successes can make one treatment look best, and then the rule gives it all the future observations. The other treatment receives no chance to correct the mistake. The trouble is not that Bayesian updating is missing; the trouble is that the posterior has been collapsed too soon into a single winner.

The opposite rule is to alternate. That avoids the catastrophe of permanent commitment, but it is also wrong in a different way. If the evidence already says one treatment is very probably better, alternation still sends half the stream to the other one. It explores without listening. So I have two failures facing each other: the plug-in rule listens too sharply and may freeze, while the fixed exploration rule refuses to let the evidence change the allocation.

Let me name the uncertainty instead of hiding it. Suppose `P` is the probability, given the data now, that the first treatment is better than the second. Then `Q = 1 - P` is the probability that the second is better. If I eliminate the second treatment now and use the first treatment forever, the expected future sacrifice, measured in units of the treatment gap, is `Q` for every later individual. That is a large price when `P` is not close to one, and small samples are exactly where it is not close to one.

So the action should not be all-or-nothing. It should have an intensity. I can let some increasing function `f(P)` set the fraction of individuals assigned to the first treatment, or equivalently the probability that the next individual receives it. When the evidence is balanced, `f(P)` should be near one half. When the evidence strongly favors the first treatment, `f(P)` should move toward one. This is already a different kind of decision rule: the posterior uncertainty is not merely a report; it becomes the control signal.

Which `f` should I use? I can invent many monotone functions, but the identity is the only one that adds no new scale and no new threshold. Set `f(P) = P`. Then I assign the first treatment with probability equal to the present probability that it is better, and I assign the second with probability `Q`. Now the expected chance of assigning an individual to the inferior treatment is `P Q + Q P = 2 P Q`. This is at most one half, and it is strictly less than one half whenever the evidence gives any preference at all. So the rule is never worse than alternation in this immediate sacrifice calculation, and it improves as soon as the evidence is informative. Against immediate final choice, it avoids putting the whole future stream behind a small-sample apparent winner.

This is the scientific move. Bayes gives me a posterior over unknown probabilities, and a bandit payoff gives me a loss for choosing the inferior treatment, but neither of those alone tells me to randomize the next action according to the posterior probability of being best. The new step is to use a posterior event, "this treatment is the better one", as the allocation law itself. Randomness is not decorative. It is how the rule keeps testing treatments in proportion to their remaining claim to be optimal.

Now I need the probability `P` for small samples. Otherwise the decision principle is only a slogan. Take one treatment with `r` occurrences and `s` failures. With a uniform prior on the unknown chance `p`, Bayes's rule gives the posterior density

`((n+1)! / (r! s!)) p^r (1-p)^s`, with `n = r + s`.

This is the beta density with parameters `r+1` and `s+1`. The posterior tail above a fixed value is an incomplete beta integral, and Pearson's binomial identity turns it into a finite sum:

`Pr(p > x | r,s) = sum_{a=0}^r C(n+1,a) x^a (1-x)^(n+1-a)`.

That identity matters because it removes the small-sample obstacle. I do not need a normal approximation or a vague large-number argument. I can compute exact finite terms.

For two treatments I write the probability that the second unknown chance exceeds the first as an integral over the first posterior times the upper tail of the second posterior. Substituting the finite tail sum makes the remaining integral another beta integral term by term. After collecting factorials, I get

`Pr(p2 > p1) = [sum_{a=0}^{r2} C(r1+r2-a, r1) C(s1+s2+1+a, s1)] / C(n1+n2+2, n1+1)`.

The complementary probability gives `Pr(p1 > p2)` because equality has probability zero under the continuous posteriors. So the exact small-sample computation and the allocation principle meet: compute the posterior probability that each treatment is better, then use that probability to assign the next individual.

There is an even cleaner operational picture. Instead of computing `P` explicitly and then flipping a `P`-coin, I can draw one plausible value from each treatment's posterior and choose the treatment with the larger draw. The probability that the first posterior draw exceeds the second is exactly `Pr(p1 > p2)`, the same integral. So posterior sampling implements the same matching rule. With more than two actions, I draw one plausible parameter for each action and play the action whose draw is best; the chance an action is selected is then its posterior chance of being optimal.

I need to be precise about what is sampled. I sample a plausible value of the treatment probability, not a possible next success or failure. The decision is about which treatment's underlying chance could be best, so the sampled object has to be the parameter that defines that chance. Sampling a binary outcome would confuse a one-step observation with the uncertainty about the treatment itself.

This also explains why deterministic exploration is the wrong contrast. A fixed exploration schedule explores arms because a clock says so, even if one arm is already almost certainly poor and another remains genuinely plausible. A greedy plug-in rule does the reverse: it exploits a current summary and may never collect the observation that would overturn it. The posterior-matching rule explores only where the posterior still assigns some chance of being best. As evidence accumulates, a bad treatment's probability of being best shrinks and its allocation shrinks with it. No separate stopping rule is needed; the posterior concentration turns exploration down.

So the method is more than a recombination of Bayesian updating and payoffs. The posterior update supplies beliefs, and the payoff gap supplies the cost of being wrong, but the key invention is a decision law that refuses to turn uncertainty into either paralysis, fixed dithering, or premature certainty. It makes uncertainty itself choose the next experiment, in exactly the proportion that the current evidence says each action could still be the best.
