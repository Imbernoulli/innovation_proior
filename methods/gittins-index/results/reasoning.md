Let me get the problem in front of me as plainly as I can. I have several activities — call them arms, or projects, or jobs, it doesn't matter — and one unit of effort per time step. Each arm is a little Markov machine: when I work on it, it pays me a random reward and shuffles its own internal state forward; when I don't work on it, it pays nothing and stays exactly where I left it. Future reward is discounted by β, with 0 < β < 1. I see what happens before I choose again. I want a rule that says, at every step, which arm to push on, to maximize total expected discounted reward over an infinite horizon.

The honest first move is just to write down the dynamic program, because this is a Markov decision process and I know those have optimal stationary policies — Blackwell proved that for discounted MDPs with finite action sets and bounded rewards the supremum is attained by a deterministic stationary Markov policy solving the functional equation. So the answer exists and it satisfies

  R(F, x) = max over actions of { immediate reward + β · E[ R(F, next state) ] },

where x is the state of the whole family and the action is which arm to advance. Fine. Now I try to actually compute it and I hit the wall immediately. The state of the family is the *joint* state — one coordinate per arm. If each arm has k states and I have n arms, the joint state space is k^n. And for the case I care most about, the Bayesian clinical-trial bandit, each arm's state is a posterior, say exponents (a,b) with density proportional to θ^a(1−θ)^b, and those exponents ratchet every time I pull. So each arm's state space is already infinite, and the product of n of them is hopeless. Backward induction over this product is exponential in n. I can solve two arms, maybe. I cannot solve ten. This is exactly Bellman's own two-armed-bandit analysis from the fifties: he got properties of the optimal policy for n = 2, one arm known and one unknown, and there it stopped, because there was no structure that scaled.

So the brute force is out, and I need structure. Let me stare at what's special about *this* MDP, because not every MDP can have nice structure or the whole field of dynamic programming would be trivial.

The thing I keep coming back to is that, in a general MDP, choices are irrevocable: when I pick a fork in the road, the roads I didn't take recede behind me, and the state of the world moves on regardless of what I do. But in my problem the arms I *don't* touch don't move. They're frozen. An arm I decline to pull right now is sitting there, untouched, and if I come back to it in ten steps it will give me the *identical* sequence of rewards it would have given me now — only shifted ten steps later in time, hence only discounted by an extra β^10. Nothing is lost by waiting except discounting. Compare that to choosing a route for a car journey: there, if I take the motorway I'm committed, the slow road I skipped isn't waiting unchanged at the next junction — the unchosen exits are gone. The car journey is a Markov decision process where forwards-greedy reasoning fails precisely because the decisions are irrevocable. My problem is the opposite. The decisions are *not* irrevocable. There is no later disadvantage to compensate for the immediate disadvantage of postponing an arm.

That asymmetry has to be the whole ballgame, so let me lean on it hard. Suppose I'm at the start and I'm wondering how good arm D is, in its current state, considered entirely on its own. I want a number — one scalar per arm, computed from that arm alone, ignoring the others — such that the right thing to do is always to pull the arm with the biggest number. If such a number exists, the k^n problem collapses to n separate one-dimensional problems, and that would be the entire game.

What would the number have to measure? Let me think about what I actually get from committing to an arm for a while. If I start pulling arm D in state x and keep pulling it for some number of steps and then stop, I collect a stream of discounted rewards, and I've spent a stream of discounted "time." The natural figure of merit is reward per unit time — but discounted, since that's the currency the objective is in. So define, for a single arm D in state x,

  ν(x) = sup over stopping rules τ ≥ 1 of  E[ Σ_{t=0}^{τ-1} β^t R(x(t)) | x(0) = x ]  /  E[ Σ_{t=0}^{τ-1} β^t | x(0) = x ].

The numerator is the expected discounted reward I rack up if I commit to D and pull it until the stopping time τ; the denominator is the expected discounted *time* I spend doing it. I optimize over when to stop, τ being any rule that depends only on what I've seen so far. I force τ ≥ 1 so the denominator isn't zero and so the number means "the value of *starting* to play." This is the best discounted reward-rate the arm can offer me, if I get to choose the most favorable window in which to enjoy it. Let me call it the index of D in state x.

Why discounted time in the denominator, not just the count of steps? Because the whole problem is discounted, and the trade-off I'm pricing is "is the reward I'd get in this window worth the discounted time it costs me." A perpetuity that pays a constant c per step is worth c·Σβ^t = c/(1−β); its reward-rate, by this very formula, is c·Σβ^t over Σβ^t = c. If I'd used plain step-count in the denominator the rate of a constant-c arm would come out as c·Σβ^t / E[τ], which depends on how long I happen to play and doesn't equal c — the units wouldn't line up and the comparison I'm about to build would not close. Discounted reward over discounted time is the only ratio for which a constant arm has reward-rate equal to its constant. Hold that thought, because the constant arm is about to become the measuring stick.

Now, does the supremum-of-a-ratio number actually govern the multi-arm problem? Let me build the intuition before I trust it. Forget cleverness; think about the first stage of play. Suppose I decide on a window — a stopping time — and over that window I'm allowed to spread my pulls across several arms. The expected discounted reward per unit of discounted time over that window is some *weighted average* of the per-arm reward-rates of the arms I touched, weighted by how much discounted time each got. But a weighted average is never larger than the largest thing being averaged. So I can never do better, in reward-rate terms over the first window, than to pour the entire window into the single arm with the highest reward-rate, and stop. There is no point averaging two arms; the lower one only drags me down. So the first stage of an optimal-rate policy continues exactly one arm — the one whose ν is largest — up to the stopping time that achieves its supremum. And by the non-irrevocability I keep harping on, doing the high-rate thing first costs me nothing later: the arms I postponed are unchanged, waiting. So I repeat the argument from the new state, and again I should pour effort into the currently-highest-ν arm. That's the policy: at every step, pull the arm of greatest ν. The index decouples the problem.

I want to nail down the stopping rule hiding in that ν, because the policy depends on it. When I commit to arm D from state x, until when should I keep pulling to realize the supremum reward-rate? The candidate is to keep going as long as the arm still looks at least as good as it did when I started, and stop the moment it looks worse. In symbols, the stopping time is

  τ_x = inf{ t ≥ 1 : ν(x(t)) < ν(x) },

so the stopping set is { states y : ν(y) < ν(x) }. I need to prove it carefully, because there is an easy trap here: the ratio of an arbitrary prefix is not automatically ν(x). The safe move is to compare prefixes and tails only through the elementary mediant inequality. For positive a, b, c, d,

  a/c < (a+b)/(c+d) < b/d  ⇐⇒  a/c < b/d.

First, suppose a candidate stopping time σ stops, with positive probability, in a state j whose index is higher than the starting index: ν(j) > ν(x). From j I can choose a further stopping time whose reward-rate is strictly between ν(x) and ν(j). On those paths I append a better tail to the old prefix; the mediant inequality says the combined ratio rises. So a stopping time that stops while the current state has index above ν(x) cannot attain the supremum.

Second, suppose a candidate stopping time keeps playing after the first time it reaches a state j with ν(j) < ν(x). Condition on that first hit. From j onward, no tail can have reward-rate above ν(j), by the definition of ν(j). If the original stopping time is close enough to optimal from x, its overall ratio is above ν(j), so that tail is dragging the ratio down; replacing the original stop by the first hit cuts off a lower-ratio tail and raises the whole ratio, again by the mediant inequality. So an optimal stopping time cannot pass through a lower-index state before stopping.

Equal-index states are the boundary: including or excluding them does not change the ratio. With the higher-index and lower-index cases both controlled, the supremum in ν is attained by τ_x, and also by any stopping time that stops no later than τ_x and only stops with boundary index at most ν(x). The operational rule is the one I wanted: play until the index first drops below its starting value.

I have a policy and an intuition. But "weighted averages and non-irrevocability" is a heuristic, and I've seen heuristics for priority rules before — the cμ rule, Sevcik's job-scheduling index — that are optimal only in special myopic cases and fail when the future matters. I need a proof that greedy-on-ν is genuinely optimal, not just plausible, for the general stochastic family. And I'd like the proof to *explain* why, not just verify it.

Let me try to make the per-arm number concrete in a way I can compute and reason about, because the sup-over-stopping-times definition is a little abstract. Pair my arm D against an artificial arm that pays a *constant* known amount λ every single step, forever, regardless — a "standard" arm whose state never changes and whose index, by my own formula, is just λ (constant reward-rate λ). Now consider the trivial two-arm family {D, standard(λ)}. By the policy I conjectured, I should start on whichever has the larger index: start on D if ν(D) > λ, start on the constant arm if ν(D) < λ. If ν(D) = λ exactly, I'm indifferent — I could start either way and do equally well.

So the index of D is the threshold value of λ that makes me *indifferent* between beginning with D and beginning with the sure thing λ. That's a calibration. And the useful part is that this little two-arm family has the *same* state space as D alone — the standard arm has only one state, and once I switch to it I never leave it — so the dynamic program for the pair is one-dimensional, no product blow-up. Write its Bellman equation. If I retire to the constant arm immediately I collect λ/(1−β) forever. Otherwise I pull D once, collect R(x,1), and continue with the pair:

  R({D,λ}, x) = max[ λ/(1−β) ,  R(x,1) + β · E_x[ R({D,λ}, y) ] ].

The index ν(D,x) is the largest λ for which the play-on branch is still at least as good as cashing out; at the threshold, in the ordinary finite-state case, the two branches tie. I can solve this by value iteration over D's own states for trial values of λ and bisect to that threshold. So the index is computable arm-by-arm, cheaply, exactly as I wanted.

Now let me reread my own indifference picture and I think the constant arm wants to be a *charge*, not a competing reward. Flip the sign. Instead of "an arm that pays λ," think "a fee λ I pay every time I pull D." Then ν(D,x) is the largest fee per pull at which I'd *still* be willing to play D at all — the break-even fee. Below that fee, optimal play of D is strictly profitable; above it, every way of playing D loses money. Let me write that down as the defining property, in net-of-charge form. Define the fair charge γ(x) as

  γ(x) = sup{ γ : sup over stopping rules τ ≥ 1 of  E[ Σ_{t=0}^{τ-1} β^t ( R(x(t)) − γ ) | x(0) = x ] ≥ 0 }.

In words: the largest constant per-pull charge γ such that, playing optimally and stopping when it's no longer worth continuing, my expected discounted (reward minus charge) is still nonnegative — exactly break-even at the top. This is the same number as my reward-rate ν. To see it, the inside condition sup_τ E[ Σ β^t (R − γ) ] ≥ 0 rearranges, since Σβ^t(R−γ) = Σβ^t R − γ Σβ^t, to: there exists τ with E[Σβ^t R] ≥ γ E[Σβ^t], i.e. γ ≤ E[Σβ^t R]/E[Σβ^t] for some stopping rule. The largest γ for which some stopping rule clears this bar is precisely the supremum of the reward-rate ratio — my ν. So fair charge = best discounted reward-rate. Two names for one number. (If I want it on a "lump sum" scale I can divide by (1−β) — a perpetuity of γ per period is worth γ/(1−β) — and call G(x) = γ(x)/(1−β); that's pure rescaling and never changes which arm is biggest, so it never changes the policy. I'll carry γ for the proof since the per-pull picture is cleaner.)

Now the charge formulation hands me a proof of optimality that actually explains itself, and it turns on a property I almost overlooked. Let g_t = γ(x(t)) be the fair charge after t pulls of a single arm, and let the prevailing charge be the running minimum

  \bar g_t = min_{0≤s≤t} g_s.

Start at g_0 = γ(x). At that fixed charge, optimal play breaks even. If the fair charge later falls below the fixed charge, I would stop; if instead I lower the charge to the new fair charge, I am again at break-even from that state. Repeating this gives the running-minimum charge sequence above. It is nonincreasing in the number of pulls, because it only moves downward; it is random, because the states are random; and it is policy-independent, because the k-th charge this arm will ever present is determined only by this arm's own first k transitions. Interleaving other arms changes calendar time, not this arm's sequence of charges.

The single-arm inequality I need is precise. If I play this arm up to any stopping time σ while being charged its prevailing charge, then

  E[ Σ_{t=0}^{σ-1} β^t R(x(t)) ] ≤ E[ Σ_{t=0}^{σ-1} β^t \bar g_t ].

Equality holds exactly when I stop only at times where the fair charge has come down to the prevailing charge; stopping while g_t > \bar g_t leaves positive continuation value on the table. This is the fair-game statement with the sign right: the prevailing charges are an upper bound on the reward earned by any way of playing the arm, and optimal play makes the bound tight. The same inequality still holds if the arm is played intermittently, because the arm's state is frozen between its own pulls and the pull times are measurable from information outside the arm plus the arm's observed past.

Now bring back all n arms, each with its own random prevailing-charge stream. For any policy, summing the single-arm inequalities over arms gives

  expected discounted reward ≤ expected discounted charges paid.

So charges give a universal upper bound on what any policy can earn. The only remaining question is which policy makes that upper bound as large as possible. Whatever policy I use, the charges I pay are an interleaving of n nonincreasing streams. A discounted sum puts larger weight on earlier terms, so the largest possible charge total is the nonincreasing rearrangement of all those stream values.

Now check what the maximal-index rule does to the paid charges. The index is the current fair charge. If the arm I am playing has fair charge strictly above its prevailing charge, the stopping-set result says I should keep playing it; frozen arms cannot change, and any arm previously left idle at a reset has fair charge equal to its prevailing charge. A switch happens only when the played arm's fair charge has fallen to its prevailing charge, and every idle arm's fair charge is no larger than the one just abandoned. Therefore the next charge paid cannot exceed the previous charge paid. The maximal-index rule produces a nonincreasing paid-charge sequence, so it realizes the largest possible discounted charge total. It also never leaves an arm idle while its fair charge is strictly above its prevailing charge, so the single-arm reward-versus-charge inequality is tight. It both maximizes the charge upper bound and earns exactly that bound. Therefore the maximal-index policy is optimal. The index of each arm is computed from that arm alone; the joint state never enters. The charges of arm j do not care what I am doing with arm i, so the problem genuinely splits.

Let me sanity-check the index against cases I can compute by hand, to make sure the abstract definition reduces to sensible things.

Take an arm that only gets worse the more I pull it — after one pull, the next state's index is almost surely no larger than the current one. Then any tail after the first pull has reward-rate no better than what I already have available now, so the mediant comparison says the tail cannot improve the one-step ratio. The supremum is achieved at τ = 1, and

  ν(x) = R(x,1),

the immediate expected reward. The index degenerates to "pull the arm with the largest immediate reward" — the one-step-lookahead rule is optimal in this deteriorating case. That's reassuring, and it's exactly the monotone case from optimal stopping where myopic play is optimal. Scheduling jobs with a non-increasing completion hazard p(t) is of this type: the index becomes p_i(t)·V_i, the expected immediate completion reward, and pulling the largest of those is optimal — which is the classical priority rule, now derived rather than assumed.

Take the opposite, an arm that only gets better as I pull it. Then the lower-index stopping set is not reached before the relevant improvement point, so I should not stop early; the index is the discounted reward divided by discounted time over that whole improving run, stopping only when the process reaches the first state where the comparison no longer improves. And take the genuinely interesting middle, the Bayesian Bernoulli arm with posterior exponents (a,b): a success moves to (a+1,b) and makes me more optimistic, while a failure moves to (a,b+1) and makes me less optimistic. Here the index strictly exceeds the immediate expected reward (a+1)/(a+b+2), because there is option value in continuing to learn — a myopic rule would undervalue exactly the arms still worth exploring. The index can be computed by the calibration: truncate the horizon at a large N, force states with a+b = N−1 to stop so there ν = (a+1)/(a+b+2), then sweep backward, at each layer solving the indifference equation against a standard arm and refining ν(a,b). As N grows the truncated index increases to the true one. Each iteration is over the arm's own two-parameter state, not the product of n of them.

And to see the payoff in dimensionality crisply, take Normal rewards: an arm whose unknown mean has a N(ξ, m^{-1}) posterior, observations of known variance. Two elementary facts — adding a constant to all rewards adds it to every weighted average, so ν(ξ,m,β) = ξ + ν(0,m,β); and multiplying all rewards by a constant multiplies every weighted average by that constant, so ν(0,m,β) = β · ν(0,m,1) — give the index a shift-and-scale law,

  ν(ξ, m, β) = ξ + β · ν(0, m, 1),

so I only ever need to tabulate the single-variable function ν(0, m, 1). Computing the index this way iterates over functions of *one* real variable. Solving the original family by the joint Bellman equation would iterate over functions of 2n real variables and is impracticable past n = 2. That gap — one variable versus 2n — is the concrete cash value of the decoupling.

So the causal chain, start to finish: I want one rule for committing a single resource among competing Markov reward processes, the exact dynamic program exists but its state is the product of the arms' states and is exponential to solve; the arms I don't touch stay frozen, so postponing an arm costs only discounting and never forecloses its future, which means choices aren't irrevocable; that lets me ask how good each arm is alone, and the right scalar is the best discounted reward-per-discounted-time it can deliver over a stopping time of my choosing — equivalently the largest constant per-pull charge at which optimal play still breaks even; pairing any arm against a constant yardstick calibrates this number through a one-dimensional Bellman equation; and the prevailing-charge construction gives a policy-independent nonincreasing charge stream for each arm. Every policy's reward is bounded by the charges it pays; the maximal-index policy sorts those charges as early as possible and makes the reward-charge bound tight. The exponential family problem becomes n independent scalar computations.

For a single Markov reward process D with state x, discount β ∈ (0,1), and per-step expected reward R(x), I can state the final scalar cleanly:

  ν(x) = sup_{τ ≥ 1}  E[ Σ_{t=0}^{τ-1} β^t R(x(t)) | x(0)=x ] / E[ Σ_{t=0}^{τ-1} β^t | x(0)=x ]

and the same scalar is the fair charge

  γ(x) = sup{ γ : sup_{τ ≥ 1} E[ Σ_{t=0}^{τ-1} β^t ( R(x(t)) − γ ) | x(0)=x ] ≥ 0 },

with γ(x)=ν(x). On the lump-sum scale, G(x)=γ(x)/(1−β). The supremum is attained by continuing until the index first falls below ν(x). For a family of independent such processes — exactly one advanced per step, the rest frozen — advancing a process whose current index is maximal at every step is optimal.

```python
# Compute a single process's index by calibration against a constant-charge
# yardstick, then play the family greedily on the index. One process at a time;
# the joint state never appears.

def gittins_index_by_calibration(states, transition, reward, beta,
                                 lo, hi, tol=1e-9):
    """Index of every state of ONE process D.
    Calibrate: lambda reaches the index threshold of state x when the two arms
    of the pair's Bellman max are equal there, i.e. retiring to a
    constant-lambda arm (worth lambda/(1-beta)) ties with playing D on and
    continuing.
      R_pair(x) = max[ lambda/(1-beta),  R(x) + beta * E_x[R_pair(y)] ].
    For fixed lambda, value-iterate R_pair over D's OWN states; the index of x
    is the largest lambda for which it is still optimal to PLAY (left arm not
    strictly bigger). Bisection per state recovers that crossing lambda."""
    def plays_on(x0, lam):
        # value iteration on the pair, restricted to D's states (1-D, no product)
        V = {s: reward(s) / (1 - beta) for s in states}
        retire = lam / (1 - beta)
        while True:
            newV, delta = {}, 0.0
            for s in states:
                cont = reward(s) + beta * sum(p * V[s2]
                                              for s2, p in transition(s).items())
                newV[s] = max(retire, cont)         # the pair's Bellman max
                delta = max(delta, abs(newV[s] - V[s]))
            V = newV
            if delta < tol:
                break
        # at the start state x0, is playing D on at least as good as retiring?
        cont0 = reward(x0) + beta * sum(p * V[s2]
                                        for s2, p in transition(x0).items())
        return cont0 >= retire                      # True => index(x0) >= lam

    index = {}
    for x0 in states:
        a, b = lo, hi                               # bracket [lo, hi] for lambda
        while b - a > tol:                          # largest lam with plays_on
            m = 0.5 * (a + b)
            if plays_on(x0, m):
                a = m
            else:
                b = m
        index[x0] = a
    return index


def greatest_index_policy(processes, states, indices):
    """Advance the process whose current state has the largest index."""
    return max(range(len(processes)), key=lambda j: indices[j][states[j]])
```
