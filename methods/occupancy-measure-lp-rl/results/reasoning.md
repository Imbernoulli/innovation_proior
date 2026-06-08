Let me start from the thing that actually bothers me, which is the *form* of the problem and not any particular instance. I have a Markov decision process — states $s$, actions $a$, a transition law $P(s'\mid s,a)$, a per-step reward $r(s,a)$, a start distribution $\mu$, a discount $\gamma<1$ — and I want the policy $\pi$ that maximizes $J(\pi)=\mathbb{E}[\sum_t\gamma^t r(s_t,a_t)]$. The textbook answer is dynamic programming: the optimum is the fixed point of $V^\star=TV^\star$, where $(TV)(s)=\max_a[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')]$, and I solve it by value iteration or policy iteration. That works. But I keep staring at it and feeling that the *structure* is wrong for the tools I trust most. There is a $\max_a$ buried inside $T$, so the equation I have to solve is nonlinear. And if I instead think of $J$ as a function of $\pi$ directly and try to optimize over the policy simplex, the map from $\pi$ to return runs through $(I-\gamma P_\pi)^{-1}$ — a matrix inverse that depends on $\pi$ — so $J(\pi)$ is some non-convex rational function of $\pi$. I have a whole arsenal of linear-programming machinery — duality, simplex, dual prices, sensitivity analysis — that I can't point at any of this, because nothing here is linear or even convex in the obvious variables. So the real question is: can I rewrite "find the optimal policy" as a linear program? If I can, the optimum stops being a fixed point I iterate toward and becomes the solution of a single numerical program.

Let me attack the nonlinearity first, because the $\max_a$ is the visible culprit. The optimality equation says $V^\star(s)=\max_a[\,r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V^\star(s')\,]$. A maximum over a finite set is the smallest number that dominates every element of the set. So $V^\star(s)$ is the smallest $V(s)$ such that $V(s)\ge r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')$ holds for *every* $a$. That single nonlinear equation has just unpacked into $|S||A|$ inequalities, and every one of them is linear in the unknowns $V$. The $\max$ is gone — absorbed into "one inequality per action, and the value sits at the tightest one."

I should be careful that pushing the values down to satisfy these inequalities actually lands me on $V^\star$ and not somewhere else. Write the inequalities collectively as $V\ge TV$. Suppose $V\ge TV$. By monotonicity of $T$ — if $V'\le V$ then $TV'\le TV$, which is immediate because raising the future values can only raise the one-step lookahead — I can apply $T$ repeatedly: $V\ge TV\Rightarrow TV\ge T^2V\Rightarrow\cdots$, so $V\ge TV\ge T^2V\ge\cdots\ge T^\infty V$. And $T$ is a $\gamma$-contraction, so $T^k V\to V^\star$ regardless of where I started. Therefore any $V$ satisfying $V\ge TV$ already satisfies $V\ge V^\star$, coordinatewise. So $V^\star$ is the *pointwise least* element of the feasible set $\{V: V\ge TV\}$.

That's exactly the setup for a minimization. If I minimize any strictly positively-weighted sum $\sum_s c(s)V(s)$ with $c(s)>0$ over $\{V\ge TV\}$, the optimizer must be $V^\star$ — because $V^\star$ is below every feasible $V$ in every coordinate, so it minimizes every positive combination simultaneously, and there's nothing feasible below it. For the start-distribution problem I actually care about, the natural weights are not an arbitrary full-support vector but $(1-\gamma)\mu$. If $\mu$ assigns zero weight to states that never matter from the start distribution, I should not pretend the LP objective uniquely determines their values; what remains true is enough: every feasible $V$ lies above $V^\star$, $V^\star$ itself is feasible, and the optimum objective is $(1-\gamma)\mu^\top V^\star$. The factor $(1-\gamma)$ is harmless for the primal argmin and will make the dual flow normalize cleanly. So I'll write

$$\min_V\ (1-\gamma)\sum_s\mu(s)V(s)\quad\text{s.t.}\quad V(s)\ge r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')\ \ \forall s,a.$$

This is a linear program. $|S|$ variables, $|S||A|$ linear constraints, linear objective. With full-support objective weights it pins down $V^\star$ uniquely; with the start-distribution objective it pins down the optimal start value and has $V^\star$ as an optimal solution. A greedy one-step lookahead on $V^\star$ gives an optimal deterministic policy. Good — I've turned planning into an LP. But this LP has the value function as its variable, and what I actually wanted was the *policy*, and a convex view of the *policy* search. The greedy read-off at the end still hides a $\max$. So this can't be the whole story. Let me not stop here.

The instant I have that LP, the next object to inspect is its dual. The primal has one constraint per $(s,a)$ pair, so the dual has one variable per $(s,a)$ pair — call it $\lambda(s,a)\ge 0$, nonnegative because the primal constraints are inequalities of the form (something $\le V(s)$). Let me actually form the Lagrangian and read the dual off, rather than quote a formula. Using the start-distribution objective weights $c(s)=(1-\gamma)\mu(s)$, the Lagrangian is

$$L(V,\lambda)=(1-\gamma)\sum_s\mu(s)V(s)+\sum_{s,a}\lambda(s,a)\Big[r(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V(s')-V(s)\Big],\quad\lambda\ge0.$$

$V$ is a free variable here, so for the dual to be bounded the coefficient of each $V(s')$ must vanish — that stationarity condition *is* the dual constraint. Collect the coefficient of a fixed $V(s')$. It appears in $(1-\gamma)\mu(s')V(s')$, in the $-\lambda(s',a)V(s')$ terms (the $-V(s)$ piece with $s=s'$), and in the $+\gamma\lambda(s,a)P(s'\mid s,a)V(s')$ terms (the lookahead piece, summed over the predecessors $(s,a)$). Setting the total coefficient to zero:

$$(1-\gamma)\mu(s')-\sum_a\lambda(s',a)+\gamma\sum_{s,a}P(s'\mid s,a)\lambda(s,a)=0,$$

which rearranges to

$$\boxed{\ \sum_a\lambda(s',a)=(1-\gamma)\mu(s')+\gamma\sum_{s,a}P(s'\mid s,a)\lambda(s,a)\ }\qquad\forall s'.$$

And whatever is left of $L$ after the $V$-terms cancel is $\sum_{s,a}\lambda(s,a)r(s,a)$. So the dual is

$$\max_{\lambda\ge0}\ \sum_{s,a}\lambda(s,a)r(s,a)\quad\text{s.t.}\quad\sum_a\lambda(s',a)=(1-\gamma)\mu(s')+\gamma\sum_{s,a}P(s'\mid s,a)\lambda(s,a)\ \ \forall s'.$$

Now I want to understand what I'm looking at, because right now $\lambda$ is just "the dual variable." Stare at the constraint. On the left, the total $\lambda$-mass sitting on state $s'$ (summed over the action taken there). On the right: a fixed injection $(1-\gamma)\mu(s')$, plus $\gamma$ times the $\lambda$-mass that *arrives* at $s'$ from every predecessor $(s,a)$ weighted by the transition probability. Mass into $s'$ equals injected mass plus discounted mass flowing in from upstream. That is a flow-balance equation. It is the discounted, controlled version of the stationary-distribution balance $d^\top=d^\top P_\pi$ I already know from Markov-chain theory — except with a source term $(1-\gamma)\mu$ feeding the start distribution in and a $\gamma$ throttling the recirculated flow. The dual variables are *flows*. They are sitting on $(s,a)$ pairs, they're nonnegative, and they obey a continuity equation.

So let me guess what $\lambda$ is and then verify, because the form is suggestive. The natural flow object for a policy $\pi$ is its discounted visitation frequency. Define

$$\lambda^\pi(s,a)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t\,\mathbb{P}[s_t=s,a_t=a\mid s_0\sim\mu,\pi].$$

The $(1-\gamma)$ out front is bookkeeping I want to check: $\sum_{s,a}\lambda^\pi(s,a)=(1-\gamma)\sum_t\gamma^t\cdot 1=(1-\gamma)\cdot\frac{1}{1-\gamma}=1$. So $\lambda^\pi$ is a probability distribution over $S\times A$ — that normalization is *why* I put the $(1-\gamma)$ there, and now I also see why I chose the primal weights $c=(1-\gamma)\mu$: it makes the right-hand side of the dual constraint sum to $(1-\gamma)\sum_{s'}\mu(s')=1-\gamma$, the constant that exactly turns the flow equations into "$\lambda$ lives in the simplex." Let me confirm $\lambda^\pi$ satisfies the flow equation, not just the normalization. Condition on the first step. At $t=0$ the mass at $(s,a)$ is $(1-\gamma)\mu(s)\pi(a\mid s)$. For $t\ge1$, the probability of being at state $s$ at time $t$ is $\sum_{s',a'}P(s\mid s',a')\cdot\mathbb{P}[s_{t-1}=s',a_{t-1}=a']$, and then action $a$ is taken with $\pi(a\mid s)$. Folding the $\gamma^t$ and resumming,

$$\lambda^\pi(s,a)=(1-\gamma)\mu(s)\pi(a\mid s)+\gamma\,\pi(a\mid s)\sum_{s',a'}P(s\mid s',a')\lambda^\pi(s',a').$$

Sum this over $a$. Since $\sum_a\pi(a\mid s)=1$, the $\pi(a\mid s)$ factors collapse and I get exactly $\sum_a\lambda^\pi(s,a)=(1-\gamma)\mu(s)+\gamma\sum_{s',a'}P(s\mid s',a')\lambda^\pi(s',a')$ — the dual flow constraint. So *every* policy's discounted occupancy is dual-feasible. The dual variables aren't just "flows" by analogy; they are precisely the occupancy measures of policies.

This is the moment where the whole thing starts to flip for me. I came in trying to optimize over policies and the difficulty was that $J(\pi)$ is non-convex in $\pi$. But for the $\lambda$ objects that actually come from policies, the dual objective $\sum_{s,a}\lambda(s,a)r(s,a)$ is *linear*. Re-encode a policy as its discounted visitation frequency and the return, which was a rational function of $\pi$, becomes a flat linear functional of $\lambda$. Let me verify the objective really is the return so I'm not fooling myself:

$$(1-\gamma)V^\pi(\mu)=(1-\gamma)\,\mathbb{E}\Big[\sum_t\gamma^t r(s_t,a_t)\Big]=(1-\gamma)\sum_{s,a}r(s,a)\sum_t\gamma^t\mathbb{P}[s_t=s,a_t=a]=\sum_{s,a}r(s,a)\lambda^\pi(s,a)=\langle\lambda^\pi,r\rangle.$$

So on the policy-generated occupancies, $\langle\lambda,r\rangle$ is exactly $(1-\gamma)$ times discounted return. If the flow constraints contain exactly those occupancies and nothing else, then the dual LP *is* policy optimization, written convexly.

I have to be honest about a gap before I celebrate, though. I've shown each policy gives a feasible $\lambda$ — that's one direction. But the dual ranges over *all* nonnegative $\lambda$ satisfying the flow constraints, and an LP optimum could in principle land on a feasible point that isn't the occupancy of any policy at all, in which case "$\max\langle\lambda,r\rangle$ over the polytope" would be optimizing over phantom points and the recovered policy could be meaningless. So I need the converse: every feasible $\lambda$ is the occupancy of some policy. That's the real content — the claim that the flow polytope *is* the set of valid occupancies, exactly, with nothing extra.

Take any $\lambda\ge0$ satisfying the flow equations. First, it's automatically normalized: sum the $|S|$ flow equations over $s'$. The left side is $\sum_{s'}\sum_a\lambda(s',a)=\sum_{s,a}\lambda(s,a)$. The right side is $(1-\gamma)\sum_{s'}\mu(s')+\gamma\sum_{s,a}\lambda(s,a)\big(\sum_{s'}P(s'\mid s,a)\big)=(1-\gamma)+\gamma\sum_{s,a}\lambda(s,a)$, using $\sum_{s'}P(s'\mid s,a)=1$. Setting the two equal: $\sum\lambda=(1-\gamma)+\gamma\sum\lambda$, so $(1-\gamma)\sum\lambda=(1-\gamma)$, hence $\sum_{s,a}\lambda(s,a)=1$. The flow constraints by themselves force the total mass to one — the $\gamma$ throttling and the $(1-\gamma)$ source are exactly balanced. Now $\lambda(s):=\sum_a\lambda(s,a)$ is the state marginal, and where it's positive I can define

$$\pi_\lambda(a\mid s)=\frac{\lambda(s,a)}{\sum_b\lambda(s,b)}.$$

This is just Bayes' rule: if $\lambda(s,a)=\lambda(s)\pi(a\mid s)$ for some policy, then conditioning on the state recovers $\pi$. (Where $\lambda(s)=0$ the state is never visited, so the policy there is irrelevant and I set it arbitrarily.) The thing I have to check is that the occupancy *generated by* $\pi_\lambda$ comes back to the very $\lambda$ I started with — that the map policy$\to$occupancy and this recovery are inverse on the polytope. Plug $\pi_\lambda$ into the occupancy fixed-point I derived above: $\lambda^{\pi_\lambda}$ is the unique solution of $x(s,a)=(1-\gamma)\mu(s)\pi_\lambda(a\mid s)+\gamma\pi_\lambda(a\mid s)\sum_{s',a'}P(s\mid s',a')x(s',a')$. But my starting $\lambda$ satisfies exactly this: by definition $\lambda(s,a)=\pi_\lambda(a\mid s)\sum_b\lambda(s,b)=\pi_\lambda(a\mid s)\big[(1-\gamma)\mu(s)+\gamma\sum_{s',a'}P(s\mid s',a')\lambda(s',a')\big]$, the bracket being the flow equation it obeys. So $\lambda$ solves the same fixed-point system that $\lambda^{\pi_\lambda}$ uniquely solves. The uniqueness is just discounted contraction: the operator $M_{\pi_\lambda}$ with $(M_{\pi_\lambda}x)(s,a)=\pi_\lambda(a\mid s)\sum_{s',a'}P(s\mid s',a')x(s',a')$ is nonnegative with column sums one, so $\|\gamma M_{\pi_\lambda}\|_1=\gamma<1$ and $I-\gamma M_{\pi_\lambda}$ is invertible. Therefore $\lambda^{\pi_\lambda}=\lambda$. Both directions hold. The feasible set of the dual is *precisely* the set of discounted occupancy measures of stationary policies — the flow constraints characterize valid occupancies exactly, no phantoms.

Now everything closes. To plan: solve the dual LP for an optimal occupancy $\lambda^\star$, then read off

$$\pi^\star(a\mid s)=\frac{\lambda^\star(s,a)}{\sum_b\lambda^\star(s,b)}.$$

Why is this $\pi^\star$ optimal and not merely feasible? Strong duality. Both the primal and dual are linear programs; the primal is feasible (take $V$ large) and bounded below by $(1-\gamma)\mu^\top V^\star$, because every feasible $V$ lies above $V^\star$ coordinatewise and $\mu\ge0$. Hence LP duality gives $\mathrm{val(dual)}=\mathrm{val(primal)}=(1-\gamma)\mu^\top V^\star=(1-\gamma)J^\star$. Since $\langle\lambda^\star,r\rangle$ attains this value and equals $(1-\gamma)$ times the return of $\pi_{\lambda^\star}$, the policy $\pi_{\lambda^\star}$ achieves the optimal return. There's a complementary-slackness reading too that I like: $\lambda^\star(s,a)>0$ forces the corresponding primal constraint tight at the optimal value function, $V^\star(s)=r(s,a)+\gamma\sum P(s'\mid s,a)V^\star(s')$, i.e. the dual puts visitation mass only on actions that are greedy with respect to $V^\star$. The occupancy and the value function are dual to each other in the literal LP sense: $\lambda$ is the dual variable of the value-function LP, and $V$ is the dual variable of the occupancy LP. Two faces of one optimum.

One more thing, because allowing randomized policies should not make me lose the deterministic optimum that dynamic programming already suggests. The flow feasible set is a polytope, and a linear objective over a polytope can be optimized at a vertex. Take any vertex $\bar\lambda$. Because it is a vertex of a polytope, there is some reward vector $\bar r$ for which $\bar\lambda$ is the unique maximizer of $\langle\lambda,\bar r\rangle$ over the flow constraints; choose a vector in the interior of its normal cone. But for that very reward vector, the Bellman optimality equation still has a deterministic greedy optimal policy: solve for $\bar V^\star$ and choose one maximizing action in each state. Its occupancy $\lambda^{\delta}$ is feasible and maximizes the same linear objective, since the objective is exactly $(1-\gamma)$ times the return. Uniqueness of the exposed maximizer forces $\bar\lambda=\lambda^\delta$. If a state has zero marginal, the action chosen there is arbitrary because it contributes no occupancy. So the vertices of the discounted flow polytope are deterministic-policy occupancies. Even though I deliberately allowed randomized policies to make the feasible set convex, an LP optimum can be taken to be pure.

Let me also note the discount-factor bookkeeping is internally consistent, because it's the easiest place to make a sign or factor error. The $\gamma$ multiplies the recirculated flow on the right-hand side of the constraint, mirroring the $\gamma$ that multiplies the future value in the primal Bellman inequality — that's the same $\gamma$ appearing on the two sides of the duality. The $(1-\gamma)$ appears in exactly two roles: as the normalizer of $\lambda$ (so that the occupancy is a probability distribution and not the un-normalized $\sum_t\gamma^t(\cdots)=1/(1-\gamma)$ total), and equivalently as the primal weight $c=(1-\gamma)\mu$ that drops $1-\gamma$ into the source term so the flow equations sum to one. They are the same constant doing one job from two sides. If I'd dropped it, I'd still have a valid LP over un-normalized occupancies $\hat\lambda=\lambda/(1-\gamma)$ with source $\mu$ instead of $(1-\gamma)\mu$ and objective equal to the raw return; the policy I recover is identical, since the recovery ratio is scale-invariant.

So the whole tangle I started with — nonlinear operator, non-convex objective in $\pi$ — dissolves through one change of variable. The optimal value function is the least solution of the Bellman *inequalities*, which is a linear program in $V$ (primal). Dualizing it produces a linear program in the dual variables $\lambda$, whose constraints are discounted flow-balance equations, whose feasible set is exactly the occupancy measures of policies, and whose linear objective $\langle\lambda,r\rangle$ is the discounted return. Solve the dual, recover the policy by normalizing the occupancy within each state, and strong duality certifies it optimal — with the optimum automatically a deterministic policy. Planning is a linear program over state–action visitation frequencies.

The empty "solve for an optimal policy" slot in the planning harness now has a concrete fill-in.

```python
import numpy as np
from scipy.optimize import linprog

# An MDP M = (P, r, mu, gamma):  P[s,a,s'] = Pr(s'|s,a); r[s,a]; mu[s]; gamma in [0,1).

def solve_primal_value_LP(P, r, mu, gamma):
    """Primal: min (1-gamma) <mu, V>  s.t.  V(s) >= r(s,a) + gamma sum_s' P(s'|s,a) V(s').
    Returns an optimal value-LP solution; with full-support weights this pins V* itself."""
    nS, nA = r.shape
    c = (1.0 - gamma) * mu                                  # objective weights give the scaled start value
    # constraint r(s,a) + gamma sum P V <= V(s)  ==>  (gamma P - E) V <= -r, in linprog's A_ub V <= b_ub form
    A_ub = np.zeros((nS * nA, nS))
    b_ub = np.zeros(nS * nA)
    row = 0
    for s in range(nS):
        for a in range(nA):
            A_ub[row, :] = gamma * P[s, a, :]               # gamma * P(.|s,a)
            A_ub[row, s] -= 1.0                             # minus V(s)  (the E matrix copies V(s))
            b_ub[row] = -r[s, a]
            row += 1
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * nS, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x                                            # optimal value-LP solution

def solve_occupancy_dual_LP(P, r, mu, gamma):
    """Dual: max <lambda, r>  s.t.  sum_a lambda(s,a) = (1-gamma) mu(s) + gamma sum_{s',a'} P(s|s',a') lambda(s',a'),
    lambda >= 0.  The equality constraints are the discounted Bellman-flow (continuity) constraints; their
    feasible set is exactly the occupancy measures of policies, and the objective is the (scaled) return."""
    nS, nA = r.shape
    idx = lambda s, a: s * nA + a                           # flatten (s,a) -> row in lambda
    c = -r.reshape(-1)                                      # linprog minimizes; negate to maximize <lambda, r>
    # flow:  sum_a lambda(s,a) - gamma sum_{s',a'} P(s|s',a') lambda(s',a') = (1-gamma) mu(s),  for each state s
    A_eq = np.zeros((nS, nS * nA))
    b_eq = (1.0 - gamma) * mu
    for s in range(nS):                                     # one continuity equation per state s
        for a in range(nA):
            A_eq[s, idx(s, a)] += 1.0                       # mass placed on state s (sum over a)
        for sp in range(nS):                                # minus gamma * discounted inflow from predecessors
            for ap in range(nA):
                A_eq[s, idx(sp, ap)] -= gamma * P[sp, ap, s]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (nS * nA), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x.reshape(nS, nA)                            # lambda*(s,a)

def policy_from_occupancy(lam):
    """Recover pi*(a|s) = lambda*(s,a) / sum_a lambda*(s,a)  (Bayes within each visited state)."""
    nS, nA = lam.shape
    state_mass = lam.sum(axis=1, keepdims=True)
    pi = np.where(state_mass > 1e-12, lam / np.maximum(state_mass, 1e-12), 1.0 / nA)
    return pi

def solve_optimal_policy(mdp):
    # dual route: occupancy LP -> normalize within each state -> optimal (deterministic at an LP vertex) policy
    lam_star = solve_occupancy_dual_LP(mdp.P, mdp.r, mdp.mu, mdp.gamma)
    return policy_from_occupancy(lam_star)
```
