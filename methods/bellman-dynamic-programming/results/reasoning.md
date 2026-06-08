Let me start from the thing that actually hurts, which is that I keep meeting the same kind of problem and keep solving it the same clumsy way. A quantity of stock to order this month, then next month, then the month after. A budget to split between two activities, each of which pays something and then shrinks, after which I split again. A machine to keep or replace, year after year. A continuous system to steer optimally from now to time $T$. Two gamblers spending their bankrolls to ruin each other. These look like different problems and they live in different journals, but when I squint they are all the same shape: there is a *state*, I make a *decision* that both earns me something and moves the state, and I want to maximize the total of what I earn across all the stages. The unknown is never a single number. It is a whole sequence of decisions, tangled together across time, because what's best to do now depends on what will be best later, which depends on where now leaves me.

The textbook move is to enumerate. Call a complete sequence of decisions a *policy*; for every feasible policy, run it forward, add up the return; then take the maximum over all policies. That's unimpeachable and it's useless. If there are $N$ stages and at each stage I have $k$ choices, there are $k^{N}$ sequences. Even modest $N$ and $k$ and that number is a wall. And for a continuous process — where I must decide at every instant of $[0,T]$ — the "policy" is a *function* of time and I'm maximizing over a function space. The dimension of the maximization grows with the length of the horizon. That's the heart of the discomfort: the cost is exponential in the number of *stages*, and I haven't even said anything yet about the decision at a single stage being hard.

And there's a second, worse failure that the enumerative picture hides. Suppose the transitions are random — a decision determines not the next state but a *distribution* over next states. Then a fixed sequence of decisions isn't just expensive, it's *meaningless*. I cannot commit now to my stage-3 action, because I don't yet know what state stage-2 will actually deposit me in; that's a coin not yet flipped. So the whole "enumerate all sequences" framing doesn't even apply to the stochastic case. That should be a clue. It's telling me the *sequence* is the wrong unknown.

Let me stop being a mathematician for a second and be a practical man about it. Where is the cost coming from? It's coming from my demanding too much information. I'm insisting on knowing the *entire* future sequence of decisions all at once. But do I actually need that? Picture myself sitting at some stage, in some state. To act well right now, what do I need? I need to know what to do *now*. I do not need to have already decided what I'll do three stages hence — when three stages hence arrives, I'll be in some state, and *then* I'll decide. If at every moment I know the right thing to do given where I am, I never need a pre-committed list of future moves at all. The future takes care of itself, stage by stage.

That reframes the unknown completely. The thing I should be solving for is not a sequence of decisions fixed in advance — an open-loop plan — but a *rule*: a decision as a function of the current state. A feedback rule. And notice this is exactly the object that survives the stochastic objection: a rule "in state $s$, do this" is perfectly well defined under randomness, because it reads the state that actually occurred. The open-loop sequence and the closed-loop rule coincide when everything is deterministic, but the moment there's uncertainty the rule is strictly the larger, the only sensible, class. Good. So I'm hunting for a rule on states, and the cost of that should scale with the size of a *single-stage* decision, not with $k^{N}$.

Now, how do I get a handle on the best rule? Here's where I want to be careful about what I make the central object. I could try to parameterize rules and optimize over them directly, but a rule is an awkward thing to optimize — there can be ties, several rules can be equally good, the space of rules is large. Let me instead track the *consequence* of behaving optimally, which is a single well-defined number for each starting point. Define

$$f_N(p) = \text{the } N\text{-stage return obtained by using an optimal policy, starting from state } p.$$

This is a function of the state $p$ and the number of stages $N$ to go, and nothing else — that's the whole point of having reframed around state. The maximum return is unique even when the optimal rule isn't, so $f$ is a cleaner object than the rule. And there's a duality I can already feel: if I know all the $f_N$, I can read off the optimal decisions (do whatever achieves the max), and conversely any optimal rule yields the $f_N$. So I'll make $f$ — the *value* of being in a state with so many stages to go — the thing I solve for, and recover the rule from it afterward.

Now the real question: does $f$ satisfy some relation that lets me build it up without enumerating sequences? Let me just think about what it *means* for a policy to be optimal. Suppose I'm following an optimal policy. My first decision lands me in some new state. Look at everything that happens after that first decision: it's an $(N-1)$-stage process starting from that new state. Could the tail of my optimal policy be *suboptimal* for that $(N-1)$-stage subproblem? If it were, I could swap it out — replace the tail with the genuinely optimal $(N-1)$-stage policy from that new state — and strictly increase the total return, while leaving the first decision untouched. But that would mean my original "optimal" policy wasn't optimal. Contradiction. So the tail must itself be optimal for the subproblem it faces.

That's the whole principle, and it's worth saying once, cleanly, because everything hangs on it: *an optimal policy has the property that, whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.* It's almost embarrassingly obvious once you see the swap argument — but it's exactly the lever that turns the global problem into a local one. I keep finding myself using this same exchange argument over and over to crank out an equation for each new problem; let me name the technique so I stop re-deriving it. The principle of optimality.

Let me now actually turn the principle into an equation, in the simplest case: a deterministic process, state vector $p$ in some region $D$, and a set of transformations $\{T_k\}$, where choosing decision $k$ in state $p$ sends me to $T_k(p)$. I want $f_N(p)$. Make my first decision: pick some $k$, landing at $T_k(p)$. After that I have an $(N-1)$-stage process starting from $T_k(p)$, and — by the principle — I should run it optimally, which by the *definition* of $f$ earns me exactly $f_{N-1}(T_k(p))$. So the only thing left to choose well is that first $k$, and I should choose it to maximize what follows:

$$f_N(p) = \max_k \, f_{N-1}\big(T_k(p)\big), \qquad N = 2, 3, \dots$$

with $f_1(p)$ being the one-stage optimum (just the best immediate return from $p$). That's it. That's the functional equation. No $k^{N}$ anywhere — to get $f_N$ from $f_{N-1}$ I do, at each state, a maximization over the *single* decision $k$. The exponential-in-horizon cost is gone, dissolved by the principle of optimality. I traded "optimize over all sequences" for "given the value function one stage shorter, optimize over one decision."

Let me immediately stress-test it on the case that broke enumeration: stochastic transitions. Now decision $k$ in state $p$ doesn't give a point; it gives a random new state $z$ with distribution $dG_k(p,z)$. So "the return" isn't a number, and I have to commit to *measuring* a policy by an average. Fine — define $f_N(p)$ via the *expected* $N$-stage return under an optimal policy. Apply the principle again: I take decision $k$, nature draws $z\sim dG_k(p,z)$, and from $z$ I have an $(N-1)$-stage problem whose optimal expected return is $f_{N-1}(z)$. The expected tail return for choosing $k$ is therefore the average of $f_{N-1}(z)$ over the draw,

$$\int f_{N-1}(z)\, dG_k(p,z),$$

and I pick the best $k$:

$$f_N(p) = \max_k \int f_{N-1}(z)\, dG_k(p,z), \qquad N=2,3,\dots$$

And look — the deterministic equation is just the degenerate case where $dG_k$ puts all its mass on the single point $T_k(p)$, so the integral collapses to $f_{N-1}(T_k(p))$. One equation covers both. The stochastic case, which was *impossible* under enumeration because fixed sequences are undefined, is here completely routine, because the value function is a function of *state*, and a state is exactly what nature hands me before I have to decide. The reframing earned its keep.

If I let the horizon run to infinity, the sequence $f_N$ should settle to a single function $f(p)$ that no longer carries a stage index, and the equation becomes self-referential:

$$f(p) = \max_k \int f(z)\, dG_k(p,z).$$

I'll have to come back and worry about whether that fixed-point equation actually has a solution and a unique one — there's no "last stage" to start a backward sweep from. Hold that thought.

Most of my real problems have a return earned *at each stage*, not just at the end, so let me write the equation in the form I'll actually use. Take the allocation problem: I have $x>0$, split it into $y$ and $x-y$; from $y$ I get $g(y)$ but $y$ shrinks to $ay$ (with $0<a<1$), from $x-y$ I get $h(x-y)$ but it shrinks to $b(x-y)$; then I repeat with the new total $ay+b(x-y)$. Let $f(x)$ be the total return under an optimal policy from $x$. The first split earns me $g(y)+h(x-y)$ right now and moves me to state $ay+b(x-y)$, from which I'll earn $f(ay+b(x-y))$ optimally. By the principle, choose $y$ to maximize the sum of the immediate return and the value of where it leaves me:

$$f(x) = \sup_{0\le y\le x}\Big[\, g(y) + h(x-y) + f\big(ay+b(x-y)\big)\,\Big], \qquad f(0)=0.$$

There's the general shape: *value now equals best over the current decision of (reward earned now) plus (value of the next state)*. Every one of my problems is going to be a special case of this. And notice the constraint $0\le y\le x$ sits inside the maximization without any trouble at all — I just maximize over the feasible set. That's worth flagging, because constraints are exactly what wreck the classical continuous theory, and here they cost nothing.

Which brings me to the continuous case, the calculus of variations, and whether this functional-equation viewpoint says anything there. Take the canonical problem: maximize $\int_0^T F(x,y)\,dt$ over the control $y$, where $x$ and $y$ are linked by $dx/dt=G(x,y)$ with $x(0)=c$. The classical technique treats the optimizing $y(\cdot)$ as a point in function space and characterizes it by the Euler equation — it describes $y$ as a function of *time*. I want instead to describe the decision at any instant as a function of the *state*, exactly as I've been doing. So set

$$f(c,T) = \max_y \int_0^T F(x,y)\,dt,$$

the optimal return over a duration $T$ starting from state $c$. Now apply the principle of optimality across a tiny initial slice. Split $[0,S+T]$ into $[0,S]$ and the rest. Whatever I do on $[0,S]$ earns $\int_0^S F\,dt$ and leaves me at state $c(S)$ with duration $T$ remaining, from which I optimally earn $f(c(S),T)$. So

$$f(c, S+T) = \max\Big[\, \int_0^S F(x,y)\,dt + f\big(c(S),T\big)\,\Big],$$

the max over the controls used on the initial interval. This is the principle of optimality again, now over a continuous split. Let $S\to 0$. Write $v=y(0)$ for the control applied right at the start. Over the infinitesimal slice, $\int_0^S F\,dt \approx F(c,v)\,S$, and the state moves by $c(S)\approx c + G(c,v)\,S$, so

$$f(c,S+T) \approx \max_v\Big[\, F(c,v)\,S + f\big(c + G(c,v)\,S,\; T\big)\,\Big].$$

Expand $f$ to first order in $S$: $f(c+G S, T) \approx f(c,T) + G(c,v)\,f_c\,S$, and also $f(c,S+T)\approx f(c,T)+f_T\,S$. Subtract $f(c,T)$ from both sides and divide by $S$, then let $S\to0$:

$$f_T = \max_v\big[\, F(c,v) + G(c,v)\, f_c\,\big].$$

A first-order partial differential equation for the value function. (This is the Hamilton–Jacobi-type equation, falling out of the principle of optimality rather than being imposed.) Now carry out the inner maximization over $v$ by stationarity — differentiate the bracket and set it to zero:

$$F_v + G_v\, f_c = 0.$$

I have two relations: the PDE $f_T = F(c,v) + G(c,v) f_c$ and the optimality condition $F_v + G_v f_c = 0$. The maximizing $v$ in the bracket is itself a function of the state $c$ (and of $f_c$) — that's the feedback law. Eliminate $f_c$ between the two. From the optimality condition, $f_c = -F_v/G_v$; substitute into the PDE:

$$f_T = F(c,v) - G(c,v)\,\frac{F_v(c,v)}{G_v(c,v)} = \frac{F\,G_v - G\,F_v}{G_v}\bigg|_{(c,v)}.$$

That's a first-order partial differential equation for $f(c,T)$ written entirely in the known data $F,G$ and the optimal feedback $v=v(c)$ — no trace of $f$ left on the right except through $v$ (in the nondegenerate interior case $G_v\ne0$; at a boundary, or if $G_v=0$, the maximization condition itself replaces the division). To compare this with the classical variational equations, I have to keep the time convention straight. The argument uses $T$ as *time remaining*. Along a real path with elapsed time $t$, put $\tau=T-t$ and define the multiplier

$$\lambda(t)=f_c(x(t),\tau).$$

Differentiate the PDE with respect to $c$ at the maximizing $v$; by the stationarity condition the derivative of the maximizing $v$ drops out, leaving

$$f_{Tc}=F_c+G_c f_c+G f_{cc}.$$

Along the optimal path $\dot x=G(x,v)$ and $\dot\tau=-1$, so

$$\dot\lambda=f_{cc}\dot x+f_{cT}\dot\tau
=G f_{cc}-f_{Tc}
=-F_c-G_c\lambda.$$

Together with the stationarity condition

$$F_v+G_v\lambda=0,$$

this is exactly the classical extremal system. Indeed, if I write the augmented integrand with the sign convention that keeps the multiplier equal to the marginal return,

$$\mathcal L=F(x,v)+\lambda(t)\big(G(x,v)-\dot x\big),$$

variation with respect to $v$ gives $F_v+\lambda G_v=0$, and the Euler equation for $x$ gives $d(-\lambda)/dt=F_x+\lambda G_x$, or $\dot\lambda=-F_x-\lambda G_x$. So the costate is not a new object hiding outside the state formulation; it is $f_c$, the marginal value of the state. My functional-equation approach doesn't contradict the calculus of variations — it contains it: the Euler equation is what I read from the characteristics of the value function's PDE. That's reassuring, but the real payoff is at the edges where the classical theory fails. Add the constraint $0\le y\le x$ — an allocation can't exceed what's on hand. Free variation is no longer allowed when $y$ sits at a boundary $0$ or $x$, so the Euler equalities must be replaced by inequalities, and the classical machinery stumbles. But my equation doesn't care: the maximization $\max_v[F(c,v)+G(c,v)f_c]$ is just taken over the feasible set for $v$, boundary or not. Same equation, constrained max. And the stochastic continuous case the variational calculus can't even formulate is, for me, just the place where the integral becomes an expectation. The value function is the unifying object; Euler and Hamilton-Jacobi were one face of it.

So I've tamed the horizon. But I should be honest about where the difficulty *went*, because it didn't vanish — it moved. My value function $f(p)$ is a function on the *state space*. To actually compute and store it I have to represent it over the states, and if the state is an $M$-dimensional vector, tabulating $f$ over a grid in $M$ dimensions costs exponentially in $M$. I beat the exponential in the number of *stages*; I'm now exposed to an exponential in the *dimension of the state*. This is the price of excessive dimensionality, the thing that can make even a fast computing machine cringe, and it's the honest cost of the method: it's a tremendous win exactly when the state is low-dimensional, which — recognizing the right, minimal state variables for a problem — is often the real art. There's no free lunch; I've converted one curse into another, and the second is the one worth paying whenever the state is small.

Now back to the worry I parked: for an infinite-horizon problem there's no last stage, so I can't just sweep backward from the end to build $f$. I have the self-referential equation $f = T(f)$, where $T$ is the operator "max over the decision of (reward plus value-of-next-state)." How do I get a solution, and is it unique? The functional equations I keep deriving are, frankly, analytically intransigent — I'm not going to solve them in closed form in general. So I reach for the workhorse: successive approximations. Guess an initial $f_0$, and iterate

$$f_{n+1}(p) = T\big(f_n(p)\big), \qquad n=0,1,2,\dots$$

When does this converge, and to the right thing? The physical structure usually supplies what I need: there's discounting, or each stage shrinks things, so that applying $T$ contracts distances between functions. Concretely, when there's a discount factor that makes the "value of next state" term enter with a multiplier $\alpha\in(0,1)$ — write the operator as $T(W)(p) = \max_k\big[\text{reward}(p,k) + \alpha\int W(z)\,dG_k(p,z)\big]$ — then for any two candidate value functions $W$ and $V$, set

$$A_k=\text{reward}(p,k)+\alpha\int W(z)\,dG_k(p,z),\qquad
B_k=\text{reward}(p,k)+\alpha\int V(z)\,dG_k(p,z).$$

For each $k$, $|A_k-B_k|\le \alpha\|W-V\|_\infty$, because the immediate-reward terms cancel and $\int dG_k=1$. Also

$$\max_k A_k-\max_k B_k\le \max_k(A_k-B_k)\le \alpha\|W-V\|_\infty,$$

and swapping $W$ and $V$ gives the reverse inequality. Hence

$$\|T W-T V\|_\infty\le \alpha\|W-V\|_\infty.$$

With $\alpha\in(0,1)$, $T$ is a contraction. By the contraction-mapping theorem it has a unique fixed point $f$, and the iterates $f_n = T^n f_0$ converge to it geometrically from *any* starting guess. That settles existence, uniqueness, and a constructive algorithm all at once — this is the iteration I'll call approximation in function space, since I'm approximating the value function directly.

There's a second, in some ways nicer, way to iterate, and it comes from remembering that the value $f$ isn't really the point — the *policy* is. So instead of refining my estimate of the value, let me refine the *policy*. Start with a crude policy — on the allocation problem, say, the policy "always take $y=0$." Its value $f_0$ satisfies its own (now linear, no max) equation; on the allocation problem that's $f_0(x) = h(x) + f_0(bx)$, the return of forever taking $y=0$. Now choose the next policy by doing the one-decision maximization while scoring the future with $f_0$:

$$f_1(x) = \max_{0\le y\le x}\big[\, g(y) + h(x-y) + f_0(ay+b(x-y))\,\big].$$

Because $y=0$ is available in that maximization, the one-step score of the improved policy is at least $f_0(x)$ for every $x$. But I need the value of the *whole* improved policy, not just its first step. Write $T_\pi$ for the no-max operator of a fixed policy and $V_\pi$ for its fixed point. If $\pi'$ is greedy with respect to $V_\pi$, then

$$T_{\pi'}V_\pi=TV_\pi\ge T_\pi V_\pi=V_\pi.$$

The fixed-policy operator $T_{\pi'}$ is monotone, so starting from $V_\pi$ gives

$$V_\pi\le T_{\pi'}V_\pi\le T_{\pi'}^2V_\pi\le\cdots,$$

and, under the same discounting or shrinkage that makes the fixed-policy equation converge, this sequence tends to $V_{\pi'}$. Therefore $V_{\pi'}\ge V_\pi$ pointwise. Now the monotonicity claim is honest: evaluate the current policy, improve greedily, evaluate the improved policy, and the value never decreases. Iterate this and I get a monotone improving sequence converging to the optimum. That structural advantage over raw value iteration is why approximating in policy space tends to be the more natural and more practical route. So I have two constructive schemes falling straight out of the same functional equation: iterate the value (approximation in function space), or iterate the policy (approximation in policy space, monotone).

And the finite-horizon case ties the bow: there it's literally backward induction. At the terminal stage the value is just the best immediate return, $f_1(p)=\max_k(\text{return})$; then $f_2$ from $f_1$ via the functional equation, $f_3$ from $f_2$, and so on back to the start. Each step is a one-decision maximization at each state. The value functions are doing the job of shadow prices — each $f_{N-1}(z)$ summarizes the entire future consequence of landing in state $z$, so that my present decision only ever has to weigh "reward now" against "summary value of where I land." That's the principle of optimality made operational, and it's why the same swap argument that justified the equation also tells me the constructed rule is optimal in *every* subproblem it can ever reach, not just on average — there's no reachable state at which a better continuation was left on the table.

Let me recap the causal chain, because it's the spine of all of it. The pain was that sequential decision problems, solved by enumerating policies, cost exponentially in the number of stages and don't even make sense under uncertainty. Being practical about *what information I actually need* showed the right unknown is a feedback rule on the state, not a fixed sequence — and the right thing to *solve for* is the value function $f$ on the state, because it's unique and the rule recovers from it. The principle of optimality — provable by a one-line swap argument — says any optimal policy's tail is optimal for the subproblem it faces, and applying that turns the global optimization into the local functional equation $f_N(p)=\max_k\int f_{N-1}(z)\,dG_k(p,z)$, i.e. *value = best over the current decision of reward-now plus value-of-next-state*. That single equation covers deterministic and stochastic alike, swallows the calculus of variations (its characteristics give the Euler equation) while shrugging off the constraints and randomness that defeat the variational calculus, and exposes the one honest cost — the curse of dimensionality, now in the *state* dimension rather than the horizon. Finally, because the equation is a contraction under discounting, successive approximation in function space converges geometrically to the unique value function, and approximation in policy space gives a monotone-improving alternative — with finite horizons reducing to backward induction. The whole edifice stands on that one principle and the value function it acts on.

```python
# The recursion in the forms I actually compute.

def _best_action(actions, score, prefer=None, tol=0.0):
    best_action, best_value = None, float("-inf")
    preferred_value = None
    for action in actions:
        value = score(action)
        if prefer is not None and action == prefer:
            preferred_value = value
        if value > best_value:
            best_action, best_value = action, value
    if best_action is None:
        raise ValueError("each state must have at least one feasible decision")
    if preferred_value is not None and preferred_value >= best_value - tol:
        return prefer, preferred_value
    return best_action, best_value


def _expected(values, distribution):
    return sum(prob * values[state] for state, prob in distribution.items())


def value_iteration_finite(states, decisions, transition, reward, terminal, horizon):
    states = list(states)
    f_next = {p: terminal(p) for p in states}
    policy = [dict() for _ in range(horizon)]

    for stage in range(horizon - 1, -1, -1):
        f = {}
        for p in states:
            best_d, best_v = _best_action(
                decisions(p, stage),
                lambda d: reward(p, d, stage)
                + f_next[transition(p, d, stage)],
            )
            f[p], policy[stage][p] = best_v, best_d
        f_next = f
    return f_next, policy


def value_iteration_stochastic(states, decisions, next_dist, reward, terminal, horizon):
    states = list(states)
    f_next = {p: terminal(p) for p in states}
    policy = [dict() for _ in range(horizon)]

    for stage in range(horizon - 1, -1, -1):
        f = {}
        for p in states:
            best_d, best_v = _best_action(
                decisions(p, stage),
                lambda d: reward(p, d, stage)
                + _expected(f_next, next_dist(p, d, stage)),
            )
            f[p], policy[stage][p] = best_v, best_d
        f_next = f
    return f_next, policy


def value_iteration(states, decisions, next_dist, reward, alpha, tol=1e-10, max_iters=100000):
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    states = list(states)
    f = {p: 0.0 for p in states}

    for _ in range(max_iters):
        f_new = {}
        delta = 0.0
        for p in states:
            _, best_v = _best_action(
                decisions(p),
                lambda d: reward(p, d) + alpha * _expected(f, next_dist(p, d)),
            )
            f_new[p] = best_v
            delta = max(delta, abs(best_v - f[p]))
        f = f_new
        if delta < tol:
            policy = {
                p: _best_action(
                    decisions(p),
                    lambda d: reward(p, d) + alpha * _expected(f, next_dist(p, d)),
                )[0]
                for p in states
            }
            return f, policy
    raise RuntimeError("value iteration did not converge before max_iters")


def _solve_linear_system(matrix, rhs):
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular policy-evaluation system")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= scale
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def _evaluate_policy(states, policy, next_dist, reward, alpha):
    index = {p: i for i, p in enumerate(states)}
    n = len(states)
    matrix = [[0.0 for _ in states] for _ in states]
    rhs = [0.0 for _ in states]

    for p, i in index.items():
        d = policy[p]
        matrix[i][i] = 1.0
        rhs[i] = reward(p, d)
        for z, prob in next_dist(p, d).items():
            matrix[i][index[z]] -= alpha * prob

    solution = _solve_linear_system(matrix, rhs)
    return {p: solution[index[p]] for p in states}


def policy_iteration(states, decisions, next_dist, reward, alpha):
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    states = list(states)
    policy = {p: _best_action(decisions(p), lambda d: 0.0)[0] for p in states}

    while True:
        f = _evaluate_policy(states, policy, next_dist, reward, alpha)
        new_policy = {}
        stable = True
        for p in states:
            best_d, _ = _best_action(
                decisions(p),
                lambda d: reward(p, d) + alpha * _expected(f, next_dist(p, d)),
                prefer=policy[p],
                tol=1e-12,
            )
            new_policy[p] = best_d
            stable = stable and best_d == policy[p]
        if stable:
            return policy, f
        policy = new_policy
```
