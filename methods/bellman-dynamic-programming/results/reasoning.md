I keep running into the same shape under different names. There is a state, I choose an action, the state changes, and the return I care about is spread across stages. If I order too much inventory now, later periods inherit the stock. If I spend a resource in one activity, only the transformed remainder is available later. If I steer a system over time, the next choice depends on where the current choice leaves the trajectory. The immediate action and the future are not separable.

The first formulation that suggests itself is to choose the whole sequence. I write down every feasible plan, run each one forward, score it, and pick the best. That is mathematically clean in a deterministic finite problem, but it asks for the wrong amount of information. With `N` stages and `k` choices per stage, I am optimizing over roughly `k^N` action sequences. In continuous time I am optimizing over a whole function. The difficulty is not only that a single stage may be hard; it is that I have made the object of choice as large as the entire future.

The stochastic case exposes something worse than expense. If an action gives a distribution over next states, then a fixed future sequence is not just costly to search. It is the wrong object entirely. I cannot sensibly decide today what I will do at stage three without knowing which state chance will hand me at stage two. A rule that waits for the realized state is meaningful; a precommitted list is not. So I stop asking for the full sequence and ask what information I actually need at the moment of action.

It seems plausible that at any stage I only need to know the current state, provided I have some rule that tells me a good action from any state. The future can be handled when it becomes present, as long as the rule applies there too. That would change the unknown from an open-loop sequence into a feedback rule, an action as a function of state, and it would dissolve the stochastic objection, because after chance acts I observe the state and apply the rule. I should be careful here: this is a hope, not yet a justification. The claim that the current state is enough, and that a state-by-state rule can be optimal, is exactly what I will have to earn below.

Optimizing directly over rules still feels awkward as a first move. A rule can be nonunique, and the space of rules can be large. What does seem unique is the best return obtainable from a state, even when several actions tie for it. So let me make that the primary unknown and recover the rule afterward. Define a value:

$$
f_N(p)=\text{best return obtainable from state }p\text{ with }N\text{ stages to go}.
$$

If I can find this function, then at any state I can recover an action by choosing one that attains the maximum. The value function is meant to be the compact summary of all future consequences — but again, whether one number per state really is enough is the thing in question.

Now I need the step that would make the global problem local. Suppose a policy is truly optimal from the initial state. I take its first decision and arrive at a new state. Look at the tail of the policy from that new state onward. If that tail is not optimal for the subproblem starting at the new state, then I could replace the tail with a better one while leaving the first decision unchanged. The first decision is unaffected by what I do afterward, so the swap is admissible, and it strictly raises the total return — contradicting the assumed optimality of the whole policy. Therefore the tail of an optimal policy must itself be optimal for the state that the first decision creates.

That argument is what earns the value function. It says the future portion of an optimum is already an optimum for the continuation problem it actually faces, so I am allowed to summarize everything after the first action by the single number `f` of the resulting state. The first action can then be treated separately from the rest. This is the lever; the equations below are consequences of it.

In the simplest deterministic case, action `k` transforms `p` into `T_k(p)`. If I choose `k` first, then the best return from the remaining `N-1` stages is, by the argument just made, `f_{N-1}(T_k(p))`. So I only have to choose the best first action:

$$
f_N(p)=\max_k f_{N-1}(T_k(p)).
$$

The exponential sequence search has become a one-step maximization at each state, repeated backward through the horizon.

Before I trust this, I want to run it on a problem small enough to check by hand, because the whole construction stands on the tail-swap argument and I would like to see it actually produce the right number. Take a stock that can be `0,1,2,3`. Two actions: `hold` keeps the stock and earns its current size; `spend` drops the stock to `0` and earns twice its size, but only once. Horizon `2`, terminal value `0`. The interesting case is the largest stock, `p=3`.

Start at the last stage (`N=1`, one decision left, then terminal `0`). From `3`: `hold` earns `3` then `0`, total `3`; `spend` earns `6` then `0`, total `6`. So `f_1(3)=6`, and the last action there is `spend`. Now the stage with two decisions to go (`N=2`). From `3`: `hold` earns `3` now and leaves the stock at `3`, giving `3+f_1(3)=3+6=9`; `spend` earns `6` now and leaves the stock at `0`, giving `6+f_1(0)=6+0=6`. The recursion picks the larger, `f_2(3)=9`, with first action `hold`.

So the rule it produces is: at the first stage `hold`, at the second stage `spend`. The total return of that plan, simulated forward by hand, is `3` (hold from 3) then `6` (spend from 3) `= 9`. The naive alternative of spending immediately gives `6`. The recursion's `9` matches the hand simulation, and it has, on its own, discovered that delaying the spend is worth more — exactly the temporal coupling I was worried about. When I run the same recursion as code over all four states it returns `f_2 = \{0,3,6,9\}` with first-stage policy all-`hold` and second-stage policy `spend` for the nonzero stocks, which agrees with the by-hand value at `p=3` and is monotone in the stock as it should be. That is enough to convince me the backward recursion is doing what the principle claims, not just rearranging symbols.

Most problems also have a reward earned immediately. Then the current action contributes now and changes the state for the continuation. The same separation gives:

$$
V_t(p)=\max_k\{r_t(p,k)+V_{t+1}(T_t(p,k))\}.
$$

Value now equals the best current reward plus the value of the state I create for the future. The future is not ignored; it is compressed into the value function — and the hand check above is exactly an instance of this form, with `r` the earnings and `T` the stock update.

For stochastic transitions, action `k` leads to a random next state `z` with distribution `G_k(p,dz)`. I cannot maximize the realized final return before I know the realization, so I commit to a criterion — expected return. If I choose `k`, the tail value is the expectation of `f_{N-1}(z)` under that distribution, since the tail is optimal from wherever I land. Therefore

$$
f_N(p)=\max_k \int f_{N-1}(z)\,G_k(p,dz).
$$

With immediate rewards and discounting this becomes

$$
V(p)=\max_k\left\{r(p,k)+\alpha\int V(z)\,P(dz\mid p,k)\right\}.
$$

The deterministic equation is the special case where the transition distribution puts all its mass on one state: the integral collapses to `f_{N-1}(T_k(p))`. This is also why the state-feedback view was the right move and the open-loop list was not. The equation does not commit to an action before the realization; it values every possible next state in advance and then waits to see which one occurs.

The same reasoning covers constrained allocation without any new principle. If I have quantity `x`, split it as `y` and `x-y`, earn `g(y)+h(x-y)`, and the next quantity is `ay+b(x-y)`, the feasible set is simply `0 <= y <= x`. The value should satisfy

$$
f(x)=\sup_{0\le y\le x}\{g(y)+h(x-y)+f(ay+b(x-y))\}.
$$

The constraint is not an obstacle to the principle; it just restricts the set the maximization runs over. That is a real difference from a free-variation calculation, where a boundary optimum breaks the usual equality conditions — here a boundary optimum is fine, because the `\sup` over the feasible interval handles it directly.

I also want the continuous deterministic case to fit, because otherwise I have only handled discrete problems and the calculus of variations sits outside. Let

$$
f(c,T)=\max \int_0^T F(x,y)\,dt
$$

subject to `dx/dt=G(x,y)` and `x(0)=c`. I split the interval into a small first slice of length `S` and the remaining interval of length `T`. Whatever I do on the first slice earns the integral over that slice and leaves a new state `c(S)`. From there the best continuation is `f(c(S),T)`, by the same tail argument. So the value over `S+T` is the best over the initial slice of immediate payoff plus continuation value.

Let the first slice shrink. With initial control `v`, the immediate payoff is approximately `F(c,v)S`, and the state changes by approximately `G(c,v)S`. Expanding `f` to first order in `S` and cancelling the common factor gives

$$
f_T=\max_v\{F(c,v)+G(c,v)f_c\}.
$$

The stationarity condition for an interior maximizer is

$$
F_v+G_v f_c=0.
$$

I should sanity-check that this interior condition actually produces a sensible control on a solvable instance, rather than just looking like the Euler setup. Take a resource being drawn down: `dx/dt=-y` so `G=-y`, with concave instantaneous payoff `F=\sqrt{y}` and `y\ge 0`. Then `F_v=1/(2\sqrt{v})` and `G_v=-1`, so the condition reads `1/(2\sqrt{v})-f_c=0`, giving `v^*=1/(4f_c^2)`. That is a genuine interior maximum: positive whenever the marginal value of stock `f_c` is finite, and decreasing in `f_c`, so the optimal extraction rate slows as the remaining stock becomes more valuable. That is the behavior a depletion problem should have, and it falls straight out of the partial differential equation. From this PDE and its characteristics the classical Euler equations can be recovered, so the continuous extremal theory is not a separate world; it is the smooth, deterministic, interior case of the same value-based reasoning. When the control is constrained, the maximum is just taken over the feasible controls, and the boundary case is a constrained maximum rather than a forced free-variation equality.

For an infinite stationary problem there is no last stage from which to start a backward sweep, so the recursion alone gives me nothing to anchor on. The equation has to be read as a fixed point:

$$
V=TV,
$$

where `T` maps a bounded trial value function to the result of one-step maximization plus discounted continuation value. I need to know when this fixed point even exists and whether I can reach it by iterating. With bounded rewards and a discount `alpha` in `[0,1)`, change a trial value function by at most `epsilon` in the sup norm. Inside `T`, that perturbation is multiplied by `alpha` in the continuation term, and the immediate reward is untouched; taking a maximum over actions cannot enlarge a uniform bound. So

$$
\|TW-TV\|_\infty\le \alpha \|W-V\|_\infty.
$$

That is a contraction on the bounded-function space, so by the contraction mapping theorem the fixed point exists, is unique, and is reached geometrically from any starting guess.

I want to see the geometric rate as a real number rather than take the inequality on faith, so I run the iteration on a two-state instance: states `A,B`; in each you can `stay` (reward `1` in `A`, `4` in `B`, self-loop) or `switch` to the other state at a cost of `0.5`; discount `alpha=0.9`. I can solve this in closed form to have something to check against. Once in `B`, staying is clearly best, so `V_B=4+\alpha V_B`, giving `V_B=4/(1-0.9)=40`. From `A`, staying forever gives `1/(1-0.9)=10`, while switching to `B` gives `(1-0.5)+0.9\cdot 40=36.5`; since `36.5>10`, the optimal action in `A` is `switch` with `V_A=36.5`. Running value iteration to tolerance `10^{-12}` returns `V_A=36.4999\ldots`, `V_B=39.9999\ldots` and policy `\{A:\text{switch},\,B:\text{stay}\}` — matching the closed form to the tolerance, and the error shrinking by roughly a factor `0.9` per sweep, exactly the contraction rate. So the operator analysis is not just formally a contraction; it converges to the right answer at the predicted speed.

There is another computational route, and it is worth taking because the object I ultimately want is a policy, not a value table. Start with any policy, evaluate the value it produces by solving its linear system, then improve it by choosing actions greedily with respect to that value. The old action is always available in the improvement step, so the greedy step can never select something worse than the current policy's own action; the one-step improved value is therefore at least as high as before, and under the same discounted conditions evaluating the improved policy gives a value at least as high state-by-state. I test the monotonicity claim by deliberately starting from a bad policy on the same two-state problem: `stay` in both states. Its evaluated value is `V_A=10`, `V_B=40`. One round of policy iteration moves `A` to `switch`, lifting `V_A` from `10` to `36.5` while `V_B` holds at `40` — improvement at `A`, no regression at `B` — and it terminates at the same optimum value iteration found. So the improvement really is monotone here, not merely asserted to be.

The success is not free. I removed the explosion over whole future sequences by making the state the sufficient summary. But if the state has many dimensions, storing or even approximating the value over that state space can itself be enormous. The cost has not vanished; it has moved from the length of the horizon to the dimension of the state. That is the honest limitation. The method is powerful exactly when the state captures all relevant history without becoming too large, and it is precisely the size of the state and action spaces, exercised in the small checks above, that governs where computation becomes hard.

Pulling it together: stop choosing complete future sequences; represent the needed history by the current state; solve for the value of each state rather than for a particular policy first; lean on the fact that the tail of an optimal policy is optimal for the state it reaches; from that, derive a one-step maximization plus continuation value; then compute the resulting fixed point or backward recursion, by value iteration or by policy iteration. The displayed equations are the artifact, but the tail-optimality argument is what makes them true, and the small deterministic, continuous, and discounted checks are what convinced me they are also right.
