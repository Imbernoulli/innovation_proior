I keep running into the same shape under different names. There is a state, I choose an action, the state changes, and the return I care about is spread across stages. If I order too much inventory now, later periods inherit the stock. If I spend a resource in one activity, only the transformed remainder is available later. If I steer a system over time, the next choice depends on where the current choice leaves the trajectory. The immediate action and the future are not separable.

The first formulation that suggests itself is to choose the whole sequence. I write down every feasible plan, run each one forward, score it, and pick the best. That is mathematically clean in a deterministic finite problem, but it asks for the wrong amount of information. With `N` stages and `k` choices per stage, I am optimizing over roughly `k^N` action sequences. In continuous time I am optimizing over a whole function. The difficulty is not only that a single stage may be hard; it is that I have made the object of choice as large as the entire future.

The stochastic case exposes the deeper mistake. If an action gives a distribution over next states, then a fixed future sequence is not just expensive. It is the wrong object. I cannot sensibly decide today what I will do at stage three without knowing which state chance will hand me at stage two. A rule that waits for the realized state is meaningful; a precommitted list is not. So I stop asking for the full sequence and ask what information I actually need at the moment of action.

At any stage I only need to know the current state and the best action from that state. The future can be handled when it becomes present, provided I have the same kind of rule there. This changes the unknown from an open-loop sequence into a feedback rule, an action as a function of state. That already handles the stochastic objection, because after chance acts I observe the state and apply the rule.

But optimizing directly over rules still feels awkward. A rule can be nonunique, and the space of rules can be large. What is unique is the best return obtainable from a state, even when several actions tie. I define a value:

$$
f_N(p)=\text{best return obtainable from state }p\text{ with }N\text{ stages to go}.
$$

If I can find this function, I can recover the rule by choosing an action that attains the maximum. The value function is the compact summary of all future consequences.

Now I need the step that makes the global problem local. Suppose a policy is truly optimal from the initial state. I take its first decision and arrive at a new state. Look at the tail of the policy from that new state onward. If that tail is not optimal for the subproblem starting at the new state, then I can replace the tail with a better one while leaving the first decision unchanged. The whole policy improves, contradicting its optimality. Therefore the tail of an optimal policy must itself be optimal for the state that the first decision creates.

This is the principle I need. It is not merely a slogan about writing an equation. It is the reason the equation is valid. It says that the future portion of an optimum is already an optimum for the continuation problem it actually faces. That lets me treat the first action separately and summarize everything after it by the value of the resulting state.

In the simplest deterministic case, action `k` transforms `p` into `T_k(p)`. If I choose `k` first, then the best return from the remaining `N-1` stages is, by definition, `f_{N-1}(T_k(p))`. I only have to choose the best first action:

$$
f_N(p)=\max_k f_{N-1}(T_k(p)).
$$

The exponential sequence search has become a one-step maximization at each state, repeated backward through the horizon. The equation follows from the principle, not the other way around.

Most problems also have a reward earned immediately. Then the current action contributes now and changes the state for the continuation. The form becomes:

$$
V_t(p)=\max_k\{r_t(p,k)+V_{t+1}(T_t(p,k))\}.
$$

This is the real pattern: value now equals the best current reward plus the value of the state I create for the future. The future is not ignored; it is compressed into the value function.

For stochastic transitions, action `k` leads to a random next state `z` with distribution `G_k(p,dz)`. I cannot maximize the realized final return before I know the realization, so I choose a criterion such as expected return. If I choose `k`, the tail value is the expectation of `f_{N-1}(z)` under that distribution. Therefore

$$
f_N(p)=\max_k \int f_{N-1}(z)\,G_k(p,dz).
$$

With immediate rewards and discounting this becomes

$$
V(p)=\max_k\left\{r(p,k)+\alpha\int V(z)\,P(dz\mid p,k)\right\}.
$$

The deterministic equation is just the special case where the transition distribution puts all mass on one state. This is exactly why the state-feedback view is necessary: the equation waits for the realized state by valuing every possible state in advance.

The same reasoning explains constrained allocation without any new principle. If I have quantity `x`, split it as `y` and `x-y`, earn `g(y)+h(x-y)`, and the next quantity is `ay+b(x-y)`, the feasible set is simply `0 <= y <= x`. The value must satisfy

$$
f(x)=\sup_{0\le y\le x}\{g(y)+h(x-y)+f(ay+b(x-y))\}.
$$

The constraint is not an obstacle to the principle. It just restricts the maximization. That is an important difference from a free-variation calculation, where a boundary optimum breaks the usual equality conditions.

I also want the continuous deterministic case to fit, because otherwise I have only solved a discrete problem. Let

$$
f(c,T)=\max \int_0^T F(x,y)\,dt
$$

subject to `dx/dt=G(x,y)` and `x(0)=c`. I split the interval into a small first slice of length `S` and the remaining interval of length `T`. Whatever I do on the first slice earns the integral over that slice and leaves a new state `c(S)`. From there the best continuation is `f(c(S),T)`. Thus the value over `S+T` is the best over the initial slice of immediate payoff plus continuation value.

Let the first slice shrink. With initial control `v`, the immediate payoff is approximately `F(c,v)S`, and the state changes by approximately `G(c,v)S`. Expanding the value to first order gives

$$
f_T=\max_v\{F(c,v)+G(c,v)f_c\}.
$$

The stationarity condition for an interior maximizer is

$$
F_v+G_v f_c=0.
$$

The classical Euler equations can be recovered from this partial differential equation and its characteristics. So the continuous extremal theory is not a separate world. It appears as the smooth, deterministic, interior case of the same value-based reasoning. When the control is constrained, the maximum is simply taken over the feasible controls, and the boundary case is handled as a constrained maximum rather than forced into a free-variation equality.

For an infinite stationary problem there is no last stage from which to start a backward sweep. The equation is now a fixed point:

$$
V=TV,
$$

where `T` maps a bounded trial value function to the result of one-step maximization plus discounted continuation value. I need to know when this fixed point is meaningful and computable. With bounded rewards and a continuation multiplier `alpha` satisfying `0 <= alpha < 1`, changing a trial value function by at most `epsilon` changes the continuation term by at most `alpha epsilon`. Taking a maximum over actions cannot enlarge that bound. Therefore

$$
\|TW-TV\|_\infty\le \alpha \|W-V\|_\infty.
$$

On the bounded-function space, the operator is a contraction. It has a unique fixed point there, and repeated application from any starting guess converges geometrically. This gives successive approximation in function space: guess a value function, apply the one-step operator, and repeat.

There is another computational route. Since the real object I want is a policy, I can start with any policy, evaluate the value it produces, then improve it by choosing actions greedily with respect to that value. The old action is feasible in the improvement step, so the one-step improved value cannot be worse. Under the same bounded discounted conditions, evaluating the improved policy gives a value at least as high as before. This gives approximation in policy space: evaluate, improve, repeat.

The success is not free. I have removed the explosion over whole future sequences by making the state the sufficient summary. If the state has many dimensions, storing or approximating the value over that state space can itself be enormous. The cost has moved from the length of the horizon to the dimension of the state. That is the honest limitation. The method is powerful exactly when the state captures all relevant history without becoming too large.

So the construction is: stop choosing complete future sequences; represent the needed history by the current state; solve for the value of each state rather than for a particular policy first; use the principle that the tail of an optimal policy is optimal for the state it reaches; derive a one-step maximization plus continuation value; then compute the resulting fixed point or backward recursion. The displayed equation is the artifact, but the principle of optimality is the lever that makes it true.
