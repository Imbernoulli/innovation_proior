The problem is how to make a sequence of decisions when each decision changes the state of the system and the return depends on the whole sequence. The naive approach is to enumerate every possible sequence of actions, but that grows exponentially with the horizon and becomes impossible even for modest problems. In stochastic settings it is worse, because a fixed sequence ignores the random states that will be observed before future actions are chosen. The right object is not a precommitted plan but a rule that maps each observed state to an action. The difficulty is then to find such a rule without searching over all possible futures.

The key insight is that the future can be summarized by a single number: the best return obtainable from each state. If the tail of a policy were not optimal from the state it reaches, replacing that tail with a better one would improve the whole policy, contradicting optimality. Therefore the tail of an optimal policy must itself be optimal. This lets us write a local equation in which the value of a state equals the best one-step reward plus the value of the resulting next state. Dynamic programming, founded on Bellman's principle of optimality, turns global sequence optimization into recursive state-value computation.

The method works by defining a value function and deriving a backward recursion or fixed-point equation from the principle of optimality. For a finite-horizon problem with immediate reward r and deterministic transition T, the value at time t is V_t(p) = max_k { r_t(p,k) + V_{t+1}(T_t(p,k)) }, with a terminal condition at the final stage. For stochastic transitions the continuation value becomes an expectation over the next-state distribution. For infinite-horizon discounted problems the value satisfies V = TV, where T applies one-step maximization plus discounted continuation. Because T is a contraction when rewards are bounded and the discount factor lies in [0,1), value iteration converges to the unique fixed point. Policy iteration alternates between evaluating a fixed policy and greedily improving it, also converging under the same conditions.

The artifact I want to hand over is not a program but the value-equation apparatus itself, since nothing about this method is tied to a particular implementation: it is a way of converting any multistage decision problem into a one-step recursion. Write $p$ for the current state, $k$ for the current feasible decision, and $f_N(p)$ for the optimal return with $N$ stages still to go. When decision $k$ moves a deterministic state $p$ to $T_k(p)$, the principle of optimality gives the backward recursion

$$
f_N(p)=\max_k f_{N-1}\bigl(T_k(p)\bigr), \qquad N=2,3,\ldots,
$$

seeded by the one-stage return $f_1$. When the transition is random instead, with decision $k$ sending $p$ to a next state drawn from $G_k(p,dz)$, the same argument gives

$$
f_N(p)=\max_k\int f_{N-1}(z)\,G_k(p,dz), \qquad N=2,3,\ldots,
$$

which reduces to the deterministic recursion exactly when $G_k$ puts all its mass on $T_k(p)$. Folding in an explicit current reward $r_t(p,k)$ and transition kernel $P_t(dz\mid p,k)$ gives the additive form used in finite-horizon applications,

$$
V_t(p)=\max_k\left\{r_t(p,k)+\int V_{t+1}(z)\,P_t(dz\mid p,k)\right\},
$$

together with a terminal condition fixing the value at the last stage. In the stationary, infinite-horizon case with discount factor $\alpha\in[0,1)$ this collapses to a single fixed-point equation for one value function,

$$
V(p)=\max_k\left\{r(p,k)+\alpha\int V(z)\,P(dz\mid p,k)\right\} =: (TV)(p).
$$

For bounded rewards and $0\le\alpha<1$, the Bellman operator $T$ satisfies

$$
\|TW-TV\|_\infty \le \alpha\,\|W-V\|_\infty
$$

for any two bounded value functions $W,V$, so $T$ is a contraction in the sup norm: it has a unique fixed point, value iteration (repeated application of $T$ from any starting guess) converges to it, and policy iteration reaches the same fixed point by alternating exact evaluation of the current policy with greedy improvement. Carrying the same construction to continuous deterministic control, where the state obeys $dx/dt=G(x,v)$ under control $v$ with running payoff $F(x,v)$ over $[0,T]$, the value function $f(c,T)$ for starting state $c$ satisfies, in the differentiable limit,

$$
f_T=\max_v\bigl\{F(c,v)+G(c,v)\,f_c\bigr\},
$$

whose interior maximizers obey $F_v+G_v f_c=0$, with any boundary or inequality constraints on $v$ left inside the maximization rather than handled by a separate case. This is the complete result: a value function defined on states, the principle of optimality that turns a maximization over whole policies into the one-step recursion above, and the contraction argument that certifies convergence to it whenever the horizon is stationary and discounted.
