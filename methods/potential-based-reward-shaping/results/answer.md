# Potential-Based Reward Shaping

## Problem

In reinforcement learning the task is encoded entirely in the reward $R$. To accelerate learning, practitioners add an auxiliary *shaping reward* $F$, training on $M'=(S,A,T,\gamma,R+F)$ instead of $M=(S,A,T,\gamma,R)$. This optimizes $R+F$, not $R$, and can silently change the optimal policy — the agent exploits the shaping rather than solving the task (e.g. a "progress"-rewarded bicycle riding in tiny circles; a "touch-the-ball"-rewarded robot vibrating against the ball). The question: for which $F$ is every optimal policy of $M'$ guaranteed optimal in $M$ (policy invariance)?

## Key idea

Every shaping bug is a **positive-reward cycle**: a closed loop of states $s_1\to\cdots\to s_n\to s_1$ with $\sum F>0$ that the agent farms forever. In the undiscounted picture, a **difference of a state potential** $\Phi$ makes every closed-loop sum cancel. Matched to the discounted Bellman recursion, the correct form is

$$\boxed{\,F(s,a,s')=\gamma\,\Phi(s')-\Phi(s)\,}$$

for an arbitrary (bounded) $\Phi:S\to\mathbb R$. This adds a per-state, action-independent offset to $Q^*$, which the greedy argmax ignores — so the optimal policy is left exactly invariant. And this form is *necessary* as well as sufficient: any other $F$ can be made to corrupt the optimum.

## Theorem (necessary and sufficient)

Fix $S,A,\gamma$ and a bounded $F:S\times A\times S\to\mathbb R$. Call $F$ **potential-based** if there exists bounded $\Phi:S\to\mathbb R$ with $F(s,a,s')=\gamma\Phi(s')-\Phi(s)$ for all $s\in S\setminus\{s_0\},a\in A,s'\in S$ (with $S\setminus\{s_0\}=S$ when $\gamma<1$). Then $F$ being potential-based is necessary and sufficient to guarantee, for **all** proper $T$ and **all** $R$, that every optimal policy of $M'=(S,A,T,\gamma,R+F)$ is optimal in $M=(S,A,T,\gamma,R)$ (and vice versa).

### Proof of sufficiency

$Q^*_M$ solves the Bellman optimality equation $Q^*_M(s,a)=\mathbb E_{s'}[R(s,a,s')+\gamma\max_{a'}Q^*_M(s',a')]$. Define $\hat Q(s,a):=Q^*_M(s,a)-\Phi(s)$. Subtracting $\Phi(s)$ and inserting $\pm\gamma\Phi(s')$ (which pulls through the max since $\Phi(s')$ is action-independent):

$$\hat Q(s,a)=\mathbb E_{s'}\big[R(s,a,s')+\underbrace{\gamma\Phi(s')-\Phi(s)}_{F(s,a,s')}+\gamma\max_{a'}\hat Q(s',a')\big]=\mathbb E_{s'}\big[R'(s,a,s')+\gamma\max_{a'}\hat Q(s',a')\big],$$

which is exactly the Bellman optimality equation for $M'$. In the undiscounted proper case, replace $\Phi$ by $\Phi-\Phi(s_0)$; because $\gamma=1$, this leaves $F(s,a,s')=\Phi(s')-\Phi(s)$ unchanged and makes the absorbing boundary match, $\hat Q(s_0,a)=0$. In the discounted case no absorbing boundary condition is needed. By uniqueness of the Bellman optimality solution, $\hat Q=Q^*_{M'}$, i.e.

$$Q^*_{M'}(s,a)=Q^*_M(s,a)-\Phi(s),\qquad V^*_{M'}(s)=V^*_M(s)-\Phi(s).$$

Since $-\Phi(s)$ is the same for every action at $s$,
$$\arg\max_a Q^*_{M'}(s,a)=\arg\max_a\big(Q^*_M(s,a)-\Phi(s)\big)=\arg\max_a Q^*_M(s,a),$$
so the optimal-policy sets coincide; swapping $M\leftrightarrow M'$ (shaping $-F$, potential $-\Phi$) gives the converse. The identity $V^\pi_{M'}=V^\pi_M-\Phi$ holds for *all* $\pi$, so near-optimality transfers within the same $\varepsilon$. $\square$

### Proof of necessity

Suppose $F$ is not potential-based; construct proper $T,R$ in which no optimal policy of $M'$ is optimal in $M$.

**Action-dependent case (Lemma).** If $\exists\,s,s',a,a'$ with $F(s,a,s')\ne F(s,a',s')$, relabel actions so $\Delta:=F(s,a,s')-F(s,a',s')>0$. Let both actions go $s\to s'$ w.p. 1; set $R(s,a,s')=0$, $R(s,a',s')=\Delta/2$. In $M$: $Q_M(s,a)=0<Q_M(s,a')=\Delta/2$, so $\pi^*_M(s)=a'$. In $M'$: $R'(s,a,s')=F(s,a,s')$ and $R'(s,a',s')=\Delta/2+F(s,a',s')=F(s,a,s')-\Delta/2<R'(s,a,s')$, so $\pi^*_{M'}(s)=a$. Opposite. $\square$

**Action-independent case** $F(s,a,s')=F(s,s')$, not potential-based. Pick reference $\hat s_0$ ($=s_0$ if $\gamma=1$); for $\gamma<1$, subtract the constant $F(\hat s_0,\hat s_0)$ from all shaped rewards, which leaves optimal policies unchanged, so $F(\hat s_0,\hat s_0)=0$. Define $\Phi(s):=-F(s,\hat s_0)$. Non-potential $\Rightarrow$ $\exists\,s_1,s_2$ with $\gamma\Phi(s_2)-\Phi(s_1)\ne F(s_1,s_2)$, i.e.
$$\Delta:=F(s_1,s_2)+\gamma F(s_2,\hat s_0)-F(s_1,\hat s_0)\ne 0.$$
Construct $M$: from $s_1$, action $a\to\hat s_0$ and action $a'\to s_2$ (each w.p. 1); from $s_2$ and, when $\gamma<1$, from $\hat s_0$, both actions $\to\hat s_0$. Set $R(s_1,a,\hat s_0)=\Delta/2$, else $0$; then $V^*_M(\hat s_0)=V^*_{M'}(\hat s_0)=0$ by absorption when $\gamma=1$ and by the zero-reward self-loop plus $F(\hat s_0,\hat s_0)=0$ when $\gamma<1$. Then
$$Q_M(s_1,a)=\tfrac{\Delta}{2},\quad Q_M(s_1,a')=0,$$
$$Q_{M'}(s_1,a)=\tfrac{\Delta}{2}+F(s_1,\hat s_0)=F(s_1,s_2)+\gamma F(s_2,\hat s_0)-\tfrac{\Delta}{2},\quad Q_{M'}(s_1,a')=F(s_1,s_2)+\gamma F(s_2,\hat s_0).$$
So $\pi^*_M(s_1)=a$ iff $\Delta>0$, while $\pi^*_{M'}(s_1)=a'$ iff $\Delta>0$ — opposite either way. $\square$

## Choosing $\Phi$ and final form

Invariance holds for *any* $\Phi$, so choose it to speed learning. Since $V^*_{M'}=V^*_M-\Phi$, taking $\Phi\approx V^*_M$ flattens the shaped value function (easiest to learn); a crude estimate suffices and a wrong $\Phi$ still cannot corrupt the optimum.
- **Distance heuristic** (undiscounted grid, $-1$/step, 80/20 moves): $\Phi(s)=-\mathrm{manhattan}(s,\text{goal})/0.8\approx V^*$. Rewards progress *symmetrically* — no tiny-circle money pump.
- **Subgoal heuristic**: $\Phi(s)=-((5-n_s-0.5)/5)\,t$ jumps by $t/5$ per subgoal, so $F$ pays exactly $t/5$ per subgoal (principled magnitude, not farmable).
- **SMDP**: $F(s,a,s',\tau)=e^{-\beta\tau}\Phi(s')-\Phi(s)$ for action duration $\tau$, discount rate $\beta$.

```python
def shaping_reward(s, a, s_next, gamma, Phi):
    # F(s,a,s') = gamma*Phi(s') - Phi(s):
    #   discounted difference of a state-only potential => shaped returns differ
    #   by the start-state offset -Phi(s), leaving argmax_a exactly unchanged.
    return gamma * Phi(s_next) - Phi(s)

def manhattan_potential(goal, p_intended=0.8):
    # Phi(s) ~ V*(s) for a -1/step minimum-cost-to-goal grid
    def Phi(s):
        return -manhattan(s, goal) / p_intended
    return Phi

def learn(mdp, episodes, Phi, alpha=0.02, eps=0.1):
    Q = {(s, a): 0.0 for s in mdp.states for a in mdp.actions}
    for _ in range(episodes):
        s = mdp.reset(); done = False
        while not done:
            a = eps_greedy(Q, s, mdp.actions, eps)
            s2, r_env, done = mdp.step(s, a)
            r = r_env + shaping_reward(s, a, s2, mdp.gamma, Phi)   # add F
            target = r + (0.0 if done else mdp.gamma * max(Q[(s2, b)] for b in mdp.actions))
            Q[(s, a)] += alpha * (target - Q[(s, a)])
            s = s2
    return Q
```
