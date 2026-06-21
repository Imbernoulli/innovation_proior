The DQN floor lands at 79% median human-normalized score — a stable, working agent, but on the typical game it sits below the human reference, and the discipline of the median tells me where to push next: the median is not moved by squeezing more out of games already won, it is moved by fixing whatever quietly degrades the agent across *many* games at once. So I want a defect baked into the update I run on all 57 titles, not one specific to a single game. The most suspect piece of the floor is the bootstrap target itself, $Y_t=R_{t+1}+\gamma\max_a Q(S_{t+1},a;\theta^-)$, because it is computed the same way everywhere and the only part of it that depends on my noisy estimates is the $\max$.

And the $\max$ is biased upward. Suppose the per-action estimates are individually *unbiased*, $Q(s',a)=Q_*(s',a)+\epsilon_a$ with each $\epsilon_a$ zero-mean. Then $\max$ is convex and Jensen runs the wrong way, $\mathbb{E}[\max_a(Q_*+\epsilon_a)]\ge\max_a(Q_*+\mathbb{E}[\epsilon_a])=\max_a Q_*$ — the expected max of noisy values is at least the max of the true values. The sharper intuition: the $\max$ hunts across actions and selects the largest estimate, preferentially picking whichever action's noise landed high; positive noise is selected, negative noise discarded. That inflated value is then bootstrapped into $Q(S_t,A_t)$, which becomes part of the next state's target, so the inflation propagates backward through the chain — over 200M frames on a function approximator whose errors are never zero, this is the steady state, not a rare event. To make it quantitative, take the worst case: all true action values tied, $Q_*(s,a)=V_*(s)$, so there is genuinely nothing to choose and any apparent winner is pure noise. With balanced errors $\sum_a\epsilon_a=0$ of mean-squared spread $\frac1m\sum_a\epsilon_a^2=C>0$ over $m\ge2$ actions, $\max_a Q(s,a)\ge V_*(s)+\sqrt{C/(m-1)}$, and the bound is tight. The companion typical case — errors i.i.d. uniform on $[-\epsilon,\epsilon]$ — gives $\mathbb{E}[\max_a\epsilon_a]=\epsilon\frac{m-1}{m+1}$, which *increases* with the action count $m$. That is the cross-suite lever: the inflation is present on every game with more than one action and is worst where the action set is largest. And it does real damage to the policy, because there is no reason the overestimation is uniform — it depends on the action count, the error shape, the data each state got — so it is a non-uniform additive distortion that scrambles the *relative* ordering of actions, which is the only thing the greedy policy reads off.

I propose **Double DQN**, which fixes exactly this. Rewrite the target to expose its structure: $\max_a Q(S_{t+1},a;\theta^-)=Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta^-);\theta^-\big)$. Written this way the disease is obvious — one set of numbers does two jobs, *selecting* the greedy next action (the inner $\arg\max$) and *evaluating* it (the outer $Q$), both on $\theta^-$. The selected action is by construction the one whose $\theta^-$-estimate is largest, i.e. whose noise is most positive, and then I read off that same inflated estimate as its value: selection and evaluation are perfectly correlated in their error, and that correlation is what turns "noisy" into "biased high." If I select an action *because* its estimate is highest, I should not trust that same estimate to tell me its worth. The fix is to evaluate with a *different* set of estimates. With an idealized second value function $\theta'$ whose errors were independent of $\theta$'s, I could select with $\theta$ and evaluate with $\theta'$: the action $a^\star$ chosen by $\theta$ is some particular action, and $\theta'$'s error on $a^\star$ is independent of *why* $a^\star$ was selected, so the evaluation is not conditioned-on-being-large and is unbiased for $Q_*(S_{t+1},a^\star)$. In the all-tied stress state $\theta'$ neither knows nor cares which action $\theta$ inflated, so its expected value is $V_*(S_{t+1})$ — where the single max overshot by $\sqrt{C/(m-1)}$, the decoupled estimator's floor is zero. (This is the two-estimator idea of van Hasselt, 2010: two tables, one to pick, one to score.)

Training and storing a whole second network with symmetric random-assignment updates would be a lot of machinery, and it would muddy the question I want answered — whether *just* fixing the max helps, not whether "DQN plus a second network" helps. But the floor already hands me a second set of weights: the target network $\theta^-$, a frozen copy of $\theta$, sitting there to hold the regression target still. So I let $\theta$ do the selection and $\theta^-$ the evaluation,
$$Y_t=R_{t+1}+\gamma\,Q\big(S_{t+1},\arg\max_a Q(S_{t+1},a;\theta);\theta^-\big).$$
The online net picks the greedy next action, the target net scores it. Compared to plain DQN — both jobs on $\theta^-$ — the *only* change is whose $\arg\max$ I use: $\theta$ instead of $\theta^-$. No new network, no new parameters, no extra forward pass beyond one I essentially already do; replay, $\epsilon$-greedy, and the target-sync rule are all untouched. That minimalism is the point — it isolates the effect of the decoupled target so any change in median HNS is attributable to fixing the max and nothing else.

I should be honest about how much it buys. $\theta$ and $\theta^-$ are not independent the way the idealized $\theta'$ was — $\theta^-$ is a *stale copy* of $\theta$, so their errors are correlated, and right after a sync $\theta^-=\theta$ exactly and the target reverts to a plain max for that interval. So this removes much, not all, of the bias. But the staleness gives partial decoupling: between syncs $\theta$ has moved on while $\theta^-$ has not, so the action current-$\theta$ thinks best is no longer guaranteed to be the one $\theta^-$ inflated, and that is enough to take a real bite out of the overestimation. The fix is broad — every multi-action game, most on the large action sets — and it is free. What stays open: I have only changed the *target*. The value object is still a single scalar per action, exploration is still $\epsilon$-greedy, replay is still uniform. Those are the remaining levers, but fixing the systematic upward bias in the one update I run on all 57 games is the highest-leverage single change to the floor.

```python
# Double DQN target: online net selects, target net evaluates. The ONLY change from DQN is the argmax net.
# Code home: vwxyzjn/cleanrl (cleanrl/ddqn_atari.py); excerpted from methods/double-dqn/results/answer.md.
import torch


def double_dqn_target(rewards, discounts, next_obs, online_net, target_net):
    """y = r + discount * Q_target(s', argmax_a Q_online(s', a)).

    `discounts` is gamma for nonterminal transitions and 0 for terminal ones.
    """
    with torch.no_grad():
        selector_q = online_net(next_obs)
        next_actions = selector_q.argmax(dim=1, keepdim=True)        # SELECT with online theta
        evaluator_q = target_net(next_obs)
        next_q = evaluator_q.gather(1, next_actions).squeeze(1)      # EVALUATE with target theta-
        return rewards + discounts * next_q


def clipped_td_error_loss(td_error, grad_error_bound=1.0):
    """Huber/clipped-error loss matching the DQN clipped TD-error gradient (cross-suite stability)."""
    abs_error = td_error.abs()
    quadratic = torch.minimum(abs_error, torch.full_like(abs_error, grad_error_bound))
    linear = abs_error - quadratic
    return (0.5 * quadratic.pow(2) + grad_error_bound * linear).mean()


def double_dqn_loss(batch, gamma, online_net, target_net):
    obs, actions, rewards, next_obs, dones = batch
    discounts = gamma * (1.0 - dones.float())
    target = double_dqn_target(rewards, discounts, next_obs, online_net, target_net)
    q_sa = online_net(obs).gather(1, actions.long().view(-1, 1)).squeeze(1)
    return clipped_td_error_loss(target - q_sa)
```
