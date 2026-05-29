Let me start where it actually hurts. I have a continuous-control actor-critic — a deterministic actor that ascends a learned critic's gradient, the whole deep-Q apparatus of replay and target networks bolted on — and it half-works. On a good seed it learns a decent gait; on a neighboring seed it falls over and never recovers, and when it does the value the critic reports keeps climbing past anything the policy could possibly collect. That second symptom is the thread I want to pull. The critic says a state is worth, say, a thousand, and when I actually roll the policy out from that state and sum the discounted rewards I get four hundred. The estimate isn't noisy-around-the-truth, it's biased upward, and it's getting worse over time. So the question isn't "why is learning noisy," it's "why does the value estimate systematically inflate, and why does that inflation feed on itself."

I've seen this story before in the discrete-action world. Q-learning forms its target by maximizing over actions, $y = r + \gamma \max_{a'} Q(s',a')$, and there's an old observation about that max that I should write out carefully because I think it's the seed of everything. Suppose the estimate at the next state isn't exact — it's the true value plus some error, $Q(s',a') = Q_{\text{true}}(s',a') + \epsilon_{a'}$, and let me even be generous and assume the error is zero-mean, $\mathbb E[\epsilon_{a'}] = 0$ for every action. Now I take the max. Is the max of the noisy estimates an unbiased estimate of the max of the true values? Take expectations: $\mathbb E_\epsilon[\max_{a'}(Q_{\text{true}}(s',a') + \epsilon_{a'})]$. The max is a convex function of its arguments, and Jensen pushes the expectation of a convex function above the function of the expectation, so $\mathbb E_\epsilon[\max_{a'}(Q_{\text{true}} + \epsilon)] \ge \max_{a'}\mathbb E_\epsilon[Q_{\text{true}} + \epsilon] = \max_{a'} Q_{\text{true}}(s',a')$. So even with zero-mean error, the target overshoots the true max. The intuition behind the inequality is sharper than Jensen, actually: the action that attains the max is, disproportionately, the one whose $\epsilon$ happened to be large and positive. The max operator doesn't just pass noise through, it actively selects for the upward errors. So the estimator is biased high, structurally, and not because anyone made a mistake — because function approximation guarantees the $\epsilon$ is there.

And then temporal-difference learning takes that biased target and bakes it into the estimate at $(s,a)$, which becomes the next state's bootstrap for some earlier state, and so on. The bias doesn't stay local; the Bellman backup carries it backward through the whole reachable set. That's the compounding I'm watching.

But wait — my problem is continuous control. There is no $\max_{a'}$ in my update. The actor isn't enumerating actions and picking the best; it's taking a gradient step, $\nabla_\phi J(\phi) = \mathbb E_s[\nabla_a Q_\theta(s,a)|_{a=\pi_\phi(s)}\,\nabla_\phi \pi_\phi(s)]$, a small soft nudge in the direction the critic says is uphill. So maybe the discrete argument simply doesn't apply, and my inflation comes from somewhere else. I should not assume; I should try to prove whether a gradient-driven actor inherits the same bias.

I can set the comparison up so the question is well-posed. The actor update uses the *approximate* critic $Q_\theta$, but I can also imagine the update it *would* have taken if it had the true critic $Q^\pi$. Two policies after one step from the same $\phi$:
$$\phi_{\text{approx}} = \phi + \frac{\alpha}{Z_1}\,\mathbb E_{s}\big[\nabla_\phi \pi_\phi(s)\,\nabla_a Q_\theta(s,a)|_{a=\pi_\phi(s)}\big],\qquad \phi_{\text{true}} = \phi + \frac{\alpha}{Z_2}\,\mathbb E_{s}\big[\nabla_\phi \pi_\phi(s)\,\nabla_a Q^\pi(s,a)|_{a=\pi_\phi(s)}\big],$$
where I normalize each gradient by $Z_1, Z_2$ so the two steps have the same length and I'm comparing directions, not magnitudes. Call the resulting policies $\pi_{\text{approx}}$ and $\pi_{\text{true}}$. The thing I want to know: is $\mathbb E[Q_\theta(s,\pi_{\text{approx}}(s))]$, the value I actually believe in after my real update, bigger than $\mathbb E[Q^\pi(s,\pi_{\text{approx}}(s))]$, the true value of that same policy?

Two facts, each just "a gradient step in the steepest-ascent direction of an objective improves that objective for small enough step." First, $\phi_{\text{approx}}$ moves uphill on the *approximate* objective, so among the two, the approximate policy scores at least as high under the approximate critic: there's an $\epsilon_1>0$ such that for $\alpha\le\epsilon_1$,
$$\mathbb E[Q_\theta(s,\pi_{\text{approx}}(s))] \ge \mathbb E[Q_\theta(s,\pi_{\text{true}}(s))].$$
Symmetrically, $\phi_{\text{true}}$ moves uphill on the *true* objective, so the true policy scores at least as high under the true critic: there's an $\epsilon_2>0$ such that for $\alpha\le\epsilon_2$,
$$\mathbb E[Q^\pi(s,\pi_{\text{true}}(s))] \ge \mathbb E[Q^\pi(s,\pi_{\text{approx}}(s))].$$
Now I need one bridge between the $\theta$-world and the $\pi$-world. The natural one: at the true policy, the critic is on average not below the truth, $\mathbb E[Q_\theta(s,\pi_{\text{true}}(s))] \ge \mathbb E[Q^\pi(s,\pi_{\text{true}}(s))]$. (This is the assumption I'll have to revisit, but grant it for now.) Then chain the three, for $\alpha < \min(\epsilon_1,\epsilon_2)$:
$$\mathbb E[Q_\theta(s,\pi_{\text{approx}}(s))] \ge \mathbb E[Q_\theta(s,\pi_{\text{true}}(s))] \ge \mathbb E[Q^\pi(s,\pi_{\text{true}}(s))] \ge \mathbb E[Q^\pi(s,\pi_{\text{approx}}(s))].$$
The leftmost is what I believe; the rightmost is the truth of the policy I actually deployed; and belief $\ge$ truth. Overestimation, with a gradient-driven actor, no $\max$ anywhere. So the discrete intuition survives the translation. The mechanism is the same selection-for-upward-error, just routed through the gradient: the actor climbs wherever the critic bulges upward, and the critic bulges upward wherever its error is positive.

I leaned on that bridge assumption — that the estimate isn't below the truth at $\pi_{\text{true}}$. What if I don't get to normalize the gradients, which is the honest case? Then I need a slightly stronger condition but I get the same conclusion. Write $V_\theta(\psi)=\mathbb E_s[Q_\theta(s,\pi_\psi(s))]$ and $V^\pi(\psi)=\mathbb E_s[Q^\pi(s,\pi_\psi(s))]$, and now let $\phi_{\text{true}}$ and $\phi_{\text{approx}}$ denote the unnormalized true-gradient and approximate-gradient steps. Suppose the approximate value equals the true value along the ray in the true-gradient direction, so $V_\theta(\phi+\beta(\phi_{\text{true}}-\phi))=V^\pi(\phi+\beta(\phi_{\text{true}}-\phi))$ for every $\beta\ge0$ in the small region I can trust, and in particular $V_\theta(\phi)=V^\pi(\phi)$. Define the small-step value changes $\Delta^\pi_{\text{true}}=V^\pi(\phi_{\text{true}})-V^\pi(\phi)$, $\Delta^\pi_{\text{approx}}=V^\pi(\phi_{\text{approx}})-V^\pi(\phi)$, and $\Delta^\theta_{\text{approx}}=V_\theta(\phi_{\text{approx}})-V_\theta(\phi)$. Under that local condition, the true-gradient step gives the largest first-order increase of $V^\pi$, so for small enough $\alpha$, $\Delta^\pi_{\text{true}}\ge\Delta^\pi_{\text{approx}}$. Along the true-gradient ray the two value surfaces agree, so that same true-direction increase is also available to $V_\theta$; the approximate-gradient step is the steepest first-order increase of $V_\theta$, hence $\Delta^\theta_{\text{approx}}\ge\Delta^\pi_{\text{true}}$. Starting from the common anchor $V_\theta(\phi)=V^\pi(\phi)$, I get $V_\theta(\phi_{\text{approx}})\ge V^\pi(\phi_{\text{true}})\ge V^\pi(\phi_{\text{approx}})$. Same overestimation, without relying on equal step lengths from normalization.

Is any of this real or is it a small-$\alpha$ curiosity? I should look. If I take the actor-critic as it stands and, every so often, freeze and roll out the current policy many times to Monte-Carlo the true discounted return from states I pull out of the replay buffer, I can plot the critic's average estimate against that truth as it learns. When I picture doing this over a million steps, the estimate doesn't hover near the Monte-Carlo curve — it pulls away and stays above, more and more. So the theory's "minimal per-update bias" is not staying minimal; it's the compounding I worried about. Two distinct dangers, and I should name them precisely because they call for different fixes. One, the bias grows over many updates if nothing checks it. Two — and this is the nastier one — the actor is defined by ascending this critic, so an inflated bump in the critic doesn't just mislead a readout, it *attracts the policy*. The policy moves toward the overrated actions, generates more transitions there, the critic fits that region and can inflate it further. A feedback loop where a suboptimal action gets rated highly by a suboptimal critic and is thereby reinforced. So I need to attack the bias at the source, in how the target is formed.

The obvious move is to borrow the discrete-world cure. The reason $\max$ inflates is that the same estimator both *selects* the action and *evaluates* it, so it's evaluating its own optimistic pick. Decouple them. Double Q-learning keeps two estimators and crosses them: $a^* = \arg\max_{a'} Q^A(s',a')$, but the value used is $Q^B(s',a^*)$. Since $Q^B$ had no hand in choosing $a^*$, its error on that action is just a draw, not a draw conditioned on being large; $\mathbb E[Q^B(s',a^*)] \le \max_{a'} Q^B(s',a')$, and the systematic inflation is gone. The deep, one-network shortcut, Double DQN, notices you already keep a target network lying around, so use it as the second estimator: select with the online net, evaluate with the target net, $y = r + \gamma Q_{\theta'}(s', \arg\max_{a'} Q_\theta(s',a'))$.

Let me transcribe Double DQN into my setting. The role of "the action the value net would pick" is played by the actor. So the analog is: select the action with the *current* actor, evaluate with the *target* critic, $y = r + \gamma Q_{\theta'}(s', \pi_\phi(s'))$, using $\pi_\phi$ rather than the target actor $\pi_{\phi'}$ so that the selection and evaluation come from different time-slices. I expect the overestimation curve to flatten.

It barely moves. The reason is in the mechanics. Double DQN's whole leverage is that the online network and the target network are *different enough* to count as independent estimates. In discrete deep Q-learning the online net changes fast, so the target lags meaningfully behind. But here the "selector" is the policy, and my policy crawls — soft target updates, slow optimization — so $\pi_\phi(s')$ and $\pi_{\phi'}(s')$ pick almost the same action, and $Q_{\theta'}$ evaluated at almost the same action is almost the same number it would have given anyway. Current and target are too similar to decouple anything. The slow policy, which I wanted for stability, is exactly what kills the decoupling. Wall.

So go back to the older, heavier version: real Double Q-learning with genuinely two of everything. Two critics $Q_{\theta_1}, Q_{\theta_2}$ and two actors $\pi_{\phi_1},\pi_{\phi_2}$, each actor ascending its own critic, and cross the targets so each critic is valued by the other's actor through the *opposite* critic:
$$y_1 = r + \gamma\, Q_{\theta'_2}(s', \pi_{\phi_1}(s')), \qquad y_2 = r + \gamma\, Q_{\theta'_1}(s', \pi_{\phi_2}(s')).$$
Now $\pi_{\phi_1}$ optimizes $Q_{\theta_1}$, so the action plugged into $y_1$ was chosen with respect to $Q_{\theta_1}$ but is *evaluated* by $Q_{\theta_2}$, which had no say in selecting it. That's the proper decoupling, and the overestimation curve does come down — more than Double DQN's. But it doesn't go to zero, and I want to understand the residual because it'll tell me what to do next. The two critics aren't actually independent. They share one replay buffer, so they see the same transitions, and worse, each one's target is built from the *other* critic, so their errors are correlated by construction. Independence was the entire premise of the unbiasedness. Without it, here's the failure: $Q_{\theta_2}$ is supposed to be the less-biased estimate of $Q^\pi(s,\pi_{\phi_1}(s))$, but for some states it overshoots its partner, $Q_{\theta_2}(s,\pi_{\phi_1}(s)) > Q_{\theta_1}(s,\pi_{\phi_1}(s))$. And $Q_{\theta_1}$ itself already tends to overestimate the truth. So in those states I'm using a number that is *above* an already-inflated estimate — overestimation piled on overestimation, and exactly in the local pockets where the policy is most likely to be drawn. The cross-estimator is supposed to be the safety valve, and in those pockets it's the opposite. Wall again, but a very informative one.

Stare at what just went wrong. The trouble is only ever that $Q_{\theta_2}$ on the selected action comes out *higher* than $Q_{\theta_1}$. When $Q_{\theta_2}$ is lower, it's doing its job — pulling the inflated estimate down, just like Double Q-learning wants. So I don't actually need $Q_{\theta_2}$ to be unbiased; I need it to never make things worse than the already-overestimating $Q_{\theta_1}$. There's a clean way to phrase that: treat the more-biased estimate as an approximate *upper bound* on the value and never let the target exceed it. Which is just: take the smaller of the two.
$$y_1 = r + \gamma\, \min_{i=1,2} Q_{\theta'_i}(s', \pi_{\phi_1}(s')).$$
Now reconsider the two cases at the selected action. If $Q_{\theta_2} \ge Q_{\theta_1}$, the min picks $Q_{\theta_1}$, which is just the standard single-critic target — so the min can never *add* overestimation beyond what plain Q-learning already has; that's the worst case and it's bounded. If $Q_{\theta_2} < Q_{\theta_1}$, the min picks $Q_{\theta_2}$, which is the Double-Q correction pulling the estimate down. So the min keeps the good behavior and caps the bad. The price is that I might now *underestimate*. I sat with that for a second worried it just trades one bias for another, but the two biases are not symmetric in this system, and that asymmetry is the whole justification. An overestimated action gets selected by the actor and its inflation is propagated through every subsequent policy update — overestimation is actively chased. An underestimated action tends to be avoided by the actor, so it is not explicitly selected and amplified by the policy update in the same way. Biasing toward underestimation is not a wash; it is the safer direction for this feedback loop.

There's a bonus hiding in the min that I want to make explicit, because it connects to the variance problem I still owe myself. Think of the two critic outputs at a given state-action as random variables with some estimation-error variance. The expected minimum of a set of random variables goes *down* as their variance goes *up* — a wider spread drags the min lower. So the min target penalizes states where the value estimate is high-variance and rewards states where it's tight. As the critic learns from those targets, the actor that climbs the critic is nudged toward regions where the estimate is steadier. The min doesn't only fight bias; it gives a quiet preference for low-variance, stable targets. That's a second reason to like it, and the first hint that bias and variance aren't separate problems here.

Before I build on the min I want to know it's not just a heuristic that happens to work — does it converge at all? Strip to the tabular finite-MDP core: two tables $Q^A, Q^B$, pick $a^* = \arg\max_a Q^A(s',a)$, target $y = r + \gamma\min(Q^A(s',a^*), Q^B(s',a^*))$, and update both tables toward $y$ with step $\alpha_t(s,a)$. I'll lean on the standard stochastic-approximation lemma used to prove SARSA and Double Q-learning converge: if a process $\Delta_{t+1}(x_t) = (1-\zeta_t)\Delta_t(x_t) + \zeta_t F_t(x_t)$ has step sizes summing to infinity but squares summing finite, and $\|\mathbb E[F_t\mid P_t]\| \le \kappa\|\Delta_t\| + c_t$ with $\kappa\in[0,1)$ and $c_t\to 0$, with bounded conditional variance, then $\Delta_t\to 0$. Set $\Delta_t = Q^A_t - Q^*$ and $\zeta_t = \alpha_t$. The update of $\Delta_t$ rearranges into exactly the lemma's form with
$$F_t(s_t,a_t) = r_t + \gamma\min(Q^A_t(s',a^*), Q^B_t(s',a^*)) - Q^*(s_t,a_t).$$
I add and subtract the standard-Q term: write $F_t = F^Q_t + c_t$ where $F^Q_t = r_t + \gamma Q^A_t(s',a^*) - Q^*(s_t,a_t)$ is precisely $F_t$ for ordinary Q-learning, for which $\|\mathbb E[F^Q_t\mid P_t]\|\le \gamma\|\Delta_t\|$ is the well-known contraction, and the leftover is
$$c_t = \gamma\big(\min(Q^A_t(s',a^*), Q^B_t(s',a^*)) - Q^A_t(s',a^*)\big).$$
The contraction part already satisfies the lemma's condition with $\kappa=\gamma$; I just need $c_t\to 0$. And $c_t$ is either $0$ (when $Q^A$ is the smaller) or $\gamma(Q^B-Q^A)(s',a^*)$ (when $Q^B$ is smaller), so $c_t\to 0$ once the gap $\Delta^{BA}_t = Q^B_t - Q^A_t \to 0$. Now track that gap. Both tables are updated toward the *same* target $y$ on the same visited pair, so
$$\Delta^{BA}_{t+1}(s_t,a_t) = \Delta^{BA}_t(s_t,a_t) + \alpha_t\big[(y - Q^B_t(s_t,a_t)) - (y - Q^A_t(s_t,a_t))\big] = (1-\alpha_t)\,\Delta^{BA}_t(s_t,a_t).$$
The $y$ cancels — that's the payoff of feeding both tables the same target — and the gap contracts by $(1-\alpha_t)$ each visit, so with $\sum\alpha_t = \infty$ it goes to zero. Hence $c_t\to0$, the lemma applies, $Q^A\to Q^*$, and by the symmetric argument $Q^B\to Q^*$ too. So clipped double-Q is convergent in the tabular case, and the proof also quietly tells me the practical shortcut is fine: I don't need two separate actors and two separate targets — I can use a single actor and feed *both* critics the *same* min-based target, and it's that shared target that makes the critics agree. One actor $\pi_\phi$ optimized against $Q_{\theta_1}$, both critics regressed to a shared target, using the actor's target copy once I add target networks for stability. If $Q_{\theta_2} \ge Q_{\theta_1}$ this is the standard update and adds no bias beyond the single-critic target; if $Q_{\theta_2} < Q_{\theta_1}$ the value is pulled down, Double-Q style. Cheaper, and exactly the case analysis I wanted.

That handles bias. But the overestimation curve climbing over a million steps was never *only* bias — it was bias *accumulating*, and accumulation is a variance story. Let me make the accumulation precise. In function approximation the Bellman equation is never satisfied exactly; each fit leaves a residual TD-error $\delta(s,a)$, so what I actually learn obeys $Q_\theta(s,a) = r + \gamma\,\mathbb E[Q_\theta(s',a')] - \delta(s,a)$. Unroll it one step, substituting the same relation at $(s_{t+1},a_{t+1})$:
$$Q_\theta(s_t,a_t) = r_t + \gamma\,\mathbb E\big[r_{t+1} + \gamma\,\mathbb E[Q_\theta(s_{t+2},a_{t+2})] - \delta_{t+1}\big] - \delta_t,$$
and continuing the substitution to the horizon,
$$Q_\theta(s_t,a_t) = \mathbb E_{s_i\sim p_\pi,\,a_i\sim\pi}\Big[\sum_{i=t}^{T}\gamma^{i-t}\big(r_i - \delta_i\big)\Big].$$
So the learned value isn't the expected return; it's the expected return *minus the expected discounted sum of all future TD-errors*. Its variance is therefore proportional to the variance of future rewards *and* of future estimation errors, all weighted by $\gamma^{i-t}$. With a discount near one those weights barely decay, so the errors from far-future states pile in at nearly full strength, and the variance of the estimate can balloon across updates if the per-step error isn't held down. And each gradient step only shrinks error on the minibatch it saw — it gives no guarantee anywhere else in state space, so errors elsewhere are free to grow between visits. So the discipline I need is: keep the error small at *every* update, and don't let the bootstrap chase a moving estimate.

This is exactly what a target network is *for*, and I want to nail down the connection rather than treat it as folklore. A deep critic needs many gradient steps to fit a target; if the target is the online network itself, then every step both chases the target and moves it, leaving residual error that the next step inherits and compounds — precisely the accumulation above, now driven by my own optimization rather than the environment. Freeze the target — soft-update it, $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$ with small $\tau$ — and the critic fits a stationary objective over many steps, so the residual per window is small and doesn't snowball. I can see the effect by watching the value estimate under different $\tau$. With a *fixed* policy, every $\tau$ converges to about the same place — a faster target is just more jittery on the way. But the instructive run is with a *learning* policy trained against the current value estimate: there, a fast target ($\tau=1$, i.e. no real target network) doesn't merely jitter, it diverges. So the divergence isn't intrinsic to bootstrapping; it's the *interaction* of a high-variance value estimate with a policy that keeps updating against it. Which lines up with the feedback loop from before: the value diverges through overestimation when the policy is bad, and the policy gets bad when it's trained on an inaccurate value. The two failure modes are one mode.

If a slow target reduces error over many critic updates, and policy updates against a still-high-error critic are what trigger divergence, then I should not update the policy at the same breakneck cadence as the critic. Let the critic settle first. So update the actor — and the target networks — only once every $d$ critic updates. With $d$ steps of critic fitting between policy moves, each policy move sees a critic that has had time to drive its error down, and I stop wastefully nudging the actor against an essentially-unchanged critic. This is the two-timescale structure that the linear-actor-critic convergence results ask for: the critic on the fast timescale, near-converged relative to the slowly-moving actor. There's a tension in choosing $d$ — larger $d$ means cleaner, lower-variance policy updates, but the critic is still only trained once per environment step, and starving the actor of updates would cripple it. A small delay, $d=2$, keeps the actor moving while giving each policy step a fresher critic. Cheap and it helps on both counts.

One source of target variance is still untouched, and it's peculiar to deterministic policies. The target action is a single point, $\pi_{\phi'}(s')$, and the critic is a function approximator with bumps. A deterministic actor will happily find and sit on a narrow spurious peak in the critic — an action where function-approximation error happens to spike the value upward — and then the target is read off exactly at that sharp, unreliable point, injecting high variance into the regression and exactly the kind of inflated value I've been fighting. What I'd want is for the target to reflect the value of a small *neighborhood* of the action, not one knife-edge point, so a spurious spike gets averaged away and similar actions are forced to similar values. That is, in fact, an old idea wearing different clothes: SARSA-style targets, which evaluate the action actually taken rather than an idealized best action, are known to give "safer" values precisely because they account for the value of nearby, perturbed actions. So fit the value of a region around the target action: $y = r + \gamma\,\mathbb E_\epsilon[Q_{\theta'}(s', \pi_{\phi'}(s') + \epsilon)]$. I can't integrate that expectation, but I can approximate it the cheap way — perturb the target action with a little noise and let averaging over minibatches do the smoothing:
$$y = r + \gamma\, Q_{\theta'}(s', \pi_{\phi'}(s') + \epsilon),\qquad \epsilon \sim \mathrm{clip}(\mathcal N(0,\sigma),\,-c,\,c).$$
The noise is clipped to a small range $[-c,c]$ so the target action stays in the genuine neighborhood of the policy's action and I don't sample some wildly different action and call it "similar." And the perturbed action should be clipped back into the valid action range too, so I never read the critic at an action the environment can't execute. This is Expected-SARSA-flavored — averaging the target over a distribution of actions — except it's off-policy and the smoothing noise is chosen on its own terms, decoupled from whatever noise I use for exploration. The effect is a value surface that's smooth across nearby actions, denying the deterministic actor those narrow peaks to overfit, and as a side effect giving policies that are more robust to action perturbations, which is the kind of safety SARSA values are known for.

Now everything I've derived stacks onto one base — a deterministic actor-critic with replay and soft target networks — as independent modifications, and they reinforce each other rather than collide. Forming the target: take the target actor's action, smooth it with clipped noise, evaluate it under *both* target critics, take the min:
$$\tilde a = \mathrm{clip}\big(\pi_{\phi'}(s') + \epsilon,\,-a_{\max},\,a_{\max}\big),\quad \epsilon\sim\mathrm{clip}(\mathcal N(0,\tilde\sigma),-c,c),\qquad y = r + \gamma\,\min_{i=1,2} Q_{\theta'_i}(s',\tilde a).$$
Regress both critics to $y$ by minimizing $N^{-1}\sum(y - Q_{\theta_i}(s,a))^2$ every step. Then only every $d$ steps, update the actor by the deterministic policy gradient through the *first* critic, $\nabla_\phi J = N^{-1}\sum \nabla_a Q_{\theta_1}(s,a)|_{a=\pi_\phi(s)}\nabla_\phi\pi_\phi(s)$, and soft-update the target networks $\theta'_i \leftarrow \tau\theta_i + (1-\tau)\theta'_i$, $\phi'\leftarrow \tau\phi + (1-\tau)\phi'$. Twin critics with a min target for the bias; delayed actor and target updates plus the soft target for the accumulated-error variance; clipped-noise smoothing of the target action for the deterministic-peak variance. Three knobs, all aimed at the single disease — function-approximation error, and the way an actor turns that error into a self-reinforcing overestimating loop.

Let me write the core of it, grounded in how the deterministic actor-critic is actually built, with one critic module that carries both Q-heads so the twin estimates and the convenient single-head readout for the actor live in one place.

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    # deterministic policy s -> a, tanh-squashed to the action range
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, s):
        a = F.relu(self.l1(s))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    # twin Q-networks: the target uses the smaller estimate to cap overestimation
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        q1 = self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))
        q2 = self.l6(F.relu(self.l5(F.relu(self.l4(sa)))))
        return q1, q2

    def Q1(self, s, a):
        # single head used for the actor's gradient (the actor ascends Q1)
        sa = torch.cat([s, a], 1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(sa)))))


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action,
                 discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise      # sigma for target-action smoothing
        self.noise_clip = noise_clip          # the [-c, c] clip on that noise
        self.policy_freq = policy_freq         # the delay d between policy updates
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        state, action, next_state, reward, not_done = (
            state.to(device), action.to(device), next_state.to(device),
            reward.to(device), not_done.to(device)
        )

        with torch.no_grad():
            # target policy smoothing: perturb the target action, clip the noise to [-c, c]
            noise = (torch.randn_like(action) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise
                           ).clamp(-self.max_action, self.max_action)  # stay in valid range

            # clipped double-Q: take the min of the two target critics
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # regress BOTH critics to the same min-based target (this shared target makes them agree)
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # delayed policy & target updates: only every d critic updates
        if self.total_it % self.policy_freq == 0:
            # deterministic policy gradient through Q1
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft-update the frozen targets
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
```

The concrete PyTorch implementation here uses two hidden layers of 256 units, Adam at $3\times10^{-4}$, minibatches of 256, soft-update $\tau=0.005$, smoothing noise $\sigma=0.2$ clipped to $\pm0.5$ for normalized action ranges, delay $d=2$, discount $0.99$, and plain uncorrelated Gaussian exploration $\mathcal N(0,0.1)$ in the outer training loop; for non-unit action ranges, the smoothing noise and clip are passed in action units. So the causal chain, start to finish: function approximation guarantees value error; selecting an action by the very estimate that's noisy biases the value upward, and routed through a gradient-ascending actor this overestimation becomes a self-reinforcing loop; the discrete cures fail because a slow policy can't supply an independent estimator; clipping the less-biased estimate by the more-biased one with a min caps the bias at the worst case of standard Q-learning and leans toward the safer side of underestimation; the same error accumulates as variance through bootstrapping, which a frozen target network restrains, which in turn says only update the policy once the critic has settled, hence the delay; and the deterministic actor's habit of overfitting sharp value peaks is defused by smoothing the target over a clipped neighborhood of the action. One disease, three coordinated corrections, dropped onto the deterministic actor-critic without disturbing its sample efficiency.
