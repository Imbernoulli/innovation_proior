Let me start from the thing that's bothering me. I have a deep Q-network learning to play these games from pixels, and it works — it reaches strong scores on a lot of them. But when I instrument it, something is off. During training I periodically freeze the policy and roll it out, and I record two numbers: the value the network *predicts* for its own greedy policy, averaged over the states it visits, and the *actual* discounted return that same policy collects when I let it play. If the value function were any good these two should sit on top of each other — the predicted value of a policy is, by definition, the return you expect from running it. Instead the predicted value keeps climbing, well above the realized return, and on a couple of games (Asterix, Wizard of Wor) it doesn't just sit high, it blows up — the predicted values shoot off on a log scale while the score *falls*. So this isn't harmless calibration error. The estimates are biased upward, the bias is large, and it's coinciding with the policy getting worse.

Where could a systematic *upward* bias come from? Not from the optimizer per se — least-squares regression toward a target is unbiased for the target. So it has to be in the target itself. Let me write down the target I'm regressing onto. For a transition $(S_t, A_t, R_{t+1}, S_{t+1})$ the Q-learning target is
$$
Y_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta^-),
$$
where $\theta^-$ is the frozen target network. The reward is what it is; $\gamma$ is a constant. The only thing in there that's a function of my noisy estimates is $\max_a Q(S_{t+1}, a; \theta^-)$. So the bias, if there is one, lives in that max.

Let me think about what the max does to estimation error. Say the true optimal value at $s' = S_{t+1}$ is some $Q_*(s', a)$, and my network gives noisy estimates $Q(s',a) = Q_*(s',a) + \epsilon_a$, where the $\epsilon_a$ are the errors — could be from approximation, from the data being limited, from the targets themselves being stale, doesn't matter yet. Suppose for the moment the errors are zero-mean: $\mathbb{E}[\epsilon_a] = 0$ for each $a$, so my estimates are individually unbiased. Is $\max_a Q(s',a)$ then an unbiased estimate of $\max_a Q_*(s',a)$?

No. And the reason is just that $\max$ is convex. For any convex function $f$ and random vector $X$, Jensen gives $\mathbb{E}[f(X)] \ge f(\mathbb{E}[X])$. The pointwise maximum of a set of values is convex in those values. So
$$
\mathbb{E}\big[\max_a Q(s',a)\big] = \mathbb{E}\big[\max_a (Q_*(s',a)+\epsilon_a)\big] \ge \max_a \big(Q_*(s',a) + \mathbb{E}[\epsilon_a]\big) = \max_a Q_*(s',a).
$$
The expected max of my noisy values is at least the max of the true values. Averaging *after* taking the max gives something at least as large as taking the max of the averages. So even with perfectly unbiased per-action estimates, the bootstrap target is biased *upward*. That's the leak.

I can feel why intuitively, too: the max operator goes hunting through all $m$ actions and picks out the largest value, which means it preferentially picks the action whose noise $\epsilon_a$ happened to land on the high side. Noise that's positive gets selected; noise that's negative gets discarded by the max. So the single largest estimate is a biased view of "the value of the best action." And then I bootstrap: this inflated target gets regressed into $Q(S_t,A_t)$, which becomes part of the next state's target, and the inflation propagates backward through the chain of states. That's the compounding I'm seeing in the curves.

Now, I should be careful before I call this fatal, because there's a counterargument I half-believe. If the overestimation were *uniform* — every action in every state lifted by the same constant — then the greedy policy $\arg\max_a Q(s,a)$ wouldn't change at all, since adding a constant to every action preserves the ordering. The values would be wrong but the policy would be fine. And being optimistic about things you're *uncertain* about is a respectable exploration strategy — optimism in the face of uncertainty drives an agent to go check out under-explored actions. So why am I worried?

Two reasons, and they're exactly what separates this from benign optimism. First, this bias is not optimism about uncertain actions *before* I act on them — it appears *after* I've updated, baked into values I'm now treating as settled, overoptimism in the face of apparent certainty. Second, and decisively: there's no reason the bias is uniform. It depends on how many actions there are, on the shape of the errors at each state, on which states got more data. A state with ten noisy actions gets a bigger upward kick from the max than a state with two; a poorly-fit region of state space gets a bigger kick than a well-fit one. So the overestimate is *state- and action-dependent*, and a non-uniform additive distortion to $Q$ scrambles the relative ordering — which is the only thing the greedy policy reads off. Combine that with bootstrapping and I'm propagating wrong *relative* judgments about which states are worth more than others. That can absolutely produce a worse policy, which is what the diverging-value/falling-score games are showing me.

I want to know how big this can get, because "it's biased upward" isn't quantitative. Let me try to nail it down. There's a known handle on this: take a state where the value estimates have errors uniformly distributed in $[-\epsilon,\epsilon]$ and $m$ actions; one can show the maximum is overestimated by about $\gamma\,\epsilon\,\frac{m-1}{m+1}$. That's an upper bound for a specific error distribution. It already tells me the bias grows with the number of actions and is proportional to the error magnitude. But an upper bound is the wrong direction for my purposes — it tells me the bias *can't exceed* something. I'd much rather show the bias *must be at least* something, because that's what proves overestimation is unavoidable, not just possible. Let me try to build a lower bound.

Here's the cleanest setting I can make to isolate the max's contribution. Take a single state $s$ where all the true optimal action values are equal, $Q_*(s,a) = V_*(s)$ for every $a$. (If the truth itself has a unique best action with a gap, the max is more forgiving; the worst case for "the max manufactures value out of noise" is when there's nothing to actually choose between, so all-equal is the right stress test.) Now let my estimates $Q_t(s,a)$ be wrong but *unbiased on the whole* in the sense that they're balanced around the truth,
$$
\sum_a \big(Q_t(s,a) - V_*(s)\big) = 0,
$$
and not all correct, with a fixed amount of squared error,
$$
\frac{1}{m}\sum_a \big(Q_t(s,a) - V_*(s)\big)^2 = C > 0,\qquad m \ge 2.
$$
I want to lower-bound $\max_a Q_t(s,a)$. Define the per-action errors $\epsilon_a = Q_t(s,a) - V_*(s)$, so $\sum_a \epsilon_a = 0$ and $\sum_a \epsilon_a^2 = mC$. The claim I'll try to prove is
$$
\max_a Q_t(s,a) \ge V_*(s) + \sqrt{\tfrac{C}{m-1}}.
$$

Let me argue by contradiction. Suppose the errors could be arranged so that the max is *strictly below* that, i.e. $\max_a \epsilon_a < \sqrt{\frac{C}{m-1}}$, while still satisfying $\sum_a\epsilon_a = 0$ and $\sum_a\epsilon_a^2 = mC$. I'll show the two constraints can't both hold, so no such arrangement exists.

Split the errors by sign. Let $\{\epsilon^+_i\}_{i=1}^n$ be the strictly positive errors and $\{\epsilon^-_j\}_{j=1}^{m-n}$ the strictly negative ones. (Zeros I can throw in with either group without affecting the sums; what matters is how many are positive.) First, $n \le m-1$. Why: if $n = m$, every error is positive, but then $\sum_a \epsilon_a = 0$ forces every $\epsilon_a = 0$, contradicting $\sum_a \epsilon_a^2 = mC > 0$. So at least one error is non-positive, $n \le m-1$.

Now bound the positive mass. Each positive error is at most the max, which I assumed is $< \sqrt{\frac{C}{m-1}}$, so
$$
\sum_{i=1}^n \epsilon^+_i \le n\,\max_i \epsilon^+_i < n\sqrt{\tfrac{C}{m-1}}.
$$
The zero-sum constraint says the negative mass exactly cancels the positive mass: $\sum_j |\epsilon^-_j| = \sum_i \epsilon^+_i < n\sqrt{\frac{C}{m-1}}$. In particular each individual negative magnitude is bounded by the total, $\max_j|\epsilon^-_j| < n\sqrt{\frac{C}{m-1}}$.

I need the sum of *squares*, not the sum of magnitudes, so let me convert. For the negative part, $\sum_j (\epsilon^-_j)^2 \le \big(\sum_j |\epsilon^-_j|\big)\cdot \max_j|\epsilon^-_j|$ — that's just bounding each $(\epsilon^-_j)^2 = |\epsilon^-_j|\cdot|\epsilon^-_j| \le |\epsilon^-_j|\cdot\max_j|\epsilon^-_j|$ and summing (it's Hölder with the $\ell_1$/$\ell_\infty$ split). Plugging both bounds,
$$
\sum_{j} (\epsilon^-_j)^2 < n\sqrt{\tfrac{C}{m-1}}\cdot n\sqrt{\tfrac{C}{m-1}} = \frac{n^2 C}{m-1}.
$$
For the positive part, similarly $\sum_i (\epsilon^+_i)^2 \le n\big(\max_i\epsilon^+_i\big)^2 < n\cdot\frac{C}{m-1}$. Add them:
$$
\sum_{a} \epsilon_a^2 = \sum_i (\epsilon^+_i)^2 + \sum_j (\epsilon^-_j)^2 < \frac{nC}{m-1} + \frac{n^2 C}{m-1} = \frac{n(n+1)}{m-1}\,C.
$$
And since $n \le m-1$, the factor $\frac{n(n+1)}{m-1} \le \frac{(m-1)m}{m-1} = m$. So
$$
\sum_a \epsilon_a^2 < mC,
$$
which directly contradicts $\sum_a \epsilon_a^2 = mC$. The assumed arrangement is impossible. Therefore $\max_a \epsilon_a \ge \sqrt{\frac{C}{m-1}}$, i.e.
$$
\max_a Q_t(s,a) \ge V_*(s) + \sqrt{\tfrac{C}{m-1}}.
$$

Is this tight, or did I throw away too much? Let me try to hit it with equality. Put all the positive error on $m-1$ of the actions equally and dump the rest on the last one: $\epsilon_a = \sqrt{\frac{C}{m-1}}$ for $a = 1,\dots,m-1$, and $\epsilon_m = -\sqrt{(m-1)C}$. Check zero-sum: $(m-1)\sqrt{\frac{C}{m-1}} = \sqrt{(m-1)^2\cdot\frac{C}{m-1}} = \sqrt{(m-1)C}$, which cancels $\epsilon_m$. Check the squared error: $(m-1)\cdot\frac{C}{m-1} + (m-1)C = C + (m-1)C = mC$. Both constraints hold, and $\max_a \epsilon_a = \sqrt{\frac{C}{m-1}}$ exactly. So the bound is tight — there genuinely exist unbiased-on-the-whole error patterns whose max sits exactly at $V_*(s) + \sqrt{C/(m-1)}$, and none can do better.

So here's what I've got: in a state where all actions are truly equal, *any* error pattern that's balanced around the truth with mean-squared error $C$ forces the single-estimator target to overshoot the truth by at least $\sqrt{C/(m-1)}$. I never assumed the errors were independent, never assumed a distribution — just balance and a nonzero spread. Estimation error of *any source*, as long as it's there, drives the max-target up. And in practice the estimates are always wrong early in training, so this is the norm, not a corner case.

One thing about this bound nags at me: it *decreases* in $m$ — more actions, smaller $\sqrt{C/(m-1)}$. That feels backwards; I argued earlier the max should be more inflated with more actions to choose from, and the $\frac{m-1}{m+1}$ upper bound *increases* in $m$. What's going on? It's because this is a *lower* bound, the floor over all error patterns, and to make the max as *small* as possible you spread the squared error across as many actions as you can — with more actions you can dilute the peak. So the floor drops with $m$. It's an artifact of asking "how small can the overestimation possibly be"; it doesn't describe the typical case. Let me also work out a *typical*-case number to see the other direction.

Take the same all-equal state, $Q_*(s,a) = V_*(s)$, but now let the errors be genuinely random and independent: $\epsilon_a = Q_t(s,a) - V_*(s)$ i.i.d. uniform on $[-1,1]$. What's $\mathbb{E}[\max_a \epsilon_a]$? The max of i.i.d. variables is an order statistic, so go through the CDF. For a single uniform$[-1,1]$, $P(\epsilon_a \le x) = \frac{1+x}{2}$ for $x\in(-1,1)$. By independence,
$$
P\big(\max_a \epsilon_a \le x\big) = \prod_{a=1}^m P(\epsilon_a \le x) = \left(\frac{1+x}{2}\right)^m,\quad x\in(-1,1).
$$
Differentiate to get the density of the max: $f_{\max}(x) = \frac{m}{2}\left(\frac{1+x}{2}\right)^{m-1}$. Then
$$
\mathbb{E}[\max_a \epsilon_a] = \int_{-1}^{1} x\,\frac{m}{2}\left(\frac{1+x}{2}\right)^{m-1} dx.
$$
Let me just verify the antiderivative they'd use is $\left(\frac{x+1}{2}\right)^m \frac{mx-1}{m+1}$. Differentiate it: $\frac{d}{dx}\big[(\tfrac{x+1}{2})^m\big]\cdot\frac{mx-1}{m+1} + (\tfrac{x+1}{2})^m\cdot\frac{m}{m+1}$. The first derivative is $\frac{m}{2}(\tfrac{x+1}{2})^{m-1}$, so the whole thing is $(\tfrac{x+1}{2})^{m-1}\big[\frac{m}{2}\cdot\frac{mx-1}{m+1} + \frac{x+1}{2}\cdot\frac{m}{m+1}\big] = (\tfrac{x+1}{2})^{m-1}\frac{m}{2(m+1)}\big[(mx-1)+(x+1)\big] = (\tfrac{x+1}{2})^{m-1}\frac{m}{2(m+1)}(m+1)x = x\,\frac{m}{2}(\tfrac{x+1}{2})^{m-1}$. Good, that's exactly the integrand. Evaluate at the ends: at $x=1$, $(\tfrac{2}{2})^m\frac{m-1}{m+1} = \frac{m-1}{m+1}$; at $x=-1$, $(0)^m(\cdots) = 0$. So
$$
\mathbb{E}[\max_a \epsilon_a] = \frac{m-1}{m+1}.
$$
There's the other direction: the *expected* overestimation rises with $m$ (toward $1$ as $m\to\infty$), confirming that with more actions the typical inflation gets worse, even though the worst-case floor gets smaller. And scaling errors to $[-\epsilon,\epsilon]$ and pushing through the discounted bootstrap recovers the $\gamma\,\epsilon\,\frac{m-1}{m+1}$ figure I started from — so that old upper bound is just this expectation under the uniform model.

Both analyses point the same way: the single-estimator max manufactures positive bias out of nothing but estimation error, and there's no setting of a flexible deterministic approximator that escapes it — the function-fitting experiment I can run in my head confirms it. Fit each action's value with a polynomial through ground-truth samples at scattered states, with all actions truly equal to the same $V_*(s)$. Even with *no noise* and exact values at the sampled states, the fits disagree between actions at the *unsampled* states (different actions saw different samples), and the pointwise max over those fits rides above the true curve almost everywhere — and the more flexible the polynomial, the more it overfits between samples and the higher the max climbs. So it's not specifically a noise phenomenon and not specifically an under-capacity phenomenon; it's the max picking the upper envelope of a bundle of imperfect estimates.

So now the real question: how do I get a target that estimates the value of the best action *without* this upward bias? Stare at the target again:
$$
\max_a Q(S_{t+1},a;\theta^-) = Q\big(S_{t+1},\ \arg\max_a Q(S_{t+1},a;\theta^-);\ \theta^-\big).
$$
Writing it this second way makes the disease obvious. The max is doing two jobs with one set of numbers: it *selects* which action is greedy (the $\arg\max$), and it *evaluates* that action's value (the $Q(\cdot;\theta^-)$ on the outside). Both use $\theta^-$. So the action I pick is, by construction, the one whose $\theta^-$-estimate is largest — which biases me toward exactly the actions whose noise is most positive — and then I read off that same inflated $\theta^-$-estimate as the value. The selection and the evaluation are perfectly correlated in their error, and that correlation is what turns "noisy" into "biased high." If I select an action *because* its estimate is the highest, I should not trust *that same estimate* to tell me its value; I've conditioned on it being large.

What if I used a *different*, independent set of estimates to do the evaluation? Suppose I had a second value function $\theta'$ whose errors are independent of $\theta$'s. Select with $\theta$, evaluate with $\theta'$:
$$
Y_t = R_{t+1} + \gamma\,Q\big(S_{t+1},\ \arg\max_a Q(S_{t+1},a;\theta);\ \theta'\big).
$$
Now the action chosen by $\theta$ is some particular $a^\star$, and I read off $Q(S_{t+1}, a^\star;\theta')$. Since $\theta'$'s error on $a^\star$ is independent of why $a^\star$ was selected, that evaluation isn't conditioned-on-being-large — it's just an unbiased estimate of $Q_*(S_{t+1},a^\star)$. In the all-equal stress state this is clean to see: $a^\star$ is whichever action $\theta$ happened to inflate most, but $\theta'$ doesn't know or care, so $\mathbb{E}[Q(S_{t+1},a^\star;\theta')] = V_*(S_{t+1})$ — no upward bias. In fact, in the tight example I built for the lower bound, I can set the second estimator's value on the selected action to exactly $V_*$, so the *lower bound on the absolute error of the decoupled estimate is zero* — it can be exactly right where the single max is forced to overshoot by $\sqrt{C/(m-1)}$. Decoupling selection from evaluation is the fix. (This is the idea behind the tabular two-estimator scheme of van Hasselt (2010): keep two tables, assign each experience randomly to update one of them, use one to pick and the other to score.)

The catch with the literal two-estimator recipe is that it doubles the value functions and I'd have to train and store a whole second network, and arrange the symmetric random-assignment updates. That's a lot of machinery to bolt onto a system that's already delicate, and it changes the comparison — I want to know whether *just* fixing the max helps, not whether "DQN plus a second network" helps.

But wait — I already have a second set of weights lying around. The target network $\theta^-$ is a frozen copy of the online network $\theta$, sitting there precisely so the regression target doesn't move every step. What if I let $\theta$ do the selection and $\theta^-$ do the evaluation?
$$
Y_t = R_{t+1} + \gamma\,Q\big(S_{t+1},\ \arg\max_a Q(S_{t+1},a;\theta);\ \theta^-\big).
$$
The online net picks the greedy action; the target net scores it. Compare to plain DQN, which is $R_{t+1} + \gamma\,Q(S_{t+1}, \arg\max_a Q(S_{t+1},a;\theta^-);\theta^-)$ — selection and evaluation both on $\theta^-$. The *only* change is whose argmax I use: $\theta$ instead of $\theta^-$ for the selection. No new network, no new parameters, no extra forward passes beyond one I'm basically already doing, and the target-network update rule stays exactly as it was — a periodic copy. This is about the smallest possible edit to the existing system that introduces the decoupling.

Let me be honest about how much this buys me, because $\theta$ and $\theta^-$ are not independent the way the idealized $\theta'$ was — $\theta^-$ is a *stale copy* of $\theta$, so their errors are correlated, and right after a target-network refresh $\theta^- = \theta$ exactly and the whole thing collapses back to plain Q-learning for that interval. So I won't fully kill the bias the way two truly independent estimators would. But the staleness gives me partial decoupling: between refreshes, $\theta$ has moved on while $\theta^-$ has not, so the action the current $\theta$ thinks is best is no longer guaranteed to be the one $\theta^-$ inflated, and that's enough to take a real bite out of the overestimation. And if I want to push the decoupling further, the lever is obvious from this analysis: increase the gap $\tau$ between target-network copies, so $\theta$ and $\theta^-$ drift further apart and spend less of the time freshly-synced — which is exactly the tuning knob I'd reach for to reduce the bias more.

That's the whole idea, and it's grounded in the bound: the single max overshoots by at least $\sqrt{C/(m-1)}$ whenever there's estimation error $C$, and that overshoot comes entirely from selecting and evaluating with the same values; swap the evaluator for a different (even partially-independent) set and the floor on the error drops to zero. The non-uniformity of that overshoot across states was corrupting the relative value ordering and dragging the policy down; removing it should give both more accurate values and better policies, at essentially no cost.

Now the code. I'll start from a standard DQN training loop and change exactly one thing — how the next-state bootstrap value is formed. Everything else (the conv net, the replay buffer, $\epsilon$-greedy, the periodic target copy) stays identical, because the point is to isolate the effect of the decoupled target.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class QNetwork(nn.Module):
    # stacked-frame state -> per-action values; same net for online and target
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, num_actions),
        )
    def forward(self, x):
        return self.net(x / 255.0)

def train_step(batch, gamma, online_net, target_net, optimizer):
    obs, actions, rewards, next_obs, dones = batch
    with torch.no_grad():
        # decouple selection from evaluation:
        # ONLINE net (theta) selects the greedy next action -> this is the
        # only line that differs from DQN, which would argmax with target_net
        next_actions = online_net(next_obs).argmax(dim=1, keepdim=True)
        # TARGET net (theta^-) evaluates the action the online net chose
        next_q = target_net(next_obs).gather(1, next_actions).squeeze(1)
        td_target = rewards + gamma * next_q * (1.0 - dones)
    q_sa = online_net(obs).gather(1, actions).squeeze(1)        # Q(S_t, A_t; theta)
    loss = F.mse_loss(td_target, q_sa)                          # regress onto the decoupled target
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss

# acting / storage / target sync are unchanged from DQN:
#   epsilon-greedy on online_net, store transitions in a uniform replay buffer,
#   every train_frequency steps sample a minibatch and call train_step,
#   every tau steps copy online_net weights into target_net (a larger tau
#   spends less time freshly-synced, decoupling theta and theta^- further).
```

The causal chain: the predicted values ran above the realized returns because the bootstrap target $\max_a Q(S_{t+1},a;\theta^-)$ selects and evaluates the greedy action with the same noisy estimates, and by Jensen the max of unbiased-but-noisy values is biased upward — provably by at least $\sqrt{C/(m-1)}$ in a state with all-equal true values and mean-squared error $C$, an overshoot that is uneven across states and so corrupts the relative ordering the greedy policy depends on. Decoupling selection from evaluation removes that floor (it drops to zero with an independent evaluator); reusing the already-present target network as the evaluator — online net selects, target net evaluates — gets most of that benefit as a one-line change with no new parameters, and a longer target-update period sharpens it.
