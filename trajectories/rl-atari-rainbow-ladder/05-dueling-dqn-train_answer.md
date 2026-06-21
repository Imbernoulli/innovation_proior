Prioritizing replay was the broad win I hoped for — 140% median, the biggest step since the decoupled target — by reallocating the gradient budget toward the typical game's surprising transitions. I have now fixed the target's bias, the exploration, and how the buffer samples; the loss and discount are reasonable. Two axes are still untouched: the *architecture of the head* and the *value object itself* (still a scalar). I save the scalar object for last because it is a deeper, representational change, so the cheaper structural move comes first. Right now the head is a single undifferentiated stream — the 512 features feed one linear map to $|\mathcal A|$ outputs and each action's value is estimated independently — and that is the wrong shape for what I am learning. In a large fraction of Atari states the *action barely matters*: the ball is mid-flight nowhere near the paddle, the screen is between events, whatever I do for a few frames the outcome is the same. There the action values are all nearly equal and the only quantity that carries information is their common level — *how good is it to be here at all*, the state value $V(s)$ — while the action choice matters only in the sparse, decisive states. So the object I am learning is really two things glued together: a state value $V(s)$ that matters *everywhere*, and a per-action advantage $A(s,a)=Q(s,a)-V(s)$ that matters only in the decisive states and is near zero elsewhere.

The single-stream head learns $V(s)$ terribly. In a TD update on a transition $(s,a,\dots)$ only the $a$-th output gets a gradient — I regress $Q(s,a)$ toward its target and the other actions' outputs are untouched. So the state value is not its own quantity; it is implicit, smeared across all $|\mathcal A|$ outputs, corrected only through whichever single action was sampled. The one quantity a bootstrapping algorithm needs accurate at *every* state — because every target is $r+\gamma V(s')$-shaped through the next-state value — is the most *diluted* thing in the network, updated $1/|\mathcal A|$ as often as it should be and only ever through the lens of one action. This is not an algorithm bug; the target is fine. It is the *shape* of the function approximator fighting the structure of the problem.

So I propose **Dueling DDQN**: reshape the head, not the algorithm. Split the 512 features into two streams — a **value stream** producing a scalar $V(s)$ and an **advantage stream** producing a vector $A(s,a)$ — then recombine them into the single $Q$ output the rest of the agent expects, so replay, the target, the $\epsilon$-noise all see the identical state$\to Q$ interface and nothing else has to change. The point of splitting is that $V(s)$ now sits *underneath all $|\mathcal A|$ outputs*: every transition, whatever action it used, backpropagates into the value stream, so the shared state value is learned from *all* the data instead of a $1/|\mathcal A|$ slice. And the advantage stream is free to drive the many redundant actions toward $A\approx0$ rather than independently fitting each one's value — it only has to represent the *differences*, genuinely low-dimensional where actions are interchangeable.

The naive recombination $Q=V+A$ has a problem I must fix or the split is meaningless: it is *unidentifiable*. Add a constant $c$ to $V$ and subtract $c$ from every $A(s,a)$ and $Q$ is unchanged, so gradient descent has no pressure to make the value stream learn the *actual* $V$ — it could park any offset there and compensate in the advantages, and then $V$ is junk plus a constant, not the state value at all. I need to *pin the offset* by subtracting a per-state reference from the advantage before adding it to $V$, so the streams can no longer trade a free constant. Two choices. The clean-semantics one subtracts the max, $Q(s,a)=V(s)+\big(A(s,a)-\max_{a'}A(s,a')\big)$; then at the greedy action the bracket is zero so $Q(s,a^\star)=V(s)$ and $V$ is *exactly* $\max_a Q$, which is what I want it to mean — but $\max_{a'}A$ jumps discontinuously whenever the best action flips, and that jump destabilizes training. The practical choice subtracts the *mean*,
$$Q(s,a)=V(s)+\Big(A(s,a)-\tfrac1{|\mathcal A|}\textstyle\sum_{a'}A(s,a')\Big).$$
This gives up the exact "$V=\max Q$" semantics — $V$ becomes the value plus the mean advantage offset, and the centered advantage becomes $Q(s,a)-\operatorname{mean}_{a'}Q(s,a')$ — but the mean is a *smooth* reference that does not jump when the argmax flips, so the advantage stream only has to track a slowly-moving baseline. More stable, and the only cost is a reinterpretation of what $V$ absorbs, not a loss of the identifiability fix. The crucial invariant: subtracting a per-state *constant* (max or mean, same for all actions in that state) never changes the *rank order* of actions, so $\arg\max_a Q(s,a)$ is identical to the naive sum's — the greedy and $\epsilon$-greedy policies are exactly what they would have been. The aggregator is purely a training-time offset-control device; it changes how the streams are *learned*, not the policy they parameterize. Same policy class, same algorithm, only the head's internal structure and gradient flow differ.

A couple of stability details the split forces. Both streams backprop into the *shared* conv trunk, so the trunk now receives roughly twice the incoming gradient it did with one stream; I rescale the gradient entering the trunk by $1/\sqrt2$ to keep its magnitude in the range the encoder was stable at. And I clip the global gradient norm ($\le10$) with a slightly lower learning rate than plain Double DQN, because the two-stream head is a little more delicate early on. The streams are each a 512-unit FC layer + ReLU off the shared trunk; value stream $\to1$, advantage stream $\to|\mathcal A|$, combined with the mean aggregator inside the forward pass. I keep the Double-DQN target underneath — the head reshape is orthogonal to how the target is built, so it composes unchanged.

The bar: this should help on the broad middle of the suite for a general reason, since *every* Atari game has many states where the action is nearly irrelevant, so every game benefits from learning $V$ from all transitions instead of a $1/|\mathcal A|$ slice — and the benefit is larger the *larger* the action set, because that is where the value dilution under the single-stream head was worst. So I expect another broad lift clearing 140%, of the same character as the prioritized-replay gain rather than the tail-only noisy-nets one. What it cannot fix — and I note it because it is the last rung's job — is that $Q$ (and now $V$ and $A$) are still *scalars*, point estimates of a return whose whole distribution the agent never sees. The head is now well-shaped, but it is shaped to predict a mean.

```python
# Dueling head: value + advantage streams, mean-subtraction aggregator, 1/sqrt(2) trunk-grad rescale.
# Code home: vwxyzjn/cleanrl; excerpted from methods/dueling-dqn/results/answer.md.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.trunk_grad_scale = 1.0 / math.sqrt(2.0)
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.value_stream = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(),
                                              nn.Linear(512, n_actions))

    def forward(self, x):
        feat = self.feature(x / 255.0)
        if torch.is_grad_enabled() and feat.requires_grad:
            feat.register_hook(lambda grad: grad * self.trunk_grad_scale)   # both streams feed the trunk
        value = self.value_stream(feat)                                     # (B, 1)
        advantage = self.advantage_stream(feat)                            # (B, |A|)
        # Q = V + (A - mean_a' A): blocks the V+c / A-c trade, preserves action ranks.
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


def train_step(q_network, target_network, optimizer, data, gamma, max_grad_norm=10.0):
    with torch.no_grad():
        next_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)   # Double-DQN
        next_q = target_network(data.next_observations).gather(1, next_actions).squeeze(1)
        y = data.rewards.flatten() + gamma * next_q * (1 - data.dones.flatten())
    q = q_network(data.observations).gather(1, data.actions).squeeze(1)
    loss = F.mse_loss(y, q)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_network.parameters(), max_grad_norm)
    optimizer.step()
    return loss
```
