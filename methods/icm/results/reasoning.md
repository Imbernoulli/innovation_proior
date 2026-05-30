Let me start from the failure that actually bites: the extrinsic reward is sparse — a single $+1$ on reaching a vest 350 steps away, zero everywhere else. The policy gradient $\mathbb{E}[\nabla\log\pi\,(\sum_t r_t)]$ is therefore zero almost always; the agent only learns if it *happens* to reach the goal, and stochastic action noise won't carry it across hundreds of reward-free steps. So I need to manufacture a reward from the agent's own experience — an intrinsic signal $r^i$ that the policy maximizes alongside the (mostly zero) extrinsic $r^e$, so the total is $r_t=r_t^i+r_t^e$ and the agent is pulled to explore even when the environment says nothing.

What should $r^i$ be? The two obvious families are novelty (reward rarely-visited states, needs a density model over images) and prediction error (reward states whose consequences the agent predicts badly, needs a dynamics model). Prediction error appeals to me because it directly captures "the agent is learning something about how the world responds to it." So: build a forward model that predicts $s_{t+1}$ from $s_t$ and $a_t$, and reward the agent in proportion to how wrong it is. The cleanest version predicts the next *observation* — the pixels — and uses the pixel error as $r^i$.

Let me think about whether that actually does what I want, by imagining the agent in a room with a window through which tree leaves are moving in a breeze. The leaf motion is driven by wind I can't model and the agent certainly can't; the pixel locations of the leaves are essentially unpredictable. So the pixel-prediction error there stays high *forever*, the intrinsic reward stays high forever, and the agent is permanently rewarded for gawking at the leaves. It never moves on. The same thing happens with a television showing static, with shadows from other moving objects, with distractors — any source of visual variation that is inherently unpredictable but completely inconsequential to the agent becomes an inexhaustible curiosity reward. Novelty-counting has the identical disease: every noisy frame looks novel. So pixel-prediction error is worse than useless here; it actively traps exploration on noise. And there's a separate objection — predicting raw pixels is hard, and it's not even clear it's the right thing to spend capacity on.

The remedy "reward only what's hard-but-learnable" is the right intuition, but estimating learnability has no feasible algorithm. Let me instead diagnose *what kind* of variation I want the reward to respond to. Everything that can change the agent's observation is one of three things: (1) stuff the agent can control, (2) stuff it can't control but that affects it — another vehicle bearing down on it, say, (3) stuff out of its control and not affecting it — the leaves. I want curiosity to care about (1) and (2) and be completely *blind* to (3). The trouble with pixel prediction is that pixels mix all three indiscriminately, and (3) dominates the error budget precisely because it's unpredictable.

So the real question isn't "forward model or not" — it's "forward model in *what space*?" If I had a feature map $\phi(s)$ that represented only the action-relevant content of the observation and threw away (3), then forward-predicting $\phi(s_{t+1})$ and rewarding *that* error would respond only to the parts of the world I care about. The leaves wouldn't be in $\phi$ at all, so their unpredictability couldn't generate reward. The problem reduces to: how do I learn such a $\phi$ without hand-designing it per environment?

Suppose I just train a forward model in some learned $\phi$, jointly learning $\phi$ to make the forward prediction easy. Watch what happens — the optimizer will discover the cheapest way to make forward prediction error small is to make $\phi$ *constant*. A $\phi$ that maps everything to the same vector is perfectly predictable (error zero) and utterly useless. A learned feature space optimized to be *predictable* collapses. So I can't anchor $\phi$ with the forward task; I need a different task that *forces* $\phi$ to retain action-relevant information and forbids the trivial collapse.

The inverse problem is exactly that anchor. Instead of predicting the future from the action, predict the *action* from the present and the future: given $\phi(s_t)$ and $\phi(s_{t+1})$, recover $a_t$. This is self-supervised — the tuples $(s_t,a_t,s_{t+1})$ come for free from the agent acting — and it pins $\phi$ in the way I want. To recover the action that took $s_t$ to $s_{t+1}$, $\phi$ *must* encode whatever in the scene changed because of the action: that's category (1), the controllable stuff, and to the extent (2) matters for distinguishing actions it keeps that too. But it has *no incentive* to encode (3): the leaves don't help predict which action the agent took, so a $\phi$ trained purely to predict the action will simply not represent them. And it can't collapse to a constant, because a constant $\phi$ destroys the information needed to recover the action and the inverse loss would be terrible. The inverse task is the thing that makes $\phi$ both non-degenerate and noise-blind. (Prior work built joint inverse-forward models but used the forward model merely to *regularize* the inverse features; here I want the opposite — the forward error is going to *be* the reward.)

Let me write the two models. The encoder produces $\phi(s)$. The inverse model $g$ takes the two encodings and predicts the action,
$$\hat a_t=g\big(\phi(s_t),\phi(s_{t+1});\theta_I\big),$$
trained to minimize a discrepancy $L_I(\hat a_t,a_t)$. For discrete actions $g$ outputs a softmax over actions and minimizing $L_I$ is maximum likelihood under a multinomial — ordinary cross-entropy. Now, *in the feature space that $g$ has carved out*, train the forward model $f$ to predict the next features from the current features and the action,
$$\hat\phi(s_{t+1})=f\big(\phi(s_t),a_t;\theta_F\big),\qquad L_F=\tfrac12\big\|\hat\phi(s_{t+1})-\phi(s_{t+1})\big\|_2^2.$$
And the intrinsic reward is precisely that forward error, scaled:
$$r_t^i=\frac{\eta}{2}\big\|\hat\phi(s_{t+1})-\phi(s_{t+1})\big\|_2^2,\qquad \eta>0.$$
Read what this reward means now: it is large exactly when the agent encounters a transition whose *action-relevant consequence* it cannot yet predict — a genuinely novel bit of controllable dynamics — and it decays as the forward model learns that transition, so the agent is pushed onward to the next unmastered piece of controllable structure. The leaves contribute nothing because they're not in $\phi$. That's the noisy-TV cure, and it falls out of the choice of feature space, not from any extra mechanism.

I should be careful about one thing: the forward error feeds the policy reward, but I must *not* train the ICM to maximize that reward. If I let the policy objective push on $\theta_I,\theta_F$, the system could cheat — make $\phi$ unpredictable on purpose to inflate the reward, or otherwise game it. The forward error has to be an honest measure of the model's ignorance, so the ICM is trained *only* on its own losses $L_I,L_F$ from observed transitions, and the policy is trained *only* to maximize the (intrinsic + extrinsic) reward. They share the reward signal but not the gradient through it. So the overall objective composes three pieces with their own parameters:
$$\min_{\theta_P,\theta_I,\theta_F}\Big[\,-\lambda\,\mathbb{E}_{\pi(s_t;\theta_P)}\big[\textstyle\sum_t r_t\big]\;+\;(1-\beta)\,L_I\;+\;\beta\,L_F\,\Big],$$
where the first term is the policy-gradient objective written as a minimization (it touches only $\theta_P$ through the reward, with $r_t^i$ treated as a fixed signal w.r.t. the policy), and the ICM terms touch only $\theta_I,\theta_F$. Here $\beta\in[0,1]$ trades the inverse loss against the forward loss and $\lambda>0$ trades policy learning against learning the reward model. I'd set $\beta$ small — the inverse task is the one that must *shape* $\phi$ correctly, so it should dominate — say $\beta=0.2$, and $\lambda=0.1$. The policy optimizer is A3C: scalable, on-policy, and it consumes the scalar $r_t$ regardless of where it came from.

Now the architecture, concretely. Inputs are grayscale $42\times42$, four frames stacked for temporal context. The encoder $\phi$ is four convolutions, 32 filters each, $3\times3$, stride 2, pad 1, ELU after each. Track the spatial size: $42\to21\to11\to6\to3$, and $3\times3\times32=288$, so $\phi(s)\in\mathbb{R}^{288}$. The inverse model concatenates $\phi(s_t),\phi(s_{t+1})$ into a $576$-vector, runs it through one FC of 256 units (ReLU) and an output FC to $n_{\text{actions}}$ logits, softmax cross-entropy against $a_t$. The forward model concatenates $\phi(s_t)$ (288) with the one-hot action and runs through one FC of 256 (ReLU) and an output FC back to 288, then the half-squared-error to $\phi(s_{t+1})$. The A3C policy net mirrors the same conv stack but feeds an LSTM of 256 units and then two heads (policy, value). Learning rate $10^{-3}$, Adam.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

PHI_DIM = 288    # 42->21->11->6->3 with k3 s2 p1; 3*3*32 = 288
HID = 256
BETA = 0.2       # inverse-vs-forward loss weight
LAMBDA = 0.1     # policy-gradient vs ICM weight
ETA = 1.0        # intrinsic-reward scale

class Encoder(nn.Module):
    # phi(s): four convs, 32 filters, 3x3, stride 2, pad 1, ELU. Shared by inverse+forward.
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
        )
    def forward(self, s):
        return self.conv(s).flatten(1)                 # (B, 288)

class ICM(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.phi = Encoder()
        # inverse model: [phi(s_t), phi(s_{t+1})] -> action logits  (anchors phi)
        self.inverse = nn.Sequential(nn.Linear(2 * PHI_DIM, HID), nn.ReLU(),
                                     nn.Linear(HID, n_actions))
        # forward model in feature space: [phi(s_t), a_t] -> phi_hat(s_{t+1})
        self.forward_net = nn.Sequential(nn.Linear(PHI_DIM + n_actions, HID), nn.ReLU(),
                                         nn.Linear(HID, PHI_DIM))

    def encode(self, s):
        return self.phi(s)

    def losses_and_reward(self, s_t, a_t, s_tp1):
        phi_t, phi_tp1 = self.phi(s_t), self.phi(s_tp1)
        a_onehot = F.one_hot(a_t, self.n_actions).float()
        # inverse: predict the action -> cross-entropy (MLE under multinomial)
        logits = self.inverse(torch.cat([phi_t, phi_tp1], dim=1))
        L_I = F.cross_entropy(logits, a_t)
        # forward: predict next features from current features + action -> 1/2 MSE
        # (L_F is allowed to shape phi too; the inverse loss dominates via beta so phi can't collapse)
        phi_hat = self.forward_net(torch.cat([phi_t, a_onehot], dim=1))
        per_sample = 0.5 * (phi_hat - phi_tp1).pow(2).sum(dim=1)
        L_F = per_sample.mean()
        # intrinsic reward = (eta/2)||phi_hat - phi(s_{t+1})||^2 ; detached (a signal, not a path)
        r_i = (ETA * per_sample).detach()
        return L_I, L_F, r_i

def icm_objective(L_I, L_F):
    # ICM trained ONLY on its own losses, never on the policy reward
    return (1 - BETA) * L_I + BETA * L_F

# total reward fed to A3C:  r_t = r_t^i + r_t^e   (r^e mostly zero in sparse tasks)
# full objective: min over (theta_P, theta_I, theta_F) of
#   -LAMBDA * E_pi[sum_t r_t]  +  (1-BETA) L_I + BETA L_F
# where the policy term updates only theta_P (r^i is a fixed signal w.r.t. the policy),
# and the ICM term updates only theta_I, theta_F.
```

The causal chain in one breath: with sparse extrinsic reward the policy gradient is mostly zero, so manufacture an intrinsic reward from prediction error → but pixel-prediction error is captured forever by unpredictable-yet-irrelevant variation (the noisy-TV trap), because pixels mix controllable, affecting, and irrelevant change indiscriminately → so predict in a learned feature space $\phi$ that keeps only action-relevant content → a feature space trained to be *predictable* collapses to a constant, so anchor $\phi$ with the self-supervised *inverse* task of recovering the action from $\phi(s_t),\phi(s_{t+1})$, which forces $\phi$ to encode what the action affects and to ignore what it doesn't → train a *forward* model in this $\phi$-space and use its error $\frac{\eta}{2}\|\hat\phi(s_{t+1})-\phi(s_{t+1})\|^2$ as the intrinsic reward, noise-blind by construction → train the ICM only on $L_I,L_F$ (never on the reward, so it can't be gamed) and the A3C policy only to maximize $r^i+r^e$, combined as $-\lambda\mathbb{E}[\sum r]+(1-\beta)L_I+\beta L_F$ with $\beta=0.2,\lambda=0.1$.
