The no-bonus run told me exactly what's missing, and it told me in numbers. On Frostbite all three seeds found reward (mean 215, tight 96–280) — there the first payoff is reachable by entropy jitter alone, so vanilla PPO is fine. But on Tutankham two of three seeds flatlined at *exactly* zero on every metric (mean 36.5 is really "one lucky seed, two that never saw a reward"), and on Private Eye the per-seed returns were {0, 100, −1000} for a *negative* mean of −300 — the −1000 is the tell: with no directed exploration the agent wanders into the game's penalties as readily as its rewards. So the diagnosis is sharp, and it is not a learning-rate or a credit-assignment problem. Where the reward is sparse the policy gradient $\mathbb{E}[\nabla\log\pi\,(\sum_t r_t)]$ is zero almost everywhere, because there is nothing to be advantaged over; the agent only learns if it *happens* to stumble onto the goal, and stochastic action noise will not carry it across hundreds of reward-free steps. It's a *signal* problem. So I have to manufacture a reward from the agent's own experience — an intrinsic signal $r^i$ the policy maximizes alongside the (mostly zero) extrinsic $r^e$, total $r_t=r_t^i+r_t^e$, large in states the agent hasn't mastered, so "go somewhere new" becomes something the policy gradient can actually ascend. That should rescue exactly the games PPO couldn't touch — the ones where the first reward is out of reach of random jitter.

What should $r^i$ be? The two obvious families are novelty (reward rarely-visited states, needs a density model over images) and prediction error (reward states whose consequences the agent predicts badly, needs a dynamics model). Prediction error appeals to me because it directly captures "the agent is learning something about how the world responds to it." So: build a forward model that predicts $s_{t+1}$ from $s_t$ and $a_t$, and reward the agent in proportion to how wrong it is. The cleanest version predicts the next *observation* — the pixels — and uses the pixel error as $r^i$.

Let me check whether that actually does what I want before I commit to it, because a bad bonus is worse than none — PPO's −1000 Private Eye seed already shows that the wrong drive is *actively harmful*. Picture the agent on one of these Atari screens with some patch of the image flickering in a way it cannot control — a scrolling background, an animated sprite, the boundary churn that sticky actions induce. The pixel content there is essentially unpredictable, so the pixel-prediction error stays high *forever*, the intrinsic reward stays high forever, and the agent is permanently paid to gawk at the flicker. It never moves on. Novelty-counting has the identical disease: every noisy frame looks novel. So pixel-prediction error is worse than useless here; it would trap exploration on noise, which on Private Eye is precisely how you end up camping next to a penalty instead of crossing the gap.

The remedy "reward only what's hard-but-learnable" is the right intuition, but estimating learnability has no feasible algorithm. Let me instead diagnose *what kind* of variation I want the reward to respond to. Everything that can change the agent's observation is one of three things: (1) stuff the agent can control, (2) stuff it can't control but that affects it — a hazard bearing down on it, say, (3) stuff out of its control and not affecting it — the flicker. I want curiosity to care about (1) and (2) and be completely *blind* to (3). The trouble with pixel prediction is that pixels mix all three indiscriminately, and (3) dominates the error budget precisely because it's unpredictable.

So the real question isn't "forward model or not" — it's "forward model in *what space*?" If I had a feature map $\phi(s)$ that represented only the action-relevant content and threw away (3), then forward-predicting $\phi(s_{t+1})$ and rewarding *that* error would respond only to the parts of the world I care about. The flicker wouldn't be in $\phi$ at all, so its unpredictability couldn't generate reward. The problem reduces to: how do I learn such a $\phi$ without hand-designing it per game?

Suppose I just train a forward model in some learned $\phi$, jointly learning $\phi$ to make the forward prediction easy. Watch what happens — the optimizer discovers that the cheapest way to make forward error small is to make $\phi$ *constant*. A $\phi$ that maps everything to the same vector is perfectly predictable (error zero) and utterly useless. A feature space optimized to be *predictable* collapses. So I can't anchor $\phi$ with the forward task; I need a different task that *forces* $\phi$ to retain action-relevant information and forbids the trivial collapse.

The inverse problem is exactly that anchor. Instead of predicting the future from the action, predict the *action* from the present and the future: given $\phi(s_t)$ and $\phi(s_{t+1})$, recover $a_t$. This is self-supervised — the tuples $(s_t,a_t,s_{t+1})$ come for free from the agent acting — and it pins $\phi$ the way I want. To recover the action that took $s_t$ to $s_{t+1}$, $\phi$ *must* encode whatever in the scene changed because of the action: category (1), and to the extent (2) matters for distinguishing actions it keeps that too. But it has *no incentive* to encode (3): the flicker doesn't help predict which action the agent took, so a $\phi$ trained purely to predict the action will simply not represent it. And it can't collapse to a constant, because a constant $\phi$ destroys the information needed to recover the action. The inverse task is the thing that makes $\phi$ both non-degenerate and noise-blind.

Let me write the two models. The encoder produces $\phi(s)$. The inverse model $g$ takes the two encodings and predicts the action,
$$\hat a_t=g\big(\phi(s_t),\phi(s_{t+1});\theta_I\big),$$
trained to minimize a discrepancy $L_I(\hat a_t,a_t)$. For discrete actions $g$ outputs a softmax over actions and minimizing $L_I$ is maximum likelihood under a multinomial — ordinary cross-entropy. Now, *in the feature space $g$ has carved out*, train the forward model $f$ to predict the next features from the current features and the action,
$$\hat\phi(s_{t+1})=f\big(\phi(s_t),a_t;\theta_F\big),\qquad L_F=\tfrac12\big\|\hat\phi(s_{t+1})-\phi(s_{t+1})\big\|_2^2.$$
And the intrinsic reward is precisely that forward error, scaled:
$$r_t^i=\frac{\eta}{2}\big\|\hat\phi(s_{t+1})-\phi(s_{t+1})\big\|_2^2,\qquad \eta>0.$$
Read what this reward means: it is large exactly when the agent meets a transition whose *action-relevant consequence* it cannot yet predict — a genuinely novel piece of controllable dynamics — and it decays as the forward model learns that transition, so the agent is pushed onward to the next unmastered piece of controllable structure. The flicker contributes nothing because it's not in $\phi$. That's the noisy-TV cure, and it falls out of the choice of feature space, not from any extra mechanism.

One thing I must be careful about: the forward error feeds the policy reward, but I must *not* train the ICM to maximize that reward. If I let the policy objective push on $\theta_I,\theta_F$, the system could cheat — make $\phi$ unpredictable on purpose to inflate the reward. The forward error has to be an honest measure of the model's ignorance, so the ICM is trained *only* on its own losses $L_I,L_F$ from observed transitions, and the policy is trained *only* to maximize the (intrinsic + extrinsic) reward. They share the reward signal but not the gradient through it:
$$\min_{\theta_P,\theta_I,\theta_F}\Big[\,-\lambda\,\mathbb{E}_{\pi(s_t;\theta_P)}\big[\textstyle\sum_t r_t\big]\;+\;(1-\beta)\,L_I\;+\;\beta\,L_F\,\Big],$$
where the first term touches only $\theta_P$ through the reward (with $r_t^i$ a fixed signal w.r.t. the policy) and the ICM terms touch only $\theta_I,\theta_F$. $\beta\in[0,1]$ trades inverse against forward loss; I keep it small — the inverse task is the one that must *shape* $\phi$ — say $\beta=0.2$, and $\lambda=0.1$. The policy optimizer is the fixed PPO loop; it consumes the scalar $r_t$ regardless of where it came from. The architecture is the inverse/forward pair over a shared conv encoder; concretely:

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

    def losses_and_reward(self, s_t, a_t, s_tp1):
        phi_t, phi_tp1 = self.phi(s_t), self.phi(s_tp1)
        a_onehot = F.one_hot(a_t, self.n_actions).float()
        logits = self.inverse(torch.cat([phi_t, phi_tp1], dim=1))   # predict action -> CE
        L_I = F.cross_entropy(logits, a_t)
        phi_hat = self.forward_net(torch.cat([phi_t, a_onehot], dim=1))  # predict next features
        per_sample = 0.5 * (phi_hat - phi_tp1).pow(2).sum(dim=1)
        L_F = per_sample.mean()
        r_i = (ETA * per_sample).detach()              # intrinsic reward: a signal, not a path
        return L_I, L_F, r_i

def icm_objective(L_I, L_F):
    return (1 - BETA) * L_I + BETA * L_F               # ICM trained ONLY on its own losses
# total reward to PPO: r_t = r_t^i + r_t^e ; policy term updates only theta_P.
```

So the delta from step 1 is concrete: replace the empty bonus module with one that learns a controllable-state encoder by inverse dynamics, predicts the next controllable state with a forward model, and emits the forward error as $r^i$; mix the intrinsic advantage in alongside the extrinsic one. Reading the step-1 shape, here is what I expect this to fix and where I'm unsure. Tutankham should *stabilize* — every seed should at least find the reward now that "explore the controllable world" is rewarded, so the two dead seeds that flatlined under PPO should come alive. Frostbite could jump, because curiosity keeps pushing the agent into new controllable configurations beyond the first ice-floe reward PPO already found. Private Eye is the open question, and I can already feel the risk in the construction: the bonus is a forward-prediction error that *decays as the forward model masters local dynamics*, and Private Eye's payoff is not just far but sits past a long stretch whose local dynamics are quickly learned — so curiosity may run out of drive at the first mastered region before it crosses the gap. If that happens, the diagnosis for the next step is already written: I'd need a novelty signal that doesn't decay the moment local dynamics are learned.

The causal chain in one breath: PPO's measured failure is a *signal* problem — sparse reward means a zero policy gradient, so two of three Tutankham seeds and all of Private Eye find nothing and one Private Eye seed goes to −1000 → manufacture an intrinsic reward from prediction error → but pixel-prediction error is captured forever by uncontrollable flicker (noisy-TV), because pixels mix controllable, affecting, and irrelevant change → so predict in a learned feature space $\phi$ that keeps only action-relevant content → a $\phi$ trained to be *predictable* collapses, so anchor it with the *inverse* task of recovering $a_t$ from $\phi(s_t),\phi(s_{t+1})$, which forces action-relevant content and forbids collapse → use the forward error in that space as $r^i$, noise-blind by construction, trained only on $L_I,L_F$ so it can't be gamed, fed to the fixed PPO loop — expecting Tutankham and Frostbite to recover, and watching Private Eye for the decay I suspect is coming.
