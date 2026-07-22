Let me start from the failure that actually bites: the extrinsic reward is sparse — a single $+1$ on reaching a vest 350 steps away, zero everywhere else. The policy gradient $\mathbb{E}[\nabla\log\pi\,(\sum_t r_t)]$ is therefore zero almost always; the agent only learns if it *happens* to reach the goal, and stochastic action noise won't carry it across hundreds of reward-free steps. So I need to manufacture a reward from the agent's own experience — an intrinsic signal $r^i$ that the policy maximizes alongside the (mostly zero) extrinsic $r^e$, so the total is $r_t=r_t^i+r_t^e$ and the agent is pulled to explore even when the environment says nothing.

What should $r^i$ be? The two obvious families are novelty (reward rarely-visited states, needs a density model over images) and prediction error (reward states whose consequences the agent predicts badly, needs a dynamics model). Prediction error appeals to me because it directly captures "the agent is learning something about how the world responds to it." So: build a forward model that predicts $s_{t+1}$ from $s_t$ and $a_t$, and reward the agent in proportion to how wrong it is. The cleanest version predicts the next *observation* — the pixels — and uses the pixel error as $r^i$.

Let me think about whether that actually does what I want, by imagining the agent in a room with a window through which tree leaves are moving in a breeze. The leaf motion is driven by wind I can't model and the agent certainly can't; the pixel locations of the leaves are essentially unpredictable. So the pixel-prediction error there stays high *forever*, the intrinsic reward stays high forever, and the agent is permanently rewarded for gawking at the leaves. It never moves on. The same thing happens with a television showing static, with shadows from other moving objects, with distractors — any source of visual variation that is inherently unpredictable but completely inconsequential to the agent becomes an inexhaustible curiosity reward. Novelty-counting has the identical disease: every noisy frame looks novel. So pixel-prediction error is worse than useless here; it actively traps exploration on noise. And there's a separate objection — predicting raw pixels is hard, and it's not even clear it's the right thing to spend capacity on.

The remedy "reward only what's hard-but-learnable" is the right intuition, but estimating learnability has no feasible algorithm. Let me instead diagnose *what kind* of variation I want the reward to respond to. Everything that can change the agent's observation is one of three things: (1) stuff the agent can control, (2) stuff it can't control but that affects it — another vehicle bearing down on it, say, (3) stuff out of its control and not affecting it — the leaves. I want curiosity to care about (1) and (2) and be completely *blind* to (3). The trouble with pixel prediction is that pixels mix all three indiscriminately, and (3) dominates the error budget precisely because it's unpredictable.

So the real question isn't "forward model or not" — it's "forward model in *what space*?" If I had a feature map $\phi(s)$ that represented only the action-relevant content of the observation and threw away (3), then forward-predicting $\phi(s_{t+1})$ and rewarding *that* error would respond only to the parts of the world I care about. The leaves wouldn't be in $\phi$ at all, so their unpredictability couldn't generate reward. The problem reduces to: how do I learn such a $\phi$ without hand-designing it per environment?

The first thing I'd reach for is to just train a forward model in some learned $\phi$, jointly learning $\phi$ to make the forward prediction easy. Let me check what the optimizer actually does with that freedom before trusting it. The forward loss is $\frac12\|\hat\phi(s_{t+1})-\phi(s_{t+1})\|^2$, and $\phi$ is one of the things being optimized to drive it down. Consider the degenerate $\phi\equiv c$, a constant vector independent of the input. Then $\phi(s_{t+1})=c$ for every state, and a forward model that simply outputs $c$ (e.g. all weights zero, bias $c$) predicts it with error exactly $0$. So the global minimum of the forward objective is reached by a $\phi$ that has thrown away the entire image — the loss isn't just small, it's literally zero, and gradient descent has every incentive to walk toward it. A feature space optimized to be *predictable* collapses. So I can't anchor $\phi$ with the forward task; I need a different task that *forces* $\phi$ to retain action-relevant information and forbids that trivial collapse.

The inverse problem is the natural candidate. Instead of predicting the future from the action, predict the *action* from the present and the future: given $\phi(s_t)$ and $\phi(s_{t+1})$, recover $a_t$. This is self-supervised — the tuples $(s_t,a_t,s_{t+1})$ come for free from the agent acting. Does it actually rule out the collapse that broke the forward-only idea? Put the same degenerate $\phi\equiv c$ into the inverse task. Now both inputs to the classifier are $c$ regardless of which action was taken, so the predictor sees an identical input on every transition and the best it can do is output the marginal action distribution. Under a uniform action prior over $n$ actions that is the uniform softmax, and its cross-entropy is $-\sum_{a}\frac1n\log\frac1n=\log n$. For Doom's four actions that floor is $\log 4\approx1.386$ nats; for Mario's fourteen it's $\log 14\approx2.639$ — and that is the *worst* the inverse loss can be (chance level), the value it's pinned at when $\phi$ carries no information. So the forward task sends the collapse to its global *minimum* (loss $0$) while the inverse task sends the very same collapse to its *maximum* (loss $\log n$). The two objectives push $\phi$ in opposite directions on degeneracy, and the inverse one is exactly the wall against collapse I was missing.

And it keeps the right information for the right reason. To pull the inverse cross-entropy below that $\log n$ floor, $\phi$ has to make $\phi(s_t)$ and $\phi(s_{t+1})$ differ in a way that depends on $a_t$ — i.e. it has to encode whatever in the scene changed because of the action: that's category (1), the controllable stuff, and to the extent (2) matters for distinguishing actions it keeps that too. But it has *no incentive* to encode (3): the leaves don't help predict which action the agent took, so a $\phi$ trained purely to predict the action gains nothing from representing them and (under any capacity pressure) will simply not. The inverse task is the thing that makes $\phi$ both non-degenerate and noise-blind. (Prior work built joint inverse-forward models but used the forward model merely to *regularize* the inverse features; here I want the opposite — the forward error is going to *be* the reward.)

Let me write the two models. The encoder produces $\phi(s)$. The inverse model $g$ takes the two encodings and predicts the action,
$$\hat a_t=g\big(\phi(s_t),\phi(s_{t+1});\theta_I\big),$$
trained to minimize a discrepancy $L_I(\hat a_t,a_t)$. For discrete actions $g$ outputs a softmax over actions and minimizing $L_I$ is maximum likelihood under a multinomial — ordinary cross-entropy. Now, *in the feature space that $g$ has carved out*, train the forward model $f$ to predict the next features from the current features and the action,
$$\hat\phi(s_{t+1})=f\big(\phi(s_t),a_t;\theta_F\big),\qquad L_F=\tfrac12\big\|\hat\phi(s_{t+1})-\phi(s_{t+1})\big\|_2^2.$$
And I'll take the intrinsic reward to be precisely that forward error, scaled:
$$r_t^i=\frac{\eta}{2}\big\|\hat\phi(s_{t+1})-\phi(s_{t+1})\big\|_2^2,\qquad \eta>0.$$
What does this reward actually do over the course of training? Two properties I'm relying on are that the bonus is high on an unfamiliar transition and that it *fades* once that transition has been seen enough — otherwise the agent would loop forever on whatever it found first. Let me trace the fading on the simplest case I can: one fixed transition, a single scalar feature, target $\phi(s_{t+1})=1$ with the forward prediction starting at $0$ and one gradient step (lr $0.3$) on $\frac12(w-1)^2$ each time the transition recurs. The half-squared error — which is exactly $r^i/\eta$ — goes $0.500,\,0.245,\,0.120,\,0.059,\,0.029,\,0.014$ over six visits: monotonically toward zero, roughly halving each pass. So a mastered transition does stop paying out, on its own, with no extra bookkeeping; the bonus is self-extinguishing. It is large exactly when the agent meets a transition whose *action-relevant consequence* it cannot yet predict — a genuinely novel bit of controllable dynamics — and shrinks as the forward model fits it, so the agent is pushed onward to the next unmastered piece of controllable structure. The leaves contribute nothing because they're not in $\phi$ in the first place. The noisy-TV failure I started from doesn't reappear, and the cure came entirely from the choice of feature space, not from any extra mechanism bolted on.

I should be careful about one thing: the forward error feeds the policy reward, but I must *not* train the ICM to maximize that reward. If I let the policy objective push on $\theta_I,\theta_F$, the system could cheat — make $\phi$ unpredictable on purpose to inflate the reward, or otherwise game it. The forward error has to be an honest measure of the model's ignorance, so the ICM is trained *only* on its own losses $L_I,L_F$ from observed transitions, and the policy is trained *only* to maximize the (intrinsic + extrinsic) reward. They share the reward signal but not the gradient through it. So the overall objective composes three pieces with their own parameters:
$$\min_{\theta_P,\theta_I,\theta_F}\Big[\,-\lambda\,\mathbb{E}_{\pi(s_t;\theta_P)}\big[\textstyle\sum_t r_t\big]\;+\;(1-\beta)\,L_I\;+\;\beta\,L_F\,\Big],$$
where the first term is the policy-gradient objective written as a minimization (it touches only $\theta_P$ through the reward, with $r_t^i$ treated as a fixed signal w.r.t. the policy), and the ICM terms touch only $\theta_I,\theta_F$. Here $\beta\in[0,1]$ trades the inverse loss against the forward loss and $\lambda>0$ trades policy learning against learning the reward model. I'd set $\beta$ small — the inverse task is the one that must *shape* $\phi$ correctly, so it should dominate — say $\beta=0.2$, and $\lambda=0.1$. The policy optimizer is A3C: scalable, on-policy, and it consumes the scalar $r_t$ regardless of where it came from.

Now the architecture, concretely. Inputs are grayscale $42\times42$, four frames stacked for temporal context. The encoder $\phi$ is four convolutions, 32 filters each, $3\times3$, stride 2, pad 1, ELU after each. I need the flattened feature dimension to size the two MLP heads, so let me push the spatial size through the stride-2 layers with the standard formula $\lfloor(n+2p-k)/s\rfloor+1=\lfloor(n+2-3)/2\rfloor+1=\lfloor(n-1)/2\rfloor+1$: $42\to21\to11\to6\to3$ (e.g. $\lfloor10/2\rfloor+1=6$, then $\lfloor5/2\rfloor+1=3$). So the final map is $3\times3\times32$, and $\phi(s)\in\mathbb{R}^{288}$. The inverse model then concatenates $\phi(s_t),\phi(s_{t+1})$ into a $576$-vector, runs it through one FC of 256 units (ReLU) and an output FC to $n_{\text{actions}}$ logits, softmax cross-entropy against $a_t$. The forward model concatenates $\phi(s_t)$ (288) with the one-hot action ($+n_{\text{actions}}$ on the input, $292$ for Doom) and runs through one FC of 256 (ReLU) and an output FC back to 288, then the half-squared-error to $\phi(s_{t+1})$ — the same $288$ on input and output, as it must be for the residual to make sense. The A3C policy net mirrors the same conv stack but feeds an LSTM of 256 units and then two heads (policy, value). Learning rate $10^{-3}$, Adam.

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
        # (L_F is allowed to shape phi too; collapse is still forbidden because a constant phi
        #  is L_F's minimum but L_I's MAXIMUM -- chance-level log(n_actions) -- so L_I pins phi)
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
