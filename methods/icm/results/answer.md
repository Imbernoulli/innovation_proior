# ICM (Intrinsic Curiosity Module)

**Problem.** With sparse or absent extrinsic reward, the policy gradient is mostly zero and random
exploration never reaches the goal. An intrinsic reward is needed — but pixel-prediction-error
curiosity is permanently captured by unpredictable-yet-irrelevant visual variation (the noisy-TV
trap: TV static, moving leaves, distractors), so exploration stalls on noise.

**Key idea.** Generate curiosity as the prediction error of a forward dynamics model, but compute
it in a *learned feature space* $\phi(s)$ that encodes only action-relevant information. Learn
$\phi$ by self-supervision on an *inverse dynamics* task — predict the action from
$\phi(s_t),\phi(s_{t+1})$. Because $\phi$ need only recover the action, it represents what the
agent controls/what affects it and ignores irrelevant variation; it also cannot collapse to a
constant (that would destroy the action information).

**The two models.**
- Inverse model: $\hat a_t=g(\phi(s_t),\phi(s_{t+1});\theta_I)$, loss $L_I$ = cross-entropy
  (softmax over discrete actions; MLE under a multinomial).
- Forward model (in feature space): $\hat\phi(s_{t+1})=f(\phi(s_t),a_t;\theta_F)$,
  $L_F=\frac12\|\hat\phi(s_{t+1})-\phi(s_{t+1})\|_2^2$.
- Intrinsic reward: $r_t^i=\frac{\eta}{2}\|\hat\phi(s_{t+1})-\phi(s_{t+1})\|_2^2$, $\eta>0$.

**Training.** Policy $\pi(s;\theta_P)$ (A3C) maximizes $\mathbb{E}[\sum_t r_t]$ with
$r_t=r_t^i+r_t^e$ ($r^e$ mostly zero). The ICM is trained on its own losses only (never on the
reward, so it cannot be gamed). Joint objective:
$$\min_{\theta_P,\theta_I,\theta_F}\Big[-\lambda\,\mathbb{E}_{\pi}\big[\textstyle\sum_t r_t\big]+(1-\beta)L_I+\beta L_F\Big],$$
$\beta\in[0,1]$ weighs inverse vs forward loss, $\lambda>0$ weighs policy learning vs the reward
model. The first term updates only $\theta_P$ (treating $r^i$ as a fixed signal); the ICM terms
update only $\theta_I,\theta_F$.

**Why it works.** $\phi$ has no incentive to encode environment factors the agent's actions
neither affect nor are affected by, so inherently-unpredictable irrelevant variation produces no
intrinsic reward — exploration is robust to distractors and noise. The reward is high on
transitions whose controllable consequences are not yet predicted, and decays as the forward model
learns them, pushing the agent onward.

**Architecture / hyperparameters.** Inputs grayscale $42\times42$, 4-frame stack. Encoder: 4 conv
layers, 32 filters, $3\times3$, stride 2, pad 1, ELU; $\phi\in\mathbb{R}^{288}$
($42\to21\to11\to6\to3$, $3\cdot3\cdot32$). Inverse: concat $\phi(s_t),\phi(s_{t+1})\to$ FC 256
(ReLU) $\to$ FC $n_{\text{actions}}$. Forward: concat $\phi(s_t),a_t\to$ FC 256 (ReLU) $\to$ FC
288. A3C torso mirrors the conv stack into an LSTM-256 with policy/value heads. $\beta=0.2$,
$\lambda=0.1$, lr $=10^{-3}$ (Adam), 20 asynchronous workers; action repeat 4 (Doom) / 6 (Mario).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

PHI_DIM, HID = 288, 256
BETA, LAMBDA, ETA = 0.2, 0.1, 1.0

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU())
    def forward(self, s):
        return self.conv(s).flatten(1)                       # (B, 288)

class ICM(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.phi = Encoder()
        self.inverse = nn.Sequential(nn.Linear(2 * PHI_DIM, HID), nn.ReLU(),
                                     nn.Linear(HID, n_actions))
        self.forward_net = nn.Sequential(nn.Linear(PHI_DIM + n_actions, HID), nn.ReLU(),
                                         nn.Linear(HID, PHI_DIM))

    def losses_and_reward(self, s_t, a_t, s_tp1):
        phi_t, phi_tp1 = self.phi(s_t), self.phi(s_tp1)
        a_onehot = F.one_hot(a_t, self.n_actions).float()
        logits = self.inverse(torch.cat([phi_t, phi_tp1], dim=1))      # predict action
        L_I = F.cross_entropy(logits, a_t)
        phi_hat = self.forward_net(torch.cat([phi_t, a_onehot], dim=1))# predict next features
        per_sample = 0.5 * (phi_hat - phi_tp1).pow(2).sum(dim=1)
        L_F = per_sample.mean()
        r_i = (ETA * per_sample).detach()                              # intrinsic reward (signal only)
        return L_I, L_F, r_i

def icm_objective(L_I, L_F):
    return (1 - BETA) * L_I + BETA * L_F                               # ICM trained on its own losses
# A3C consumes r_t = r_t^i + r_t^e; full objective adds -LAMBDA * E_pi[sum_t r_t] over theta_P.
```
