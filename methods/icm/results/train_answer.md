The problem is reinforcement learning with sparse or absent extrinsic reward. In these settings the policy gradient is effectively zero almost everywhere, because the environment returns a reward only on the rare occasion when the agent happens to reach a distant goal. Undirected exploration such as entropy regularization or epsilon-greedy dithering cannot carry the agent across hundreds of reward-free steps, so the agent learns nothing until it randomly stumbles onto success. The natural fix is to generate an intrinsic reward from the agent's own experience, but simple ideas have a fatal flaw. Rewarding the prediction error of a forward model that predicts raw pixels is permanently hijacked by inherently unpredictable but irrelevant visual variation: television static, moving leaves, flickering shadows, or other agents. This is the noisy-TV trap. The pixel error stays high forever, so the agent is rewarded forever for staring at noise and exploration stalls. Count-based novelty bonuses suffer from the same pathology, because every noisy frame looks novel, and building a density model directly over raw images is difficult.

The right curiosity signal should respond to what the agent can control or what affects it, and be blind to what is out of its control and inconsequential. Everything that changes an observation falls into one of three categories: controllable change, uncontrollable but affecting change, and uncontrollable irrelevant change. Pixel prediction mixes all three, and the irrelevant sources dominate the error budget precisely because they are unpredictable. The question therefore reduces to finding a feature space that keeps the action-relevant content and discards the rest.

The method is the Intrinsic Curiosity Module, or ICM. It builds a learned feature representation phi(s) of the observation and computes curiosity as the forward-dynamics prediction error in that feature space, not in pixel space. The crucial ingredient is how phi is learned. If phi were trained only to make forward prediction easy, it could collapse to a constant vector and become useless. To prevent collapse and force the right content, phi is anchored with a self-supervised inverse-dynamics task: given phi(s_t) and phi(s_{t+1}), predict the action a_t that produced the transition. This task is free from the agent's own rollouts and has exactly the right incentive structure. To recover the action, phi must encode whatever the action changed, which is the controllable part of the scene and, to the extent it matters, the uncontrollable but affecting part. It has no reason to encode irrelevant noise, because the leaves do not help predict which action the agent took. It also cannot collapse to a constant, because a constant feature would make action recovery impossible.

The module has two small networks. The inverse model g takes the concatenation of phi(s_t) and phi(s_{t+1}) and outputs logits over actions; it is trained with cross-entropy against the true action. The forward model f takes phi(s_t) and a one-hot action and predicts the next feature phi_hat(s_{t+1}); it is trained with half the squared Euclidean error. The intrinsic reward is exactly that forward error, scaled by a constant eta. Because phi ignores noise, the reward is high only on transitions whose controllable consequences the agent has not yet mastered, and it decays as the forward model learns those transitions, pushing the agent onward. The reward is detached before being fed to the policy, so the ICM is trained only on its own losses and cannot game the reward by making phi deliberately unpredictable. The full objective combines the A3C policy-gradient objective on the total reward r_t = r_t^i + r_t^e with the inverse and forward losses, with the inverse loss weighted more heavily so that the forward task cannot distort phi.

Concretely, the encoder uses four convolutional layers with 32 filters, 3x3 kernels, stride 2, and ELU activations, mapping a 42x42 grayscale 4-frame stack to a 288-dimensional feature vector. The inverse model is a single hidden layer of 256 ReLU units followed by an output layer over actions. The forward model is a single hidden layer of 256 ReLU units followed by a linear projection back to 288 dimensions. The hyperparameters used in the original experiments are beta = 0.2 for the forward-loss weight, lambda = 0.1 trading policy learning against the reward model, eta = 1.0 for the intrinsic-reward scale, and a learning rate of 1e-3 with Adam.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

PHI_DIM = 288
HID = 256
BETA = 0.2
LAMBDA = 0.1
ETA = 1.0

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ELU(),
        )

    def forward(self, s):
        return self.conv(s).flatten(1)

class ICM(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.phi = Encoder()
        self.inverse = nn.Sequential(
            nn.Linear(2 * PHI_DIM, HID), nn.ReLU(),
            nn.Linear(HID, n_actions)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(PHI_DIM + n_actions, HID), nn.ReLU(),
            nn.Linear(HID, PHI_DIM)
        )

    def losses_and_reward(self, s_t, a_t, s_tp1):
        phi_t = self.phi(s_t)
        phi_tp1 = self.phi(s_tp1)
        a_onehot = F.one_hot(a_t, self.n_actions).float()

        logits = self.inverse(torch.cat([phi_t, phi_tp1], dim=1))
        L_I = F.cross_entropy(logits, a_t)

        phi_hat = self.forward_net(torch.cat([phi_t, a_onehot], dim=1))
        per_sample = 0.5 * (phi_hat - phi_tp1).pow(2).sum(dim=1)
        L_F = per_sample.mean()

        r_i = (ETA * per_sample).detach()
        return L_I, L_F, r_i

def icm_objective(L_I, L_F):
    return (1 - BETA) * L_I + BETA * L_F

def total_reward(r_extrinsic, r_intrinsic):
    return r_extrinsic + r_intrinsic
```

In use, an A3C agent consumes the total reward r_t = r_t^i + r_t^e. The ICM updates are performed on sampled transitions using the icm_objective, while the policy updates treat r_i as a fixed scalar signal. This separation is essential: it keeps the intrinsic reward an honest measure of the model's ignorance, and it makes the agent robust to distractors while still driven to explore and master the controllable structure of its environment.