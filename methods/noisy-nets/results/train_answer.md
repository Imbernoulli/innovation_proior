I propose Noisy Networks as the canonical method for replacing action-space dithering with learnable parameter noise in deep reinforcement-learning agents. A reinforcement-learning agent in an MDP has to improve its value function or policy from collected data while also choosing actions that gather useful future data. The second job is awkward in deep RL because the function approximator is powerful, the state space is large, and the agent usually sees only a thin stream of its own current behaviour. The dominant exploration mechanisms are local dithering rules. In a value-based agent, epsilon-greedy acts greedily most of the time and picks a uniformly random action with probability epsilon. In a policy-gradient agent like A3C, an entropy bonus keeps the softmax policy from collapsing. Both inject stochasticity right at the action distribution, which is exactly the wrong place. A fresh random action is drawn at every step with no memory of why the previous exploratory action was taken, and for epsilon-greedy the exploratory action is even state-independent. If the useful information lies at the end of a coordinated sequence of choices, independent per-step jitter spends almost all of its budget on local wiggling rather than on a coherent exploratory trajectory. The scale of exploration is also wrong: epsilon schedules and entropy weights are hand-set numbers, usually shared across tasks and states, even though how much one should explore differs enormously between a familiar state and an unseen one.

The nearby alternatives each fall short. Optimism and confidence-bound methods are clean in small or linear settings but do not drop cleanly into large nonlinear networks. Intrinsic-reward methods add novelty or information-gain bonuses, but that introduces a second reward scale that must be balanced against the environment reward and can quietly change the objective if misweighted. Randomized value functions are the most relevant route: sample a plausible value function and act greedily under it, committing the agent to a sampled estimate instead of dithering action by action. Bootstrapped DQN realizes this with several value heads and bootstrap masks, but those duplicated heads and masks are extra machinery. Plain parameter-space perturbation makes behaviour state-dependent and temporally coherent, yet if the perturbation scale is fixed or externally adapted, the amount of exploration is still not learned by the RL loss itself. What I want is exploratory variation structured over multi-step trajectories, learned to grow or shrink from the same loss that trains the network, and realized by touching only linear heads in a standard DQN, dueling-DQN, or A3C agent.

The idea is to move the stochasticity upstream. Instead of corrupting the selected action, I corrupt the function that maps states to values or policies. I make a network parameter vector noisy as theta equals mu plus Sigma times epsilon, with elementwise multiplication, epsilon a zero-mean noise variable with fixed statistics, and zeta equals the pair mu and Sigma the learnable parameters. A fixed draw of epsilon induces a different function, and because the perturbation flows through the network, its effect on behaviour depends on the input state. That immediately recovers the two properties action dithering lacks: the variation is state-dependent, and while a sample is held fixed the agent behaves according to one coherent sampled function rather than re-rolling a random action rule at every state. This is deliberately not a posterior or Bayesian belief; it is a parameterized source of noise whose scale is trained by gradient descent. The loss becomes the noise-averaged loss L bar of zeta equals the expectation over epsilon of L of mu plus Sigma times epsilon. The key point that makes it trainable is that the distribution of epsilon does not depend on zeta, so the gradient is a plain reparameterized expectation and can be estimated with a single sample xi. The derivative with respect to mu is just the ordinary weight derivative; the derivative with respect to Sigma is that same local derivative multiplied by the sampled noise, so backpropagation can discover, per parameter, whether the agent wants more or less injected variation there. That is the whole reason for choosing learnable parameter noise over a fixed-variance heuristic: the RL loss itself sets the exploration scale.

Concretely the perturbation lives in the fully connected heads. For a linear layer with p inputs and q outputs, the ordinary map y equals w times x plus b becomes y equals the sum of mu weight and sigma weight times epsilon weight times x, plus mu bias plus sigma bias times epsilon bias, where the mu and sigma weight tensors have shape q by p and the mu and sigma bias tensors have shape q. There are two ways to supply the noise. The independent case makes every weight-noise entry and bias-noise entry an independent standard Gaussian; that costs p times q plus q Gaussian draws per layer, which is acceptable in the distributed A3C setting and is the choice for the main A3C variant. For a single-threaded value agent, drawing a full q by p matrix of Gaussians for every noisy layer is too expensive relative to the matrix multiply itself, so I factor the noise. I draw only p input noises and q output noises, pass each through the transform f of x equals sign of x times the square root of the absolute value of x, and set each weight-noise entry to the product of the transformed output noise for that row and the transformed input noise for that column, while each bias-noise entry is just the transformed output noise. The weight-noise tensor is therefore an outer product, and the draw count drops from p times q plus q to p plus q. The transform keeps each factor zero-mean and order-one: for a standard normal Z, the expectation of f of Z is zero and the expectation of f of Z squared equals the expectation of the absolute value of Z, which is the square root of two over pi, so a factorized weight entry has variance two over pi. That is not exactly one, but it is reliably order one while the random-number cost collapses.

Initialization is handled per case. For independent noise the means are drawn from a uniform distribution between minus the square root of three over p and plus the square root of three over p, whose variance is one over p, and every scale is set to the inherited constant zero point zero one seven. For factorized noise the means are drawn from a uniform distribution between minus one over the square root of p and plus one over the square root of p, and the scales are set to sigma zero over the square root of p with sigma zero equal to zero point five. The community Rainbow reference code that the implementation below follows instead initializes the weight sigma with std init over the square root of the input dimension and the bias sigma with std init over the square root of the output dimension, which is a code convention rather than a separate derivation.

Wiring the sampled functions into the RL losses is where the on-policy and off-policy distinctions matter, and the governing rule is to hold one sampled epsilon fixed between optimization steps, not literally forever. For DQN I delete epsilon-greedy entirely and act greedily under the current sampled value network. Because a DQN agent does one replay update per action step, holding the sample fixed between updates means resampling before each action, and a whole replay batch shares one current sample. The replay loss uses one noise draw for the online network and an independent draw for the target network: the squared temporal-difference error between r plus gamma times the maximum over next actions of the target Q-value and the current Q-value. The independence is load-bearing: sharing the same random variables between online and target would correlate the bootstrapped target with the current estimate, so the algorithm explicitly keeps online noise, target noise, and the separate acting noise independent of one another. For dueling DQN the value and advantage streams and the mean-subtracted combination are untouched; only the target switches to the double-DQN form, where the online sampled network selects the best next action and the target sampled network evaluates that action, again with independent noise for selection, evaluation, and acting. For A3C I remove the entropy bonus, since the stochasticity now comes from sampling the parameters, and make the policy and value heads noisy. The on-policy subtlety is the critical one: if I resampled inside a rollout, the actions and the bootstrap value would no longer come from a single fixed policy, so for the k-step return I force every epsilon across the rollout to be the same, and take the policy-gradient and value-gradient updates with that one sampled network held fixed for the whole rollout. The result is not a Bayesian posterior sampler and not a fixed-variance parameter-noise trick; it is ordinary deep-RL optimization with a reparameterized, learnable source of parameter noise sitting in the network heads, with the separate epsilon schedule and the entropy bonus removed.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer used in Noisy Networks for Exploration."""

    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        return F.linear(input, self.weight_mu, self.bias_mu)


class NoisyDQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = NoisyLinear(3136, 512)
        self.head = NoisyLinear(512, n_actions)

    def forward(self, x):
        h = self.torso(x.float() / 255.0)
        h = F.relu(self.fc(h))
        return self.head(h)

    def reset_noise(self):
        self.fc.reset_noise()
        self.head.reset_noise()


def act(net, x):
    net.reset_noise()
    with torch.no_grad():
        return net(x).argmax(dim=1)


def dqn_loss(online, target, obs, actions, rewards, next_obs, dones, gamma):
    online.reset_noise()
    target.reset_noise()
    q = online(obs).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        target_q = target(next_obs).max(dim=1).values
        y = rewards + gamma * target_q * (1.0 - dones)
    return F.mse_loss(q, y)


def dueling_target(online, target, rewards, next_obs, dones, gamma):
    online.reset_noise()
    target.reset_noise()
    with torch.no_grad():
        next_actions = online(next_obs).argmax(dim=1)
        target_q = target(next_obs).gather(1, next_actions[:, None]).squeeze(1)
        return rewards + gamma * target_q * (1.0 - dones)


def verify_noisy_layer():
    torch.manual_seed(0)
    layer = NoisyLinear(10, 3)
    x = torch.randn(5, 10)
    layer.train()
    out1 = layer(x)
    layer.reset_noise()
    out2 = layer(x)
    assert out1.shape == (5, 3)
    assert not torch.allclose(out1, out2)
    layer.eval()
    out3 = layer(x)
    out4 = layer(x)
    assert torch.allclose(out3, out4)
    print("NoisyLinear behaves correctly in train and eval modes.")


if __name__ == "__main__":
    verify_noisy_layer()
    net = NoisyDQN(n_actions=4)
    dummy_obs = torch.zeros(2, 4, 84, 84)
    actions = act(net, dummy_obs)
    assert actions.shape == (2,)
    print("NoisyDQN forward pass and greedy action selection work.")
```
