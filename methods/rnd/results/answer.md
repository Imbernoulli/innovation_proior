# RND (Random Network Distillation)

**Problem.** Sparse-reward exploration (e.g. Montezuma's Revenge) needs a directed exploration
bonus that is cheap and scalable. Prediction-error bonuses are cheap but a forward-dynamics bonus
falls into the *noisy-TV trap*: it is permanently attracted to transitions whose answer is
stochastic (irreducible aleatoric error), so the agent farms entropy instead of exploring.

**Key idea.** Use a prediction problem whose answer is a *deterministic* function of the
observation and lies *inside the predictor's model class*, so the error is dominated by epistemic
uncertainty (lack of similar training data = novelty). Two networks:
- *Target* $f:\mathcal{O}\to\mathbb{R}^k$ — fixed, randomly initialized, **frozen**.
- *Predictor* $\hat f:\mathcal{O}\to\mathbb{R}^k$ — trained by SGD to distill the target,
  $\min_{\theta}\mathbb{E}_x\|\hat f(x;\theta)-f(x)\|^2$.

Intrinsic reward $i_t=\|\hat f(s_{t+1})-f(s_{t+1})\|^2$ — small on seen-like states, high on novel
ones. A deterministic target removes aleatoric error; a target inside the model class removes
misspecification. (SGD does not overgeneralize to mimic $f$ everywhere — verified on an MNIST toy.
The distillation error is also an uncertainty estimate à la randomized prior functions, which
motivates making the predictor *deeper* than the target.)

**Combining intrinsic + extrinsic returns (two value heads).** Treat the intrinsic stream as
*non-episodic* (so the agent isn't made risk-averse by game-overs zeroing future novelty) and the
extrinsic stream as *episodic* (so it can't suicide-farm an early reward). Since the return is
linear in rewards, $R=R_E+R_I$: fit two value heads $V_E,V_I$ on their own returns, $V=V_E+V_I$,
allowing different discounts. Advantages summed: $A=c_E A_E+c_I A_I$ ($c_E=2,c_I=1$). Discounts
$\gamma_E=0.999>\gamma_I=0.99$.

**Normalization.**
- *Intrinsic reward*: divide $i_t$ by a running estimate of the std of intrinsic *returns* (scale
  consistency).
- *Observation* (for predictor/target, **not** policy): whiten each dim by running mean/std, clip
  to $[-5,5]$; initialize the stats by stepping a random agent for $M$ steps. (Critical because
  the frozen random target cannot adapt to the observation scale.)

**Algorithm (PPO + RND).** Initialize obs-norm with a random rollout. Per rollout (length 128, 128
parallel envs): act under $\pi$, observe $s_{t+1},e_t$; compute $i_t$; update reward-norm.
Normalize intrinsic rewards; compute non-episodic $R_I,A_I$ and episodic $R_E,A_E$; $A=A_I+A_E$;
update obs-norm. For 4 epochs / 4 minibatches: optimize the policy by PPO clip loss and the
predictor by the distillation loss (predictor trained on 25% of the batch). PPO: clip $[0.9,1.1]$,
$\lambda_{\text{GAE}}=0.95$, entropy coef $0.001$, Adam lr $10^{-4}$. Sticky actions $p=0.25$
(non-determinism); extrinsic reward clipped $[-1,1]$, intrinsic unclipped; $k=512$.

```python
import torch
import torch.nn as nn

K = 512

class RND(nn.Module):
    def __init__(self):
        super().__init__()
        def conv():
            return nn.Sequential(
                nn.Conv2d(1, 32, 8, stride=4), nn.LeakyReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.LeakyReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(), nn.Flatten())
        self.target = nn.Sequential(conv(), nn.Linear(3136, K))                      # frozen, single dense
        self.predictor = nn.Sequential(conv(), nn.Linear(3136, K), nn.ReLU(),
                                       nn.Linear(K, K), nn.ReLU(), nn.Linear(K, K))   # deeper, trainable
        for p in self.target.parameters():
            p.requires_grad = False

    def intrinsic_reward(self, next_obs):
        with torch.no_grad():
            return (self.predictor(next_obs) - self.target(next_obs)).pow(2).sum(dim=1)

    def distillation_loss(self, obs):
        return (self.predictor(obs) - self.target(obs).detach()).pow(2).sum(dim=1).mean()

class RunningNorm:
    def __init__(self, shape):
        self.mean = torch.zeros(shape); self.var = torch.ones(shape); self.count = 1e-4
    def update(self, x):
        bm, bv, bn = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean; tot = self.count + bn
        self.mean += d * bn / tot
        self.var = (self.var * self.count + bv * bn + d ** 2 * self.count * bn / tot) / tot
        self.count = tot

def normalize_obs(x, stats):                          # predictor/target input only
    return ((x - stats.mean) / (stats.var.sqrt() + 1e-8)).clamp(-5.0, 5.0)

# Training (PPO + RND), two reward streams / two value heads:
#   init obs-norm by stepping a random agent M steps.
#   per rollout: a_t~pi; observe s_{t+1}, e_t (clip [-1,1]);
#                i_t = rnd.intrinsic_reward(normalize_obs(s_{t+1}, obs_stats)); update return-std.
#   i_t /= return_std
#   R_I, A_I  non-episodic, gamma_I=0.99   (head V_I)
#   R_E, A_E  episodic,     gamma_E=0.999  (head V_E)
#   A = 2*A_E + 1*A_I;  V = V_E + V_I
#   for 4 epochs x 4 minibatches:
#       PPO clip update on theta_pi (Adam 1e-4); distillation_loss update on theta_predictor (25% of batch)
```
