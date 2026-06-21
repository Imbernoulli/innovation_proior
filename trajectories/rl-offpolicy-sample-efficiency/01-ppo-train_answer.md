The task is to train a humanoid to stand, walk, and run and reach the highest mean episode return within a fixed, small budget — 100,000 gradient steps with 128 parallel environments, evaluated with deterministic actions — and the binding constraint is sample efficiency. The whole substrate around the algorithm is an *off-policy* one: a GPU-resident replay buffer, a categorical distributional critic over the support $[-250, 250]$, target networks, per-env exploration noise. Before I design anything that exploits that machinery, I want an honest floor: the method a robotics practitioner reaches for first, that is reliable and will not blow up, but that spends the budget in the one currency this task cannot afford. That method is Proximal Policy Optimization, and it is the floor precisely because it is the one fill on this ladder that *cannot* touch the replay buffer.

I propose to use PPO as the baseline, run as its own self-contained on-policy script rather than as a fill of the editable off-policy contract — and the value of doing so is that PPO measures, by being unable to reuse a single frame, exactly what experience reuse is worth on this task. Let me write down what PPO optimizes so the budget argument is exact. PPO maximizes a clipped surrogate

$$L^{CLIP}=\hat{\mathbb{E}}_t\!\left[\min\!\big(r_t\,\hat A_t,\ \operatorname{clip}(r_t,\,1-\epsilon,\,1+\epsilon)\,\hat A_t\big)\right],$$

where $r_t=\pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$ is the probability ratio between the updated policy and the one that generated the rollout, $\hat A_t$ is a generalized-advantage estimate, and $\epsilon=0.2$. The clip is a trust region expressed as a *flat spot in the loss*: once a sample's ratio moves past $1\pm\epsilon$ in the direction its advantage favors, the gradient through that sample dies, which is what lets me run several epochs of minibatch SGD over a single batch of rollouts without the policy collapsing. The advantages come from GAE, $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$ with $\delta_t=R_t+\gamma V(s_{t+1})-V(s_t)$ and $\lambda=0.95$, so the value head trades bias against variance through $\lambda$; an entropy bonus on the Gaussian policy keeps it from committing too early. This is a clean, reliable algorithm.

The catch is the word *on-policy*. The advantages $\hat A_t$ are computed under the policy that generated the rollouts, and the surrogate is only an honest proxy for improvement while $\pi_\theta$ stays close to $\pi_{old}$. After a handful of epochs I have extracted what I safely can from this batch, and I must throw it away and collect fresh rollouts with the updated policy. Every transition feeds at most a few gradient steps and is then gone. Now put that against the budget: about 12.8M environment frames. To stay stable PPO wants large, decorrelated batches and a modest epoch count; to fit the humanoid's high-dimensional, contact-rich dynamics it wants *many* such batches; under a tight frame budget those two demands fight, so I can have stable updates or many updates, not both. An off-policy method keeps each transition in a replay buffer and revisits it many times, converting the same frames into far more gradient signal — which is the entire reason the editable surface here is an off-policy actor-critic and not a policy-gradient loop. PPO is the baseline that demonstrates, by being the one method that cannot touch that buffer, what the buffer is worth.

That structural fact dictates how PPO is run, and it is worth being explicit because it says what PPO is *not*. PPO does not fill the editable `Actor`/`Critic`/`update_*` contract — those functions assume a replay buffer, a distributional critic, target networks, and a per-env noise scheme, none of which an on-policy method uses. So PPO runs as a separate stable-baselines3 on-policy script: an `MlpPolicy` with separate $[256, 256]$ tanh policy and value heads, `n_steps=2048` per env across 16 parallel environments, 10 epochs per batch, `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.01`, learning rate $3\times10^{-4}$, trained for 800,000 PPO steps to match the same 12.8M-frame budget. That is the fair way to give PPO its shot, and it is also exactly why PPO is the floor: it opts out of the one mechanism — experience reuse — that the rest of the ladder is built to exploit.

What I expect across the three tasks follows the axis PPO is most sensitive to: how much directed, sustained exploration the reward demands before the behavior is found. Standing balance is gentlest — dense reward, almost any small correction changes the height, and "do not fall" is a short-horizon behavior a stochastic policy stumbles into early — so PPO should make the most progress on **stand**. Walking is harder: forward velocity requires first not falling *and* coordinating a fragile, longer-horizon gait that must hold together for many steps before the reward confirms it, and with no reuse PPO gets fewer attempts per frame to lock it in. Running is hardest, walking pushed to a regime where small coordination errors are catastrophic. So I expect a descending pattern, stand best and run worst, all three pressed down by the budget, with meaningful seed-to-seed variance because whether a fragile gait is found in time is partly luck under undirected exploration.

The crucial point for what comes next is the *nature* of PPO's weakness. It is not instability — the clip makes PPO reliable, the curves will not diverge — and it is not a sparse-reward hard-exploration failure, since the HumanoidBench rewards are dense and the policy gradient always has a nonzero signal to ascend. PPO fails at *rate*: the throughput of usable policy improvement per frame. That is a quiet, insidious failure, because the return curve rises and the agent looks like it is learning, it simply tops out lower than it should within the budget. So the right way to read PPO is not "did it work" but "where did its curve flatten relative to the off-policy methods, and on which tasks did the gap open widest." If the gap is largest on walk and run and smallest on stand, that confirms the binding constraint is sample efficiency on the harder-to-explore behaviors, and it points the next step squarely at the replay buffer. No amount of clip-tuning or GAE-tuning changes the fact that the rollouts are discarded; the fix is to stop throwing transitions away, which means moving onto the editable off-policy surface. PPO's own value head also gives an architecture-free read of roughly what return each task admits within the budget — a sanity anchor for the off-policy numbers and for the critic's fixed $[-250, 250]$ support.

```python
# PPO baseline: self-contained on-policy script (scripts/ppo.sh) — bypasses the editable surface.
import gymnasium as gym
import humanoid_bench
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

env_name = "h1hand-stand-v0"   # one of stand / walk / run
seed = 1
num_envs = 16
total_timesteps = 800_000      # 12.8M frames / 16 envs, matching the off-policy budget
max_episode_steps = 1000


def make_env(rank):
    def _init():
        import humanoid_bench  # re-register envs in subprocess
        env = gym.make(env_name)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.unwrapped.seed(seed + rank)
        return env
    return _init


envs = SubprocVecEnv([make_env(i) for i in range(num_envs)])

model = PPO(
    "MlpPolicy", envs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=seed,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)
model.learn(total_timesteps=total_timesteps)

# Final evaluation: mean episode return over 3 deterministic rollouts (8 eval envs).
```
