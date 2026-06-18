**Problem.** Train a humanoid to stand/walk/run under a fixed budget (~12.8M frames). The baseline
must be reliable and give an honest floor; the binding constraint on this task is sample
efficiency, and the floor is the method that uses the budget least efficiently.

**Key idea (the floor).** PPO: a clipped-surrogate on-policy policy gradient,
$L^{CLIP}=\hat{\mathbb{E}}_t[\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]$
with $r_t=\pi_\theta/\pi_{old}$, $\epsilon=0.2$, GAE advantages ($\lambda=0.95$), a value head, and
an entropy bonus. The clip is a trust region as a flat spot in the loss, so one batch is safely
reused for $K$ epochs — but being on-policy, every rollout is discarded after those epochs. PPO
cannot touch a replay buffer, so it converts the same frames into far less gradient signal than any
off-policy method.

**Step-1 "edit".** PPO does *not* fill the editable off-policy contract (`Actor`/`Critic`/
`update_*` assume a replay buffer, distributional critic, target nets, per-env noise). It runs as
its own self-contained stable-baselines3 on-policy script over 16 envs for 800k steps (matching the
12.8M-frame budget), bypassing the off-policy infrastructure entirely. It is the floor by
construction — it opts out of experience reuse, the mechanism the rest of the ladder exploits.

**What to watch.** Expect a descending pattern stand > walk > run: dense, short-horizon balance
suits undirected Gaussian exploration; long-horizon gait coordination needs the sample efficiency
PPO lacks. All three should land below an off-policy actor-critic on the same budget; the size of
that shortfall — widest on the harder-to-explore tasks — motivates moving onto the off-policy
surface. PPO's value head also calibrates the reachable return scale (a sanity anchor for the
critic's `[-250, 250]` support).

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
