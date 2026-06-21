Offline reinforcement learning gives us a fixed batch of trajectories and forbids any new environment interaction. The standard recipe is to fit a value function with temporal-difference backups and then act greedily, but that path is fragile here. With frozen data, the max over actions queries states outside the dataset support, optimistic extrapolation errors are selected and then bootstrapped into targets, and the errors compound because no fresh rollouts can correct them. Behavior cloning avoids all of this by pure supervised regression, yet it simply reproduces the average behavior in the log and cannot lift out the better fragments scattered through the data.

The way out is to keep the stability of supervised learning while still conditioning on outcome. Every logged action is relabeled with the return its trajectory actually achieved from that point onward, and the model learns to map a desired future return to the action that historically produced it. Because the same state can now be paired with different actions under different return targets, the model is no longer forced into a single average policy. The central design choice is to treat the trajectory as a token stream and train a causal Transformer with a plain action-prediction loss, which removes Bellman backups, target networks, and learned critics entirely.

The method is the Decision Transformer. It interleaves each trajectory into a sequence of tokens

(Ř_1, s_1, a_1, Ř_2, s_2, a_2, ..., Ř_T, s_T, a_T),

where Ř_t is the undiscounted return-to-go from timestep t. Separate linear embeddings project returns, states, and actions into the same hidden dimension, and a shared per-timestep embedding is added to all three tokens at a given step so the model knows they belong to the same moment. The tokens are stacked in return-state-action order and fed through a causal GPT-style Transformer. The action head is attached to the state-token positions, so the prediction for a_t has causal access to the full prefix plus the current return-to-go and state. Training minimizes MSE or cross-entropy on the logged actions over the valid, non-padded timesteps.

At test time we prompt the model with a high target return-to-go and the initial state. The model emits an action, we execute it in the environment, observe the reward, and update the target by subtracting that reward: Ř_{t+1} = Ř_t - r_t. This recursion keeps the conditioning consistent with what the model saw during training. The context is cropped or padded to the last K timesteps, and the model and environment jointly generate the rest of the episode. No discount factor is needed for the conditioning labels because the objective is pure regression rather than a Bellman fixed-point iteration.

This design inherits several advantages from casting RL as sequence modeling. Attention can directly relate distant events, so sparse or delayed rewards are handled without slow credit propagation. The model can stitch together useful trajectory fragments that share the same state and return-to-go, even if they originated in different episodes. And the whole stable language-model training stack—AdamW, weight decay, warmup, gradient clipping, causal masking—comes for free.

```python
import numpy as np
import torch
import torch.nn as nn
import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class DecisionTransformer(TrajectoryModel):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs,
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.hidden_size = hidden_size

        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs,
        )
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=states.device
            )

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])
        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            pad = self.max_length - states.shape[1]
            attention_mask = torch.cat(
                [torch.zeros(pad, device=states.device),
                 torch.ones(states.shape[1], device=states.device)]
            ).to(dtype=torch.long).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((1, pad, self.state_dim), device=states.device), states], dim=1
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((1, pad, self.act_dim), device=actions.device), actions], dim=1
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((1, pad, 1), device=returns_to_go.device), returns_to_go], dim=1
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((1, pad), device=timesteps.device), timesteps], dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states, actions, None, returns_to_go, timesteps,
            attention_mask=attention_mask, **kwargs
        )
        return action_preds[0, -1]


def discount_cumsum(x, gamma):
    y = np.zeros_like(x)
    y[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        y[t] = x[t] + gamma * y[t + 1]
    return y


def train_step(model, get_batch, optimizer, batch_size):
    states, actions, rewards, dones, rtg, timesteps, attention_mask = get_batch(batch_size)
    action_target = actions.clone()

    _, action_preds, _ = model.forward(
        states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask
    )

    act_dim = action_preds.shape[2]
    action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    loss = torch.mean((action_preds - action_target) ** 2)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()
    return loss.detach().cpu().item()


def evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
):
    model.eval()
    model.to(device=device)
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0.0, 0
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != "delayed":
            next_target = target_return[0, -1] - reward / scale
        else:
            next_target = target_return[0, -1]
        target_return = torch.cat([target_return, next_target.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1
        )

        episode_return += reward
        episode_length += 1
        if done:
            break

    return episode_return, episode_length
```
