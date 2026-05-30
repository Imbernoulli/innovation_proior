# Decision Transformer

## The Problem

Offline reinforcement learning asks for a good controller from a fixed dataset of trajectories, with no new environment interaction. TD and Q-learning become fragile in this setting because function approximation, bootstrapping, and off-policy data combine with extrapolation error on unsupported actions. Behavior cloning avoids that instability, but it imitates the average behavior in the dataset.

## The Key Idea

Treat reinforcement learning as conditional sequence modeling. Represent each trajectory as an interleaved token stream

$$\tau=\big(\widehat{R}_1,s_1,a_1,\widehat{R}_2,s_2,a_2,\ldots,\widehat{R}_T,s_T,a_T\big),\qquad \widehat{R}_t=\sum_{t'=t}^{T}r_{t'},$$

where $\widehat{R}_t$ is the return still to be collected from timestep $t$. Train a causal Transformer with a supervised action-prediction loss. At deployment, prompt the model with a desired return-to-go, execute the predicted action, subtract the received reward,

$$\widehat{R}_{t+1}=\widehat{R}_t-r_t,$$

and continue autoregressively.

The return-to-go token is the control knob: each logged action is relabeled with the future return it actually achieved, so the model learns behavior conditioned on outcome rather than averaging all behavior into one policy. The Transformer supplies long-context trajectory modeling without Bellman backups, target networks, or learned value maximization.

## The Algorithm

**Token layout.** Use three tokens per timestep: return-to-go, state, action. The hidden state at the state token has causal context $(\widehat{R}_1,s_1,a_1,\ldots,\widehat{R}_t,s_t)$, so the action head models

$$p_\theta(a_t\mid \widehat{R}_1,s_1,a_1,\ldots,\widehat{R}_t,s_t).$$

**Embeddings.** Use separate embeddings for returns, states, and actions; visual states use a convolutional encoder. Add the same learned timestep embedding to all three tokens from one environment timestep, then layer-normalize the stacked token sequence.

**Training.** Sample length-$K$ sub-trajectories. Compute undiscounted returns-to-go with $\gamma=1$, keep one extra return label in the batch, and feed the first $K$ return tokens alongside the $K$ states and actions. Predict actions from state-token hidden states and minimize MSE for continuous actions or cross-entropy for discrete actions, masked over non-padding positions.

**Generation.** Start from a target return-to-go and the initial state. Repeatedly append an action placeholder, query the model for the last action, execute it, append the reward and next state, and update the target return-to-go by subtracting the observed reward. Crop or left-pad to the context length $K$.

## Code

Continuous-action implementation using the custom GPT-2 backbone whose token positional embeddings are removed so the timestep embeddings below are the only positional signal.

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
            vocab_size=1,  # unused because we pass embeddings directly
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

        return_preds = self.predict_return(x[:, 2])  # next return after seeing state and action
        state_preds = self.predict_state(x[:, 2])    # next state after seeing state and action
        action_preds = self.predict_action(x[:, 1])  # action after seeing return and state
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
                [torch.zeros(pad, device=states.device), torch.ones(states.shape[1], device=states.device)]
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

The data loader computes returns-to-go as `discount_cumsum(rewards, gamma=1.0)` and divides them by a fixed `scale` before embedding. The calling evaluation code passes `target_return / scale`, so the online update subtracts `reward / scale`.

For discrete actions and image states, replace the state linear layer with a convolutional encoder, replace the action linear embedding with an action lookup table, and train logits with cross-entropy. The token order, timestep embeddings, state-token action head, and return-to-go recursion stay the same.

## Why It Works

- **No bootstrapping:** the action objective is supervised, so there is no Bellman target or learned critic for the policy to exploit.
- **Return conditioning:** the same state can map to different actions under different desired outcomes instead of being averaged into one behavior.
- **Long context:** a trajectory prefix helps infer which behavior mode generated the current data and lets attention connect distant events directly.
- **Undiscounted returns:** $\gamma=1$ is used for the conditioning labels because no contraction argument is needed for supervised sequence modeling.
- **Stitching:** local state and return-to-go conditioning can compose useful trajectory fragments from mixed data without explicit dynamic programming.
