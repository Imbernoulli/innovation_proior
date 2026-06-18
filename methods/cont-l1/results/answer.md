# OpenVLA-OFT (Cont-L1), distilled

OpenVLA-OFT is a recipe for fine-tuning a large pretrained vision-language-action model (VLA)
to a new robot. Its continuous-L1 instantiation (Cont-L1) combines three design choices on top
of the base OpenVLA trunk: **parallel decoding with action chunking**, a **continuous action
representation**, and an **L1 regression objective**. Together they replace the base model's
slow, autoregressive, discretized action generation with a single-pass continuous regressor —
raising action throughput by ~26x while improving task success on the LIBERO benchmark.

## Problem it solves

The base VLA generates a 7-DoF action as 256-bin discrete tokens, decoded one token at a time
under a causal mask. This costs `D` sequential decoder passes per timestep (`K·D` for a chunk
of `K` timesteps), capping throughput at 3-5 Hz — far below the 25-50+ Hz a high-frequency
controller needs — and making action chunking impractical. The 256-bin discretization also
caps action precision. The goal: fast, precise, chunked action generation suitable for
real-time control, obtained by fine-tuning.

## Key idea

1. **Parallel decoding + action chunking.** Action coordinates have no intrinsic left-to-right
   ordering, so sequential decoding is unnecessary. Fill the `K·D` action-token positions with
   *empty* action embeddings distinguished only by positional encoding, and replace the causal
   attention pattern with **bidirectional** attention over those slots. Now all `K·D` action
   positions are predicted in a **single forward pass**, collapsing the sequential-pass factor
   from `K·D` to 1. Chunking still lengthens the sequence, but it no longer multiplies decoder
   calls; predicting `K` future timesteps and executing them open-loop becomes fast enough to
   use the known action-chunking benefits: shorter effective horizon, less compounding error,
   and temporal dependencies.

2. **Continuous actions.** Replace the bin-logit output layer with a small MLP head that maps
   the action-position hidden states directly to real-valued normalized actions, removing the
   lossy discretization bottleneck (more bins → rarer tokens → worse generalization; fewer
   bins → coarser precision).

3. **L1 regression.** Train the head by the mean absolute error between predicted and
   ground-truth action chunks (actions normalized to `[-1,+1]`):
   `L = mean |predicted − ground_truth|`. L1 (median-seeking) is more robust to noisy
   demonstrations and more precise than L2 (mean-seeking). Versus diffusion: a denoising
   process represents multimodal action distributions but needs many sequential denoising
   steps at inference (re-introducing latency) and trains slower; the high-capacity 7B trunk
   makes simple L1 regression competitive in success rate. **Limitation:** L1 learns the
   median mode, so it cannot represent genuinely multimodal action distributions the way
   diffusion can — it suits focused, consistent-strategy demonstrations.

## Architecture

- **Trunk (base OpenVLA):** fused SigLIP + DINOv2 vision backbone (256 patch embeddings per
  view, projected by a 3-layer GELU MLP), Llama-2 7B decoder. Adapted via **LoRA** (rank 32)
  on the trunk's linear layers; full fine-tuning is wasteful on a few hundred demonstrations.
- **Action head:** a 4-layer-deep **MLP-ResNet** (input projection → 2 pre-norm residual
  blocks → output projection). It consumes the `D` action-position hidden states of each
  timestep jointly: the `(B, K·D, hidden)` action hidden states are regrouped to
  `(B, K, D·hidden)` and mapped to `(B, K, action_dim)`. The head is tiny relative to the 7B
  trunk, so it adds negligible inference cost. (For LIBERO: `K = NUM_ACTIONS_CHUNK = 8`,
  `D = ACTION_DIM = 7`, `hidden = 4096`.)
- **Inputs (optional):** extra camera views and robot proprioceptive state can be added —
  proprio is projected by a small 2-layer GELU MLP — and folded into the same single forward
  pass thanks to the headroom parallel decoding creates.

## Training procedure

1. Forward the trunk in the parallel-decoding branch: action-token IDs mark the `K·D` positions,
   but their input embeddings are zeroed when no noisy diffusion actions are supplied; the
   design uses bidirectional attention over those action positions.
2. Take the last-layer hidden states, drop the vision-patch prefix, and select the `K·D`
   action-token positions — at train time via a `current | next` action mask from the labels;
   at eval time by slicing the contiguous action positions after the prompt.
3. Regroup to `(B, K·D, hidden)`, run the MLP-ResNet head to get `(B, K, action_dim)`.
4. Minimize `L1Loss(ground_truth_actions, predicted_actions)`.

Actions are normalized to `[-1,+1]`, which makes the L1 scale meaningful (a typical
convergence criterion is mean normalized L1 < 0.01). At inference, predict a full chunk and
execute all `K` actions open-loop before requerying.

## Working code

The MLP-ResNet head and the L1 training step:

```python
import torch
import torch.nn as nn

# Robot constants (LIBERO single-arm)
ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 8


class MLPResNetBlock(nn.Module):
    """Pre-norm residual feedforward block: LayerNorm -> Linear -> ReLU, + input."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        identity = x
        x = self.ffn(x)
        return x + identity


class MLPResNet(nn.Module):
    """MLP with residual blocks: input projection -> blocks -> output projection."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [MLPResNetBlock(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class L1RegressionActionHead(nn.Module):
    """Continuous-action head trained via L1 regression over an action chunk."""
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        super().__init__()
        self.action_dim = action_dim
        # Input width = hidden * ACTION_DIM: the D action-position hidden states
        # of a timestep are consumed jointly to predict that timestep's D-dim action.
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states):
        # actions_hidden_states: (B, K * ACTION_DIM, hidden_dim)
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.model(rearranged)  # (B, K, action_dim)


def zero_action_token_embeddings(input_embeddings, all_actions_mask):
    """Inner OpenVLA-OFT forward step for L1 regression.
    The action token IDs keep the positions; their embeddings are zeroed so
    they carry no teacher-forced action content."""
    return input_embeddings * ~all_actions_mask.unsqueeze(-1)


def l1_training_step(vla, action_head, batch, num_patches,
                     get_current_action_mask, get_next_actions_mask):
    # Ground-truth action chunk, normalized to [-1, +1]: (B, K, action_dim)
    ground_truth_actions = batch["actions"].to(torch.bfloat16)

    # Parallel decode: vla.forward zeroes action-token input embeddings in the
    # L1 path; the design uses bidirectional action-token attention,
    # so all K*D action positions are predicted in a single forward pass.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"].to(torch.bfloat16),
            labels=batch["labels"],
            output_hidden_states=True,
        )

    # Select the K*D action-token hidden states from the last layer.
    last_hidden_states = output.hidden_states[-1]                 # (B, seq_len, hidden)
    text_hidden_states = last_hidden_states[:, num_patches:-1]    # drop vision prefix
    ground_truth_token_ids = batch["labels"][:, 1:]
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )                                                            # (B, K*D, hidden)

    # Predict the continuous action chunk and take the mean L1 loss.
    predicted_actions = action_head.predict_action(actions_hidden_states)  # (B, K, D)
    loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)
    return loss, predicted_actions
```

## Relation to prior methods

- **Base OpenVLA (discrete, autoregressive):** Cont-L1 keeps the trunk but replaces causal +
  256-bin + next-token-prediction with bidirectional + continuous + L1, and adds chunking.
- **Action Chunking with Transformers (ACT):** source of single-pass chunk prediction via
  fixed per-timestep queries and of the L1 reconstruction choice (L1 over L2 for precision);
  here those ideas ride on a large pretrained VLA instead of a from-scratch transformer.
- **Diffusion Policy:** the expressive alternative objective; matched in success by L1 here
  (given the trunk's capacity) at a fraction of the inference cost, at the price of not
  modeling truly multimodal action distributions.
