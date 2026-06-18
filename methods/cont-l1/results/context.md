# Context: adapting a pretrained vision-language-action model to a new robot (circa 2024-2025)

## Research question

A vision-language-action model (VLA) is a robot policy built by fine-tuning a large
pretrained vision-language model on robot demonstration data so that it maps an image plus a
language instruction to low-level robot actions. The base model used here, OpenVLA, is a
7B-parameter policy that already follows language and generalizes semantically across many
tasks. The problem is deployment on a *new* robot setup: such models almost always need
fine-tuning on a few hundred task demonstrations before they work, and it is unclear what the
most effective fine-tuning approach is. Two failures dominate. First, **speed**: the base
model generates actions one discrete token at a time, autoregressively, which runs at only
3-5 Hz — far below the 25-50+ Hz a high-frequency controller (especially a bimanual one)
needs for real-time, smooth control. Second, **quality**: autoregressively decoded,
coarsely-quantized actions are imprecise, and the slow generation makes it impractical to
predict a chunk of several future actions at once, even though predicting action chunks is
known to help. The goal is a fine-tuning recipe that simultaneously delivers (1) high action
throughput suitable for high-frequency control, (2) reliable task success, and (3) enough
flexibility to accept extra inputs (wrist camera, robot proprioceptive state) and emit
multi-timestep outputs without wrecking the latency budget. Each known ingredient achieves
some of this; combining them coherently into one adaptation recipe for a large pretrained VLA
is the open problem.

## Background

By this time, fine-tuning a pretrained vision-language model into a robot policy is an
established and rapidly improving paradigm (Brohan et al. 2023 RT-2; O'Neill et al. 2023 Open
X-Embodiment; Kim et al. 2024 OpenVLA; Black et al. 2024 pi0). The field's load-bearing
concepts:

- **The autoregressive-discrete formulation.** The dominant recipe, inherited from language
  modeling, represents a robot action as a short string of discrete tokens and predicts them
  left-to-right. For a 7-DoF arm the action is a 7-dimensional delta end-effector pose (3
  position, 3 orientation, 1 gripper); each dimension is normalized to `[-1, +1]` and
  uniformly binned into 256 bins, and each bin is mapped to a token in the language model's
  vocabulary. The policy is trained exactly like a language model — next-token prediction with
  cross-entropy loss over the action tokens. This is convenient (no architectural change to
  the VLM) but has two structural costs that are knowable up front.

- **The latency arithmetic of autoregressive decoding.** Generating one `D`-dimensional
  action requires `D` sequential decoder forward passes, because each token is conditioned on
  the previously generated ones. On an A100, generating even a single-timestep OpenVLA action
  takes about 0.33 s. This is the root cause of the 3-5 Hz ceiling.

- **Action chunking.** Predicting and executing a sequence of `K` future actions per query —
  rather than one action at a time — is documented to improve manipulation success rates
  (Zhao et al. 2023; Chi et al. 2023; Liu et al. 2024). The intuition has two parts: it
  reduces the *effective horizon* of the task `K`-fold, which curbs the compounding-error /
  covariate-shift problem of behavioral cloning (Ross et al. 2011), and it lets the policy
  represent temporally-correlated structure (e.g. pauses) that a single-step Markovian policy
  cannot. But under autoregressive decoding chunking is impractical: a chunk of `K` timesteps
  costs `K·D` sequential passes, a `K`-fold latency increase on top of an already slow loop.

- **The discretization-precision tradeoff.** With discrete tokens, more bins buy finer action
  resolution but make each individual token rarer in the training data, which can hurt
  generalization; fewer bins generalize better but sacrifice fine-grained action detail.
  Either way, binning is a lossy bottleneck between the model and the true continuous action.

- **Continuous-action imitation learning, from outside the VLA world.** Two prominent
  from-scratch imitation-learning methods already model continuous actions directly. Both are
  the conceptual reservoir the VLA recipe draws on (detailed as baselines below): one regresses
  an action chunk through a transformer with a simple regression loss, having found that an
  L1 (absolute-error) reconstruction loss models the action sequence more precisely than the
  more common L2; the other models the action chunk with a conditional denoising diffusion
  process, which captures multimodal action distributions but pays for it with iterative
  sampling.

- **Parameter-efficient fine-tuning.** Because downstream datasets are small (hundreds of
  demonstrations versus the ~1M used to pretrain the base VLA), low-rank adaptation (LoRA; Hu
  et al. 2021) — injecting trainable low-rank matrices into the frozen model's linear layers —
  is the standard way to adapt the 7B model without full fine-tuning.

- **Empirical state of play that motivates the work.** Fine-tuning a large autoregressive VLA
  with its native recipe is measured at 3-5 Hz for single-arm robots and lower for bimanual
  ones — below the real-time threshold — and prior work reports it underperforms on bimanual,
  high-frequency tasks (Wen et al. 2024; Liu et al. 2024; Black et al. 2024). Recent
  better-tokenization efforts (vector-quantized or DCT-compressed action tokens; Belkhale et
  al. 2024; Pertsch et al. 2025) cut token counts and reach 2-13x speedups, but remain
  inherently iterative — e.g. ~750 ms latency between chunks for the fastest — so the
  autoregressive bottleneck persists. Diffusion/flow-matching VLAs reach high-frequency
  bimanual control but introduce slower training and multiple denoising steps at inference.

## Baselines

These are the prior methods a new fine-tuning recipe would be measured against and reacts to.

**OpenVLA, fine-tuned with its native recipe (Kim et al. 2024).** The base model: a Prismatic
VLM (Karamcheti et al. 2024) with a fused SigLIP (Zhai et al. 2023) + DINOv2 (Oquab et al.
2023) vision backbone, a Llama-2 7B decoder (Touvron et al. 2023), and a 3-layer GELU MLP
projecting 256 patch embeddings per view into the language embedding space. Actions are the
256-bin discrete tokens described above, generated **autoregressively** under a **causal**
attention mask, trained by **next-token prediction with cross-entropy**. Adapted to new tasks
via LoRA. **Gap:** the autoregressive causal decode forces `D` sequential passes per timestep
(`K·D` for a chunk), capping throughput at 3-5 Hz and making action chunking impractical; the
256-bin discretization is a lossy bottleneck that limits action precision.

**Action Chunking with Transformers — ACT (Zhao et al. 2023).** A from-scratch imitation
policy for fine bimanual manipulation. It predicts a chunk of `k` future actions per query and
executes them, with *temporal ensembling* — querying every step and combining overlapping
chunk predictions by an exponentially-weighted average, `aₜ = Σᵢ wᵢ Aₜ[i] / Σᵢ wᵢ` with
`wᵢ = exp(−m·i)` — to smooth the output. The `k` future actions are produced in a single
forward pass: a transformer encoder ingests images, joint state, and a latent style variable,
and a transformer decoder reads `k` fixed learned query embeddings, one per output timestep,
so all `k` actions emerge non-sequentially. It is trained as a conditional VAE; the
reconstruction term is the mean absolute error between predicted and ground-truth action
chunks, `L_reconst = L1(â_{t:t+k}, a_{t:t+k})`, plus `β·D_KL(q_φ(z|·) ‖ N(0,I))` regularizing
the latent to a standard normal (`z` set to its zero mean at test time). The ACT study
explicitly reports switching reconstruction from the more common L2 to **L1** because L1 gives
more precise action modeling. **Gap:** trained from scratch on a single task's demonstrations,
with a small bespoke architecture — it carries no large-scale vision-language pretraining and
hence none of a VLA's semantic generalization or language following.

**Diffusion Policy (Chi et al. 2023).** Models the action chunk with a conditional denoising
diffusion process: a network learns to predict the noise added to ground-truth action chunks
under a forward diffusion schedule, and at inference the policy starts from Gaussian noise and
iteratively denoises (DDPM/DDIM) to produce an action chunk, conditioned on the observation.
It represents multimodal action distributions well. **Gap:** training is slower to converge and
inference requires many sequential denoising steps (tens), so even with single-pass
conditioning the per-chunk latency is high relative to a one-shot regressor; the algorithm is
also more complex (noise schedule, sampler, timestep conditioning).

## Evaluation settings

- **LIBERO simulation benchmark (Liu et al. 2024).** A Franka Emika Panda arm in simulation,
  with demonstrations containing camera images, robot state, language task annotations, and
  delta end-effector-pose actions. Four task suites — Spatial, Object, Goal, Long — each with
  500 expert demonstrations across 10 tasks, stressing generalization to spatial layouts,
  objects, goals, and long-horizon tasks respectively. Policies receive a third-person image
  and a language instruction (optionally also a wrist-camera image and robot proprioceptive
  state). Actions are normalized to `[-1, +1]` (1st/99th-percentile bounds). Metric:
  **success rate** per suite. Standard practice: filter unsuccessful demonstrations and
  LoRA-fine-tune per suite; chunk size `K = 8` to match the Diffusion Policy baseline, executed
  open-loop; report the best checkpoint. (In the MLS-Bench setting, the yardstick is narrowed
  to three disjoint LIBERO-Goal subsets, scored by the geometric mean of the three subset
  success rates, under a fixed reduced training-step budget.)
- **Inference-efficiency protocol.** Throughput (actions/second) and latency (time to produce
  one action or chunk) measured by querying a model many times on an A100 with a fixed
  image + instruction.
- **Comparison policies on LIBERO:** fine-tuned OpenVLA (discrete autoregressive), Diffusion
  Policy, Octo, DiT Policy, MDT, Seer, and pi0.

## Code framework

The recipe plugs into the existing OpenVLA fine-tuning harness. What already exists: the fused
vision encoder + Llama-2 decoder forward pass, the LoRA-wrapped trunk, the data pipeline that
yields a batch of `(image, instruction, ground_truth_action_chunk)` with actions normalized to
`[-1, +1]`, and the optimizer/training loop. Two things are *not yet decided* and are exactly
the slots the method fills: (a) how the decoder is driven to emit a chunk of actions, and
(b) the module that turns the decoder's action-position hidden states into actions, together
with the objective that trains it. Those are left as empty stubs below.

```python
import torch
import torch.nn as nn

# --- already exist in the harness ---
# fused_vision_encoder, llm_decoder (Llama-2), lora-wrapped trunk, projector
# data_loader yields: input_ids, attention_mask, pixel_values,
#                     actions  (B, K, action_dim)  normalized to [-1, +1]
# optimizer, lr schedule

ACTION_DIM = 7          # delta end-effector pose dims for this robot
NUM_ACTIONS_CHUNK = 8   # K: timesteps predicted per query (fixed by protocol)


def vla_forward(batch, action_token_inputs, attention_mask):
    """Existing multimodal forward pass through the pretrained trunk.
    Returns per-position last-layer hidden states for the full sequence.
    `action_token_inputs` are the embeddings placed in the action-token slots,
    and `attention_mask` controls how those slots may attend. HOW to populate
    those slots and what mask to use over them is part of what we design."""
    ...
    return last_hidden_states  # (B, seq_len, hidden_dim)


class ActionDecoder(nn.Module):
    """Turns the decoder's action-position hidden states into a chunk of
    continuous actions, and supplies the training objective. The internal
    structure and the loss are exactly what this method must decide."""

    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=ACTION_DIM):
        super().__init__()
        # Open slot: the module to design here.
        pass

    def predict(self, actions_hidden_states):
        # actions_hidden_states: (B, K * ACTION_DIM, hidden_dim)
        # Open slot: map action-position hidden states -> (B, K, action_dim)
        pass

    def loss(self, predicted_actions, ground_truth_actions):
        # Open slot: the training objective on continuous action chunks
        pass


# existing fine-tuning loop the method plugs into
def train_step(batch, trunk, action_decoder):
    action_token_inputs, attn_mask = build_action_token_inputs(batch)  # open slot
    hidden = vla_forward(batch, action_token_inputs, attn_mask)
    actions_hidden_states = extract_action_positions(hidden, batch)    # (B, K*ACTION_DIM, hidden)
    predicted_actions = action_decoder.predict(actions_hidden_states)  # (B, K, action_dim)
    loss = action_decoder.loss(predicted_actions, batch["actions"])
    loss.backward()
    return loss


def build_action_token_inputs(batch):
    # Open slot: action-token inputs and attention pattern.
    pass
```

The trunk, data pipeline, and loop are fixed. The decode-driving stub, the action decoder, and
the objective are the empty slots the method occupies.
