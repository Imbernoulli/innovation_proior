# Context: action-method design inside a fixed OpenVLA-OFT PD&AC scaffold

## Research Question

The task is to fill one local action-method slot inside an otherwise fixed vision-language-action control stack. The trunk is already an OpenVLA-style 7B vision-language model adapted for parallel decoding and action chunking. It receives images, language, optional robot state, and a block of empty action positions; it returns language-model hidden states and logits at those positions. The surrounding runtime owns prompt construction, image processing, the trunk forward pass, metric logging, and un-normalization of normalized actions back into robot units.

The open question is only the local decoder/training rule: given the trunk's action-slot outputs, how should an editable `CustomActionMethod` train and produce a normalized action chunk of shape `(batch, NUM_ACTIONS_CHUNK, ACTION_DIM)`? The fixed constants are `ACTION_DIM = 7` for end-effector delta control and `NUM_ACTIONS_CHUNK = 8` future timesteps. The method must run within the fixed budget and API, returning `(loss, metrics)` during training and `(normalized_actions, aux)` during evaluation.

## Fixed Substrate

The base policy inherits the usual VLA structure. A visual encoder turns image observations into patch features, a projector maps those features into the language-model embedding space, and a Llama-family decoder predicts tokens. The OpenVLA lineage builds this trunk from Prismatic-style components: fused SigLIP and DINOv2 visual features, a small projector, and a Llama-2 7B language backbone. The native training objective of this backbone is next-token cross-entropy over a fixed vocabulary.

The PD&AC scaffold changes the decoding schedule around that trunk. Instead of generating action coordinates one token at a time under a causal mask, the runtime inserts position-marked action slots and lets the decoder attend bidirectionally across those slots. With chunk size `K` and action dimension `D`, the trunk can produce outputs for `K * D` action positions in one forward pass. The action method may pass auxiliary action-slot features into `runtime.forward(...)`, but the rest of the trunk and protocol are fixed.

The runtime exposes the method-facing pieces: `runtime.forward(...)`, `runtime.phase`, `runtime.batch_size`, `runtime.device`, `runtime.hidden_dtype`, `runtime.num_prompt_tokens` at evaluation time, `runtime.ground_truth_actions` and masks at training time, `runtime.compute_discrete_objective(...)`, `runtime.compute_action_l1_metrics(...)`, and `runtime.model.{vocab_size, bin_centers, llm_dim}`.

## Prior Art And Constraints

RT-1 establishes that a Transformer controller can emit robot actions as categorical action tokens: each continuous action dimension is mapped into 256 uniformly spaced bins within that variable's bounds, and the controller is trained with categorical cross-entropy. Its action vocabulary is a robot-specific output space in a model trained from robot data, so it does not by itself solve how to reuse a web-pretrained VLM's own vocabulary and output head.

RT-2 moves the action string into a pretrained VLM's normal output stream. It uses the RT-1-style 256-bin action encoding, then either maps bin integers to existing number tokens or overwrites the 256 least-used vocabulary tokens when the tokenizer lacks convenient integer tokens. It also constrains decoding to valid action tokens when the prompt asks for a robot action. This keeps action prediction in the same token-output interface as language, but the public recipe and adaptation setup remain limited.

OpenVLA makes that VLA recipe open around a Prismatic-7B backbone. Its action preprocessing normalizes each action dimension with dataset statistics based on the 1st and 99th percentiles, maps normalized values to the interval `[-1, 1]`, discretizes each of the seven dimensions, and trains with the language model's next-token objective on action tokens. The released tokenizer and model code are the arithmetic source of truth for the bin edges, token ids, and de-tokenization cases.

## Evaluation And Budget

The fixed benchmark is a LIBERO-Goal fine-tuning setup with a single seed and a limited training budget. Policies are evaluated on three disjoint task subsets, and the reported score is based on success rates across those subsets. The method must fit the same training loop, same LoRA-adapted trunk, same image/state inputs, same chunk length, and same normalized-action output contract as the other action-method fills.

## Editable Interface

The editable surface is a `CustomActionMethod` class plus narrow configuration overrides. The class may define helper methods, but it must preserve the runtime contract:

```python
class CustomActionMethod(nn.Module):
    def training_step(self, runtime):
        forward = runtime.forward(...)
        loss, metrics = ...
        return loss, metrics

    def predict_actions(self, runtime):
        forward = runtime.forward(...)
        normalized_actions = ...
        return normalized_actions, aux
```

At evaluation time the action positions are the slice from `runtime.num_prompt_tokens` to `runtime.num_prompt_tokens + NUM_ACTIONS_CHUNK * ACTION_DIM`. At training time the runtime already knows which action labels should contribute to the loss. The method should not alter prompt/image preparation, trunk architecture, action un-normalization, dataset statistics, evaluation protocol, or nonlocal training machinery.
