Let me start from the deployment failure, because the model is not failing at language modeling. It is doing language modeling too well. I give it a prompt that asks for a specific response, and the pretrained autoregressive model tries to continue the distribution it saw during pretraining. A question can be followed by another question, a passage can be followed by more passage, and a request can be followed by a plausible but unhelpful continuation. The model's objective has been "predict the next token in broad text." My deployment objective is "given this prompt, produce the expert response." Those are different conditional distributions. Scaling the model makes the text continuation sharper, but it does not by itself change what distribution I am asking the model to imitate.

Now suppose I can collect examples of the behavior I want. For each prompt `q`, I have an expert demonstration `y = (y_1, ..., y_T)`: a worked solution, a useful completion, a correctly formatted answer. The data is not a reward model and it is not an interactive environment. It is just a set of target continuations. So the question becomes very concrete: how do I use these pairs to move the pretrained policy so that, when it sees `q`, the demonstrated response is the natural continuation?

The first possibility is to leave the weights alone and put examples in the context window. Few-shot prompting is useful as a diagnostic because it proves the pretrained model can often follow a pattern from context. But it is a weak endpoint. It spends context tokens every time, it is sensitive to formatting and example order, and nothing persists after the context disappears. I want the behavior to live in the weights. That pushes me back to gradient training.

Single-task finetuning already tells me that continuing training from pretrained weights can move a model cheaply. The part that is wrong for this problem is the narrowness: if I train on one classification or QA dataset, I get a specialist for that format. I need a policy that handles many prompt shapes, so the data should be demonstrations of the general behavior I want, drawn from the prompt distribution I care about. Behavior cloning gives the matching control analogy: if expert demonstrations are available, likelihood training is the simplest way to imitate them. It will be bounded by the demonstrator and blind to negative alternatives, but it is stable and direct.

So what is the likelihood of a demonstration under a causal language model? The prompt is given, and the response is generated one token at a time. That means

  π_θ(y | q) = Π_{t=1}^{T} π_θ(y_t | q, y_{<t}).

Maximizing this product is the same as maximizing the sum of the logs, or minimizing its negative:

  L = -Σ_{t=1}^{T} log π_θ(y_t | q, y_{<t}).

This already looks like pretraining, and that is the point. The model already knows how to expose next-token logits, compare them to observed next tokens, and backpropagate cross-entropy. I am not adding a head or inventing an environment. I am continuing the same next-token machinery on curated prompt-demonstration data.

But the phrase "given the prompt" matters. In the tensor I feed the model, the prompt and response are concatenated into one sequence. If I compute the ordinary language-modeling loss on every target token in that sequence, I also train the model to predict prompt tokens from earlier prompt tokens. That teaches the model to model the input distribution, not to answer the input. At deployment the prompt is supplied; the model is not supposed to generate it. The conditional likelihood `π_θ(y | q)` only scores the response tokens. So the loss must be zero for prompt targets and nonzero for demonstration targets.

I need to be precise about the shift. With input tokens `[q_1, ..., q_m, y_1, ..., y_T]`, the logits at sequence position `j` predict token `j+1`. Therefore the target `y_1` is predicted by the logits at the last prompt position. If my implementation mask is stored on prediction positions, that last prompt position must be kept because its next-token target is the first demonstration token. Positions whose next-token targets are still prompt tokens are suppressed; positions whose next-token targets are demonstration tokens are kept. In code, this is the standard causal shift: `shift_logits = logits[..., :-1, :]`, `shift_labels = input_ids[..., 1:]`, and the prediction-position mask is sliced to the same `L-1` positions before multiplying the unreduced cross-entropy.

Now I have another choice: sum the response-token losses or average them. If I sum, long demonstrations dominate just because they have more tokens. That changes the objective from "fit demonstrations" into "let length decide gradient scale." The mathematical object I want is the average negative log-likelihood per demonstrated token:

  L_SFT(θ) = -(1 / |y|) Σ_{t=1}^{|y|} log π_θ(y_t | q, y_{<t}).

In a batch this becomes the same thing operationally: compute every shifted per-token cross-entropy, zero out prompt and padding targets, sum the remaining losses, and divide by the number of valid demonstration targets. That is the token average. The sequence-sum version is the failure mode I am avoiding.

Statistically, this is maximum likelihood on expert demonstrations. Each observed token is a positive target. There is no comparison saying that one possible answer is better than another, and there is no penalty attached to plausible answers that are absent from the demonstrations. That gives the method both its strength and its ceiling. It is cheap and stable because it is pure supervised imitation. It cannot rank responses, cannot improve beyond the demonstrations, and does not train on the model's own off-trajectory mistakes.

The same point appears if I look at it through a reward-plus-demonstration objective. Suppose I write a common post-training objective

  J(θ) = E_{τ ~ π_θ}[r(τ | q)] - μ KL(π_β(. | q) || π_θ(. | q)),

where `π_β` is the behavior policy represented by the demonstration data. Differentiating the reward term gives the usual score-function gradient, `E_{τ~π_θ}[r(τ | q) ∇ log π_θ(τ | q)]`. The KL term is

  KL(π_β || π_θ) = E_{τ~π_β}[log π_β(τ | q) - log π_θ(τ | q)].

The first log does not depend on `θ`, so the derivative of `-μ KL(π_β || π_θ)` is

  μ E_{τ~π_β}[∇ log π_θ(τ | q)].

That is exactly the ascent direction for demonstration log-likelihood, or the negative of the gradient of the loss I minimize. If I write `∇ log π_θ = (1 / π_θ) ∇π_θ`, the same term has the unified-gradient shape with a `1 / π_θ` reweight in the `∇π_θ` parameterization. In the fixed-advantage view, every demonstration has advantage `+1`, scaled globally by `μ` if I keep that coefficient. There is no PPO-style clipping term in this corner, because I am not using an on-policy importance-sampling ratio from a drifting rollout policy. The data is fixed positive demonstration data, and the gradient is the likelihood gradient.

The remaining practical choices follow from that objective. Token validation loss tells me how well the model predicts held-out demonstration tokens, but the behavior I care about is the generated response to a prompt, so I should monitor both token likelihood and response-level quality. The demonstration distribution also matters: if the data only contains benchmark-style classification and short QA, the model learns that distribution, not the open-ended prompts I care about. The loss is only a conduit; the demonstrations define the behavior it can imitate.

Now I can fill the one empty slot in the training harness. I make the implementation mirror the causal-LM trainer: labels are `input_ids[:, 1:]`, logits are `logits[..., :-1, :]`, cross-entropy is unreduced, the prediction-position mask removes prompt and padding targets, and the scalar loss is the sum over valid targets divided by the valid-token count.

```python
import torch.nn as nn


def sft_loss(model, batch):
    """Masked, shifted, token-averaged next-token cross-entropy."""
    loss_mask = batch["loss_mask"][:, :-1].reshape(-1)
    labels = batch["input_ids"][:, 1:]

    output = model(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"],
                   position_ids=batch["position_ids"],
                   use_cache=False)

    logits = output.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels.contiguous()

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1).to(shift_logits.device)
    loss_mask = loss_mask.to(shift_logits.device)

    per_token_loss = loss_fct(shift_logits, shift_labels)
    per_token_loss = per_token_loss * loss_mask

    valid_tokens = loss_mask.sum()
    return per_token_loss.sum() / valid_tokens


def adapt(model, optimizer, data_loader):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        loss = sft_loss(model, batch)
        loss.backward()
        optimizer.step()
```

I have the path now: the pretrained model already optimizes next-token likelihood, but on the wrong distribution for prompt following. Expert demonstrations give me the desired conditional continuation distribution. The autoregressive factorization turns that into next-token log-likelihood of response tokens only; the prompt is conditioning context, so its targets are masked; the shift aligns logits at position `j` with token `j+1`; the token average prevents long demonstrations from dominating by length; and the unified-gradient view shows the same update as a positive-only demonstration term with advantage `+1`, `1 / π_θ` reweighting, and no PPO clip.
