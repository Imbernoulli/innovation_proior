## Research question

We have a large autoregressive language model that was pretrained by next-token prediction on a broad corpus. It is fluent and broadly capable, but its pretraining objective is to continue text, while the deployment objective is to respond to a prompt in the way a competent expert would. The mismatch shows up as instruction neglect, plausible but unhelpful continuations, hallucinated details, and answers that fit web-text statistics better than the user's intent.

Suppose the target task distribution provides pairs of a prompt and an expert demonstration of the desired response: a worked solution, a correctly formatted answer, or a helpful completion. The goal is to adapt the pretrained model so that this demonstrated behavior becomes its default response to similar prompts. The adaptation should be cheap, stable, compatible with ordinary gradient training, and stored in the weights rather than reintroduced through a fragile prompt prefix.

## Background

Autoregressive language modeling factors a token sequence as `p(x_1, ..., x_n) = Π_t p(x_t | x_<t)`. Pretraining maximizes the likelihood of observed next tokens under that factorization, usually by minimizing categorical cross-entropy at each predicted token. The same interface remains available after pretraining: the model reads a context and emits logits for the next token.

The transfer-learning lesson from the ULMFiT, GPT, and BERT lineage is that pretrained representations can be moved to a new operating point by continuing gradient training on task-specific data. This is much more sample-efficient than training from scratch, but the downstream data and objective determine what behavior is actually reinforced.

Few-shot prompting shows that much of the desired behavior is latent in the pretrained model. A frozen model can be steered by placing examples in the context window, but the behavior must be paid for in every prompt, is sensitive to exact formatting and example order, and does not persist in the weights.

Imitation learning provides another relevant precedent. In behavior cloning, a policy is fit to expert state-action demonstrations by maximum likelihood, without a reward model or environment interaction. This makes the procedure cheap and stable, but also limits it: the learner only sees positive expert behavior, cannot rank a better action above a worse one, and does not learn from states it would visit after its own mistakes.

Instruction-format training on public NLP benchmarks confirms that weight-level adaptation can teach a model to read instructions and produce task-shaped outputs. The limitation is coverage: public benchmarks are dominated by classification and question answering, while realistic prompts often ask for open-ended generation, reasoning, editing, brainstorming, and synthesis.

## Baselines

**Few-shot / in-context prompting (Brown et al. 2020).** A frozen pretrained model is conditioned on several input-output examples before the query. The core mechanism is contextual steering: no weights change, and the next-token distribution is biased by the prefix. It is fast and reversible, but the behavior is not stored in the model, consumes context length, and remains brittle under small prompt changes.

**Single-task finetuning of pretrained language models.** A pretrained model is trained further on labeled examples for one downstream task. This reliably specializes the model and reuses the pretrained initialization, but it produces a narrow task model and does not create one general prompt-following policy across many task formats.

**Multi-task instruction finetuning (FLAN, Wei et al. 2021; T0, Sanh et al. 2021).** Many public NLP datasets are reformatted with natural-language instructions and used for supervised adaptation. This demonstrates that instruction-shaped training data can generalize across held-out task families, but the benchmark mixture underrepresents the open-ended prompts that users actually issue.

**Behavior cloning from demonstrations (Pomerleau 1991 and the imitation-learning line).** A policy is fit to expert demonstrations by likelihood maximization. It avoids reward design and online rollouts, but every demonstration is treated as a positive sample of equal status. The learned policy is bounded by the demonstrations and receives no corrective signal on its own generated errors.

## Evaluation settings

The natural setup starts from a pretrained autoregressive language model and a dataset of prompt-demonstration pairs. For general assistant behavior, demonstrations are human-written responses to diverse held-out prompts; for a verifiable domain such as math, each prompt can be paired with a checked worked solution.

Evaluation uses disjoint prompts from the same intended deployment distribution. General assistant behavior is judged by human preference or quality ratings against baseline outputs; verifiable domains use answer accuracy or pass@1 on held-out benchmark splits. Public NLP tasks can also be used as regression checks for broad pretrained capability.

The protocol is a standard train-validation-test split over prompts. Validation records both token-level likelihood and response-level behavior metrics, because one measures imitation of the held-out demonstrations while the other measures whether generated answers satisfy the prompt.

## Code framework

The available machinery is the ordinary minibatch training harness for a pretrained causal language model: tokenization, concatenation of prompt and demonstration into a single sequence, attention and position tensors, an optimizer, and a loop that backpropagates a scalar objective. The open slot is the objective that turns one prepared batch into a scalar.

```python
def training_objective(model, batch):
    """Return the scalar objective for one prepared minibatch."""
    # TODO: the objective we will design.
    pass


def adapt(model, optimizer, prepared_batches):
    model.train()
    for batch in prepared_batches:
        optimizer.zero_grad()
        loss = training_objective(model, batch)
        loss.backward()
        optimizer.step()
```

Only the objective slot is open; the rest of the harness is already ordinary causal-LM training infrastructure.
