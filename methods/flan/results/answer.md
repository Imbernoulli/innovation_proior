# FLAN (Instruction Tuning)

## Problem

A large pretrained LM is a strong few-shot learner but a weak zero-shot one: given only a task description (no in-context examples), it underperforms, especially on tasks (NLI, QA, reading comprehension) whose prompts look unlike the prose it was pretrained to continue. The model has the knowledge but no foothold for a bare instruction. Goal: make a pretrained LM reliably follow natural-language instructions for task types it never saw in training, via a simple training-time change.

## Key idea

**Instruction tuning** — finetune the pretrained LM on a large mixture of *existing NLP datasets, each rephrased as natural-language instructions*, so it learns the general skill of following instructions and transfers it to unseen task types. The resulting model is FLAN.

- **Templates.** For each dataset, write ~10 natural-language instruction phrasings (plus up to 3 that "turn the task around," e.g. generate a review for sentiment). Each training example is formatted with a *randomly chosen* template, so the model learns the task behind the wording, not the wording.
- **Held-out generalization.** Group datasets into clusters by task type. A dataset is "unseen" only if no dataset from any cluster it belongs to was in training. To evaluate c clusters, train c models, each holding out a different cluster.
- **Classification with options.** Append an OPTIONS list of valid answer strings to classification prompts, so rank classification isn't distorted by probability mass leaking across paraphrases of an answer.
- **Mixing/training.** Cap each dataset at 30k examples; combine with examples-proportional mixing (mixing-rate maximum 3k); pack multiple examples per sequence. Finetune for 30k steps, batch 8,192 tokens, Adafactor at lr 3e-5, input/target lengths 1024/256. Base model: a ~137B decoder-only LM. Instruction-tuning compute is <2% of pretraining.

Two findings the design predicts: zero-shot held-out performance improves with the **number of instruction-tuning clusters** (task diversity), and the benefit is **scale-dependent** — instruction tuning helps large models generalize but can hurt small ones (their capacity is consumed by the training tasks, leaving none for the instruction-following skill). Gains concentrate on tasks naturally verbalized as instructions (NLI, QA, translation, struct-to-text) and are minimal on tasks already formatted as plain continuation (commonsense/coreference completions), confirming the mechanism fixes the instruction-format mismatch specifically.

## Code

The pipeline is templating (the core contribution) plus standard mixing/finetuning. One checkpoint is produced per held-out cluster.

```python
import random

PATTERNS = {
    "rte": [
        ('{premise}\n\nBased on the paragraph above can we conclude that "{hypothesis}"?\n\n{options_}', "{answer}"),
        ('{premise}\n\nCan we infer the following?\n{hypothesis}\n\n{options_}', "{answer}"),
        ('Read the following paragraph and determine if the hypothesis is true:\n\n{premise}\n\n'
         'Hypothesis: {hypothesis}\n\n{options_}', "{answer}"),
        # ~10 phrasings per dataset; include "turned-around" templates for instruction-shape diversity:
        ("Generate a context and a hypothesis.", "Context: {premise}\n\nHypothesis: {hypothesis}"),
    ],
    # ... one entry per dataset (over 60 datasets, twelve task clusters)
}

def options_suffix(classes):
    return "OPTIONS:\n" + "\n".join(f"- {c}" for c in classes)

def format_example(dataset_name, example, classes=None):
    template_in, template_out = random.choice(PATTERNS[dataset_name])   # random phrasing per example
    fields = dict(example)
    fields["options_"] = options_suffix(classes) if classes else ""     # classification only
    return template_in.format(**fields), template_out.format(**fields)

def make_train_eval_split(eval_cluster, clusters):
    # unseen = no dataset from any cluster the eval dataset belongs to is in training
    train = [d for c, ds in clusters.items() if c != eval_cluster for d in ds]
    return train, clusters[eval_cluster]

def instruction_tune(model, clusters, eval_cluster):
    train_datasets, _ = make_train_eval_split(eval_cluster, clusters)
    mixture = examples_proportional_mixture(
        [ [format_example(name, ex, classes_of(name)) for ex in cap(D, 30_000)]
          for name, D in train_datasets ],
        cap=3_000)                                   # mixing-rate maximum
    finetune(model, mixture, optimizer=Adafactor(lr=3e-5),
             steps=30_000, batch_tokens=8_192, max_in_len=1024, max_tgt_len=256)   # packed sequences
    return model
```

Example formatted instruction (NLI), with options:

```
<premise>

Based on the paragraph above can we conclude that "<hypothesis>"?

OPTIONS:
- yes
- no
```
→ target: `yes`
