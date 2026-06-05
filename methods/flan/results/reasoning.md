Let me start from the lopsidedness that's bugging me. A big pretrained language model is a great few-shot learner — drop three examples of a task into the prompt and it picks it up. But ask it to do the same task zero-shot, with only a description and no examples, and it falls apart, especially on things like reading comprehension, question answering, natural language inference. That's strange, because the *knowledge* clearly hasn't gone anywhere — the same model that nails the task with three examples surely "knows" how to do it with zero. So the gap isn't knowledge. It's that the zero-shot prompt doesn't connect to anything.

Why would examples be the thing that connects? Because the model was pretrained on one objective only: predict the next token of natural prose. A few-shot prompt — example, example, example, then the query — *is* a plausible continuation; the model has seen patterns like "here's a thing, here's its label, here's a thing, here's its label" and it just continues the pattern. But a bare zero-shot instruction like "Does this premise entail this hypothesis?" is not a natural continuation of web text. An NLI example phrased as a sentence to continue is genuinely awkward — that kind of thing essentially never appears in unsupervised text. So the model has the ability but no foothold: it doesn't recognize the instruction as a request it should answer. The deficit is largest exactly on the task formats that are least like natural prose, which is the tell.

So the fix shouldn't be a new architecture or a clever decoding trick. It should attack the mismatch directly: teach the model that a natural-language instruction is a request to be answered. If the reason zero-shot fails is "the model never learned to respond to instructions," then the cure is to *show it how*, with supervision. Take a pile of existing NLP datasets — which I already have, with inputs and gold outputs — and rephrase each example as an instruction. A sentiment example becomes "Is the sentiment of this review positive or negative? <review>" → "positive." A translation example becomes "Translate to French: The dog runs." → "Le chien court." Then finetune the pretrained model on this mixture. The hope: by being supervised to follow instructions across many tasks, the model learns the general *skill* of instruction-following, and that skill transfers to instructions for tasks it never saw in finetuning.

But here's the trap I have to avoid, and it's the whole experimental design. If I finetune on, say, NLI data and then test zero-shot NLI, of course it'll do better — I just trained on NLI. That tells me nothing about whether it learned a transferable skill versus just learned that task. The claim I actually want to make is *generalization to unseen task types*. So I have to be strict about what "unseen" means. Grouping the datasets into clusters by task type — one cluster for NLI, one for reading comprehension, one for translation, and so on — I'll declare a dataset unseen only if *no* dataset from *any* cluster it belongs to appeared in finetuning. To evaluate the NLI cluster, I finetune on everything *except* NLI, then test zero-shot NLI. That means I can't train one model and evaluate everything; I have to train one model per held-out cluster, each blind to a different task type. It's more expensive, but it's the only way the "unseen" claim is honest.

Now the practical question: how do I phrase the instructions? My first instinct is one instruction template per dataset. But that's brittle — the model could just memorize "when the input starts with these exact words, do this," which is the opposite of learning to follow *arbitrary* instructions, and it would make zero-shot performance hostage to whether the test phrasing happens to match. So for each dataset I write *several* templates — about ten — that all describe the same task in different natural ways. For an NLI dataset: "Based on the paragraph, can we conclude that <hypothesis>?", "Does <premise> mean that <hypothesis>?", "Read the text and determine if the sentence is true: …", and so on. At finetuning time, each example is formatted with a randomly chosen one of its templates. That forces the model to learn the task *behind* the wording rather than the wording itself.

And while I'm at it, I want the model to learn that an instruction is a flexible thing, not always "input → label." So for each dataset I also include a few templates — up to three — that turn the task around: for sentiment, instead of "classify this review," a template that says "write a movie review that is positive." Same data, inverted direction. It widens the range of instruction shapes the model has practiced.

There's a wrinkle on the classification tasks. When the answer is one of a few classes — "yes" / "no", or an entailment label — how do I read the model's prediction out? The standard move is rank classification: restrict to the valid answer strings and take whichever has higher probability. Logically clean, but it has a leak. There are many surface ways to say "yes" — "yes", "yeah", "correct", "true" — and the model's probability mass spreads across all of them, so the bare token "yes" can get an unfairly low score just because its synonyms stole mass. The model doesn't *know*, when it's responding, that I only care about a specific set of options. So I tell it: append the list of acceptable answers to the prompt, an OPTIONS block enumerating the classes. Now the model is aware of which strings are in play and concentrates its mass on them, and the rank comparison is fair. Generation tasks need none of this — the model already responds in free text, so for those I leave the output untouched.

Mixing the datasets is its own small problem. They differ enormously in size; if I sample examples uniformly, a couple of giant datasets swamp everything and the model barely sees the small ones. So I cap each dataset's contribution — limit it to a maximum number of training examples — and sample with examples-proportional mixing under a mixing-rate cap, so that beyond some threshold a dataset stops accruing extra weight from its sheer size. That keeps the mixture balanced across tasks. For throughput I pack multiple examples into each sequence, separating an input from its target, and inputs from the next example, with a delimiter token. Concretely the recipe lands at: finetune for 30k gradient steps, batch of 8,192 tokens, the Adafactor optimizer at learning rate 3e-5, inputs capped at 1024 tokens and targets at 256, cap of 30k examples per dataset, mixing-rate maximum of 3k. Modest — the instruction-tuning compute is a tiny fraction of pretraining; I'm not building a new model, I'm reshaping how an existing one responds.

Two things I genuinely don't know in advance and want to probe, because they decide whether this idea is real or a mirage.

First: does breadth matter? My whole thesis is that practicing *many* task types teaches a general skill. If that's right, then adding more clusters to the finetuning mixture should keep improving zero-shot performance on the held-out clusters. If instead one or two tasks were enough, the thesis is weaker. So I'd sweep the number of clusters, adding them in, and watch held-out performance — what I'd want to see is that it keeps climbing as I add clusters, ideally not even saturating, which would say "the more task diversity, the better the instruction-following skill."

Second, and this is the one that could sink it: does scale matter? Emergent abilities tend to show up only past some model size. It's entirely possible that instruction tuning helps a huge model but *hurts* a small one. The intuition for the bad case: a small model has limited capacity; if I make it learn forty tasks, it might spend all its capacity memorizing those tasks and have nothing left over for new ones, so it generalizes *worse* than before tuning. A large model has the capacity to both absorb the tasks *and* extract the general instruction-following skill, leaving room to apply it to unseen tasks. So I'd run the same recipe across a ladder of model sizes and look at held-out zero-shot. The hypothesis I'd be testing: at small scale, instruction tuning hurts held-out generalization (capacity all consumed by the training tasks); at large scale, it helps substantially. If that's the shape, it both validates the method and explains *why* it works — instruction tuning buys generalization only once there's spare capacity to hold the skill rather than just the tasks.

And I should be honest about where this *won't* help, because it sharpens what the method is really doing. The mechanism is "teach the model to respond to an instruction." For tasks that are already just plain continuation — finish this sentence, resolve this coreference framed as a completion — the instruction is largely redundant; the pretraining objective already covers the format, so there's little for instruction tuning to add. So I'd expect gains concentrated on tasks that are naturally verbalized as an instruction (NLI, QA, translation, struct-to-text) and little-to-no gain on tasks that are essentially language modeling already. That asymmetry isn't a failure — it's a confirmation that what instruction tuning fixes is specifically the instruction-format mismatch, not some generic "more training is better."

The code is the data pipeline plus an ordinary finetune. The heart is templating: a bank of natural-language phrasings per dataset, a random one chosen per example, with an options suffix bolted on for classification.

```python
import random

# multiple natural-language phrasings per dataset; {options_} is the OPTIONS suffix for classification.
# a few per dataset "turn the task around" (e.g. generate a premise/hypothesis) for instruction-shape diversity.
PATTERNS = {
    "rte": [
        ('{premise}\n\nBased on the paragraph above can we conclude that "{hypothesis}"?\n\n{options_}', "{answer}"),
        ('{premise}\n\nCan we infer the following?\n{hypothesis}\n\n{options_}', "{answer}"),
        ('Read the following paragraph and determine if the hypothesis is true:\n\n{premise}\n\n'
         'Hypothesis: {hypothesis}\n\n{options_}', "{answer}"),
        # ... ~10 phrasings total, incl. a "turned-around" one:
        ("Generate a context and a hypothesis.", "Context: {premise}\n\nHypothesis: {hypothesis}"),
    ],
    # ... one entry per dataset
}

def options_suffix(classes):
    # tell the model which answers are valid, so rank-classification mass isn't split across paraphrases
    return "OPTIONS:\n" + "\n".join(f"- {c}" for c in classes)

def format_example(dataset_name, example, classes=None):
    template_in, template_out = random.choice(PATTERNS[dataset_name])   # random phrasing per example
    fields = dict(example)
    fields["options_"] = options_suffix(classes) if classes else ""     # only for classification
    return template_in.format(**fields), template_out.format(**fields)

def make_train_eval_split(eval_cluster, clusters):
    # "unseen" = no dataset from any cluster the eval dataset belongs to is in training
    train = [d for c, ds in clusters.items() if c != eval_cluster for d in ds]
    eval_ = clusters[eval_cluster]
    return train, eval_

# build the instruction-tuning mixture: cap per dataset, examples-proportional mixing, then finetune
def instruction_tune(model, clusters, eval_cluster):
    train_datasets, _ = make_train_eval_split(eval_cluster, clusters)
    mixture = examples_proportional_mixture(
        [ [format_example(name, ex, classes_of(name)) for ex in cap(D, 30_000)]
          for name, D in train_datasets ],
        cap=3_000)                                  # mixing-rate maximum
    finetune(model, mixture, optimizer=Adafactor(lr=3e-5),
             steps=30_000, batch_tokens=8_192, max_in_len=1024, max_tgt_len=256)  # with packing
    return model   # FLAN: one checkpoint per held-out cluster
```

So the causal chain: a pretrained LM has the knowledge but, with no in-context examples, no foothold for a bare instruction, because instructions don't look like the prose it was trained to continue; so finetune it to respond to instructions, by rephrasing many existing datasets as natural-language instructions and training on the mixture; phrase each task many ways (and sometimes inverted) so it learns the task behind the wording, not the wording; hold out entire task *clusters* to honestly test generalization to unseen task types; expose answer options so rank-classification is fair; and expect the gains to grow with task-diversity and to appear only at sufficient model scale, concentrated on tasks where an instruction adds something beyond plain continuation.
