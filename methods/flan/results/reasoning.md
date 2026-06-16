The thing that bothers me is the asymmetry. A large pretrained language model can look smart when I give it a few examples in the prompt, but the same model becomes much less reliable when I only write the task down in words. Reading comprehension, question answering, natural language inference: these are not exotic skills for the model, because the few-shot version often works. The missing piece is not just knowledge. The missing piece is that a bare instruction does not give the model the familiar continuation pattern that examples give it.

The pretraining objective explains why. The model has practiced predicting the next token in natural text. A few-shot prompt gives it a local mini-distribution to continue: input, label, input, label, input, and now the next label. A zero-shot prompt such as "Does this premise entail this hypothesis?" has a different shape. It is a request, not a continuation. Natural language inference is the sharp case: premise-hypothesis judgments are not the kind of text that usually appears by itself on the web. So the model can contain the latent competence and still fail to recognize the form of the request.

I do not need a new decoder or a different pretraining objective to test that diagnosis. I can keep the pretrained LM and change the supervised data it sees afterward. Existing NLP datasets already have inputs and targets; I can turn each example into a natural-language request and train the model to answer it. A sentiment example becomes a question about whether a review is positive or negative. A translation example becomes "Translate to German" or "Translate to French." A reading-comprehension example becomes a passage, a question, and a requested answer. If this works, the model is not just memorizing one downstream task; it is learning that when text describes a task, the next text should carry out the task.

That last clause is easy to fool myself about. If I train on NLI and evaluate on NLI, I have only built an NLI finetuned model. The generalization claim has to be stricter than "the exact dataset was absent." I need task clusters. Put the 62 TFDS datasets into twelve clusters by task type: NLI, reading comprehension, commonsense, sentiment, closed-book QA, paraphrase, coreference, reading comprehension with commonsense, struct-to-text, translation, summarization, and miscellaneous. Then a dataset is unseen only if its whole task type is absent from training. To evaluate a cluster, I train a separate checkpoint with that cluster held out. If there are c evaluation clusters, there are c finetuning runs.

Some clusters are close enough that the literal cluster name is not enough. Paraphrase detection is close to NLI, because semantic equivalence can be viewed as entailment in both directions, so when I test NLI I should also remove paraphrase, and when I test paraphrase I should remove NLI. Reading comprehension with commonsense overlaps both of its parents, so that combined cluster has to be removed when I evaluate either parent, and both parents have to be removed when I evaluate the combined cluster. This is the price of making "unseen task type" mean what it says.

Now the dataset has to speak in instructions without collapsing into one brittle phrase per task. One template per dataset would let the model key on a surface string. I want the same task to arrive in several natural wordings, so for each dataset I write ten templates. Most describe the original input-to-output direction; up to three deliberately turn the task around. For sentiment, a normal template asks for the polarity of a review, while an inverted one can ask the model to write a positive review. The inversion matters because it teaches that instructions are not only classifiers with labels; they can ask for generation in either direction allowed by the dataset.

At training time I should not freeze one wording onto one example. Each example gets a randomly selected template for its dataset. That makes the supervision noisy in the right way: the same underlying mapping appears under different surface forms, so the model has to attend to the requested operation rather than only to a fixed prefix.

Classification needs one more adjustment. If I score only candidate strings, "yes" can lose probability mass to "true", "correct", or other paraphrases even when the model's semantic answer is yes. Ranking fixed answer strings is still the practical classifier, but I should tell the model what strings are valid. So the prompt ends with an OPTIONS block listing the classes, and the target is one listed class. For three-way NLI datasets such as ANLI and CB, the choices are variants of yes, no, and impossible-to-say; for binary RTE it is yes/no; for multiple-choice commonsense tasks it is the provided answer choices. Generation tasks do not need this suffix because their output space is free text.

The mixture also has to be balanced. If I let dataset size dominate, the largest datasets decide the training distribution and the small task types barely count. I cap each dataset at 30,000 training examples, and for sampling I use examples-proportional mixing with a mixing-rate maximum of 3,000, so examples beyond that threshold no longer add sampling weight. Multiple short examples can be packed into one sequence, with a special EOS token separating input from target. The concrete finetuning recipe is modest relative to pretraining: 30,000 gradient steps, 8,192 tokens per batch, Adafactor with learning rate 3e-5, input length 1024, target length 256, about 60 hours on 128 TPUv3 cores.

The base model should be large enough for the hypothesis to have a chance. The model I use is a 137B-parameter dense left-to-right decoder-only transformer pretrained on web documents, code-containing web text, dialog, and Wikipedia, 2.49T SentencePiece BPE tokens with a 32k vocabulary. The supervised pass is not trying to inject all task knowledge from scratch. It is trying to make an already broad LM use its knowledge when the prompt is a plain request.

There are two stress tests I have to run. Task breadth is the first one. If the learned object is really a general instruction-following behavior, then adding more task clusters to the finetuning mixture should improve held-out clusters. If performance saturates immediately, the story is weaker: maybe one neighboring task supplies all the benefit. I would hold out NLI, closed-book QA, and commonsense reasoning, keep paraphrase out because it is too close to NLI, keep reading comprehension with commonsense out because it is too close to commonsense reasoning, and add the seven eligible remaining clusters in decreasing order of number of tasks per cluster. Then I can ask whether the held-out average keeps moving upward as task breadth grows.

Scale is the second stress test, and it may go the opposite way for small models. A small model asked to learn dozens of supervised tasks might spend its capacity fitting those tasks and become worse on new task types. A larger model can absorb the mixture and still have room to represent the reusable behavior: read the instruction, identify the task, produce the requested answer. So I need the same split across a size ladder: 422M, 2B, 8B, 68B, and 137B. The expected signature is not "more finetuning always helps"; it is that the benefit emerges only once the model is large enough.

This also tells me where the recipe should be weak. If a downstream task is already just language modeling, such as choosing a sentence completion for a commonsense or coreference example, then an explicit instruction adds less. The pretraining objective already covers the shape. I should expect the largest gains where the task is naturally a request - NLI, QA, translation, struct-to-text - and smaller or mixed gains where the task already resembles continuation.

Evaluation needs to avoid making one wording look like the whole recipe. For each dataset I should report the mean across its available templates, because that approximates a typical natural-language request. If a dev set exists, I can also choose the best dev template and report the corresponding test score, but that is a separate, more prompt-engineered view. For NLU I will use accuracy or exact match except for DROP, MultiRC, and SQuAD variants, where F1 is the right metric; translation uses BLEU, and struct-to-text uses ROUGE.

I also need to separate the effect of the instructions from the effect of multi-task finetuning. If I remove the templates entirely and train only on raw inputs and outputs, the model gets supervision but not the habit of mapping a natural request to an action. If I prepend only a task or dataset name, I give it a symbolic task identifier but still not an ordinary user-facing instruction. For the no-template version, inference still has to use natural requests because otherwise the model would not know what task a bare input is asking for. For the task-name version, I can test both the natural request and the name prefix. If those controls fall short, the missing ingredient is the instruction format, not just exposure to many labels.

The final pipeline is just templating, conservative cluster splitting, option rendering for classification, capped mixture sampling, packing, and ordinary finetuning.

```python
import random

OVERLAP_BLOCKS = {
    "nli": {"paraphrase"},
    "paraphrase": {"nli"},
    "reading_comprehension": {"reading_comprehension_with_commonsense"},
    "commonsense": {"reading_comprehension_with_commonsense"},
    "reading_comprehension_with_commonsense": {"reading_comprehension", "commonsense"},
}

TASK_FORMATS = {
    # Representative entries from the ten-template-per-dataset format set.
    "anli": [
        (
            '{premise}\n\nBased on the paragraph above can we conclude that "{hypothesis}"?\n\n{options}',
            "{answer}",
        ),
    ],
    "rte": [
        (
            '{premise}\n\nBased on the paragraph above can we conclude that "{hypothesis}"?\n\n{options}',
            "{answer}",
        ),
    ],
    "imdb": [
        ("Is the sentiment of this movie review positive or negative?\n\n{review}\n\n{options}", "{answer}"),
        ("Write a {answer} movie review.", "{review}"),
    ],
    "wmt16_en_de": [
        ("{source}\n\nTranslate to German", "{target}"),
    ],
}

def options_suffix(classes):
    if not classes:
        return ""
    return "OPTIONS:\n" + "\n".join(f"- {label}" for label in classes)

def format_example(dataset_name, example, classes=None):
    template_in, template_out = random.choice(TASK_FORMATS[dataset_name])
    fields = dict(example)
    fields["options"] = options_suffix(classes)
    return template_in.format(**fields), template_out.format(**fields)

def blocked_clusters(eval_cluster):
    return {eval_cluster} | OVERLAP_BLOCKS.get(eval_cluster, set())

def make_train_eval_split(eval_cluster, clusters):
    blocked = blocked_clusters(eval_cluster)
    train_names = [
        name
        for cluster, names in clusters.items()
        if cluster not in blocked
        for name in names
    ]
    return train_names, clusters[eval_cluster]

def formatted_stream(dataset_name, dataset, classes):
    examples = take_at_most(dataset.train, 30_000)
    for example in examples:
        yield format_example(dataset_name, example, classes)

def train_checkpoint(model, clusters, datasets, eval_cluster):
    train_names, _ = make_train_eval_split(eval_cluster, clusters)
    streams = [
        formatted_stream(name, datasets[name], classes_of(name))
        for name in train_names
    ]
    mixture = examples_proportional_mixture(streams, mixing_rate_maximum=3_000)
    packed = pack_with_eos(mixture, input_length=1024, target_length=256)
    return finetune(
        model,
        packed,
        steps=30_000,
        batch_tokens=8_192,
        optimizer=Adafactor(learning_rate=3e-5),
    )
```

The causal chain is now clean: zero-shot prompts fail when a task description is too unlike a pretraining continuation; supervised examples can be rewritten as natural-language requests; many tasks and many templates teach the model to map descriptions to actions; cluster-level held-out splits keep the transfer claim honest; options make ranked classification match the intended label set; and the benefit should depend on both task diversity and model scale because the reusable instruction-following behavior has to be learned in addition to the individual tasks.
