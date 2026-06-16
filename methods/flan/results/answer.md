FLAN is instruction tuning: start from a pretrained 137B decoder-only language model, convert 62 existing TFDS NLP datasets into natural-language instruction examples, finetune on the mixture, and evaluate zero-shot on held-out task clusters.

The core move is to make supervised NLP data look like the way a user would ask for a task. Each dataset gets ten manually written instruction templates. Most templates express the original task direction; up to three per dataset turn the task around, such as asking for a positive movie review instead of asking whether a given review is positive. During finetuning, each example is formatted with a randomly selected template for that dataset.

The zero-shot protocol is cluster-leave-out, not dataset-leave-out. The datasets are grouped into twelve task clusters. A dataset is considered unseen only when no dataset from its task cluster was used in finetuning, with extra exclusions for overlapping clusters: NLI and paraphrase are held out from each other; reading comprehension with commonsense is held out when evaluating either reading comprehension or commonsense reasoning; and both parent clusters are held out when evaluating reading comprehension with commonsense. Evaluating c clusters requires c checkpoints, each finetuned with a different cluster omitted. Summarization is included in the task collection but left out of evaluation because most summarization inputs exceed the 1024-token input length.

Classification prompts append an OPTIONS suffix listing the valid answer strings, then rank the listed strings by probability. This reduces the problem where the model spreads probability mass across paraphrases such as "yes", "true", and "correct". Generation tasks use free-text decoding directly.

Training uses a cap of 30,000 examples per dataset, examples-proportional mixing with mixing-rate maximum 3,000, sequence packing with EOS separators between inputs and targets, 30,000 gradient steps, batch size 8,192 tokens, Adafactor at learning rate 3e-5, input length 1024, and target length 256.

The important empirical shape is scale and breadth dependent. Adding more instruction-finetuning clusters improves held-out zero-shot performance in the cluster-count ablation, where NLI, closed-book QA, and commonsense reasoning are held out, the too-similar paraphrase and reading-comprehension-with-commonsense clusters are also kept off the training side, and the seven eligible clusters are added in decreasing order of number of tasks per cluster. Across model sizes 422M, 2B, 8B, 68B, and 137B, the benefit emerges only at large scale: the 68B and 137B models improve on held-out tasks, while the 8B and smaller models are hurt, plausibly because their capacity is consumed by the supervised mixture instead of leaving room for transferable instruction-following behavior. Controls that remove natural instructions during finetuning, using either raw input-output pairs or only task/dataset names, perform worse on held-out clusters, which isolates the natural-language instruction format from ordinary multi-task finetuning.

```python
import random

CLUSTER_EXCLUSIONS = {
    "nli": {"nli", "paraphrase"},
    "paraphrase": {"paraphrase", "nli"},
    "reading_comprehension": {"reading_comprehension", "reading_comprehension_with_commonsense"},
    "commonsense": {"commonsense", "reading_comprehension_with_commonsense"},
    "reading_comprehension_with_commonsense": {
        "reading_comprehension_with_commonsense",
        "reading_comprehension",
        "commonsense",
    },
}

TEMPLATE_BANK = {
    # Representative entries from the ten-template-per-dataset bank.
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
    "wmt16_en_de": [
        ("{source}\n\nTranslate to German", "{target}"),
    ],
    "imdb": [
        ("Is the sentiment of this movie review positive or negative?\n\n{review}\n\n{options}", "{answer}"),
        ("Write a {answer} movie review.", "{review}"),
    ],
}

def options_suffix(classes):
    if not classes:
        return ""
    return "OPTIONS:\n" + "\n".join(f"- {label}" for label in classes)

def format_example(dataset_name, example, classes=None):
    template_in, template_out = random.choice(TEMPLATE_BANK[dataset_name])
    fields = dict(example)
    fields["options"] = options_suffix(classes)
    return template_in.format(**fields), template_out.format(**fields)

def blocked_clusters(eval_cluster):
    return CLUSTER_EXCLUSIONS.get(eval_cluster, {eval_cluster})

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
    for example in take_at_most(dataset.train, 30_000):
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
