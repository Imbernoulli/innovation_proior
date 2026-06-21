A large pretrained language model can look capable when prompted with a few input-output examples, yet the same model often stumbles when asked to follow a plain natural-language instruction. The gap is not primarily about missing knowledge; the few-shot version shows the knowledge is there. The real problem is that a bare task description does not resemble the continuation patterns the model saw during pretraining. Natural language inference, reading comprehension, and question answering are useful diagnostics because their prompts are requests rather than text the model would naturally continue. Existing supervised approaches either train on a single task or bundle many datasets without surfacing the task in ordinary language, so they do not teach the model that a human-style instruction should be treated as a command to perform an operation.

One could simply multi-task finetune on many datasets, but that only shows the model has seen many labels. To claim cross-task generalization, the evaluation must be stricter than leaving out one dataset: whole task types must be absent from training. One also needs to separate the value of natural instructions from the value of multi-task exposure. Controls that train on raw input-output pairs or on dataset-name prefixes help isolate that separation. The remaining challenge is to design a training format that makes supervised examples look like user-facing requests, keeps the evaluation honest with cluster-level held-out splits, and handles classification so that valid answers are not diluted across paraphrases.

The method is FLAN, which stands for Fine-tuned LAnguage Net. It takes a dense left-to-right decoder-only language model pretrained on web text, code, dialog, and Wikipedia, and continues finetuning it on a mixture of existing NLP datasets that have been reformatted as natural-language instructions. The base model used in the original work has 137B parameters, but the same recipe can be applied across scales. The core idea is instruction tuning: instead of training the model to predict a target given a raw input, train it to predict a target given a templated request.

Each of the 62 TFDS datasets is equipped with a bank of ten manually written instruction templates. Most templates express the original task direction, such as asking whether a premise entails a hypothesis or asking to translate a sentence to German. Up to three templates per dataset invert the task, for example asking the model to write a positive movie review rather than classify one. During training, every example is rendered with a randomly chosen template from its dataset's bank. This template randomization prevents the model from relying on a fixed surface string and forces it to attend to the requested operation. Generation tasks keep a free-text target, while classification tasks append an OPTIONS suffix listing the valid answer strings; at inference the listed strings are ranked by probability to avoid splitting mass across paraphrases such as "yes", "true", and "correct".

The zero-shot evaluation uses cluster-leave-out splits rather than dataset-leave-out splits. The datasets are grouped into twelve clusters: natural language inference, reading comprehension, commonsense reasoning, sentiment analysis, closed-book QA, paraphrase detection, coreference resolution, reading comprehension with commonsense, struct-to-text, translation, summarization, and miscellaneous tasks. A dataset counts as unseen only if no dataset from its cluster appeared during training. Additional exclusions prevent leakage between semantically close clusters: NLI and paraphrase are withheld from each other, and reading comprehension with commonsense is withheld alongside reading comprehension or commonsense reasoning whenever one of those is evaluated. Evaluating c clusters therefore trains c separate checkpoints, each with a different cluster held out.

Training follows a modest finetuning recipe relative to pretraining. Each dataset is capped at 30,000 examples, the streams are combined with examples-proportional mixing capped at a maximum mixing rate of 3,000, and short examples are packed into sequences with an EOS separator between input and target. The run uses 30,000 gradient steps, 8,192 tokens per batch, Adafactor with learning rate 3e-5, input length 1024, and target length 256. The empirical signature is that the benefit of instruction tuning depends on both task breadth and model scale. Adding more eligible clusters to the training mixture improves held-out cluster performance in a cluster-count ablation, and across model sizes the gains appear only at large scale: the 68B and 137B models improve on held-out tasks, while smaller models can be hurt because their capacity is consumed by fitting the supervised mixture rather than learning reusable instruction-following behavior.

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
