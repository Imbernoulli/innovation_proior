# Context

## Research question

Large pretrained language models can do impressive few-shot prompting: a prompt with several input-output examples gives the model a pattern to continue. Their zero-shot behavior is weaker, especially for tasks whose bare prompts do not look like ordinary text continuation: natural language inference, reading comprehension, question answering, and similar NLP tasks.

The practical question is whether a pretrained decoder-only LM can be taught, with supervised data, to treat a natural-language task description as something to answer. The goal is cross-task zero-shot generalization: after training on many described NLP tasks, the model should follow descriptions for task types that were not present in that training mixture.

## Background

**Pretraining and prompting.** A left-to-right LM is trained to predict the next token in web documents, code-containing web text, dialog, and Wikipedia. Few-shot prompts work partly because exemplars create a continuation pattern. A bare instruction has no exemplars and may be a poor match to the pretraining distribution.

**Why task format matters.** Natural language inference is a useful diagnostic. A premise-hypothesis judgment is awkward to phrase as sentence continuation and is unlikely to appear naturally in unsupervised text, so a zero-shot LM has little formatting support even if it has relevant world and language knowledge.

**Multi-task supervision.** Multi-task finetuning and unified text-to-text formulations already provide ways to train one model over many NLP datasets. Those lines mostly optimize performance on trained tasks or related domains. The open issue here is stricter: can supervision across many described tasks transfer to entirely held-out task clusters?

**Scale as a condition.** Zero-shot and few-shot abilities often improve with model size, so a supervised intervention may not help uniformly across scales. A small model may spend its capacity fitting the supervised mixture, while a much larger model may have room to learn a reusable task-following behavior.

**Task descriptions as data format.** Existing NLP datasets already contain input fields and target outputs, but a bare input field does not by itself state what operation is requested of it. Any representation that surfaces a task to the model has to preserve the dataset's semantics while remaining intelligible as ordinary text.

**Classification by ranking answer strings.** For a generative LM, classification can be done by scoring the valid answer strings and selecting the highest-probability one. This is sound but imperfect because probability mass can be split across paraphrases of the same answer, so a yes-intended answer can lose to "true" or "correct".

**Data mixture mechanics.** NLP datasets differ greatly in size. A practical multi-dataset finetuning recipe needs per-dataset caps, examples-proportional sampling with a maximum mixing rate, and sequence packing so short examples do not waste training tokens.

## Baselines

**Untuned base LM, zero-shot and few-shot.** The 137B decoder-only base model can be prompted directly. For this baseline, the prompts should follow the GPT-3-style prompts used for the base comparison, since the untuned model has not been adapted to any other prompting convention.

**External large LMs.** GPT-3 175B and GLaM 64B/64E provide scale-comparable zero-shot and few-shot reference points where their reported results are available.

**Single-task supervised systems.** T5-11B, BERT-large, and task-specific translation systems serve as supervised reference points for datasets where those baselines are reported.

**Multi-task finetuning without natural instructions.** Two control formats are useful: input-output pairs with no template, and inputs prefixed only by a task or dataset name. The no-template control still needs natural instructions at zero-shot inference, because a bare input does not identify the task. The task-name control can be tested with either natural instructions or the task and dataset name at inference. These controls isolate whether natural-language task descriptions matter beyond multi-task finetuning itself.

## Evaluation settings

The source data consists of 62 public TFDS text datasets grouped into twelve task clusters: natural language inference, reading comprehension, commonsense reasoning, sentiment analysis, closed-book QA, paraphrase detection, coreference resolution, reading comprehension with commonsense, struct-to-text, translation, summarization, and miscellaneous tasks. Summarization can be part of the tuning mixture, but summarization evaluation is left aside because most summarization inputs exceed the 1024-token input length.

The zero-shot generalization rule is cluster-level. A dataset counts as unseen only when no dataset from any task cluster associated with it appeared during finetuning. Evaluating on c clusters therefore requires c separately finetuned checkpoints, each trained with a different cluster held out. There are additional similarity exclusions: when evaluating reading comprehension with commonsense, both reading comprehension and commonsense reasoning are also omitted from training; when evaluating reading comprehension or commonsense reasoning, the combined reading-comprehension-with-commonsense cluster is omitted; NLI and paraphrase detection are omitted from each other's training splits.

For evaluation, each dataset is scored with the mean across its available natural-language templates, and also with the template selected by best dev performance when a dev set is available. NLU tasks are reported with accuracy or exact match, except DROP, MultiRC, and SQuAD v1/v2, which use F1. Translation uses BLEU. Struct-to-text uses ROUGE-1, ROUGE-2, and ROUGE-L.

For the cluster-count and scale checks, the held-out clusters are natural language inference, closed-book QA, and commonsense reasoning. The training side uses the seven remaining eligible clusters while excluding paraphrase and reading comprehension with commonsense because they are too similar to the held-out NLI and commonsense reasoning clusters. The cluster-count sweep adds those eligible clusters in decreasing order of number of tasks per cluster. The scale sweep uses the same split for models with 422M, 2B, 8B, 68B, and 137B parameters.

The base model is a dense left-to-right decoder-only transformer with 137B parameters, pretrained on 2.49T BPE tokens with a 32k SentencePiece vocabulary; roughly 10 percent of the pretraining data is non-English.

## Code framework

The available primitives are TFDS dataset loaders, a pretrained decoder-only LM, a tokenizer, examples-proportional mixture sampling, sequence packing, an optimizer, and a standard finetuning loop. The remaining pieces of the pipeline are left as open slots to be filled in.

```python
DATASETS = {...}        # TFDS-backed datasets with input fields and targets
TASK_CLUSTERS = {...}   # task_cluster -> dataset names
TASK_FORMATS = {...}    # dataset_name -> per-dataset rendering config
OVERLAP_BLOCKS = {...}  # eval_cluster -> additional clusters to omit

def options_suffix(classes):
    # TODO: optional per-task handling of the class set, if used.
    pass

def format_example(dataset_name, example, classes=None):
    # TODO: render a dataset example into input/target strings.
    pass

def blocked_clusters(eval_cluster):
    # TODO: omit the evaluation cluster and any overlapping clusters.
    pass

def make_train_eval_split(eval_cluster, clusters):
    # TODO: build one train/eval split for a held-out task cluster.
    pass

def formatted_stream(dataset_name, dataset, classes, max_examples):
    # TODO: cap a dataset and render its examples.
    pass

def examples_proportional_mixture(dataset_streams, mixing_rate_maximum):
    ...

def pack_with_eos(formatted_examples, input_length, target_length):
    ...

def train_checkpoint(model, clusters, datasets, eval_cluster, optimizer):
    # TODO: assemble streams, pack examples, and run the finetuning loop.
    pass
```
