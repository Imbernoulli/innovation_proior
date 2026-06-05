# Context

## Research question

A large language model trained only to predict the next token on web text becomes a remarkable *few-shot* learner: show it a handful of input–output examples in the prompt and it does the task. But its *zero-shot* ability — doing a task from only a description, no examples — lags far behind, especially on tasks like reading comprehension, question answering, and natural language inference. The likely reason is a format mismatch: with no in-context examples, the model is asked to respond to a prompt that looks nothing like the continuous prose it was pretrained on, so it has no foothold for what's being requested.

Zero-shot matters because it is what most users can actually do: state a task in plain language ("Is the sentiment of this review positive or negative?", "Translate 'how are you' into Chinese.") and get an answer, without curating examples. The question is whether a simple change to how the model is *taught* — not a new architecture, not test-time tricks — can make a pretrained LM reliably follow such natural-language instructions for tasks it was never trained on. The pain point: a model that has the underlying knowledge (from pretraining) but doesn't know how to *apply* it when a task is posed as an instruction rather than as a continuation.

## Background

**Pretraining and the two ways to use a model.** A decoder-only LM is pretrained on a large corpus by next-token prediction; it then gets applied in one of two paradigms. *Pretrain–finetune*: take the pretrained weights and finetune on a labeled dataset for one specific task, yielding a specialist. *Prompting*: leave the weights frozen and elicit behavior via the input — zero-shot (instruction only) or few-shot (a few in-context examples). Few-shot prompting is strong at scale; zero-shot prompting is comparatively weak, which is the gap of interest.

**Why zero-shot lags.** Without exemplars, the model only has the instruction text. If that text is phrased unlike anything in the pretraining distribution — and many task formats (NLI as a yes/no judgment, a structured QA prompt) are unnatural as raw continuations — the model has trouble. Diagnostically: NLI examples are unlikely to occur naturally in unsupervised text and are awkward to phrase as a sentence continuation, which is exactly where the largest zero-shot deficits show up.

**Multi-task learning and instruction-style formulations.** Casting many NLP tasks into a single unified text format (e.g. everything as question-answering over a context) is an established idea, as is multi-task finetuning across datasets. Prior work mostly targeted *multi-task* performance (do better on the trained tasks) rather than *zero-shot generalization to unseen task types*, and was generally not framed around exploiting the knowledge already in a large pretrained LM.

**Scale-dependence of emergent behavior.** Larger LMs show qualitatively new capabilities that small ones lack; zero- and few-shot ability improves sharply with scale. This sets up a key open question for any training-time intervention: does it help uniformly, or only past some scale?

**Classification via rank.** A common way to read a classification answer out of a generative LM is *rank classification*: restrict to the valid answer strings (e.g. "yes"/"no") and pick the higher-probability one. This is logically sound but has a known imperfection — probability mass leaks across the many surface ways of expressing the same answer, distorting the comparison.

**Mixing datasets of very different sizes.** When finetuning on a mixture of datasets with wildly different sizes, naive uniform sampling over examples over-represents huge datasets. Examples-proportional mixing with a cap (a "mixing rate maximum") is a standard recipe to keep any single large dataset from dominating. *Packing* multiple short examples into one sequence (separated by a delimiter) is a standard throughput trick.

## Baselines

A method that improves zero-shot instruction following would be compared against:

**The untuned base LM, zero-shot.** The same pretrained model prompted zero-shot with the task instruction. This is the most direct ablation — it isolates how much any improvement is due to the training intervention rather than the base model. Gap: weak zero-shot, for the format-mismatch reason above.

**The untuned base LM, few-shot.** Same model with a few in-context examples. Stronger than its own zero-shot, and a reference for "how much does the intervention close the gap to few-shot?" Gap: requires the user to supply exemplars.

**Large general LMs, zero- and few-shot.** Much larger pretrained models prompted directly (zero-shot and few-shot) as strong external reference points. Gap: their zero-shot still trails their few-shot on many tasks — the very gap under study.

**Multi-task finetuning without instructions** (ablation baselines). Finetune on the same mixture of datasets but *without* natural-language instructions — either inputs/outputs only ("no template"), or each input prefixed with just the task/dataset name ("dataset name"). These test whether the gains come from instructions specifically or merely from multi-task finetuning. Gap (hypothesized): without instructions the model learns the tasks but not how to *follow instructions*, so it generalizes poorly to unseen task types.

## Evaluation settings

The natural yardstick is *zero-shot performance on task types held out from training*, which forces a careful definition of "unseen." Group existing NLP datasets into clusters by task type, and call a dataset unseen only if **no** dataset from any cluster it belongs to was in training. To evaluate on c clusters, train c models, each holding out a different cluster. The pre-existing datasets/metrics that serve as the yardstick:

- **Natural language inference:** ANLI (R1–R3), CB, RTE, SNLI, QNLI, WNLI — accuracy.
- **Reading comprehension:** BoolQ, MultiRC, OBQA — accuracy.
- **Closed-book QA:** ARC (easy/challenge), Natural Questions, TriviaQA — accuracy / exact match.
- **Translation:** WMT'14 En–Fr, WMT'16 En–De, En–Ro (both directions) — BLEU.
- **Commonsense reasoning & coreference:** CoPA, HellaSwag, PiQA, StoryCloze, Winogrande, WSC, etc. — accuracy (sentence-completion style).
- Plus sentiment, paraphrase, struct-to-text, and others, grouped into clusters.

Protocol: for each dataset, evaluate the mean over multiple natural-language instruction phrasings (to proxy a typical instruction), and also report the best phrasing chosen on a dev set. For classification, attach the list of answer options to the prompt. (Datasets, clusters, and metrics here are pre-existing facts; no result numbers belong to this setup.) Base model: a ~137B-parameter decoder-only LM pretrained on web text, code, dialog, and Wikipedia.

## Code framework

Pre-method primitives that already exist: a corpus of NLP datasets (each an input/output mapping), a tokenizer, a standard finetuning loop with an optimizer, and the dataset-mixing/packing utilities. The contribution will fill in (1) how each dataset's examples become natural-language instruction strings, (2) how datasets are grouped and split into train/held-out for generalization, and (3) the option-listing for classification.

```python
# --- existing NLP datasets: each yields {input fields..., "answer": ...} ---
DATASETS = {...}                      # e.g. rte, boolq, arc, wmt14, ...

# --- existing finetuning loop / optimizer / packing / proportional mixing (exist) ---
def finetune(model, dataset, optimizer, steps, batch_tokens, max_in_len, max_tgt_len):
    ...

def examples_proportional_mixture(datasets, cap):    # standard mixing with a rate cap
    ...

# --- slot 1: turn an example into an instruction-formatted (input, target) string (TODO) ---
def format_example(dataset_name, example):
    pass

# --- slot 2: define task-type groups and the held-out split for generalization (TODO) ---
def make_train_eval_split(eval_cluster):
    pass

# --- slot 3: for classification, expose the valid answer options (TODO) ---
def add_options(prompt, classes):
    pass
```
