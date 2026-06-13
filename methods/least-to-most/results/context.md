# Context

## Research question

Few-shot chain-of-thought prompting — showing a large LM a handful of worked examples that spell out intermediate reasoning steps — dramatically raises accuracy on math, commonsense, and symbolic tasks, while staying fully interpretable and needing no finetuning. But it has a sharp, specific weakness: it tends to fail on test problems that are *harder than the demonstrations*. If the exemplars solve 2-step problems, the model does well on 2-step problems and degrades as the required number of steps grows; if the exemplars use short inputs, performance collapses on longer inputs. This is *easy-to-hard generalization*, and it is exactly where humans excel — a person who can solve small instances of a problem can usually solve large ones, by breaking the large one down.

The most pointed version of this failure is *compositional generalization*: a model trained or prompted on simple compositions of primitives fails to generalize to longer or novel compositions of the same primitives. The question is whether a pure prompting strategy — no architecture changes, no training, no symbolic machinery — can let a frozen LM solve problems substantially harder (more steps, longer inputs) than anything shown in its prompt. A solution would have to bridge the gap between the difficulty of the exemplars and the difficulty of the test instances, using only in-context demonstrations.

## Background

**Chain-of-thought (CoT) prompting.** Augment each few-shot exemplar with a natural-language rationale — the intermediate steps from input to answer — and the model learns to produce its own rationale before answering, raising accuracy on multi-step problems. Combined with self-consistency (sample many rationales, majority-vote the answer), it matches or beats specialized supervised models on many benchmarks while remaining interpretable. The rationale in a CoT exemplar solves the *whole* problem in one continuous generation; each exemplar is self-contained and independent of the others.

**The CoT limitation that motivates everything here.** Empirically, CoT does a perfect job when test problems match the difficulty of the prompt exemplars, and a poor job when they exceed it. On a task where the exemplars use short lists and the test lists are long, CoT accuracy drops steeply with length. On a canonical compositional-generalization benchmark, CoT with a handful of exemplars is far below what's needed. The model can execute the *kind* of reasoning shown, but cannot stretch it to a longer or deeper instance it hasn't been shown.

**Compositional generalization and its benchmark.** A widely used benchmark maps natural-language commands to action sequences ("look thrice after jump" → JUMP LOOK LOOK LOOK). Its hardest split is the *length split*: action sequences in the test set are longer than any in training. The structure of the task is compositional — long commands are built by combining short ones with connectives like "and", "after", "twice", "thrice", "around", "opposite" — so a model that understands the pieces *ought* to handle long commands, but standard sequence models trained on the full training set generalize poorly to longer test sequences.

**Easy-to-hard generalization in prior work.** Several lines of work achieve generalization from simple to complex by mechanisms that recur extra computation or apply learned logic rules at larger scale — e.g. networks that perform more recurrent steps at test time to solve bigger instances, or neural-logic systems trained on small instances that scale up. These typically require dedicated training.

**Task decomposition in prior work.** Decomposing a hard problem into subproblems is an old idea: split a multi-hop question into single-hop subquestions answered independently and then aggregate; chain LM calls so each step's output feeds the next; translate a question into a sequence of slot-filling subprompts via rules. Most of these rely on *trained* decomposition/aggregation models, and several produce *independent* subquestions answered in isolation.

**Educational psychology origin.** The phrase "least-to-most" comes from a teaching technique: a graded sequence of increasingly informative prompts that guides a learner toward a skill, starting with the least help and adding more as needed.

## Baselines

A method for easy-to-hard generalization would be compared against:

**Standard few-shot prompting.** Exemplars are just input→output pairs, no rationale. Gap: no intermediate steps; collapses on anything multi-step (often ~0% on the symbolic/compositional tasks).

**Chain-of-thought prompting** (Wei et al., 2022). Exemplars include a full rationale that solves the whole problem in one pass; exemplars are independent of each other. For the symbolic and compositional tasks, the CoT exemplars demonstrate the mapping/solving rationale. Gap: solving the whole problem in one rationale does not transfer to instances harder (longer, more steps) than the exemplars — accuracy falls steeply as test difficulty exceeds demonstration difficulty.

**Zero-shot prompting.** Just the instruction, no exemplars and no rationale. Gap: weakest; no demonstration of the reasoning format.

**Specialized neural-symbolic / data-augmentation models (compositional generalization).** Stack machines, grammar-induction, latent-grammar, and recombination-based augmentation methods that can reach high accuracy on the benchmark — but require complex training, grammar-search, or augmentation pipelines, often on the full (15k+ example) training set, and the end-to-end neural ones still fail the length split. Gap: heavy machinery and training; not a general, training-free prompting recipe.

## Evaluation settings

The natural yardstick is tasks where test difficulty can be made to exceed exemplar difficulty along a clear axis (length, number of steps, composition depth). Pre-existing datasets/metrics:

- **Symbolic manipulation — last-letter concatenation.** Input: a list of words; output: concatenation of their last letters ("thinking, machine" → "ge"). Difficulty axis: list length. Test lists of length 4–12 (500 lists per length), sampled from a word-frequency list; exemplars use short lists. Metric: exact-match accuracy by length.
- **Compositional generalization — the command→action benchmark, length split.** Difficulty axis: action-sequence length (test longer than train). Metric: exact-match accuracy on the test set.
- **Math reasoning — GSM8K and the numerical subset of DROP.** Difficulty axis: number of reasoning steps to the answer. Metric: answer accuracy, also broken down by number of steps in the gold solution. Base model: a strong code/text GPT-3-class LM (e.g. `code-davinci-002`), prompted; for math, prompts must fit the model's context limit (~2048 tokens), which constrains exemplar count and motivates compact intermediate representations.

(Datasets, splits, and metrics here are pre-existing facts of the yardstick; no result numbers belong to this setup.)

## Code framework

Pre-method primitives that already exist: an LM completion call and a few-shot exemplar bank per task. The contribution will define the prompting strategy that closes the easy-to-hard gap.

```python
def llm(prompt, stop=None):                 # frozen LM completion (exists)
    ...

# few-shot exemplar bank(s) per task (exist)
EXEMPLARS = "..."

# --- the procedure the method will design ---
def solve(question):
    # TODO: design the prompting strategy
    pass
```
