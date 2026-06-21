# Context

## Research question

A pretrained language model trained only on next-token prediction is not, out of the box, good at *following instructions* — given "Write an essay about school safety" or "Classify the sentiment of this sentence," a vanilla LM does not reliably produce the intended response. The recipe that fixes this is *instruction tuning*: fine-tune the LM on a collection of (instruction, input, output) examples spanning many tasks, after which the model generalizes to follow *new* instructions it never saw in training. Two ingredients power this: a large pretrained LM, and a large, diverse corpus of human-written instruction data.

The question is how to build and expand that instruction corpus. Human-written instruction data is collected through manual annotation, and annotators tend to write tasks in domains familiar to them — classification, QA, NLI. Generalization to genuinely unseen tasks is empirically tied to the *size and diversity* of the instruction data. Can a language model itself be used as a generator of instruction data, given a small human-authored seed, so that fine-tuning the model on that generated data improves its instruction-following ability?

## Background

**Instruction tuning and its data dependence.** Vanilla LMs become effective instruction-followers when fine-tuned on datasets of natural-language instructions paired with desired outputs (Mishra et al., 2022; Wei et al., 2022 (FLAN); Sanh et al., 2022 (T0); Wang et al., 2022 (Super-NaturalInstructions)). These works establish a direct correlation: the broader and more diverse the instructional data, the better the model generalizes to unseen tasks. The data is human-annotated — PromptSource and Super-NaturalInstructions are large hand-built collections.

**What an "instruction task" is.** A task t is defined by an instruction I_t in natural language and has n_t ≥ 1 input-output instances {(X_{t,i}, Y_{t,i})}. The model should satisfy M(I_t, X_{t,i}) = Y_{t,i}. The instruction/input boundary is soft: "Write an essay about school safety" can be a self-contained instruction (empty input X), or be split into instruction "Write an essay about the following topic" + input "school safety." Allowing empty inputs widens the format diversity.

**LMs as few-shot generators.** Large LMs (GPT-3, the davinci engine, 175B params) exhibit strong in-context learning: prompt them with a few demonstrations of a pattern and they continue the pattern. This is the engine that could, in principle, be turned on the *meta-task* of producing instruction data — given a few example tasks in the prompt, generate more tasks; given an instruction plus examples of solving tasks, generate the input/output for it.

**Prior art the method reacts to / resembles.**
- **LM-based data generation/augmentation** (Schick & Schütze, 2021; Wang et al., 2021 (WANLI); Meng et al., 2022): use LMs to synthesize training data for a specific, pre-defined task (QA, NLI). They populate an existing task; they do not invent new task *definitions*.
- **Instruction generation from examples** (Zhou et al., 2022; Honovich et al., 2022): produce an instruction *given* a few examples of one task — task-specific reverse-engineering of instructions.
- **Self-training** (He et al., 2019; Xie et al., 2020): a model labels unlabeled data and retrains on its own labels, within a fixed *target task* and a pool of unlabeled *examples* under it.
- **Knowledge distillation** (Hinton et al., 2015): transfer knowledge from a teacher to a (usually smaller) student.
- **InstructGPT** (Ouyang et al., 2022): builds a general instruction-follower via human-collected demonstrations and feedback; the role of the *data* is understudied because the data is private.
- **Concurrent: Unnatural Instructions** (Honovich et al., 2022): generates instruction data with GPT-3, seeded from Super-NaturalInstructions tasks and generated with an *already instruction-tuned* model (text-davinci-002).

## Baselines

- **Off-the-shelf vanilla LMs (GPT-3 "davinci", T5-LM).** Pretraining only, no instruction tuning.
- **Public instruction-tuned models (T0, Tk-Instruct; 11B).** Fine-tuned from T5 on human-built instruction collections (PromptSource, Super-NaturalInstructions).
- **InstructGPT (text-davinci-001/002/003).** General instruction-followers from human demonstrations and feedback.
- **Same-model fine-tuned on existing public data (GPT-3 fine-tuned on T0 / Super-NI data).** Controls for the base model, isolating the effect of the *data source*.

## Evaluation settings

- **Generation engine.** The largest GPT-3 ("davinci", 175B) accessed via API, used both to generate the data and (fine-tuned) as the resulting model.
- **Seed.** 175 manually written seed tasks (1 instruction + 1 instance each), authored without reference to existing datasets or the test tasks; 25 classification, 150 non-classification.
- **Held-out evaluation.** (1) Super-NaturalInstructions (Super-NI) test set — typical NLP tasks, *zero-shot* generalization to unseen tasks, automatic metric (ROUGE-L). (2) A separately curated set of 252 expert-written *user-oriented* instructions for novel, realistic usages, scored by human evaluation (a 4-level rating: correct / acceptable-with-minor-imperfections / responds-to-some-but-wrong / irrelevant-or-invalid). Seed-to-test ROUGE-L overlap is reported (avg 0.21 vs Super-NI, 0.34 vs user-oriented) to show the seeds don't trivially cover the tests.
- **Diversity / quality diagnostics on the generated data (pre-method facts about a synthetic corpus, not proposed-method outcomes).** Verb–noun structure of instructions via a constituency parser; ROUGE-L overlap of each generated instruction with the seeds; length distributions; an expert correctness audit of a 200-instruction sample.

## Code framework

The pieces that already exist: an API-served pretrained LM that does in-context completion, a tokenizer, a fuzzy text-similarity metric (ROUGE-L) for dedup, and a supervised fine-tuning entry point. The method has to define the bootstrapping loop — how to prompt for new instructions, how to prompt for instances, how to filter — and the fine-tuning format.

```python
import random

class LM:
    def complete(self, prompt, **decode_kwargs):
        # in-context completion from a pretrained (untuned) LM
        pass

def rouge_l(a, b):
    # fuzzy overlap between two strings, for dedup
    pass

# --- Bootstrapping loop over a growing task pool ---
def generate_instructions(task_pool, lm):
    # TODO: prompt the LM with sampled existing instructions to produce NEW instructions
    pass

def is_classification(instruction, lm):
    # TODO: do classification vs non-classification tasks need different instance generation?
    pass

def generate_instances(instruction, is_clf, lm):
    # TODO: produce (input, output) instances for an instruction;
    #       which generation ORDER avoids biased data?
    pass

def filter_and_postprocess(new_task, task_pool):
    # TODO: novelty/dedup/validity heuristics before adding to the pool
    pass

def bootstrap(seed_tasks, lm, target_size):
    task_pool = list(seed_tasks)
    while len(task_pool) < target_size:
        instr = generate_instructions(task_pool, lm)
        clf = is_classification(instr, lm)
        instances = generate_instances(instr, clf, lm)
        # TODO: filter, then add valid tasks back to the pool
    return task_pool

# --- Fine-tune the SAME LM on its own generated data ---
def finetune_to_follow_instructions(lm, generated_tasks):
    # TODO: format (instruction, input) -> output as supervised pairs; many templates
    pass
```
