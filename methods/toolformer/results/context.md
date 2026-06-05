# Context

## Research question

A large language model is, paradoxically, both very strong and embarrassingly weak. It solves new tasks from a handful of examples or a plain instruction, yet it stumbles on things a pocket calculator or a database lookup does perfectly: precise arithmetic, current facts, the date, low-resource translation, anything that depends on information not frozen into its weights at pretraining time. Scaling helps a little but cannot close these gaps — a bigger model still cannot know today's date or add two numbers reliably.

The obvious remedy is to let the model call external tools — a calculator, a search engine, a translation system, a calendar — and splice their outputs into its generation. The hard part is *how the model learns to do this*. Existing routes are unsatisfying. One route annotates large amounts of human data showing when and how to call tools, which is expensive and, worse, encodes *human* judgments of which calls are useful rather than what the *model* actually finds useful. The other route hand-builds a task-specific prompt that tells the model, for one task, which tool to use and how — which doesn't generalize and defeats the point of a general model. What a solution would have to deliver: the model decides for *itself* when a tool is worth calling, which tool, and what arguments to pass; it learns this with essentially no human annotation; and it loses none of its general language-modeling ability in the process.

## Background

**In-context learning and self-generated data.** A large LM, shown a few demonstrations in its prompt, will imitate the pattern on new inputs. This is strong enough that the model can be used to *generate entire datasets*: give it a handful of human-written examples of some annotation and let it annotate a large corpus itself. This bootstrapping idea — the model labels data that is then used to train the model — is the lever that makes "no human annotation" plausible.

**Self-training / bootstrapping.** Across NLP (word-sense disambiguation, relation extraction, parsing, retrieval, reasoning) there is a long tradition of training a model on its own predictions after a filtering step that keeps only the high-quality ones. The recurring design question is the *filter*: what objective decides which self-generated examples are good enough to train on? A good filter must be cheap, automatic, and aligned with the end goal.

**Augmenting LMs with text during pretraining.** Various methods feed extra textual signal into pretraining — metadata, HTML/markup, or passages fetched by a retriever. In all of these the extra information is provided *unconditionally*, whether or not it helps at that position. None of them lets the model itself decide, token by token, that it wants a particular piece of information right here.

**Tool use as text.** If a tool's input and output can both be written as text, then a tool call can be *inlined into a text sequence* with special marker tokens, and calling the tool is just decoding text up to a marker, pausing to run the tool, and pasting the result back in. This reframes "use a tool" as "predict the right text," which means an ordinary language-modeling objective can, in principle, supervise it — if only we had training sequences with tool calls inserted in the right places.

**Existing tool-use systems (the immediate prior art).** Some systems learn tool use from heavy human supervision (web-browsing agents, dialog systems trained to issue API calls). Others prompt the model in a few-shot, task-specific way where it is known in advance which tool to use. A closely related self-supervised approach teaches a calculator and a search engine via a similar objective, but only in the setting where the model is then finetuned for specific downstream tasks rather than learning general, task-agnostic tool use.

## Baselines

A method that learns general tool use would be measured against these.

**Vanilla pretrained LM.** A capable autoregressive model used zero-shot. Gap: cannot access fresh or external information, cannot compute precisely, hallucinates facts — exactly the systematic weaknesses tool use targets.

**Same LM, further finetuned on plain text (no API calls).** Controls for the effect of additional finetuning on the in-domain corpus. Gap: still has no tool access; isolates how much of any improvement is just "more training" versus tool use.

**Human-supervised tool use.** Train the model to call tools from large collections of human demonstrations of when/how to call them. Gap: costly to annotate, and it bakes in human notions of usefulness, which may differ from what helps the model predict.

**Task-specific few-shot tool prompting.** Provide, per task, in-context examples showing which tool to call and how to solve that concrete task with it. Gap: not general — it presupposes the user knows in advance which tool the task needs; it does not let the model decide for itself across arbitrary inputs.

**Much larger general LMs.** Far bigger models used zero-shot as a strong reference point — the question being whether a small model with tools can match a large model without them.

## Evaluation settings

Natural yardsticks are zero-shot tasks where at least one tool should help, plus a check that core language modeling is unharmed and a study of how the effect scales with model size. (Datasets and metrics here are pre-existing facts; no outcomes are part of this setup.)

- **Factual completion (LAMA: SQuAD/Google-RE/T-REx subsets).** Complete a short statement with a missing fact; left-to-right examples only. A lenient metric (correct word among the first few predicted words) accounts for zero-shot phrasing. A question-answering tool is the relevant aid.
- **Mathematical reasoning (ASDiv, SVAMP, MAWPS).** Math word problems; the answer is a number, so the metric checks the first predicted number. A calculator is the relevant aid.
- **Open-domain QA (WebQuestions, Natural Questions, TriviaQA).** Check whether the first ~20 predicted words contain the answer. A search engine is the relevant aid.
- **Multilingual QA (MLQA).** Answer (in English) a question posed in one of several languages about an English paragraph. A translation tool is the relevant aid.
- **Temporal QA (TempLAMA, and a date-arithmetic dataset).** Questions whose answers depend on the current date. A calendar is the relevant aid.
- **Language modeling (WikiText, held-out corpus subset).** Perplexity, to confirm tool-augmented finetuning does not degrade plain language modeling when tools are off.
- **Scaling.** Apply the method across a family of model sizes to see at what scale tool use begins to help.

The setup is strictly zero-shot for downstream tasks (instruction in natural language, no in-context examples). Base model: a ~6–7B-parameter autoregressive LM; the language-modeling corpus used both to bootstrap tool-call data and to finetune is a subset of a large web corpus. Tools span a question-answering system (a retrieval-augmented LM), a calculator (four basic ops, rounded to two decimals), a Wikipedia search engine (BM25 over a Wikipedia dump), a translation system (a multilingual MT model with automatic source-language detection, target always English), and a calendar (returns the current date, no input).

## Code framework

Pre-method primitives that already exist: an autoregressive LM exposing per-token probabilities/logits, a tokenizer, and a set of callable tools mapping a text input to a text output. The contribution will fill in (1) how tool-call markers are linearized into text, (2) how candidate calls are sampled, (3) the **filter** that decides which calls to keep, and (4) the inference-time interruption that runs a call and pastes the result back.

```python
import torch

# --- LM with per-token probabilities (exists) ---
def model(token_ids):            # -> logits (batch, seq, vocab)
    ...

# --- tools: text in, text out (exist) ---
def call_tool(name, args):       # e.g. Calculator, QA, WikiSearch, MT, Calendar
    ...                          # returns a single text string r

# --- linearizing a call into text with special markers (TODO: the method defines e(c), e(c,r)) ---
API_START, API_END, SEP = "<API>", "</API>", "->"
def linearize(call, response=None):
    pass

# --- step 1: sample candidate calls in a passage (TODO) ---
def sample_api_calls(x):         # pick positions; sample call candidates at each
    pass

# --- step 2: the filter that keeps only helpful calls (TODO) ---
def filter_calls(x, candidates):
    # decide which (position, call, response) actually help predict the
    # following tokens, and drop the rest
    pass

# --- step 3: build the augmented corpus and finetune on it (standard LM objective) ---
def make_augmented_dataset(corpus):
    pass

# --- step 4: inference — interrupt decoding to run a call, paste result, continue (TODO) ---
def generate_with_tools(prompt):
    pass
```
