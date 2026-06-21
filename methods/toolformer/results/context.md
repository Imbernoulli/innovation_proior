# Context

## Research question

A large language model solves new tasks from a handful of examples or a plain instruction, yet it struggles on things a pocket calculator or a database lookup does perfectly: precise arithmetic, current facts, the date, low-resource translation, anything that depends on information not frozen into its weights at pretraining time. The question is how to let a language model call external tools — a calculator, a search engine, a translation system, a calendar — and splice their outputs into its generation.

## Background

**In-context learning and self-generated data.** A large LM, shown a few demonstrations in its prompt, will imitate the pattern on new inputs. This is strong enough that the model can be used to *generate entire datasets*: give it a handful of human-written examples of some annotation and let it annotate a large corpus itself. This bootstrapping idea — the model labels data that is then used to train the model — is a lever for reducing reliance on human annotation.

**Self-training / bootstrapping.** Across NLP (word-sense disambiguation, relation extraction, parsing, retrieval, reasoning) there is a long tradition of training a model on its own predictions after a filtering step that keeps only the high-quality ones. The recurring design question is the *filter*: what objective decides which self-generated examples are good enough to train on? A good filter must be cheap, automatic, and aligned with the end goal.

**Augmenting LMs with text during pretraining.** Various methods feed extra textual signal into pretraining — metadata, HTML/markup, or passages fetched by a retriever. In all of these the extra information is provided *unconditionally*, whether or not it helps at that position.

**Tool use as text.** If a tool's input and output can both be written as text, then a tool call can be *inlined into a text sequence* with special marker tokens, and calling the tool is just decoding text up to a marker, pausing to run the tool, and pasting the result back in. This reframes "use a tool" as "predict the right text," which means an ordinary language-modeling objective can, in principle, supervise it — if only we had training sequences with tool calls inserted in the right places.

**Existing tool-use systems (the immediate prior art).** Some systems learn tool use from heavy human supervision (web-browsing agents, dialog systems trained to issue API calls). Others prompt the model in a few-shot, task-specific way where it is known in advance which tool to use. A closely related self-supervised approach teaches a calculator and a search engine via a similar objective, in the setting where the model is then finetuned for specific downstream tasks.

## Baselines

**Vanilla pretrained LM.** A capable autoregressive model used zero-shot.

**Same LM, further finetuned on plain text (no API calls).** Controls for the effect of additional finetuning on the in-domain corpus.

**Human-supervised tool use.** Train the model to call tools from large collections of human demonstrations of when/how to call them.

**Task-specific few-shot tool prompting.** Provide, per task, in-context examples showing which tool to call and how to solve that concrete task with it.

**Much larger general LMs.** Far bigger models used zero-shot as a strong reference point — the question being whether a small model with tools can match a large model without them.

## Evaluation settings

Natural yardsticks are zero-shot tasks where at least one tool should help, plus a check that core language modeling is unharmed and a study of how the effect scales with model size.

- **Factual completion (LAMA: SQuAD/Google-RE/T-REx subsets).** Complete a short statement with a missing fact; left-to-right examples only. A lenient metric (correct word among the first few predicted words) accounts for zero-shot phrasing. A question-answering tool is the relevant aid.
- **Mathematical reasoning (ASDiv, SVAMP, MAWPS).** Math word problems; the answer is a number, so the metric checks the first predicted number. A calculator is the relevant aid.
- **Open-domain QA (WebQuestions, Natural Questions, TriviaQA).** Check whether the first ~20 predicted words contain the answer. A search engine is the relevant aid.
- **Multilingual QA (MLQA).** Answer (in English) a question posed in one of several languages about an English paragraph. A translation tool is the relevant aid.
- **Temporal QA (TempLAMA, and a date-arithmetic dataset).** Questions whose answers depend on the current date. A calendar is the relevant aid.
- **Language modeling (WikiText, held-out corpus subset).** Perplexity, to confirm tool-augmented finetuning does not degrade plain language modeling when tools are off.
- **Scaling.** Apply the method across a family of model sizes to see at what scale tool use begins to help.

The setup is strictly zero-shot for downstream tasks (instruction in natural language, no in-context examples). Base model: a 6.7B-parameter autoregressive LM; the language-modeling corpus used both to bootstrap tool-call data and to finetune is a subset of a large web corpus. Tools span a question-answering system (a retrieval-augmented LM), a calculator (four basic ops, rounded to two decimals), a Wikipedia search engine (BM25 over a Wikipedia dump), a translation system (a multilingual MT model with automatic source-language detection, target always English), and a calendar (returns the current date, no input).

## Code framework

Available primitives: an autoregressive LM exposing per-token probabilities/logits, a tokenizer, and a set of callable tools mapping a text input to a text output. The missing pieces are (1) how tool-call markers are linearized into text, (2) how candidate calls are sampled, (3) the filter that decides which executed calls to keep, (4) how surviving calls from all tools are merged into the corpus, and (5) the inference-time interruption that runs a call and pastes the result back.

```python
import torch

# --- LM with per-token probabilities (exists) ---
def model(token_ids):            # -> logits (batch, seq, vocab)
    ...

# --- tools: text in, text out (exist) ---
def call_tool(name, args):       # e.g. Calculator, QA, WikiSearch, MT, Calendar
    ...                          # returns a single text string r

# --- linearizing a call into text with special markers (TODO) ---
API_START, API_END, SEP = "<API>", "</API>", "->"
def linearize(call, response=None):
    pass

# --- step 1: sample candidate calls in a passage (TODO) ---
def sample_api_calls(x):         # pick positions; sample call candidates at each
    pass

# --- step 2: the filter that decides which executed calls to keep (TODO) ---
def filter_calls(x, candidates):
    # decide which (position, call, response) to keep, and drop the rest
    pass

# --- step 3: merge surviving calls from all tools and finetune (standard LM objective) ---
def make_augmented_dataset(corpus):
    pass

# --- step 4: inference — interrupt decoding to run a call, paste result, continue (TODO) ---
def generate_with_tools(prompt):
    pass
```
