# Context: prompting a frozen LLM to solve interactive, knowledge-seeking tasks (circa 2021-2022)

## Research question

A frozen large language model has become a surprisingly capable general-purpose engine: prompt
it with a few examples and it will continue the pattern. Two distinct uses of this ability have
grown up side by side. In one, the model is asked to *reason* — to write out a chain of
intermediate steps before committing to an answer to an arithmetic, commonsense, or symbolic
problem. In the other, the model is asked to *act* — to emit, step by step, the domain-specific
actions that drive an external system: a web browser, a robot's skill library, a text-adventure
game, a shopping site. These have been pursued as separate research topics, with separate
methods and benchmarks.

The question taken up here is whether a single policy, runnable as few-shot prompting over a
frozen model, can solve both knowledge-intensive reasoning tasks and long-horizon interactive
tasks.

## Background

The field state: scaling pretrained language models has produced systems with emergent few-shot
abilities. Two lines are directly load-bearing here.

**Reasoning by prompting.** Large models can be coaxed into multi-step reasoning by showing
them, in the prompt, a handful of worked examples whose answers include the intermediate steps,
not just the final result. This reasoning is produced from the model's *internal* representations
and is read off the model's own continuation.

**Acting by prompting in interactive environments.** A frozen model can also serve as a policy:
convert the environment's observation into text, let the model generate the next domain action
(or a plan of actions), execute it, repeat. This line had begun to add *feedback* — piping the
environment's textual state back into the prompt so the model can adjust — and to *ground* the
model's outputs against the environment so it only proposes feasible actions.

Two background frames make the setting precise. First, the **agent-environment loop**: at step
`t` the agent receives an observation `o_t` from an observation space `O`, and takes an action
`a_t` from an action space `A` according to a policy `pi(a_t | c_t)`, where the context
`c_t = (o_1, a_1, ..., o_{t-1}, a_{t-1}, o_t)` is the full interaction history so far. Solving a
task is choosing actions so the trajectory reaches a goal. The mapping from context to action,
`c_t -> a_t`, can be highly implicit and computationally heavy — deciding the final action of a
multi-hop question requires reasoning over the entire question-and-evidence trajectory at once.
Second, the **few-shot in-context learning** frame: a frozen pretrained model conditioned on a
few demonstrations in its prompt will imitate their form on a new instance, with no gradient
updates. This is what makes "just write the right kind of demonstration" a viable way to install
a new behavior, and what makes annotation cost — rather than training cost — the binding
constraint.

## Baselines

The prior methods a new approach would be measured against and would respond to.

**Chain-of-thought prompting (Wei et al. 2022).** Few-shot exemplars whose answers spell out a
series of intermediate reasoning steps before the final answer; e.g. for an arithmetic word
problem the demonstration walks through the subtotals and then states the result. On sufficiently
large models this elicits a "thinking procedure" over arithmetic, commonsense, and symbolic
tasks. The chain is generated and consumed entirely inside the model.

**Self-consistency (Wang et al. 2022).** Keep chain-of-thought exemplars but replace greedy
decoding of a single chain with a sample-and-marginalize procedure: sample many diverse chains
under stochastic decoding, let each produce its own final answer, and return the majority
answer. This consistently improves chain-of-thought by exploiting that a correct answer is
often reachable by several reasoning paths.

**Zero-shot reasoning prompting (Kojima et al. 2022).** Appending a trigger phrase such as
"Let's think step by step" elicits a reasoning chain with no exemplars at all.

**Acting-only / action-generation policies (Nakano et al. 2021; Ahn et al. 2022; Huang et al.
2022).** Use the model as a policy that emits domain actions in an interactive environment.
WebGPT (Nakano et al. 2021) has a model operate a web browser to answer long-form questions,
training the policy with imitation and reinforcement learning from human feedback. SayCan
(Ahn et al. 2022) lets the model score how much each candidate skill advances the instruction
and multiplies that by a learned value/affordance function `p(skill succeeds | state)`, grounding
the model's choice in what the robot can actually do from the current state. Inner Monologue
(Huang et al. 2022) closes the loop by feeding textual environment feedback — success detection,
scene descriptions, human replies — back into the prompt so the model can replan.

## Evaluation settings

The natural yardsticks already in use, across both task families:

- **HotpotQA** (Yang et al. 2018): multi-hop question answering that requires combining facts
  from two or more Wikipedia passages. Run here in a *question-only* setup — the model gets the
  question with no supporting paragraphs and must rely on internal knowledge or on retrieval.
  Metric: exact match.
- **FEVER** (Thorne et al. 2018): fact verification; each claim is labeled SUPPORTS, REFUTES, or
  NOT ENOUGH INFO depending on whether a Wikipedia passage verifies it. Claim-only setup. Metric:
  accuracy.
- A **Wikipedia interaction API** as the QA environment, deliberately weak so that retrieval
  forces explicit reasoning rather than leaning on a strong neural retriever:
  `search[entity]` (returns the first few sentences of the entity's page, or the top similar
  entity names if the page is missing), `lookup[string]` (returns the next sentence on the
  current page containing the string, like Ctrl-F), and `finish[answer]` (ends the task).
- **ALFWorld** (Shridhar et al. 2020): a synthetic text game aligned to embodied household tasks,
  where a goal (e.g. "examine an object under a desklamp") needs navigation and object interaction over
  a long horizon. Metric: success rate.
- **WebShop** (Yao et al. 2022): an online-shopping environment with a large real-product catalog;
  the agent must buy a product matching a natural-language instruction via search and
  option-selection actions. Metrics: average attribute score and success rate.
- Backbone: a frozen instruction-capable large language model (PaLM-540B; also GPT-3
  `text-davinci-002`), prompted few-shot with greedy decoding. Prior systems compared as baselines
  include standard prompting, the reasoning baselines above, the acting-only policies above, and
  task-specific imitation / RL agents (e.g. BUTLER on ALFWorld; an imitation and an
  imitation-plus-RL agent on WebShop).

## Code framework

The substrate is the generic agent-environment harness used by the acting-only baselines, plus
the few-shot prompting machinery. An environment exposes `reset()` and a
`step(action)` that parses a domain action, executes it, and returns the next observation and a
done flag (the Wikipedia API above is one such environment). A frozen language model is wrapped in
a single `llm(prompt, stop)` call. A policy drives one task instance to completion by repeatedly
asking the model what to do next and feeding the result back. The per-step policy and prompt update
are the open slot.

```python
# --- provided: a frozen LLM behind one call ---
def llm(prompt, stop=None):
    """Greedy completion of `prompt` from a frozen large language model, stopping at `stop`."""
    ...

# --- provided: a text environment for a task family (e.g. the Wikipedia API) ---
class Environment:
    def reset(self, idx=None):
        """Start a task instance; return its initial observation (e.g. the question)."""
        ...

    def step(self, action):
        """Parse and execute one domain action string; return (observation, reward, done, info)."""
        ...

# --- provided: few-shot exemplars and a task instruction, concatenated into the prompt ---
FEWSHOT_EXEMPLARS = "..."   # human-written demonstrations of solving instances of this task
INSTRUCTION = "..."         # natural-language description of the task and the available actions


def solve(env, max_steps, idx=None):
    """Drive the frozen LLM to solve one task instance, one step at a time.

    The prompt starts as INSTRUCTION + FEWSHOT_EXEMPLARS + the initial observation.
    At each step the policy must decide what the model emits and how the running
    context is updated before the next call.
    """
    observation = env.reset(idx=idx)
    prompt = INSTRUCTION + FEWSHOT_EXEMPLARS + observation + "\n"
    for step in range(max_steps):
        # TODO: fill in the task policy.
        pass
    return prompt
```

The outer loop hands over a frozen model, a parsing environment, and a growing prompt; the single
empty slot is the per-step policy.
