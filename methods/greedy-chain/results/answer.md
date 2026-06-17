# ReAct as `greedy_chain`

ReAct ("Reason + Act") prompts a frozen large language model to solve a task by generating, in one
interleaved trajectory, both free-form reasoning traces ("thoughts") and task-specific actions.
The `greedy_chain` baseline is the single-chain inference loop: greedy decoding, no tree search,
no backtracking. A thought is an action in the space of language that has no effect on the
environment and returns no observation; it just composes information and updates the context for
the next step. Real actions touch the environment and return observations. Interleaving the two
lets each fix the other's blind spot: reasoning keeps the acting on-plan and synthesizes results
(reason-to-act), while acting pulls external facts into the context so the reasoning stays grounded
(act-to-reason). It runs as few-shot prompting over a frozen model, with no training.

## Problem it solves

A single few-shot policy over a frozen LLM that handles both knowledge-intensive reasoning tasks
and long-horizon interactive tasks, without the structural blind spots of either ingredient.
Reasoning-only prompting (chain-of-thought) is a closed box: ungrounded, it hallucinates facts
it lacks and propagates them through the chain. Acting-only prompting touches the world but has
no place to think between actions, so it cannot decompose goals, track subgoals, or synthesize a
final answer from gathered observations.

## Key idea

Augment the agent's action space from `A` to `A_hat = A ∪ L`, where `L` is the space of language.

- A normal action `a_t ∈ A` executes in the environment and yields an observation.
- A language action ("thought") `a_hat_t ∈ L` has **no** environment effect and yields **no**
  observation; it reasons over the current context `c_t` and updates it to
  `c_{t+1} = (c_t, a_hat_t)`. Thoughts can decompose the goal, inject commonsense, do small
  arithmetic, react to an observation, reformulate a failed retrieval, or state the final answer.

Because `L` is unbounded, the policy is a **frozen large language model prompted few-shot** with
human-written trajectories of interleaved thought/action/observation steps — the strong language
prior makes the augmented action space tractable; no gradient updates or reward are used.

- **Dense vs. sparse thoughts.** For reasoning-heavy tasks (multi-hop QA, fact verification),
  interleave a thought before each action. For long-horizon decision tasks, write exemplars with
  *sparse* thoughts and let the model decide when to think vs. act, so reasoning sentences don't
  bloat the trajectory past the in-context window.
- **Deliberately weak actions.** The QA environment exposes only `search[entity]` (first few
  sentences of an exactly-named page, or similar entity names if missing), `lookup[string]` (next
  sentence containing the string, like Ctrl-F), and `finish[answer]`. Keeping retrieval crude
  forces it to be *driven by* explicit reasoning rather than by a strong retriever doing the
  thinking.
- **Greedy single chain.** One trajectory, greedy decoding, stop on `Finish` or at a fixed task
  step cap; if the cap is hit, close the episode with `finish[]`. Each step is one model call
  producing thought + action that stops *before* the observation slot, so the model can never
  fabricate an observation; the environment fills it in.
- **Parse fallback.** If the thought/action continuation does not contain the expected
  `Action i:` split, keep the first line as the thought and make a second, action-only call that
  stops at the newline.

## The execution loop

```
context  <-  INSTRUCTION + few-shot interleaved exemplars + initial observation
for step i = 1 .. max_steps:
    text             <-  one greedy LLM call on context + "Thought i:", stopping before "Observation i:"
    thought, action  <-  split text on "\nAction i: " (fallback: ask for Action i alone)
    obs, done        <-  env.step(action)        # only the action touches the world
    context          <-  context + "Thought i: ... Action i: ... Observation i: <obs>"
    if done: break                                # the model emitted Finish[...]
if not done: env.step("finish[]")                 # close at the cap
```

## Working code

Filling the per-step policy slot of the agent harness with the single greedy chain:

```python
def llm(prompt, stop=None):
    """Greedy completion from a frozen large language model, temperature 0,
    halting when any string in `stop` is produced. (e.g. text-davinci-002, PaLM-540B.)"""
    ...


# Names the task and the augmented action space (Search / Lookup / Finish for the
# Wikipedia QA environment). FEWSHOT_EXEMPLARS are human trajectories interleaving
# Thought / Action / Observation -- dense for reasoning tasks, sparse (model-decided)
# for long-horizon decision tasks.
INSTRUCTION = (
    "Solve a question answering task with interleaving Thought, Action, Observation "
    "steps. Thought can reason about the current situation, and Action can be three "
    "types:\n"
    "(1) Search[entity] -- returns the first sentences of the entity's wiki page, or "
    "similar entity names if it does not exist.\n"
    "(2) Lookup[keyword] -- returns the next sentence on the current page containing "
    "keyword.\n"
    "(3) Finish[answer] -- returns the answer and finishes the task.\n"
    "Here are some examples.\n"
)
FEWSHOT_EXEMPLARS = "..."   # a few human-written interleaved trajectories for this task


def greedy_chain(env, max_steps=7, idx=None):
    """Single greedy chain alternating model-generated Thought+Action with
    real environment Observations -- one trajectory, no backtracking."""
    question = env.reset(idx=idx)
    prompt = INSTRUCTION + FEWSHOT_EXEMPLARS + question + "\n"
    n_calls, n_badcalls = 0, 0
    done = False
    reward, info = 0, {}

    for i in range(1, max_steps + 1):
        n_calls += 1
        # One call -> this step's Thought AND Action; stop BEFORE the observation slot
        # so the model cannot invent an observation -- the environment provides it.
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except ValueError:
            # No clean Action emitted: keep the thought, ask narrowly for the action.
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split("\n")[0]
            action = llm(
                prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]
            ).strip()

        # Only the action touches the environment; the thought was an inert language
        # action that merely updated the context.
        obs, reward, done, info = env.step(action[0].lower() + action[1:])
        obs = obs.replace("\\n", "")

        # Fold Thought, Action and the REAL Observation back in for the next step.
        prompt += (
            f"Thought {i}: {thought}\n"
            f"Action {i}: {action}\n"
            f"Observation {i}: {obs}\n"
        )
        if done:                       # model emitted Finish[...]
            break

    if not done:                       # hit the cap without finishing -> close the episode
        obs, reward, done, info = env.step("finish[]")

    info.update({"n_calls": n_calls, "n_badcalls": n_badcalls, "traj": prompt})
    return reward, info
```

## Relation to prior methods

- **Chain-of-thought** = ReAct with no actions and no observations: an internal reasoning chain
  only. Fluent but ungrounded, so it hallucinates.
- **Self-consistency (CoT-SC)** = chain-of-thought with sample-many-then-majority-vote decoding.
  Improves CoT but adds no external information, so it cannot fix missing knowledge.
- **WebGPT-style browser acting** uses a model to navigate web pages and answer questions, but it
  learns from imitation and human-feedback reinforcement and does not expose a first-class
  thinking step in the trajectory.
- **SayCan-style affordance grounding** lets the language model score which skill helps the
  instruction, then reranks by a learned value / affordance estimate
  `p(skill succeeds | state)`. This grounds feasible robot skills, but the model is still choosing
  among skills rather than maintaining a free-form working memory.
- **Inner Monologue-style closed-loop feedback** re-injects environment feedback (success
  detection, scene description, human replies) into the prompt; ReAct instead injects the model's
  own free-form reasoning as a first-class language action interleaved with acting.
