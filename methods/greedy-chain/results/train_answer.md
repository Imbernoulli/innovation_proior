The problem is to make a single frozen large language model solve both knowledge-heavy reasoning tasks and long interactive tasks with only a prompt. Reasoning-only prompting gives the model a chain of intermediate steps, which lets it decompose problems and synthesize answers, but the chain is generated entirely inside the model. When it needs a fact it does not hold, it hallucinates one, and because every later step conditions on the text already written, the mistake propagates. Sampling many chains and voting helps when the model knows the facts but takes an unlucky reasoning path, but it adds no new external information, so it cannot fix missing knowledge. Acting-only prompting lets the model drive an environment step by step, but with no place to think between actions it cannot decompose goals, track subgoals, or synthesize a final answer from gathered observations. Even the closest prior work only feeds environment feedback back into the prompt, not a free-form working memory.

The fix is to stop treating reasoning and acting as separate modes and instead let the model emit both in one interleaved trajectory. ReAct augments the agent's action space from A to A_hat = A ∪ L, where L is ordinary language. A normal action a_t ∈ A executes in the environment and returns an observation. A language action, called a thought, has no environment effect and returns no observation; it merely composes information and updates the running context. Because thoughts are just tokens, they can decompose the goal, inject commonsense, do small arithmetic, react to an unexpected observation, reformulate a failed retrieval, or state the final answer. The action space is unbounded, so the policy is implemented as few-shot prompting of a frozen pretrained model with human-written trajectories that interleave Thought, Action, and Observation steps. The strong language prior makes the space of possible thoughts tractable without any gradient updates or reward signal.

ReAct covers both failure modes at once. Thoughts induce, track, and adjust the plan and finally synthesize the answer, while actions pull real external facts into the context so the reasoning stays grounded. The density of thoughts is chosen per task: for multi-hop question answering or fact verification, thoughts are interleaved densely before each action; for long-horizon decision tasks, exemplars use sparse thoughts and let the model decide when to think versus act, so the context window is not bloated by reasoning about every primitive move. The actions themselves are kept deliberately weak, for example a Wikipedia API with only search, lookup, and finish, so retrieval must be driven by explicit reasoning rather than by a strong retriever doing the thinking.

The inference loop is a single greedy chain with no tree search and no backtracking. At each step the model is prompted with the running context plus "Thought i:" and generates the thought and the next action together, stopping before "Observation i:". The implementation splits the continuation into thought and action, executes only the action against the environment, and pastes the real observation back into the prompt. If the continuation does not contain a clean "Action i:" split, the first line is kept as the thought and a second narrow call prompts explicitly for the action. The loop terminates when the model emits Finish[...] or when a fixed step cap is reached, in which case the episode is closed with finish[].

```python
def llm(prompt, stop=None):
    """Greedy completion from a frozen large language model, temperature 0,
    halting when any string in `stop` is produced."""
    ...


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
FEWSHOT_EXEMPLARS = "..."   # human-written interleaved Thought/Action/Observation trajectories


def greedy_chain(env, max_steps=7, idx=None):
    """ReAct single greedy chain: alternate model-generated Thought+Action with
    real environment Observations, one trajectory, no backtracking."""
    question = env.reset(idx=idx)
    prompt = INSTRUCTION + FEWSHOT_EXEMPLARS + question + "\n"
    n_calls, n_badcalls = 0, 0
    done = False
    reward, info = 0, {}

    for i in range(1, max_steps + 1):
        n_calls += 1
        # Generate this step's Thought and Action; stop before the Observation slot
        # so the model can never fabricate an observation.
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except ValueError:
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split("\n")[0]
            action = llm(
                prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]
            ).strip()

        # Only the action touches the environment; the thought is an inert
        # language action that updates the context.
        obs, reward, done, info = env.step(action[0].lower() + action[1:])
        obs = obs.replace("\\n", "")

        # Fold Thought, Action, and the real Observation back into the context.
        prompt += (
            f"Thought {i}: {thought}\n"
            f"Action {i}: {action}\n"
            f"Observation {i}: {obs}\n"
        )
        if done:                       # model emitted Finish[...]
            break

    if not done:                       # hit the step cap without finishing
        obs, reward, done, info = env.step("finish[]")

    info.update({"n_calls": n_calls, "n_badcalls": n_badcalls, "traj": prompt})
    return reward, info
```
