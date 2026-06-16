ReAct interleaves reasoning and acting in one model. Large LMs can reason by producing multi-step chains of thought, and they can act as policies in interactive environments, but the two capabilities are usually separated. Reasoning-only chains are sealed internal computations: they hallucinate facts and propagate early errors with no way to verify or repair. Acting-only policies emit actions from an implicit context→action map with no place to plan, track progress, handle exceptions, or synthesize a final answer from what they gathered.

The key move is to augment the action space with language: Â = A ∪ L, where L is free-form text. A normal action a_t ∈ A is executed in the environment and returns an observation. A language action — a *thought* or reasoning trace — does not affect the environment and yields no observation; it only reasons over the current context c_t and updates it, c_{t+1} = (c_t, â_t). Thoughts decompose goals, inject commonsense, extract information from observations, track progress, handle exceptions, and reformulate queries; actions fetch external information. Reasoning guides acting; acting grounds reasoning.

Because L is unlimited, the policy is not trained — a frozen large LM is steered with a handful (1–6) of hand-written exemplar trajectories (thought/action/observation sequences); the annotator just writes their thoughts on top of their actions, with no special format. The cadence of thoughts is task-dependent:
- **Reasoning-heavy tasks** (multi-hop QA, fact verification): dense interleave — a thought before nearly every action (thought → action → observation, repeated).
- **Action-heavy tasks** (text games, web navigation): sparse thoughts — let the model decide where a thought is worth inserting (plan-setting, progress, exceptions), and otherwise act.

For knowledge tasks the environment is a deliberately simple Wikipedia API with three actions: **search[entity]** (first five sentences of the page, or similar entity names if absent), **lookup[string]** (next sentence containing the string, like Ctrl-F), **finish[answer]**. Keeping it weaker than a real retriever forces retrieval to be driven by explicit reasoning.

The two paths can also back each other up. Grounded acting reduces hallucination by tying claims to observations, but a bad search can derail it; free reasoning is flexible but ungrounded. A back-off heuristic runs the interleaved trajectory and falls back to self-consistency CoT when it fails to answer within the step budget (seven turns for HotpotQA, five for FEVER), or runs self-consistency first and backs off to the grounded trajectory when the majority answer appears in fewer than half of the sampled traces.

The interaction loop generates a thought and the action it implies together each turn, stops before the observation line (the observation belongs to the environment), executes the action, appends the real observation, and repeats:

```python
import os, openai, requests
import wikienv, wrappers

env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def llm(prompt, stop=["\n"]):
    response = openai.Completion.create(
        model="text-davinci-002", prompt=prompt, temperature=0,
        max_tokens=100, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=stop)
    return response["choices"][0]["text"]

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first five sentences if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples   # hand-written thought/action/observation exemplars

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            n_badcalls += 1; n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        prompt += f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info
```

An example dense ReAct trajectory (HotpotQA): each Thought reasons over what is known and decides the next Action; the Wikipedia API returns each Observation.

```
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area it extends into, then find the elevation range of that area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions.
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]
```
