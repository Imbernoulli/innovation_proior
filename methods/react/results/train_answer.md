A large language model can do two useful things, but we keep them in separate boxes. Prompted with a few worked exemplars, it produces multi-step reasoning traces — chains of intermediate thoughts that lift accuracy on arithmetic, commonsense, and multi-hop problems. Separately, it can serve as a policy in an interactive environment: read an observation, emit an action, repeat. The trouble is that each box is crippled exactly where the other is strong. A reasoning-only chain is a sealed, closed computation: it can only use what is already inside the weights, so it cannot check a fact, fetch a missing premise, or notice mid-chain that an early step was wrong. That shows up as confident hallucination and as errors that propagate to the answer — a multi-hop chain can be structurally fluent while resting on an unsupported factual bridge, and once the false bridge is written nothing in the trace can repair it. An acting-only policy is the mirror image: at step $t$ it maps the interaction history $c_t = (o_1, a_1, \dots, o_{t-1}, a_{t-1}, o_t)$ to an action under $\pi(a_t \mid c_t)$, and it can gather information, but the moment it has to *use* what it gathered it falls down. It retrieves the right paragraph and then cannot assemble the final answer, because the context-to-action map is left fully implicit and there is no place in the loop to stop, plan, decompose the goal, track progress, handle an exception, or declare that the answer is now in hand. The standard baselines isolate one half each and inherit its gap: standard input-output prompting has no working steps and no external access; chain-of-thought (Wei et al., 2022) reasons in a closed black box grounded in nothing; self-consistency CoT (Wang et al., 2022) samples many such chains and majority-votes, which is more robust but still cannot acquire a single new fact; acting-only prompting interacts with the world but never verbalizes a reasoning step. What we want is the human move — between two physical actions, talk to yourself: track where you are, adjust the plan when you are out of salt, recognize when you need to go look something up. That weave of deliberation into acting is exactly what both boxes lack, and any remedy has to supply it without training a bespoke policy per environment.

I propose ReAct, which interleaves reasoning and acting inside one frozen model by making "thinking" a kind of action. The mechanism is to augment the action space with language itself,
$$\hat{A} = A \cup L,$$
where $L$ is the space of free-form text. A normal action $a_t \in A$ is executed in the environment and returns an observation. But an action that lands in the language part — call it a *thought* — does not touch the environment and yields no observation; its only effect is internal, updating the context that conditions everything that follows:
$$c_{t+1} = (c_t, \hat{a}_t).$$
That single equation is the whole idea, and what I like is how little it adds: no second module, no controller, no new training objective. A thought is just an action whose effect is purely a write to working memory — it composes useful information by reasoning over $c_t$ and writes it back into $c_t$, where it is available to condition every future thought and action. And this immediately closes both failure modes in the *same* loop. The reasoning chain is no longer sealed: a thought can say "I should search for X," the next action searches, the real observation comes back, and a later thought reasons over *that observation* instead of an internal guess — so claims are grounded, not hallucinated, and an early mistake can be caught against the world rather than propagated. The action loop is no longer thoughtless: a thought can decompose the goal, inject a small commonsense or arithmetic step, note an exception, reformulate a failed query, or declare the gathered pieces sufficient. Reasoning guides acting; acting grounds reasoning.

There is a catch I have to respect: $L$ is unlimited — the space of all text — and learning a good policy over an infinite language action space is hopeless without strong language priors and heavy supervision. So I do not train the policy at all. I lean entirely on the prior already baked into a large pretrained LM and steer it the cheap way, with few-shot prompting: a handful (one to six) of hand-written exemplar trajectories — human-authored sequences of thoughts, actions, and observations that solve a task instance — prepended to the context. Seeing how a few trajectories interleave free-form thinking with concrete actions, the frozen model imitates that format on a new instance. Designing these prompts is almost trivial because the annotator just types their own thoughts in plain language on top of the actions they would take; there is no thought taxonomy and no rigid template — a thought is whatever you would actually think. The one design parameter that genuinely matters is *how often* a thought should appear, and here I refuse to hardcode a constant because the task structure pushes back. In multi-hop QA, reasoning is the bottleneck on essentially every turn — decide what to retrieve, read the result, decide what is still missing, synthesize — so I want a dense interleave, thought → action → observation repeated. In a long-horizon environment with dozens of low-level moves, forcing a thought before every "go to drawer 1" drowns the trajectory in narration and wastes the limited prompt budget, so thoughts should be sparse: a plan-setting thought near the start, a progress-tracking thought when the plan needs updating, an exception thought when something is off, and otherwise just act. Cadence is a knob the task sets, not a constant the method imposes.

For the knowledge tasks the environment is a deliberately simple Wikipedia API with exactly three actions, kept weaker than a real retriever on purpose so that retrieval must be *steered by reasoning* rather than by retrieval muscle: $\textbf{search[entity]}$ returns the first five sentences of that entity's page, or — and this matters — if the page does not exist it returns a few similar entity names, so a failed search is recoverable by reasoning over the suggestions; $\textbf{lookup[string]}$ returns the next sentence on the current page containing the string, like Ctrl-F; and $\textbf{finish[answer]}$ ends the episode. The prompt opens with an instruction naming these three actions and the interleaved Thought/Action/Observation format, then the hand-written exemplars, then the question. In the loop I ask the model to emit a thought *and* the action it implies together — I generate from "Thought $i$:" and let it continue through "Action $i$:", stopping generation right before the observation line, because the observation is not the model's to write; it comes from the environment. I parse the thought and action out of that one generation; if the model produced a thought but no action, I re-query for the action alone. Then I execute the action, append the real "Thought $i$ / Action $i$ / Observation $i$" to the running prompt, and repeat. The dense trajectory is capped at seven turns for HotpotQA and five for FEVER, and if the model never calls finish I force a finish at the end. One last thing the two failure modes tell me to do at the system level: grounded acting reduces hallucination by tying claims to observations but pays for it when a weak search leaves little evidence, while free reasoning is fluent but ungrounded — they fail in complementary places, so I let them back each other up. Run the interleaved trajectory and fall back to self-consistency CoT when it cannot answer within the step budget; or run self-consistency first and back off to the grounded trajectory when the majority answer appears in fewer than half of the sampled traces. A simple switching heuristic decides when to trust internal reasoning and when to ask the environment.

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
