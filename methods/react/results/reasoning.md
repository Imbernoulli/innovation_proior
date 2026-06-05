Let me look hard at how a reasoning-only chain actually fails, because that's where the real clue is. I take a multi-hop question — say, "what is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?" — and I let the model just think step by step. It writes a clean, confident chain: the Colorado orogeny extends into X, X has elevation range Y, therefore Y. Looks like reasoning. But when I check, the fact it asserted about which area the eastern sector extends into is simply wrong — it invented it. And because the chain is one forward pass with no way to verify, the invented fact propagates straight to a wrong answer. When I sample fifty such traces and label them, the dominant failure mode is exactly this: hallucination. Even some of the answers that come out *right* are resting on facts the model had no business being sure of. The reasoning structure is fine. The problem is that the whole computation is sealed off from the world — it can only ever use what's already inside the weights, so it can't look anything up, can't check itself, can't repair a step once it's been written.

Now look at the opposite design, the one where I use the LM as a policy that just acts. Drop it into an environment: it sees an observation, emits an action, sees the next observation, and so on, π(a_t | c_t) with the context c_t being the whole history (o_1, a_1, …, o_t). For a retrieval task I'd give it actions like "search this entity," "look up this string," and watch it gather information. And it does gather — but then it falls on its face at exactly the moment it has to *use* what it gathered. It retrieves the right paragraph and then can't assemble the final answer, because the mapping from a long messy interaction history to the correct next action is highly implicit and there's no place in the loop for the model to stop and reason: "I now have the elevation; I'm done; the answer is this." It also can't plan ("I need to find X, then find Y"), can't handle an exception ("that search returned nothing, let me try a different name"), can't track progress. It has actions but no working memory, no deliberation.

So I've got two halves of one competence, each crippled by the absence of the other. Reasoning that can't touch the world hallucinates and can't self-correct. Acting that can't reason has no plan, no memory, no way to synthesize. The obvious thing to want is the human move: between actions, *talk to yourself*. When I cook, between "chop the onion" and "heat the pan" I reason in language — track where I am, adjust the plan when I'm out of salt, recognize when I need to go look something up. The verbal reasoning isn't a separate phase; it's woven into the acting, and that weave is what lets a person handle a situation they've never seen. I want to give the model that weave.

How do I actually express "think between actions" in the policy formalism, though, without bolting on a second module? Here's where I get to be lazy in a good way. The agent already maps a context to an action in some space A. What if I just enlarge the action space to include language itself: Â = A ∪ L, where L is the space of free-form text. A normal action a_t ∈ A goes out to the environment and comes back with an observation. But an action that lands in the language part, call it a thought, does *not* touch the environment — it produces no observation, changes nothing outside. All it does is update the context: c_{t+1} = (c_t, â_t). The thought is reasoning that the model writes to its own working memory, available to condition every future thought and action.

That's the whole mechanism, and I like how little it adds. There's no new model, no new training objective, no controller. A "thought" is just an action whose effect is purely internal — it composes useful information by reasoning over c_t and writes it back into c_t. And now the two failure modes I started with both have a fix in the same loop. The reasoning chain is no longer sealed: a thought can say "I should search for X," the next action searches, the observation comes back, and a later thought reasons over *that real observation* instead of an internal guess — so the facts are grounded, not hallucinated. And the action loop is no longer thoughtless: a thought can decompose the goal, note an exception, decide the search to reformulate, or declare the final answer is assembled. Reasoning to act; acting to reason.

But there's a catch I have to respect. L is unlimited — the space of all text. If I just enlarge the action space and turn the model loose, learning a good policy over an infinite language action space is hopeless without enormous supervision. I don't have that and I don't want a bespoke trained policy per environment anyway. So I lean entirely on the prior already baked into a big pretrained LM, and I steer it the cheap way: few-shot prompting. I hand-write a small number of exemplar trajectories — each one a human-authored sequence of thoughts, actions, and observations that solves a task instance — and prepend them to the context. The frozen model, seeing how a few trajectories interleave free-form thinking with concrete actions, imitates that format on a new instance. The beautiful part is that designing these prompts is almost trivial: an annotator just types their own thoughts in plain language on top of the actions they'd take. No special thought taxonomy, no rigid template — the thought is whatever you'd actually think.

Now I have to decide *how often* a thought should appear, and this is where the structure of the task pushes back, so I shouldn't impose one rule. Consider a multi-hop QA episode. Here reasoning is the bottleneck on essentially every turn: I have to decide what to retrieve, read the result, decide what's still missing, and eventually synthesize. So I want a thought before nearly every action — a dense interleave, thought → action → observation, thought → action → observation. Decompose the question, extract the relevant bit from each observation, do the small commonsense or arithmetic step ("1844 is before 1989"), guide the next search, and finally state the answer. Contrast that with a long-horizon environment task — navigating a house, dozens of low-level moves. If I force a thought before every single "go to drawer 1," I drown the trajectory in verbiage and waste the limited prompt budget on narration that adds nothing. There, thoughts should be *sparse*: a plan-setting thought near the start ("to clean the mug I need to find it, then wash it, then put it back"), a progress-tracking thought when the plan needs updating, an exception thought when something's off — and otherwise just act. So I don't hardcode the cadence: for reasoning-heavy tasks I alternate thought and action; for action-heavy tasks I let the model itself decide, from the exemplars, where a thought is worth inserting. The placement of thinking is a knob the task sets, not a constant.

Let me make the QA instantiation fully concrete, since that pins down the action API and the loop. The environment is a deliberately simple Wikipedia interface with three actions. search[entity] returns the opening sentences of that entity's page, or — and this matters — if the page doesn't exist, it returns a few similar entity names, so a failed search is recoverable by reasoning about the suggestions. lookup[string] returns the next sentence on the current page containing the string, like hitting Ctrl-F. finish[answer] ends the episode. I keep this API weaker than a real neural retriever on purpose: it can only pull a small piece keyed on an exact page name, which forces the model to *reason* its way to the right query rather than leaning on retrieval muscle. The prompt opens with an instruction that names these three actions and says "solve the task with interleaving Thought, Action, Observation steps," then the hand-written exemplars, then the question.

The loop has one subtlety worth getting right. On each turn I want the model to emit a thought *and* the action it implies, together, so I ask it to generate "Thought i:" and let it continue through "Action i:", stopping its generation right before the observation line — because the observation isn't the model's to write; it comes from the environment. I parse the thought and the action out of that single generation. Then I execute the action in the environment, get the real observation, append "Thought i / Action i / Observation i" to the running prompt, and go again. If the model ever fails to produce the action on the same line as the thought, I just re-query for the action alone. I cap the episode at a small number of steps; if it never calls finish, I force a finish at the end.

```python
import os, openai
import wikienv, wrappers

# environment: Wikipedia API wrapped gym-style; actions are search[]/lookup[]/finish[]
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

# the instruction names the three actions and the interleaved turn format
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples   # hand-written thought/action/obs trajectories

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):                      # dense interleave, capped at 7 steps
        n_calls += 1
        # generate the thought AND the action together, stop before the observation line
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:                                # model didn't emit the action inline -> ask for it
            n_badcalls += 1; n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        # the observation is the environment's, not the model's
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info
```

There's one more thing the two failure modes tell me to do at the system level. Grounded acting fixes hallucination but pays for it: by tying itself to what it can retrieve, the interleaved trajectory is less free to reason in whatever shape it wants, and a non-informative search can derail it. Pure reasoning is the opposite — flexible and often structurally right, but ungrounded. They fail in complementary places. So I don't have to pick one. I can let them back each other up: run the interleaved reason-and-act trajectory, and if it can't return an answer within its step budget, fall back to sampling several pure-reasoning traces and majority-voting; or, the other direction, run self-consistency reasoning first and back off to the grounded trajectory when the reasoning samples don't agree. The model decides which knowledge to trust — internal or externally fetched — based on a simple heuristic about whether the primary method succeeded.

So the causal chain: reasoning-only LMs are sealed off from the world, so they hallucinate and can't self-correct; acting-only LMs have no working memory or plan, so they can't synthesize what they gather; the fix is to enlarge the action space with language itself, making a "thought" an action whose only effect is to update the context — no environment feedback, no new module — so reasoning can steer the next action and observations can flow back into the reasoning; steer the infinite language-action space with a few hand-written interleaved exemplars on a frozen model; make thoughts dense where reasoning is the bottleneck and sparse where actions dominate; and, since grounded acting and free reasoning fail in complementary places, combine them with a back-off heuristic.
