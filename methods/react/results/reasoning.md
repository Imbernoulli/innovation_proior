Let me look hard at how a reasoning-only chain actually fails, because that's where the real clue is. A multi-hop question like "what is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?" makes the problem visible. A chain can write a clean, confident path: the Colorado orogeny extends into X, X has elevation range Y, therefore Y. It looks like reasoning. But if the first factual bridge is wrong, the whole answer inherits the error. Nothing in the trace can ask Wikipedia what the eastern sector extends into, nothing can notice that the retrieved page would contradict the guess, and nothing can repair the path once the false premise is written. The reasoning structure is useful. The problem is that the whole computation is sealed off from the world — it can only ever use what's already inside the weights, so it hallucinates, can't check itself, and propagates early mistakes.

Now look at the opposite design, the one where I use the LM as a policy that just acts. Drop it into an environment: it sees an observation, emits an action, sees the next observation, and so on, π(a_t | c_t) with the context c_t being the whole history (o_1, a_1, …, o_t). For a retrieval task I'd give it actions like "search this entity," "look up this string," and watch it gather information. And it does gather — but then it falls on its face at exactly the moment it has to *use* what it gathered. It retrieves the right paragraph and then can't assemble the final answer, because the mapping from a long messy interaction history to the correct next action is highly implicit and there's no place in the loop for the model to stop and reason: "I now have the elevation; I'm done; the answer is this." It also can't plan ("I need to find X, then find Y"), can't handle an exception ("that search returned nothing, let me try a different name"), can't track progress. It has actions but no working memory, no deliberation.

So I've got two halves of one competence, each crippled by the absence of the other. Reasoning that can't touch the world hallucinates and can't self-correct. Acting that can't reason has no plan, no memory, no way to synthesize. What I keep coming back to is the human move: between actions, *talk to yourself*. When I cook, between "chop the onion" and "heat the pan" I reason in language — track where I am, adjust the plan when I'm out of salt, recognize when I need to go look something up. The verbal reasoning isn't a separate phase; it's woven into the acting, and that weave is what lets a person handle a situation they've never seen. I want to give the model that weave.

How do I actually express "think between actions" in the policy formalism, though, without bolting on a second module? The agent already maps a context to an action in some space A. If the missing object is language self-talk, the cheapest thing I can try is to put language into the action space itself: Â = A ∪ L, where L is the space of free-form text. A normal action a_t ∈ A goes out to the environment and comes back with an observation. But an action that lands in the language part, call it a thought, does *not* touch the environment — it produces no observation, changes nothing outside. All it does is update the context: c_{t+1} = (c_t, â_t). The thought is reasoning that the model writes to its own working memory, available to condition every future thought and action.

Before I get attached to this, I should check that it actually buys me anything — that an interleaved loop would have repaired the exact failure I opened with, not just relabeled it. Let me trace the Colorado orogeny question by hand through this design and see where each turn lands. The opening thought has to decompose: I don't know the area the eastern sector extends into, so the plan is search the orogeny, find the area, then find that area's elevation. So Action 1 = Search[Colorado orogeny]. What comes back from a five-sentence Wikipedia stub is the general definition — "an episode of mountain building in Colorado and surrounding areas" — and, importantly, it does *not* mention the eastern sector. This is precisely the spot where the sealed chain hallucinated: it had to invent what the eastern sector extends into. Here the observation is real and visibly silent on the point, so the honest next thought is "this didn't mention the eastern sector; I should look it up," i.e. Lookup[eastern sector]. That returns the sentence I actually need: the eastern sector extends into the High Plains. So the grounding did its job — the bridge fact that the chain had to guess is now read off the page instead.

That trace also surfaces a wall I didn't anticipate, and it's a useful one. The next thought is "search High Plains, find its elevation," Action 3 = Search[High Plains]. The observation comes back useless: "High Plains refers to one of two distinct land regions." The exact-page-name retrieval landed on a disambiguation stub, not the region I want. An acting-only policy has nothing to do with this — it can't represent "that search was ambiguous, pick the United States sense." But a thought can: "I need to instead search High Plains (United States)." Action 4 = Search[High Plains (United States)] returns "the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m)." Now Thought 5 reads the number straight off the observation and Finish[1,800 to 7,000 ft] ends it. Counting calls: five thought/action turns, well under any reasonable cap, and the one wrong turn (the ambiguous search) was caught and corrected *by a thought reasoning over a real observation* — which is exactly the self-correction the sealed chain could not do. So both failure modes really are addressed in this one loop, and not by assertion: the hallucinated bridge became a Lookup over real text, and the ambiguous retrieval became a reformulated search. That is enough evidence to commit to the mechanism.

And I like that it adds almost nothing: no new model, no new training objective, no controller. A "thought" is just an action whose effect is purely internal — it composes useful information by reasoning over c_t and writes it back into c_t. The reasoning chain is no longer sealed: a thought can say "I should search for X," the next action searches, the observation comes back, and a later thought reasons over *that real observation* instead of an internal guess. And the action loop is no longer thoughtless: a thought can decompose the goal, note an exception, decide the search to reformulate, or declare the final answer is assembled. Reasoning to act; acting to reason.

But there's a catch I have to respect. L is unlimited — the space of all text. If I just enlarge the action space and turn the model loose, learning a good policy over an infinite language action space is difficult without strong language priors and a lot of supervision. I don't have that and I don't want a bespoke trained policy per environment anyway. So I lean entirely on the prior already baked into a big pretrained LM, and I steer it the cheap way: few-shot prompting. I hand-write a small number of exemplar trajectories — each one a human-authored sequence of thoughts, actions, and observations that solves a task instance — and prepend them to the context. The frozen model, seeing how a few trajectories interleave free-form thinking with concrete actions, imitates that format on a new instance. The nice part is that designing these prompts is almost trivial: an annotator just types their own thoughts in plain language on top of the actions they'd take. No special thought taxonomy, no rigid template — the thought is whatever you'd actually think. The hand-traced trajectory above is, in fact, exactly the shape of exemplar I'd write down.

Now I have to decide *how often* a thought should appear, and this is where the structure of the task pushes back, so I shouldn't impose one rule. The QA trace I just walked is the evidence for one extreme: reasoning was the bottleneck on essentially every turn — decide what to retrieve, read the result, decide what's still missing, catch the ambiguous search, eventually synthesize. Every single one of those five turns needed a thought; strip any of them and the loop either retrieves blindly or fails to notice the disambiguation. So for multi-hop QA I want a thought before nearly every action — a dense interleave, thought → action → observation, thought → action → observation. Contrast that with a long-horizon environment task — navigating a house, dozens of low-level moves. If I force a thought before every single "go to drawer 1," I drown the trajectory in verbiage and waste the limited prompt budget on narration that adds nothing. There, thoughts should be *sparse*: a plan-setting thought near the start ("to clean the mug I need to find it, then wash it, then put it back"), a progress-tracking thought when the plan needs updating, an exception thought when something's off — and otherwise just act. So I don't hardcode the cadence: for reasoning-heavy tasks I alternate thought and action; for action-heavy tasks I let the model itself decide, from the exemplars, where a thought is worth inserting. The placement of thinking is a knob the task sets, not a constant.

For QA, the action API should be just strong enough to expose an external source and just weak enough that retrieval still has to be steered by reasoning. A deliberately simple Wikipedia interface gives me three actions. search[entity] returns the first five sentences of that entity's page, or — and this matters — if the page doesn't exist, it returns a few similar entity names, so a failed search is recoverable by reasoning about the suggestions. The trace already showed me why this property earns its place: the ambiguous Search[High Plains] only got rescued because the next thought could reformulate to the (United States) sense; an API that just errored out would have stranded the loop. lookup[string] returns the next sentence on the current page containing the string, like hitting Ctrl-F — that's what pulled "eastern sector extends into the High Plains" off a page whose first five sentences omitted it. finish[answer] ends the episode. If the API can only pull a small piece keyed on an exact page name, the model has to reason its way to the right query rather than lean on retrieval muscle. The prompt opens with an instruction that names these three actions and says "solve the task with interleaving Thought, Action, Observation steps," then the hand-written exemplars, then the question.

On each turn I want the model to emit a thought *and* the action it implies, together, so I ask it to generate "Thought i:" and let it continue through "Action i:", stopping its generation right before the observation line — because the observation isn't the model's to write; it comes from the environment. (My hand-trace makes that boundary concrete: if I'd let the model continue past "Action 1: Search[Colorado orogeny]" it would have *invented* Observation 1, and an invented stub could easily have "mentioned the eastern sector" — re-opening the hallucination hole. The stop token is what forces the real page in.) I parse the thought and the action out of that single generation. Then I execute the action in the environment, get the real observation, append "Thought i / Action i / Observation i" to the running prompt, and go again. If the model produces a thought but not the action in the same generation, I just re-query for the action alone. For the Wikipedia QA loop I cap the dense trajectory at seven turns — the worked example finished in five, so seven leaves slack for one or two extra reformulations; the same idea uses five turns for FEVER. If it never calls finish, I force a finish at the end.

```python
import os, openai, requests
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
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first five sentences if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples   # hand-written thought/action/observation trajectories

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
        except:                                # model didn't emit the action with the thought -> ask for it
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

There's one more thing the two failure modes tell me to do at the system level. Grounded acting should reduce hallucination by tying claims to retrieved observations, but it pays for that grounding: a weak or uninformative search can leave the trajectory with little useful evidence. I saw a small version of this risk in the trace — if Search[High Plains (United States)] had also come back uninformative, the loop could have burned turns retrieving and still finished empty. Pure reasoning is the opposite — flexible and often structurally fluent, but ungrounded. They fail in complementary places. So I don't have to pick one. I can let them back each other up: run the interleaved reason-and-act trajectory, and if it can't return an answer within its step budget — seven turns for HotpotQA, five for FEVER — fall back to sampling several pure-reasoning traces and majority-voting; or, the other direction, run self-consistency reasoning first and back off to the grounded trajectory when the majority answer appears in fewer than half of the sampled traces. A simple switching heuristic decides when to trust internal reasoning and when to ask the environment.

So the causal chain: reasoning-only LMs are sealed off from the world, so they hallucinate and can't self-correct; acting-only LMs have no working memory or plan, so they can't synthesize what they gather; enlarging the action space with language itself — making a "thought" an action whose only effect is to update the context — repaired both failures when I traced it by hand on the Colorado orogeny question, turning the hallucinated bridge into a Lookup over real text and the ambiguous retrieval into a reformulated search, with no new module and no environment feedback; steer the infinite language-action space with a few hand-written interleaved exemplars on a frozen model; make thoughts dense where reasoning is the bottleneck and sparse where actions dominate; and, since grounded acting and free reasoning fail in complementary places, combine them with a back-off heuristic.
