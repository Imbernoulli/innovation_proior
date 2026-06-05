# Context

## Research question

A capable language model can do two useful things, but the field studies them in separate boxes. In one box, prompted with worked exemplars, it produces multi-step *reasoning traces* — chains of intermediate thoughts that raise accuracy on arithmetic, commonsense, and symbolic problems. In the other box, it serves as a *policy* in an interactive environment: read an observation, emit an action, repeat. Reasoning is internal and disconnected from any world; acting is external and disconnected from any deliberation.

The question is whether keeping them separate is the bottleneck. A reasoning-only model thinks in a closed loop: it can only use what it already believes, so it cannot check a fact, fetch a missing premise, or notice mid-chain that an earlier step was wrong — which shows up as confident hallucination and as errors that propagate to the answer. An acting-only model, conversely, emits actions with no place to plan, track what it has learned, handle an exception, or decide what to look for next; it has no working memory beyond the raw observation history. What a solution would have to do is let a single model interleave the two so that thinking can guide and revise the next action, and acting can pull external information back into the thinking — without training a bespoke policy for each environment.

## Background

**Reasoning by prompting.** Given a few exemplars that show worked intermediate steps, a large LM will, at inference, generate its own step-by-step trace before the final answer, and this lifts accuracy on tasks whose input→output map is non-trivial. The trace is produced entirely from the model's internal state: it is a static, closed computation with no channel to the outside world. Two failure modes follow directly. First, *hallucination*: with nothing to check against, the model can assert a false fact and build on it. Second, *error propagation*: a wrong intermediate step is carried forward, and there is no mechanism to detect or repair it once the chain has moved on. Diagnostic inspection of reasoning-only traces on multi-hop QA bears this out — hallucination is the dominant failure category, and a sizable fraction of even the *correct*-looking answers rest on unsupported facts.

**LMs as policies in interactive environments.** A standard control setup: at step t the agent sees observation o_t ∈ O and takes action a_t ∈ A under a policy π(a_t | c_t), where the context c_t = (o_1, a_1, …, o_{t-1}, a_{t-1}, o_t) is the history. Recent systems use an LM to map this context to a domain-specific action (possibly via a controller that selects or grounds the proposal). These approaches predict actions from language priors but do not use the LM to reason abstractly about the goal or to maintain a working memory that supports acting. The closest prior systems inject limited environment feedback (a form of "inner monologue") between actions, but the inserted text is environment-reported state, not the model's own free-form deliberation.

**Reasoning-and-acting in humans.** A recurring observation about human competence is the tight interleaving of acting with verbal self-talk: between two physical actions we reason in language to track progress ("now that everything is cut, heat the pot"), handle exceptions ("no salt — use soy sauce instead"), and recognize when external information is needed ("how do I prepare dough? look it up"). This synergy lets people learn new tasks fast and act robustly under uncertainty. It is the property the separate boxes above each lack.

**Self-consistency / sampling.** For reasoning tasks, sampling several traces (e.g. with nonzero temperature) and taking the majority answer is a known way to make the final decision more robust than a single greedy chain.

## Baselines

These are the points of comparison a combined method would be measured against; each is an ablation of "reason and act together."

**Standard prompting.** Map question directly to answer with few-shot input→output exemplars and no intermediate steps, no actions, no observations. Gap: no working steps and no external access — at the mercy of one shot.

**Chain-of-thought (CoT) prompting** (Wei et al., 2022). Few-shot exemplars show a reasoning trace before the answer; at inference the model emits its own trace, then answers. Reasoning only. Gap: the trace is a closed black box grounded in nothing external — prone to hallucinated facts and to propagating an early error, with no way to fetch a missing fact or correct itself against the world.

**Self-consistency CoT (CoT-SC)** (Wang et al., 2022). Sample many CoT traces at temperature and return the majority answer. More robust than a single chain, but still purely internal reasoning — it cannot acquire new information, only re-sample the same closed computation.

**Acting-only prompting.** Few-shot exemplars show only actions and observations (the reasoning stripped out); the model emits the next action from the interaction history. Resembles prior LM-as-policy systems that interact with an environment but do not verbalize a reasoning step. Gap: no place to plan, decompose the goal, handle exceptions, or synthesize a final answer from gathered observations — the mapping context→action is left fully implicit, which is exactly where it tends to fail (e.g. it cannot assemble the final answer after retrieving the pieces).

## Evaluation settings

Natural yardsticks span both regimes the method aims to unify — knowledge-intensive reasoning and interactive decision making.

- **Multi-hop question answering (HotpotQA).** Questions requiring reasoning over two or more Wikipedia passages. Run in a *question-only* setup: the model receives only the question (no gold support paragraphs) and must use internal knowledge or retrieve. Metric: exact-match answer accuracy.
- **Fact verification (FEVER).** Each claim is labeled SUPPORTS, REFUTES, or NOT ENOUGH INFO depending on whether a Wikipedia passage verifies it; again question/claim only. Metric: label accuracy.
- **Interactive environments.** A text-based household task environment (ALFWorld) where the agent navigates and manipulates objects via text commands, and a web-navigation shopping environment (WebShop) where it browses and selects products to match an instruction. Metric: task success rate. These come with imitation/RL baselines trained on 10³–10⁵ task instances, against which a few-shot prompted method can be compared.

For the QA/verification tasks, the environment is a simple Wikipedia web API exposing three actions: **search[entity]** (returns the first sentences of the entity's page, or suggests similar entity names if it doesn't exist), **lookup[string]** (returns the next sentence on the current page containing the string, like Ctrl-F), and **finish[answer]** (ends the episode with an answer). This API is intentionally weaker than a real retriever — it forces retrieval to be driven by explicit reasoning. Base model: a large frozen LM prompted with a handful (one to six) of hand-written exemplars; for reasoning tasks the self-consistency baseline samples ~21 traces at temperature 0.7.

## Code framework

Pre-method primitives that already exist: an LM completion call, and an environment exposing `reset`/`step` with the Wikipedia-style action API and a reward/answer check. The contribution will fill the prompt format and the interaction loop that decides what the model emits at each turn.

```python
import os, openai

# --- LM access (exists) ---
def llm(prompt, stop=["\n"]):
    response = openai.Completion.create(
        model="text-davinci-002", prompt=prompt, temperature=0,
        max_tokens=100, top_p=1, stop=stop)
    return response["choices"][0]["text"]

# --- environment (exists): a Wikipedia API wrapped as a gym-style env ---
# env.reset(idx) -> question/observation
# env.step(action) -> (obs, reward, done, info), where action is one of
#   search[entity] | lookup[string] | finish[answer]
# (plus task wrappers that hold the gold answer and compute exact-match reward)

# --- the prompt + interaction loop the method will design ---
instruction = """..."""          # TODO: describe the action API and the turn format
fewshot_examples = """..."""     # TODO: hand-written exemplar trajectories

def run(idx, prompt=instruction + fewshot_examples, max_steps=...):
    question = env.reset(idx=idx)
    prompt += question + "\n"
    for i in range(1, max_steps + 1):
        # TODO: at each turn, decide what the model should emit and how the
        #       loop consumes it, then take a step in the env and append the result
        pass
```
