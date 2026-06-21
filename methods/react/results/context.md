## Research question

Language models can be prompted, via worked exemplars, to produce multi-step *reasoning traces* — chains of intermediate thoughts that improve accuracy on arithmetic, commonsense, and symbolic problems. They can also serve as *policies* in interactive environments: receive an observation, emit an action, repeat. How can a single frozen language model be used, without task-specific training, across both knowledge-intensive reasoning tasks and interactive decision-making tasks?

## Background

**Reasoning by prompting.** Given a few exemplars that show worked intermediate steps, a large LM will, at inference, generate its own step-by-step trace before the final answer, and this lifts accuracy on tasks whose input→output map is non-trivial. The trace is produced entirely from the model's internal state: it is a static, closed computation with no channel to the outside world.

**LMs as policies in interactive environments.** A standard control setup: at step t the agent sees observation o_t ∈ O and takes action a_t ∈ A under a policy π(a_t | c_t), where the context c_t = (o_1, a_1, …, o_{t-1}, a_{t-1}, o_t) is the history. Recent systems use an LM to map this context to a domain-specific action (possibly via a controller that selects or grounds the proposal). The closest prior systems inject limited environment feedback (a form of "inner monologue") between actions, where the inserted text is environment-reported state.

**Reasoning-and-acting in humans.** A recurring observation about human competence is the tight interleaving of acting with verbal self-talk: between two physical actions we reason in language to track progress ("now that everything is cut, heat the pot"), handle exceptions ("no salt — use soy sauce instead"), and recognize when external information is needed ("how do I prepare dough? look it up"). This synergy lets people learn new tasks fast and act robustly under uncertainty.

**Self-consistency / sampling.** For reasoning tasks, sampling several traces (e.g. with nonzero temperature) and taking the majority answer is a known way to make the final decision more robust than a single greedy chain.

## Baselines

**Standard prompting.** Map question directly to answer with few-shot input→output exemplars and no intermediate steps, no actions, no observations.

**Chain-of-thought (CoT) prompting** (Wei et al., 2022). Few-shot exemplars show a reasoning trace before the answer; at inference the model emits its own trace, then answers. Reasoning only, with no channel to the outside world.

**Self-consistency CoT (CoT-SC)** (Wang et al., 2022). Sample many CoT traces at temperature and return the majority answer.

**Acting-only prompting.** Few-shot exemplars show only actions and observations (the reasoning stripped out); the model emits the next action from the interaction history. Resembles prior LM-as-policy systems that interact with an environment but do not verbalize a reasoning step.

## Evaluation settings

Natural yardsticks span the two regimes: knowledge-intensive reasoning and interactive decision making.

- **Multi-hop question answering (HotpotQA).** Questions requiring reasoning over two or more Wikipedia passages. Run in a *question-only* setup: the model receives only the question (no gold support paragraphs) and must use internal knowledge or retrieve. Metric: exact-match answer accuracy.
- **Fact verification (FEVER).** Each claim is labeled SUPPORTS, REFUTES, or NOT ENOUGH INFO depending on whether a Wikipedia passage verifies it; again question/claim only. Metric: label accuracy.
- **Interactive environments.** A text-based household task environment (ALFWorld) where the agent navigates and manipulates objects via text commands, and a web-navigation shopping environment (WebShop) where it browses and selects products to match an instruction. Metrics: task success rate for ALFWorld; average score and success rate for WebShop. These come with imitation/RL baselines trained on 10³–10⁵ task instances, giving a clear reference point for a few-shot prompting setup.

For the QA/verification tasks, the environment is a simple Wikipedia web API exposing three actions: **search[entity]** (returns the first five sentences of the entity's page, or suggests similar entity names if it doesn't exist), **lookup[string]** (returns the next sentence on the current page containing the string, like Ctrl-F), and **finish[answer]** (ends the episode with an answer). This API is intentionally weaker than a real retriever — it forces retrieval to be driven by explicit reasoning. Prompting setup: a large frozen LM with a handful (one to six) of hand-written exemplars; for reasoning tasks the self-consistency baseline samples ~21 traces at temperature 0.7.

## Code framework

Available primitives: an LM completion call, and an environment exposing `reset`/`step` with the Wikipedia-style action API and a reward/answer check. The open slots are the prompt format and the interaction loop that decides what the model emits at each turn.

```python
import os, openai

# LM access
def llm(prompt, stop=["\n"]):
    response = openai.Completion.create(
        model="text-davinci-002", prompt=prompt, temperature=0,
        max_tokens=100, top_p=1, stop=stop)
    return response["choices"][0]["text"]

# Environment: a Wikipedia API wrapped as a gym-style env
# env.reset(idx) -> question/observation
# env.step(action) -> (obs, reward, done, info), where action is one of
#   search[entity] | lookup[string] | finish[answer]
# (plus task wrappers that hold the gold answer and compute exact-match reward)

# Prompt and interaction loop slots
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
