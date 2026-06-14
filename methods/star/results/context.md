# Context: organizing many LLM agents for collaborative task solving (circa 2023-2024)

## Research question

A single large language model, prompted once, produces an artifact — an answer, a snippet of
code, a paragraph — in one enclosed forward pass. It is now well established that letting
several LLM-backed agents talk to each other and revise each other's work often beats any one
of them: the back-and-forth surfaces mistakes a single pass misses, and different agents bring
different angles. But the systems that demonstrate this almost all use a handful of agents —
typically fewer than ten, occasionally a few dozen — wired together in a fixed, hand-designed
shape (a pipeline, a small hierarchy, a hand-customized graph). The neural side of the field
has a scaling law: keep adding neurons and well-trained networks keep getting better in a
predictable power-law way. The question this raises for collaboration is the obvious one and
nobody has answered it: **if I keep adding agents — tens, hundreds, a thousand — does
collaborative performance keep improving, and in what shape does it improve?**

Answering it forces two things that none of the existing systems provide. First, a way to
organize an *arbitrary* number of agents into *arbitrary* interaction patterns, uniformly,
without redesigning the system for each task or each count, and without the interaction
folding back on itself in ways that need task-specific patching. Second, a way to keep the
running cost from exploding as the count grows: if every agent can see everything every other
agent ever said, the context each agent must read grows with the size of the whole
conversation, and at a thousand agents that is fatal — you run out of context window and money
long before you run out of agents to add. A solution has to make collaboration both *general*
(any number of agents, any interaction structure, any task) and *scalable* (cost that grows
gently, not explosively, with the agent count). It also has to actually use the interaction:
simply running the model many times and taking a majority vote, or keeping the best of N
independent samples, is known to stall after a handful of samples, because independent samples
do not refine each other. Closing that gap — genuine, scalable, general multi-agent
collaboration — is the problem.

## Background

By this time the prevailing recipe for going beyond a single LLM call is to turn the base
model into an *autonomous agent* — equipping it with a persona/profile, context-aware memory,
tools, and a planning loop — and then to have several such agents collaborate by exchanging
natural-language messages. The collaboration is task-oriented: agents are given distinct
expertise and prompted to interact, and the interaction itself, not any one agent, produces
the final artifact. The pain points that frame this work:

- **Collaboration has only ever been demonstrated at tiny scale.** Almost all systems use
  fewer than ten agents; the largest stretch to a few dozen. Nobody has run collaboration at
  hundreds or thousands of agents, so the *relationship between agent count and performance*
  is simply unknown — there is no collaborative analogue of the neural scaling law.
- **Interaction structure is fixed and hand-built.** Each system bakes in one shape — a linear
  pipeline, a small tree, or a graph whose every node and edge must be hand-customized for the
  task. There is no general, structure-agnostic way to organize an arbitrary number of agents
  into an arbitrary interaction pattern that transfers across tasks without re-engineering.
- **Naive collaboration explodes in cost.** When every agent reads the whole running
  conversation, the context each agent must process grows with the total amount that has been
  said, which itself grows with the number of agents and interactions. This *context
  explosion* is exactly what caps existing systems at a few dozen agents before they run out
  of context window. Token cost, latency, and dollar cost all balloon together.
- **More samples without interaction does not help for long.** Equalizing the number of model
  calls through majority voting (for closed-ended tasks) or best-of-N (for open-ended ones)
  improves things only marginally and plateaus after roughly eight samples — because the
  samples are drawn independently and never refine one another. Effective collaboration has to
  be *interdependent*, not parallel-and-pool.

Several background ideas are on the table and load-bearing here.

The **iterative self-improvement** idea: an LLM can improve its own output by generating
feedback on it and then revising — generate, critique, refine, repeat — using natural-language
feedback and no extra training (Madaan et al. 2023). Iterating this reflect-then-refine loop
reliably beats one-shot generation. It is a single-model, single-thread procedure, but it
establishes that *reflection and refinement* is the unit of useful interaction.

The **division-of-labor** idea: giving two agents complementary roles — one that issues
instructions / critiques and one that produces the work — activates more focused, functional
behavior than a single undifferentiated agent (the instructor/assistant split of Li et al.
2023, "CAMEL"). Role specialization is what turns a conversation into progress.

The **graph view of computation and of society**: a graph is the data structure for entities
and their relations; information propagates along its edges much as it does in a social
network. A *directed acyclic graph* in particular is a set of nodes with directed edges and no
cycles, and it admits a *topological order* — a linear arrangement in which every edge points
"forward," so that a node is reached only after everything pointing into it (Kahn 1962,
topological sorting: repeatedly remove a node with no remaining incoming edges). DAGs are the
standard way to schedule tasks with dependencies.

The **complex-networks** background: networks differ sharply in how efficiently information
moves through them, measured by the *average path length* (mean shortest-path distance over
all node pairs — smaller means information reaches everywhere in fewer hops) and the
*clustering coefficient* (how densely a node's neighbors connect to each other). Watts &
Strogatz (1998) showed that adding a few random "shortcut" edges to an otherwise regular,
highly-clustered network collapses its average path length to near that of a random graph
while keeping high clustering — the *small-world* property — so a handful of long-range links
makes distant parts of a network suddenly close. This is a pre-method fact about graphs, not
about agents.

The **long-tailed-distribution** background: token frequencies (and many natural quantities)
follow a power law / Zipf distribution, a few common items and a very long tail of rare ones,
with the probability of the item at rank `r` scaling like `1/r` (Newman 2005). A basic
consequence is that rare items only show up once you draw enough samples: the chance that a
tail item of probability `p` appears at least once in `n` independent draws is `1-(1-p)^n`,
which rises toward 1 as `n` grows. Sampling more is how you reach the tail.

The **neural scaling law** background: well-trained networks improve as a power law in model
size, data, and compute (Kaplan et al. 2020), and large enough models show *emergent*
abilities — capabilities that appear as a relatively sudden jump past a scale threshold (Wei
et al. 2022). The emergence thresholds are large: roughly billion-parameter models and
upward of `10^22` training FLOPs. This is the trend the present question is consciously trying
to find an inference-time analogue of.

## Baselines

The prior methods a new collaboration scheme would be measured against and reacts to.

**Chain-of-Thought prompting (Wei et al. 2022).** Prompt a single model to emit a series of
intermediate reasoning steps before its final answer. Powerful and general for reasoning,
because the benchmark knowledge is largely already inside the foundation model. **Limitation:**
it is one model, one pass — a degenerate single-thread "chain" inside one head, with no second
perspective to catch its own errors and nothing to scale.

**ChatDev / chained communicative agents (Qian et al. 2023).** Differentiate agents into roles
(e.g. designer, coder, tester) and run them as a *chained* workflow that sequentially refines
a software artifact, passing only the latest artifact down the line. Introduced the SRDD
software-requirement benchmark and the practice of carrying forward the artifact rather than
the whole transcript. **Limitation:** the structure is a single fixed chain of a few agents;
it does not express branching, convergence, or large counts, and there is no account of what
happens as you add agents.

**AgentVerse (Chen et al. 2023).** Dynamically assembles a team of expert agents in chained or
hierarchical (tree-like) structures and has them reflect and refine collaboratively, exhibiting
emergent social behaviors. **Limitation:** in practice the coordination collapses toward a
hub-like pattern and hits context explosion past roughly thirty agents, so it cannot scale; the
structure is also assembled per task rather than being a general topology.

**GPTSwarm / language agents as graphs (Zhuge et al. 2024).** Represents a swarm of agents as
a computational graph, with nodes as manually customized operations and edges as information
flow, and optimizes node prompts and edge connectivity during reasoning. The closest prior
move to a graph organization. **Limitation:** every node and edge must be hand-customized and
task-specifically engineered, which complicates use and blocks seamless generalization across
heterogeneous tasks; the manual, energy-intensive setup also limits how far it can scale.

**Search-over-thoughts: Tree-of-Thoughts (Yao et al. 2023) and Graph-of-Thoughts (Besta et al.
2024).** Generalize chain-of-thought into a tree or graph *of intermediate thoughts* explored
by one model, with branching, backtracking, and merging of thoughts. **Limitation:** the units
are *thoughts inside a single model's inference*, not specialized agents with their own
profiles, memory, and division of labor; there is no multi-agent interaction to scale and no
mechanism to keep many parallel threads' contexts from piling up.

**Sample-and-aggregate: majority voting and best-of-N (e.g. Du et al. 2024 for debate-style
aggregation).** Draw many independent samples and combine them — vote for closed-ended tasks,
keep the best for open-ended ones; multi-agent debate adds rounds of mutual critique.
**Limitation:** pure voting/best-of-N samples are independent and do not refine each other, so
gains are marginal and plateau after roughly eight samples; debate broadcasts full context and
so inherits the same scaling ceiling.

The common thread: each prior system either is a single model (no real collaboration), or fixes
one small structure by hand (no generality and no path to large counts), or lets everyone see
everything (no scalability) — and none of them charts what happens as the agent count grows.

## Evaluation settings

The natural yardsticks already in use, across heterogeneous downstream scenarios so that no
single task type dominates:

- **MMLU** (Hendrycks et al. 2020): broad multiple-choice questions across many subjects;
  metric is **accuracy**. A closed-domain logical-reasoning probe.
- **HumanEval** (Chen et al. 2021): function-level code generation; metric is **pass@k**, the
  fraction of problems whose generated function passes the hidden unit tests. Basic
  programming skill.
- **SRDD** (the Software Requirement Description Dataset released with ChatDev, Qian et al.
  2023): repository-level software development from real-world-style textual requirements,
  spanning comprehension, design, coding, and testing; scored by a comprehensive metric over
  completeness, executability, and consistency.
- **CommonGen-Hard** (via Madaan et al. 2023): generate coherent sentences from a set of
  discrete concepts; scored by a comprehensive metric over grammar, fluency, context relevance,
  and logical consistency. An open-ended generation probe.

Protocol facts that exist at the time: multi-agent baselines run on the order of four agents,
so a default of about four nodes makes comparisons apples-to-apples; the foundation backbone
is a fixed mid-tier chat model (e.g. GPT-3.5) chosen for the efficacy/efficiency balance, with
each adjacent-agent interaction capped at about three exchange rounds; agent count is swept
geometrically, from a single agent up through `2^6` (over a thousand agents in the densest
structure). No outcomes here — these are the dials and datasets, not results.

## Code framework

The collaboration plugs into the agent machinery that already exists. A foundation chat model
is wrapped into an *agent* — a profile/system message plus a `step(message) -> reply` call.
"Agentizing" an element means attaching such a model with a role and memory. A *task* arrives
as a prompt; the desired output is an *artifact* (here, code, represented as a parsed bundle of
files). What does **not** exist yet, and is exactly what is to be designed, is *how to arrange
many agents and route work among them* — the organizing structure and the order and rules by
which agents act on and pass along artifacts. So the substrate is just: the agent wrapper, the
artifact container, a task prompt, and an empty orchestrator with one big slot for the
collaboration scheme.

```python
from foundation import ChatModel        # base LLM: .step(message) -> reply


def agentize(role_profile: str, model_name: str):
    """Wrap a foundation model into an agent: a persona + a step() call.
    Pre-existing primitive — turns an LLM into a role-played collaborator."""
    return ChatModel(system_message=role_profile, model=model_name)


class Artifact:
    """The thing agents produce and pass around (e.g. a bundle of code files).
    Pre-existing container; parses model text into addressable pieces."""
    def __init__(self, text: str = ""):
        self.pieces = parse(text)          # text -> {name: content}

    def as_text(self) -> str:
        return render(self.pieces)


def agent_refine(agent, task_prompt: str, incoming: str) -> Artifact:
    """One agent's local act: read the task and an incoming artifact, return a
    (hopefully better) artifact. The within-act loop (how an agent reflects on
    and improves a draft) can use the existing reflect-then-refine primitive."""
    reply = agent.step(format_request(task_prompt, incoming)).content
    return Artifact(reply)


class Collaboration:
    """Organize many agents on a task and produce one final artifact.
    The organizing structure, the order agents act in, how work is routed and
    combined among them, and what each agent is allowed to remember are NOT
    settled — that scheme is exactly what we will design."""

    def __init__(self, num_agents: int, task_prompt: str, model_name: str):
        self.num_agents = num_agents
        self.task_prompt = task_prompt
        self.model_name = model_name
        self.agents = [agentize(default_profile(), model_name)
                       for _ in range(num_agents)]

    def run(self) -> Artifact:
        # TODO: the collaboration scheme we will design.
        #       Given the agents and the task, decide how they are organized,
        #       in what order they act, how artifacts are routed and combined,
        #       and what each agent reads — then return the final artifact.
        pass
```

The single empty slot is `Collaboration.run` (with whatever helper structure it needs): the
scheme for arranging the agents, ordering their actions, routing and combining their artifacts,
and bounding what each one reads.
