# Context: organizing many LLM agents for collaborative task-solving (circa 2023-2024)

## Research question

A single large language model, prompted once, produces an artifact in one shot: an
answer, a function, a small program, a paragraph. The artifact inherits whatever
mistakes the model makes on its first pass — a logical slip, a hallucinated API, an
unhandled edge case — with no opportunity to catch and fix them. The empirical finding
that reframed the field is that *several* LLM-backed agents, talking to each other and
revising each other's work, reliably beat a single agent on the same task: specialization
plus iterative critique recovers errors the lone model commits. Yet almost every system
that demonstrated this used a tiny number of agents — typically fewer than ten, a few
dozen at the very most. The number of collaborators was treated as a fixed, small
constant, never as a knob to turn.

That juxtaposition is the pain point. In ordinary deep learning the dominant lever is
*scale*: add neurons, add data, add compute, and quality climbs along a smooth power law,
sometimes with abilities appearing abruptly past a threshold. If a network of
collaborating agents is even loosely analogous — agents playing the role neurons play —
then the obvious question is whether *adding more collaborating agents* keeps paying off,
and if so, how the relationship between agent count and quality is shaped. Answering it
requires more than running a few agents; it requires being able to instantiate
collaboration among *arbitrarily many* agents and measure quality as that number grows.
Nothing on the table can do that. The precise goal: a single, task-agnostic mechanism
that (1) wires up any number of agents into a collaboration structure without
hand-crafting each one for the task; (2) fixes a well-defined order in which they
interact and exchange work; (3) keeps the per-agent context (and hence cost and latency)
from exploding as the number of agents grows; and (4) is flexible enough that one
*structure* of collaboration can be swapped for another and the two compared on equal
footing. Each existing system achieves a sliver of this; none achieves all four, and in
particular none was built to be pushed past a few dozen agents.

## Background

By this time the prevailing recipe is to turn a foundation LLM into an *autonomous agent*
by wrapping it with role instructions (a "profile"), context-aware memory, and sometimes
tools, and then to have several such agents converse. The motivating phenomenon — well
established before any new structure is proposed — is that this multi-agent interaction
acts as a kind of "slow thinking": agents reflect on a draft, point out flaws, and
produce a revised draft, and the revision is usually better than the original. The gains
are attributed to two effects observed across prior systems: *specialization* (different
agents adopt different expert roles and notice different problems) and *iterative
refinement* (a critic-then-fix loop catches and repairs first-pass errors). It is also
established that this is not the same as naive *majority voting* — independently sampling
many answers and taking the most common one — which plateaus quickly: equalizing the
number of model calls through voting or best-of-N stops helping after roughly a handful
of samples. Whatever scaling exists in collaboration must come from agents *interacting*,
not from independent sampling.

Several conceptual frames are in the air. The **neural scaling law** says a well-trained
network's loss falls as a power law in the number of parameters (and data and compute),
and that qualitatively new abilities can *emerge* once a model is large enough — but only
at enormous scale (billions of parameters, on the order of 10^22 training FLOPs). A
recurring background observation about *why* collaboration helps draws on the long-tail,
Zipf-like shape of an LLM's output distribution: the most useful refinement of an
artifact often comes from a rare, "tail" consideration that any single sample is unlikely
to surface, so drawing more diverse interactions raises the chance of hitting those tail
considerations. There is also a body of **complex-network** theory ready to be borrowed:
the notion that a network's efficiency is captured by its *average path length* (mean
shortest-path distance between node pairs) and its *clustering coefficient*, and the
**small-world** phenomenon — that randomly rewiring a few edges of a regular network
sharply shortens average path length while preserving clustering. And there is the
classical algorithmic notion of a **topological order** of a directed acyclic graph (Kahn,
1962): a linear arrangement of the nodes in which every node comes after all the nodes
that point into it, which is exactly the order in which one would process a set of
dependency-respecting steps.

A diagnostic fact about the *cost* of unrestricted multi-agent talk is the load-bearing
constraint here. If every agent can see the full transcript of every interaction it
depends on, then in a densely connected group the agent at the end of the network — the
one that depends on everyone — must read a context that grows with the *number of pairwise
interactions*, which in a fully connected group of n agents scales like n^2. Context that
grows quadratically in the number of agents is fatal: it inflates token cost and latency
super-linearly and eventually exceeds the model's context window, which is precisely why
the hierarchical systems of the day were observed to break down once they passed a few
dozen agents. Any mechanism that hopes to reach hundreds or thousands of agents has to
confront this quadratic context growth head-on.

## Baselines

These are the prior approaches a new method would be compared against and would react to.

**Chain-of-Thought (Wei et al., 2022).** Prompt a single LLM to emit a coherent series of
intermediate reasoning steps before its final answer, turning a one-shot guess into a
step-by-step derivation. Powerful and general on benchmarks whose knowledge is already
baked into the foundation model. **Gap:** it is one model reasoning to itself — no second
agent, no specialization, no critique, and crucially no structural quantity to scale; you
cannot "add more" of anything to a single chain of thought.

**Communicative agents along a chat chain (Qian et al., 2023; ChatDev,
arXiv:2307.07924).** Differentiate LLM agents into expert roles and have them communicate
along a *waterfall* sequence of phases — design, then coding, then testing — each phase a
dual-role dialogue that hands its result to the next. This established that
language-mediated multi-agent collaboration works for repository-level software, and it
introduced the SRDD software-requirement benchmark. **Gap:** the structure is a single
linear sequence — purely sequential interaction, depth equal to the number of phases and
no parallel breadth, fixed hand-designed roles, and no notion of varying or scaling the
organizational structure.

**AgentVerse (Chen et al., 2023; arXiv:2308.10848).** Dynamically assemble a team of
expert agents in chained or hierarchical (tree-shaped) arrangements, with rounds of
reflection and refinement and emergent social behaviors. **Gap:** the arrangement is
hand-assembled and effectively collapses to a star (a coordinator broadcasting to
workers); it is observed to hit context explosion and break down once scaled beyond
roughly thirty agents, and it is not a general, swappable family of structures.

**Tree of Thoughts (Yao et al., 2023; arXiv:2305.10601).** Model a single model's
reasoning as a *tree* of partial "thoughts," exploring multiple branches with lookahead
and backtracking, selecting promising paths. **Gap:** it is still one model at the
thought level, not a society of role-specialized agents, and the tree is rigid — branches
cannot recombine, so independent lines of reasoning can never be merged into one.

**Graph of Thoughts (Besta et al., 2024; arXiv:2308.09687).** Argue the tree is too
rigid and model a single LLM's reasoning as an arbitrary *graph* of thoughts: vertices are
thoughts, edges are dependencies, and — unlike a tree — separate branches can be
*aggregated* (merged) into a combined thought, distilled, or improved through feedback
loops. This is the clearest prior statement that a graph buys you the ability to converge
several lines of work into one. **Gap:** it operates over the thoughts of a *single*
model — there is no multi-agent role bipartition and no study of collaboration at scale —
and it explicitly permits feedback loops/cycles, which complicate any fixed interaction
order.

**GPTSwarm (Zhuge et al., 2024; arXiv:2402.16823).** Represent language-agent systems as
*optimizable computational graphs*: each node is an operation (an LLM call or a tool
call), each edge is an information-flow channel, an agent is a sub-graph and a "swarm" a
composite graph; the work focuses on directed acyclic graphs and then *optimizes* node
prompts and edge connectivity via reinforcement learning or evolution. This is the
closest precedent for treating agent organization itself as a graph. **Gap:** every node
and edge function must be customized for the task, which makes it laborious and hard to
generalize across heterogeneous downstream tasks; its emphasis is on *learning* the
connectivity for one task rather than on holding a structure fixed and asking how
collaboration behaves as the structure and its size are varied; and it does not tackle
the quadratic context-growth problem that blocks scaling to very large numbers of agents.

The shared shape of the gaps: chains give depth but no breadth and no scaling knob; trees
and stars give some breadth but collapse or explode in context; the graph-of-thoughts and
swarm lines show that a graph is the right abstraction and that aggregating branches is
valuable, but they operate at the wrong granularity (thoughts, not role-specialized
agents) or demand per-task hand-engineering, and none keeps per-agent context bounded as
the population grows into the hundreds or thousands.

## Evaluation settings

The natural yardsticks already in use, spanning closed- and open-domain artifacts:

- **MMLU** — multiple-choice questions across many subjects and difficulties; metric is
  accuracy (correctness of the chosen option). A closed-domain logical-reasoning test.
- **HumanEval (Chen et al., 2021)** — function-level Python code generation from a
  docstring; metric is pass@k, the fraction of problems whose generated function passes
  all hidden unit tests. A closed-domain programming test.
- **SRDD (Qian et al., 2023)** — the Software Requirement Description Dataset of
  real-world software requirements for repository-level development (requirement
  comprehension, design, coding, testing); assessed by a comprehensive metric combining
  completeness, executability, and consistency. An open-domain software-engineering test;
  a natural narrower proxy is whether the produced program's entry point runs without
  crashing.
- **CommonGen-Hard (Madaan et al., 2023)** — generate coherent text containing a set of
  required discrete concepts; a comprehensive quality metric over grammar, fluency,
  context relevance, and logical consistency. An open-domain creative-writing test.
- Protocol the comparisons would adopt: a small default agent count (around four, to line
  up with existing multi-agent baselines); a fixed foundation model used for all
  interactions (GPT-3.5 was the efficiency/efficacy sweet spot at the time) with a small
  fixed cap on exchange rounds per interaction (around three); the same prompts,
  benchmark splits, and metrics across all methods. For the scaling study, the agent
  count is swept geometrically — 2^0, 2^1, ..., up to 2^6 — so that quality can be plotted
  against the (log of the) number of agents. Baselines are equalized by number of model
  calls (via majority voting on closed-domain tasks, best-of-N on open-domain ones) so the
  comparison is at matched compute.

## Code framework

A small harness already exists for running a *fixed* group of role-specialized LLM agents
that pass an artifact along: a runtime that takes a set of directed "who-talks-to-whom"
edges over numbered agent nodes, attaches a sentinel input node (the task prompt) feeding
the source agents and a sentinel output node collecting from the sink agents, schedules
the agents, runs each interaction as a critique-then-refine dialogue, and propagates the
result. All of that machinery is given. What is *not* given — and is exactly the thing to
be designed — is how to lay out the edges: given a desired number of agents, decide which
agent should pass its work to which, i.e. produce the directed edge set that organizes the
collaboration. The single empty slot is that edge-generating function. The runtime's only
requirements on what it returns: the edges must form a directed acyclic graph (no cycles),
every agent node must be reachable, and edges go from lower-numbered to higher-numbered
agents so the numbering is already a valid processing order.

```python
def generate_topology(node_num: int) -> list[tuple[int, int]]:
    """Given node_num agent nodes (numbered 0 .. node_num-1), return the directed
    edges (source, target) that organize their collaboration.

    The surrounding runtime will:
      - attach an input sentinel (-1) feeding every node that has no predecessor,
        and an output sentinel (-2) collecting from every node that has no successor;
      - schedule the agents in an order consistent with the edges;
      - run each edge as a critique-then-refine interaction and propagate the result;
      - at a node with several incoming edges, fuse the incoming artifacts before it
        refines.

    Contract on the returned edges:
      - they must form a DAG (no cycles);
      - every node 0 .. node_num-1 must be reachable;
      - edges go from a lower-numbered to a higher-numbered node.
    """
    # TODO: decide how to wire node_num agents together -- the organization we will design.
    pass


# existing runtime the edge set plugs into (already implemented, shown for context)
def run_collaboration(node_num, task_prompt, model):
    edges = generate_topology(node_num)          # <-- the one slot to fill
    graph = build_dag(edges)                      # add -1/-2 sentinels, index the nodes
    order = topological_order(graph)             # schedule respecting the edges
    for node in order:
        artifacts = [incoming.artifact for incoming in node.predecessors]
        fused = aggregate(artifacts) if len(artifacts) > 1 else artifacts[0]
        node.artifact = critique_then_refine(node, fused, task_prompt, model)
    return graph.output_sentinel.artifact
```

`build_dag`, `topological_order`, `aggregate`, and `critique_then_refine` already exist;
`generate_topology` is the empty slot the design will fill.
