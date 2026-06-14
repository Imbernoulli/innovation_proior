## Research question

A single LLM prompted once for a hard artifact — a program, a software project — commits to one
draft, and whatever omission, logic bug, or hallucinated assumption that draft carries ships with it.
Multi-agent collaboration fixes this by routing the artifact through several agents: each receives a
predecessor's solution, reviews it, and produces an improved version, and where several solutions
converge on one agent they are aggregated. The runtime that does this — actors on nodes, critics on
edges, artifact-only propagation, topological-order execution — is **fixed**. The one thing being
designed is the **collaboration topology**: the directed acyclic graph over the agent nodes that
decides who hands work to whom. The topology, and only the topology, sets the balance between depth
(iterative refinement down a path), diversity (parallel independent takes), and synthesis (merging
convergent solutions). Everything else about the system is frozen.

## Prior art before the first rung (the collaboration lineage)

The runtime the first rung fills is the resolution of a line of multi-agent and refinement methods.
These are the ancestors the first topology reacts to; the fixed substrate below is what they
converged to.

- **Self-Refine (Madaan et al. 2023).** One model improves its own output: generate `y0`, then loop
  feedback `fb_t = M(p_fb || x || y_t)` and revision `y_{t+1} = M(p_refine || x || y_t || fb_t)` until
  a stop rule. No retraining, no second model. Gap: a single model is writer, reviewer, and reviser at
  once, so the review is trapped behind the same blind spots that produced the draft.
- **CAMEL (Li et al. 2023).** Two role-played agents — an instruction-giving user and an
  instruction-following assistant — cooperate through inception prompting. The role split suppresses
  role flipping, instruction repetition, and rubber-stamp replies, because the worker that evaluates
  is not the worker that performs. Gap: it is a fixed dyad, not a recipe for organizing many workers
  or for deciding who influences whom.
- **ChatDev (Qian et al. 2023).** A software task becomes a fixed chat chain of phases (design,
  coding, testing), each an instructor/assistant dialogue; crucially the local conversation stays
  local while only the extracted solution moves forward. Gap: the structure is welded to one waterfall
  workflow, hand-built and small — no way to vary collaboration shape or team size.
- **MetaGPT (Hong et al. 2023).** Human standard operating procedures become an assembly line of role
  agents (product manager, architect, engineer) passing structured documents. Gap: same workflow
  specificity — the procedure and roles are engineered for one process, not generated from a few
  structural knobs.
- **GPTSwarm (Zhuge et al. 2024).** Agent systems as computational graphs (node = operation, edge =
  information flow), focusing on DAGs, with prompts and connectivity *optimized* per task. Validates
  the agents-as-DAG abstraction. Gap: it learns one task-specific wiring with every node and edge
  hand-customized — the wrong question for a study that wants to *fix* a runtime and vary the
  structure on equal footing.

The common object behind all of these is "nodes that produce work, connected by directed edges along
which work flows." A pair is one edge, a waterfall is a path, a branching process is a tree, a dense
committee is a highly connected graph. Forbidding cycles (a DAG) kills information backflow and
guarantees a topological order to run in. That DAG, with the machinery held fixed and only the edge
set varied, is the substrate the ladder climbs.

## The fixed substrate

A MacNet-style collaboration runtime is frozen and must not be touched. Agents are nodes of a DAG;
an **actor** on each node produces the current artifact, and a **critic** on each edge reads the
upstream artifact and instructs the downstream actor to refine it. The system runs the agents in
topological order by Kahn peeling: take the wavefront of predecessor-free nodes, drive each of their
outgoing edges through the critic-then-actor refinement, deposit the refined artifact at the child,
then delete the wavefront and repeat. **Memory control** is built in: short-term memory holds the
working back-and-forth inside one edge interaction, but long-term memory keeps **only the final
artifact** — the dialogue is forgotten, so per-agent context stays linear rather than quadratic in the
number of agents. A node with several incoming edges is **convergent**: it aggregates its incoming
artifacts into one before refining, rather than picking one or flatly concatenating. The runtime
automatically wraps the topology with an **input sentinel** (`-1`) feeding the task to every source
node and an **output sentinel** (`-2`) draining every sink node into one final artifact. The
underlying LLM backbones, the actor/critic prompts, the aggregation machinery, and the evaluators are
all fixed — none of them is the design surface.

## The editable interface

Exactly one region is editable — the body of `generate_topology(node_num)` in
`chatdev-macnet/custom_topology.py`. It returns a list of directed edges `(source, target)` forming
a DAG over nodes `0 .. node_num-1`; the runtime adds the `-1`/`-2` sentinels itself. The contract:

- The returned edges must form a valid DAG (no cycles).
- Every node `0 .. node_num-1` must be reachable from at least one path.
- Edges go from lower-indexed to higher-indexed nodes (so index order is already a valid topological
  order, and the DAG property is automatic).
- The topology is deterministic in `node_num`; cross-seed variability comes only from the LLM API.

Every topology on the ladder is a fill of this same function. The starting point is the scaffold
default: the **chain** — a sequential pipeline `0 → 1 → … → (node_num-1)`. Each later topology
replaces exactly this function body and nothing else.

```python
# EDITABLE region of custom_topology.py -- default fill (chain / sequential pipeline)
def generate_topology(node_num: int) -> list[tuple[int, int]]:
    """Design the multi-agent collaboration topology.

    Given N agent nodes (numbered 0 to node_num-1), return a list of
    directed edges (source, target) forming a DAG. The graph will
    automatically get input/output sentinel nodes (-1, -2) added.

    Constraints:
    - Must form a valid DAG (no cycles)
    - All nodes 0..node_num-1 should be reachable from at least one path
    - Edges must go from lower-indexed to higher-indexed nodes
    """
    # Default: chain topology (sequential pipeline)
    # Each agent improves upon the previous agent's solution.
    # 0 -> 1 -> 2 -> ... -> (node_num - 1)
    edges = []
    for i in range(node_num - 1):
        edges.append((i, i + 1))
    return edges
```

## Evaluation settings

Each topology is run with **4 agent nodes** (`node_num = 4`) across three settings — two code
benchmarks crossed with the MacNet backbone LLM — over a single seed (42); cross-seed variability is
not available because the topology is deterministic in `node_num` and the only stochasticity is the
LLM API.

| # | Benchmark | MacNet backbone | label |
|---|-----------|-----------------|-------|
| 1 | HumanEval (33 problems) | deepseek-chat | `humaneval-4-deepseek` |
| 2 | HumanEval (33 problems) | qwen2.5-72b-instruct | `humaneval-4-qwen` |
| 3 | SRDD (20 prompts) | deepseek-chat | `srdd-4-deepseek` |

Three metrics, higher is better on all: **`pass_at_1_deepseek`** and **`pass_at_1_qwen`** (HumanEval
— fraction of problems whose generated code passes all unit tests on the first attempt, on each
backbone) and **`srdd_exec_rate`** (SRDD — fraction of generated software projects whose entry point
`main.py` executes without crashing). A good topology should generalize across all three settings
rather than over-specialize to one model or benchmark.
