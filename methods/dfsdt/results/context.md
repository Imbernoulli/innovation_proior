## Research question

An LLM is being used as a *planner* for a multi-step tool-use task. Given a natural-language
instruction and a set of API tools (each with its documentation, required and optional
parameters, and an executable call format), the model must decide, round by round, which API
to call, with what arguments, and in what order, until it can return a final answer. This is
cast as a multi-round conversation: at round `t` the model emits an action `a_t` of the form
`Thought: ... ; API Name: ... ; Parameters: ...`, the environment actually executes the call
and returns a real response `r_t`, and the model conditions its next action on the full
history `{a_1, r_1, ..., a_{t-1}, r_{t-1}}` plus the instruction. Two special finishing
actions are always available: "Finish with Final Answer" (with the answer as a parameter) and
"Finish by Giving Up" (used when the provided APIs cannot complete the task after several
attempts).

The difficulty is the size of what must be searched. The action at each round is a
free-form thought, *and* a choice of API from a large pool, *and* a fill-in of that API's
parameters — the product of these is effectively an infinite decision space, so it cannot be
enumerated or brute-forced. On top of that, the environment is unforgiving: a real API call
can fail, return an error, time out, or return an irrelevant payload, and once the model has
committed to a bad action the conversation history now contains that mistake. Even the
strongest available models often fail to find a single valid solution path on the harder,
multi-tool instructions. The precise goal is a search/reasoning policy that (1) raises the
chance of finding at least one valid path on complex instructions, (2) tolerates a wrong
action without the whole attempt collapsing, and (3) does all of this within a bounded number
of LLM calls, because each call costs money and latency. The yardstick is the proportion of
instructions for which *some* valid path is found per unit of query budget.

## Background

By 2023 the dominant way to get multi-step behavior out of a frozen LLM is prompting, not
fine-tuning, because the models are too large and the per-task data too scarce to train a
controller with gradient descent. Several prompting paradigms for reasoning and acting are in
play, and they form the conceptual ground here.

The first relevant frame is **reasoning-as-text**. Chain-of-thought prompting (Wei et al.
2022) showed that asking a model to emit a chain of intermediate "thoughts" `z_1, ..., z_n`
before its answer dramatically improves performance on tasks where the input-to-output
mapping is non-trivial. Each thought is a coherent language step toward the solution. But this
reasoning is, in ReAct's phrase, a "static black box": the model reasons purely over its own
internal state with no access to the external world, so it cannot check a fact, cannot react
to what an environment actually returns, and can confidently hallucinate intermediate facts
that then propagate to a wrong answer.

The second frame is **acting-as-text**, and specifically interleaving it with reasoning. The
clean formalization (Yao et al. 2022, ReAct) is an agent that at step `t` receives an
observation `o_t` from the environment and takes an action `a_t` from an action space `A`,
following a policy `π(a_t | c_t)` over the context `c_t = (o_1, a_1, ..., o_{t-1}, a_{t-1},
o_t)`. The move is to *augment* the action space to `Â = A ∪ L`, where `L` is the space of
language: a "thought" `â_t ∈ L` does not touch the environment and produces no observation,
it just rewrites the context to support later reasoning, while a real action `a_t ∈ A`
executes and returns an observation. Interleaving thought and action lets the model decompose
the task, inject knowledge, track progress, and adjust plans, all grounded in real
observations. This is the natural template for a tool-use agent: a thought, then a concrete
API call, then its real response, repeated.

A third frame concerns **recovering from failure**. Once an agent can act, the question
becomes what to do when an action turns out to be wrong. One line of work (Shinn et al. 2023,
Reflexion) treats this as "verbal reinforcement": after an episode fails, the model reflects
in natural language on what went wrong, stores that reflective summary in a memory buffer, and
then runs the *whole task again from the beginning* with the reflection prepended to its
context, so the binary success/failure signal is converted into a textual hint for the next
attempt. This is a between-episode loop: each trial restarts at the root.

The motivating diagnostic findings about these existing systems are themselves part of the
ground. Pilot studies of CoT- and ReAct-style single-trajectory prompting on real multi-tool
instructions surface two recurring failure modes. **Error propagation:** a single mistaken
action — calling an API with the wrong parameters, or hallucinating an API that does not
exist — pushes the conversation into a faulty region from which the model keeps making related
mistakes (for instance, repeatedly calling the same API the same wrong way, or looping on a
nonexistent API). Because the bad action is now in the history, the model conditions on it and
compounds the error. **Limited exploration:** a single linear trajectory commits to one path
through the action space; if that path is unlucky, the attempt simply fails, and rerunning the
same linear procedure does not reliably help on the hardest instructions. These two
observations — that a flat trajectory both *propagates* early errors and *under-explores* the
space — are the empirical pressure that any better strategy has to answer.

A final piece of background is the cost structure that governs every design choice here.
There is no cheap value oracle: the only way to judge a partial trajectory or to compare two
candidate actions is to spend more LLM calls. Concretely, if a strategy generates `n`
candidate children at a node and uses the LLM to compare them pairwise to pick the best, that
sorting costs on the order of `O(n log n)` LLM calls, and this comparison is typically the
single most expensive part of any tree-structured procedure. So any exploration scheme is
implicitly trading pass rate against query count, and the number of LLM calls per instruction
is a first-class quantity, not an afterthought.

## Baselines

These are the prior strategies a new search policy is measured against and reacts to.

**Chain-of-thought / single-chain reasoning (Wei et al. 2022).** Emit thoughts then an
answer; in a tool-use harness this becomes a single chain where the model thinks, optionally
calls one tool, reads the response, and continues, with no environment grounding in the pure
CoT case. **Gap:** ungrounded reasoning over an internal black box — it cannot verify against
real API responses and is prone to hallucinating facts and APIs, with no way to react to what
actually came back.

**ReAct (Yao et al. 2022).** Interleave language thoughts with real actions over the
augmented space `Â = A ∪ L`, conditioning each new action on the real observation history.
This grounds the reasoning and is the right per-step shape for tool use. Concretely it
produces a single linear trajectory: think, act, observe, think, act, observe, ..., until a
finish action or a step limit. **Gap:** there is no mechanism for decision *retraction*. The
trajectory only ever moves forward, so an early wrong action stays in the context and
propagates, and a single committed path explores only one corner of the action space. Running
ReAct again as an independent retry ("ReAct@N", repeating until the budget is spent and
counting a pass if any run succeeds) spends the same budget but still consists of independent
flat trajectories with no shared structure.

**Reflexion (Shinn et al. 2023).** Add a between-episode self-reflection loop on top of an
acting agent: on failure, summarize the mistake verbally, store it, and retry the whole task
from the start with that summary in context. This is the first real attempt to *use* failure
information to improve the next attempt. **Gap:** the unit of backup is the entire episode —
every retry restarts at the root. There is no way to keep the part of a trajectory that was
working and revise only the step that went wrong; a single bad action late in an otherwise
good path forces redoing the whole path. The recovery is coarse-grained.

**Tree-structured deliberation with a value heuristic (Yao et al. 2023, concurrent).** Frame
problem solving as search over a tree whose nodes are partial solutions (states `s = [x,
z_{1...i}]`). A thought generator `G(p_θ, s, k)` produces `k` candidate next steps from a
state; a state evaluator `V(p_θ, S)` scores states (e.g. a 1-10 value or a comparative vote)
to serve as a search heuristic; and a classical search algorithm (breadth-first or
depth-first) uses that heuristic to choose which state to expand and to prune states whose
value falls below a threshold, backtracking when a branch is deemed hopeless. This adds the
two things a flat trajectory lacks — local exploration of *different* continuations, and
lookahead/backtracking guided by evaluation. **Gap for this setting:** the approach is
demonstrated on tasks with a small, essentially brute-forceable state space (e.g. arithmetic
puzzles, crosswords) where a value heuristic is both cheap and meaningful. In the present
setting the decision space is infinite, and the value/ranking step — comparing candidate
states with the LLM — is exactly the dominant cost identified above (`O(n log n)` LLM calls
per node). Spending the bulk of the budget on scoring nodes, in a space where one good path is
all that is needed, is the cost this setting cannot afford.

**Classical depth-first search with child ranking.** The textbook way to add exploration to a
linear agent: at each node generate several candidate children, use the LLM to rank them
(pairwise comparisons), expand the best-scoring child first, and on reaching a terminal or a
dead end backtrack to nearby unexpanded nodes. This does find valid paths on hard
instructions. **Gap:** the ranking/sorting of children is the most resource-intensive part —
the `O(n log n)` LLM-comparison cost — so the strategy buys exploration at a query budget that
scales poorly, which is precisely the quantity this setting is trying to conserve.

## Evaluation settings

The natural yardsticks are:

- **Real-world API tool-use instructions** drawn from a large marketplace of REST APIs
  (thousands of tools across dozens of categories), each instruction paired with a sampled
  set of candidate APIs and their documentation. Instructions come in single-tool and
  multi-tool variants of increasing difficulty (single-tool I1; intra-category multi-tool I2;
  intra-collection multi-tool I3), so a strategy can be measured separately on easy and hard
  regimes.
- **Backbones:** the same search policy is run unchanged across several frozen instruction-
  following LLM backbones, so the strategy's effect is isolated from the choice of model.
- **Metric: pass rate** — the fraction of instructions for which the agent produces a valid
  final answer within a fixed query budget. This is the primary signal (whether a valid path
  was found at all). A solvability judgment (is the instruction answerable with the given
  tools at all) and a final-answer judgment are made by an LLM-based evaluator with majority
  voting over several predictions.
- **Efficiency / diagnostic signals:** the average number of LLM queries per instruction
  (lower is better) and the give-up rate (fraction of instructions the agent abandons). To
  compare an exploring strategy fairly against a flat one, a budget-matched control runs the
  flat strategy repeatedly until it has spent the same number of queries.
- **Protocol:** the tool environment, the candidate-API set per instruction, the prompts, and
  the per-call decoding parameters are held fixed; only the search/reasoning policy varies.

## Code framework

The search policy plugs into a fixed tool-use harness. The harness gives a tree
representation of the interaction (each node carries its message history, its node type,
whether it is terminal, whether it has been pruned, its children, and the status code of the
last API call), a single-step primitive that performs one LLM call plus the resulting tool
execution and attaches the new leaf nodes, and a costly primitive that uses the LLM to rank a
set of candidate nodes. The budget, the per-node branching factor, the depth limit, and the
terminal/give-up bookkeeping are all provided. The one empty slot is the policy itself: how to
walk this tree, when to go deeper, when to back up, and how far.

```python
class TreeSearch:
    """Fixed harness for an LLM tool-use agent operating over a tree of partial
    action sequences. Owns the tree, the query budget, and the per-node bookkeeping.
    The search policy is the one empty slot."""

    def __init__(self, llm, env, single_chain_max_step, tree_beam_size, max_query_count,
                 answer_count):
        self.llm = llm
        self.env = env
        self.single_chain_max_step = single_chain_max_step   # tree depth limit
        self.tree_beam_size = tree_beam_size                 # children attempted per node
        self.max_query_count = max_query_count               # hard LLM-call budget
        self.answer_count = answer_count                     # stop after this many valid paths
        self.query_count = 0
        self.now_expand_num = 0                              # monotone visit counter
        self.terminal_node = []                             # nodes that produced a final answer
        self.give_up_node = []                              # nodes that gave up

    # ---- primitives the harness already provides ----
    def _step(self, node):
        """One LLM call plus any resulting tool execution from `node`; appends and
        returns the final leaf node produced by that round. Increments self.query_count.
        The leaf carries .is_terminal (final answer reached), .pruned (API error / dead
        end), and .observation_code (the API status: e.g. final-answer code, give-up
        code, hallucinated-API code)."""
        ...

    def _rank_nodes(self, candidates):
        """LLM pairwise ranking of candidate nodes; ~O(n log n) extra LLM calls. Costly."""
        ...

    # ---- the slot to fill ----
    def search(self, root_node):
        """The search/reasoning policy over the tree. Decide which node to expand next,
        when to go deeper, when to back up, and how far, subject to the query budget."""
        # TODO: the search policy we will design.
        pass
```

The harness supplies one tree, one step primitive, and an optional expensive ranking
primitive; `search()` is where the policy that drives the exploration will live.
