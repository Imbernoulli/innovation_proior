## Research question

We want a language model to act as an agent that fulfills a human instruction by calling
external tools (APIs). The interaction is inherently *multi-step*: given an instruction
`Inst`, the model must emit a sequence of actions `{a_1, ..., a_N}`, where at round `t` it
produces `a_t` conditioned on everything seen so far,
`p(a_t | a_1, r_1, ..., a_{t-1}, r_{t-1}, Inst)`, and `r_t` is the *real* response returned
by executing the chosen API. Each action `a_t` is a triple — a free-form thought, the name of
one API to call, and the concrete parameters for that call — and the model can also choose to
stop, either by declaring a final answer or by giving up. The agent succeeds only if it
arrives at a final answer that actually resolves the instruction within a fixed budget of
model queries.

This is a search problem over a large space. The number of candidate next actions at any
state is the product of (all the free-form thoughts the model could write) × (every available
API) × (every possible parameter string) — for practical purposes unbounded. The API calls
are real and irreversible: a call consumes budget, changes state, and returns a response.
Complex instructions need several tools chained over multiple rounds. The question is what
*reasoning/search strategy* the agent should follow as it works through the action space
under a model-query budget.

## Background

By this time the dominant way to make an LLM solve a task that needs intermediate work is to
have it *write out* that work. Chain-of-thought prompting (Wei et al. 2022) shows that
eliciting intermediate reasoning steps before the final answer sharply improves performance on
arithmetic and multi-step reasoning; a chain of thought is purely internal and never touches
the world. For tasks that require *acting* (querying a knowledge source, calling a tool,
navigating an environment), the agent interleaves thinking with grounded actions whose results
feed back into the next thought. That interleaving is the prevailing recipe for tool-use
agents.

One property of the interleaved recipe is that **a single trajectory is one linear sequence**.
When the agent commits to one linear sequence of (thought → action → observation), each step
sits in the context and conditions every subsequent step. On complex multi-tool instructions,
even the strongest available model of the time often does not find a valid solution path with
single-chain reasoning, which is part of why high-quality solution paths are hard to obtain.

A second property concerns **decision retraction**. Standard interleaved-reasoning agents move
forward once `a_t` is taken; after seeing `r_t`, the agent's default behavior is to condition
the next action on the full prefix and continue rather than to return to the state before
`a_t`.

A third strand reframes problem-solving for LLMs as *deliberate* search rather than left-to-
right generation. Instead of one chain, one can let the model maintain several partial
solutions, generate candidate next steps, *evaluate* how promising each partial solution is,
and use a classical search procedure to decide which to pursue, look ahead, and backtrack.
This view supplies a vocabulary of reusable primitives — generating multiple candidate
continuations from a state, scoring or comparing partial solutions with the model itself, and
running a graph search (breadth-first or depth-first) over the resulting structure. The
demonstrations of this view live on toy domains with small, countable next-step sets (combine
four numbers to make 24; fill a crossword), where the candidate set at each node can be
essentially enumerated and a single numeric self-evaluation guides the search. The tool-use
setting differs: the candidate set is unbounded, and the environment is real and stateful.

A practical constraint frames everything: model queries cost money and time. Any strategy that
explores more spends more queries, and a run is capped by a maximum number of model queries.

## Baselines

These are the prior strategies a new tool-use reasoning method would be measured against and
would react to.

**Chain-of-thought (CoT) prompting (Wei et al. 2022).** Prompt the model to emit a sequence of
intermediate reasoning steps before its final answer. Core idea: the intermediate steps give
the model "room to compute," decomposing a hard one-shot mapping into easier sub-steps, which
markedly improves multi-step reasoning. *Algorithm:* a single forward generation — reason,
then answer; no interaction with any external environment.

**ReAct (Yao et al. 2022).** Interleave reasoning and acting in one trajectory: the model
alternates between a *Thought* (free-form reasoning that tracks the plan, interprets the last
observation, handles exceptions) and an *Action* (a tool/API call), and the environment
returns an *Observation* that conditions the next Thought. Core idea: reasoning makes acting
better (the model plans and recovers from surprises) and acting makes reasoning better (the
model grounds its reasoning in real feedback rather than hallucinating). *Algorithm:* a single
chain `Thought_1 → Action_1 → Obs_1 → Thought_2 → ...` until the model emits a final answer.

**Reflexion (Shinn et al. 2023).** Add a *verbal* learning loop on top of an interleaved-
reasoning agent: after a trial fails, the model reflects in natural language on *why* it
failed, stores that self-reflection in an episodic memory buffer, and retries the whole task
with the reflection prepended — a "semantic gradient" that steers the next attempt without any
weight updates. Core idea: failures carry usable information; reading your own post-mortem
before retrying improves the next trial. *Algorithm:* run a trial → if it fails, generate a
reflection → append to memory → restart the trial.

**Deliberate tree search over thoughts (Yao et al. 2023).** Cast solving as search over a tree
whose nodes are partial solutions ("thoughts"). Four plug-in choices: how to decompose a
problem into thought-steps; a *thought generator* that produces candidate next thoughts from a
state (either *sample* several diverse ones, or *propose* a set deterministically); a *state
evaluator* `V(p_θ, S)(s)` that scores how promising each state is — either *value* mode (a
prompt makes the model emit a scalar, e.g. 1–10, or a class like sure/likely/impossible,
turned into a value) or *vote* mode (a prompt shows the model several states and has it vote
for the most promising); and a *search algorithm* over the tree. Two searches are given:
breadth-first, which keeps the `b` most promising states at each depth (Algorithm 1), and
depth-first, which expands the most promising state first and, when a state's value falls below
a threshold `v_th`, prunes that subtree and *backtracks to the parent* to continue (Algorithm
2). On Game-of-24 this turns a few-percent chain-of-thought success rate into a large one. The
demonstrations are set where the next-step set at each node is small and essentially
enumerable, the environment is simulated and resettable, and a single scalar per-state value
drives pruning via a fixed threshold `v_th`.

## Evaluation settings

The natural yardsticks for a tool-use reasoning strategy:

- **Instruction families over a large real-world API pool.** Instructions are grouped by how
  many tools they require: single-tool (I1), multi-tool drawn from the same API *category*
  (I2), and multi-tool drawn from the same API *collection* (I3). Generalization is probed by
  splitting into instructions over seen instructions/tools versus previously unseen
  instructions and unseen tools, so a strategy is tested both in-distribution and on novel
  APIs it must read documentation for at inference time.
- **Tool environment.** Each instruction comes with a set of available APIs whose
  documentation is fed to the model; the agent issues calls and receives real responses
  from the live endpoints, so the protocol mirrors deployment.
- **Answer judgment.** An automatic evaluator, backed by a strong LLM, decides whether a
  solution path *passes* — i.e. whether the final answer actually resolves the instruction,
  with explicit rules for solvable vs. unsolvable instructions and for the two finish types
  (final-answer vs. give-up), aggregated by majority vote over several judge samples. A
  pairwise *win-rate* judgment compares two solution paths for relative quality. The evaluator
  is validated against human agreement.
- **Budget and efficiency.** Each run is capped by a maximum number of model queries; the
  number of queries spent and the rate at which the agent gives up are recorded as efficiency
  and diagnostic signals.
- **Protocol.** The same reasoning/search strategy is run across multiple agent backbones and
  reported per instruction family; the metric of record is answer quality (pass / win rate),
  with query count and give-up rate as secondary signals.

## Code framework

The strategy plugs into an existing tool-use agent harness. The harness already supplies an
interaction-state record (with its parent pointer, child list, depth, terminal/stop flags, and the
status code from the last tool call), a one-step primitive that asks the model for the next
thought/action and executes any resulting tool call (returning the new state(s)), finish handling
for answer vs. give-up, a couple of optional helpers — one that perturbs the next model call toward
a different action than ones already tried from a state, and one that uses the model to compare a
set of candidate states (at the cost of extra model calls) — and counters for the model-call budget.
What remains empty is the control policy that decides how to use these ordinary agent primitives
under the budget.

```python
class AgentState:
    """A point in the interaction. Already provided by the harness."""
    def __init__(self):
        self.children = []           # successor states already generated from here
        self.father = None           # parent state
        self.messages = []           # conversation prefix that produced this state
        self.is_terminal = False     # reached a finish state
        self.pruned = False          # this partial interaction should not continue
        self.observation_code = 0    # status code from the last tool call

    def get_depth(self):
        ...

    def make_finish(self, inter_val):
        ...


class ToolAgentController:
    """Owns the interaction state and a model-call budget.

    Provided primitives:
      self._step(state)               -> one model call + tool execution; returns new leaf state(s)
      self._add_diversity_prompt(s)   -> perturb the next call away from prior actions at s;
                                         returns whether a prompt was added
      self._rank_nodes(candidates)    -> model-based comparison of candidate states (costs calls)
    Provided state:
      self.query_count, self.max_query_count
      self.now_expand_num
      self.terminal_node (list), self.give_up_node (list)
      self.answer_count              # stop once this many answers are found
      self.single_chain_max_step     # maximum interaction depth
      self.tree_beam_size            # how many candidate continuations to draw per state
      self.status                    # 1 once an answer is found
    """

    def search(self, root_state):
        # TODO: the control policy we will design.
        #       Decide which state to extend next, in what order, when to back up,
        #       and when to stop — given the primitives and budget above.
        pass
```

The harness supplies state construction, one-step model/tool execution, the perturb/compare
helpers, finish handling, and budget bookkeeping; `search` is where the missing policy will live.
