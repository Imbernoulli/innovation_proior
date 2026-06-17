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

The difficulty is that this is a search problem over an enormous space. The number of
candidate next actions at any state is the product of (all the free-form thoughts the model
could write) × (every available API) × (every possible parameter string) — for practical
purposes unbounded. The API calls are real and irreversible: a wrong call consumes budget,
changes state, and returns a possibly useless or error response. Complex instructions need
several tools chained over multiple rounds, so the chance of taking at least one wrong step
along the way is high, and the cost of a single bad early step is large. The precise goal is a
*reasoning/search strategy* that (1) lets the agent recover from a bad action instead of being
locked into it, (2) explores more than one line of attack through the action space, and
(3) does so without spending an unreasonable number of model queries — ideally collapsing to
the cheapest possible behavior on easy instructions and only paying more when the instruction
is genuinely hard. Existing strategies achieve at most one of these; closing the gap is the
problem.

## Background

By this time the dominant way to make an LLM solve a task that needs intermediate work is to
have it *write out* that work. Chain-of-thought prompting (Wei et al. 2022) shows that
eliciting intermediate reasoning steps before the final answer sharply improves performance on
arithmetic and multi-step reasoning — but a chain of thought is purely internal; it never
touches the world. For tasks that require *acting* (querying a knowledge source, calling a
tool, navigating an environment), the agent needs to interleave thinking with grounded
actions whose results feed back into the next thought. That interleaving is the prevailing
recipe for tool-use agents, and the practical lessons that come with it are the load-bearing
facts here.

The first lesson is that **a single trajectory is brittle**. When the agent commits to one
linear sequence of (thought → action → observation), an early mistake has nowhere to go: it
sits in the context and conditions every subsequent step. Two concrete failure modes are
observed repeatedly with single-chain agents on real APIs. (a) *Error propagation*: one
mistaken action propagates, and the model gets trapped in a faulty loop — for instance,
calling the same API the wrong way over and over, or hallucinating an API name that does not
exist and then continuing as if it had. (b) *Limited exploration*: a single chain explores
exactly one direction in the action space, so if that direction is unproductive the agent
never tries the alternatives that would have worked. These are not hypothetical; on complex
multi-tool instructions even the strongest available model of the time fails to find a valid
solution path with single-chain reasoning, which is precisely why high-quality solution paths
are hard to obtain at all.

The second lesson is about **decision retraction**. Standard interleaved-reasoning agents have
no mechanism to *undo* a committed action. Once `a_t` is taken, the agent moves forward; it
cannot decide, after seeing `r_t`, that `a_t` was a wrong turn and that it should go back to
the state before `a_t` and try something else. This is the structural root of error
propagation: without retraction, every error is permanent.

A third strand reframes problem-solving for LLMs as *deliberate* search rather than left-to-
right generation. Instead of one chain, one can let the model maintain several partial
solutions, generate candidate next steps, *evaluate* how promising each partial solution is,
and use a classical search procedure to decide which to pursue, look ahead, and backtrack.
This view supplies a vocabulary of reusable primitives — generating multiple candidate
continuations from a state, scoring or comparing partial solutions with the model itself, and
running a graph search (breadth-first or depth-first) over the resulting structure. The
demonstrations of this view, however, live on toy domains with small, countable next-step sets
(combine four numbers to make 24; fill a crossword), where the candidate set at each node can
be essentially enumerated and a single numeric self-evaluation is reliable enough to guide the
search. The tool-use setting violates both assumptions: the candidate set is unbounded, and
the environment is real and stateful.

A practical constraint frames everything: model queries cost money and time. Any strategy that
explores more must justify the extra queries. A method that, on an easy instruction, spends as
little as the cheapest single-chain agent — and only pays the exploration premium when the
instruction actually demands it — is far more attractive than one that pays a fixed high cost
everywhere.

## Baselines

These are the prior strategies a new tool-use reasoning method would be measured against and
would react to.

**Chain-of-thought (CoT) prompting (Wei et al. 2022).** Prompt the model to emit a sequence of
intermediate reasoning steps before its final answer. Core idea: the intermediate steps give
the model "room to compute," decomposing a hard one-shot mapping into easier sub-steps, which
markedly improves multi-step reasoning. *Algorithm:* a single forward generation — reason,
then answer; no interaction with any external environment. **Gap:** a chain of thought is
self-contained text; it never executes anything, so it cannot obtain real API responses, and
it is a single linear pass with no notion of trying an action, observing a result, and
revising. For tool use it cannot even get off the ground.

**ReAct (Yao et al. 2022).** Interleave reasoning and acting in one trajectory: the model
alternates between a *Thought* (free-form reasoning that tracks the plan, interprets the last
observation, handles exceptions) and an *Action* (a tool/API call), and the environment
returns an *Observation* that conditions the next Thought. Core idea: reasoning makes acting
better (the model plans and recovers from surprises) and acting makes reasoning better (the
model grounds its reasoning in real feedback rather than hallucinating). *Algorithm:* a single
chain `Thought_1 → Action_1 → Obs_1 → Thought_2 → ...` until the model emits a final answer.
**Gap:** it is still one trajectory and has no mechanism for decision retraction. An initial
mistaken action stays in the context and can lead to a cascade of subsequent errors; the agent
cannot back up to a prior state and choose a different action. It explores a single path
through the action space.

**Reflexion (Shinn et al. 2023).** Add a *verbal* learning loop on top of an interleaved-
reasoning agent: after a trial fails, the model reflects in natural language on *why* it
failed, stores that self-reflection in an episodic memory buffer, and retries the whole task
with the reflection prepended — a "semantic gradient" that steers the next attempt without any
weight updates. Core idea: failures carry usable information; reading your own post-mortem
before retrying improves the next trial. *Algorithm:* run a trial → if it fails, generate a
reflection → append to memory → restart the trial. **Gap:** the unit of revision is the *whole
trial*. Each retry starts over from the root with extra advice; the method does not branch
within a single trajectory, does not keep several partial paths alive at once, and does not
*select* among candidate next steps at an intermediate state. It corrects between attempts, not
within one.

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
2). On Game-of-24 this turns a few-percent chain-of-thought success rate into a large one.
**Gap:** the demonstrations live where the next-step set at each node is small and essentially
enumerable, the environment is simulated and resettable, and a single scalar per-state value
is calibrated enough to drive pruning via a fixed threshold `v_th`. None of that holds for an
agent acting on real APIs, where the candidate continuations from a state are unbounded
free-form (thought, API, parameters) triples, the actions are real and irreversible, and a
lone numeric self-score on a tool trajectory is noisy. The fixed breadth `b` and the
value-threshold pruning are tuned to brute-force-searchable puzzles, not to an open-ended,
budget-constrained tool-use loop.

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
