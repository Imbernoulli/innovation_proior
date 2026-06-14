Let me start from the failure mode I keep seeing when I ask one LLM for a hard artifact. It writes one draft and commits to it. If the draft has a missing case, a logic bug, or a hallucinated assumption, nothing in the one-shot generation forces a second look. Self-Refine gives me the first useful primitive: generate `y0 = M(p_gen || x)`, then ask for feedback `fb_t = M(p_fb || x || y_t)`, then revise `y_{t+1} = M(p_refine || x || y_t || fb_t)`, and repeat until the result is good enough or the budget ends. That is already a big conceptual shift: the object I care about is not a single response, it is a draft being repeatedly improved. But the same model is still writing, judging, and rewriting. It sees the artifact through the same lens that produced the artifact, so the review can miss exactly the blind spots I need it to catch.

CAMEL shows why a second role matters. If I split the conversation into an instruction-giving agent and an instruction-following agent, the two sides have different jobs. That is not cosmetic. It suppresses role flipping, instruction repetition, and fake replies, because the worker that evaluates or directs the next move is not also trying to be the worker that performs it. So I want the refinement loop, but I want the review and the production to be separated. The problem is that CAMEL is only a pair. It does not tell me how to add a third expert, or a hundred, or how to decide which worker should influence which other worker.

ChatDev gets closer because it composes these interactions into a longer process. A software task becomes a fixed chain of phases such as design, coding, and testing. Each phase contains an instructor/assistant dialogue, and, crucially, the whole dialogue is not dumped into every later phase. The local conversation stays local, while the extracted solution moves forward. That instinct is exactly right: pass the artifact, not all the chatter. Still, the structure is welded to one waterfall workflow. MetaGPT has the same shape with more standardized roles and documents. These are useful pipelines, but they are hand-built and small. They do not let me ask the scaling question: if I add more collaborating workers, and if I change the way they are connected, does the artifact keep improving?

I need one language for "who can hand work to whom." The natural object is a graph: nodes are workers, and a directed edge means one worker's artifact informs another worker. That immediately contains the old cases: a pair is one edge, a waterfall is a path, a branching process is a tree-like structure, and a dense committee is a highly connected structure. But arbitrary directed edges create a wall. If there is a cycle, an artifact can flow back into one of its own ancestors. Then a revises after b, b revises after a, and I either need an arbitrary stopping rule or a task-specific cycle breaker. That is exactly the kind of bespoke machinery I am trying to avoid. So I forbid cycles. The structure is a directed acyclic graph,

  G = (V, E),  V = {v_i},  E = {<v_i, v_j> : i != j},  with no directed cycles.

Acyclicity is doing real work. It makes information backflow impossible, so every artifact moves downstream. It also gives me a legal schedule: process each node only after the nodes feeding it are done. In other words, the structure gives both data flow and termination.

Now I have to decide what lives on the node and what happens on the edge. If I put a full self-refining agent on each node and make the edge a passive wire, I lose the role split that CAMEL taught me to preserve. An edge is not just a wire; it is the moment when one artifact is inspected on its way to another worker. That is exactly where the reviewing role belongs. The node should produce the artifact, and the edge should supply the review directive. Formally, I can agentize both graph elements:

  a_i = rho(v_i),  for each node v_i,
  a_ij = rho(<v_i, v_j>),  for each edge <v_i, v_j>.

The node agent is the producer of the current artifact. The edge agent is the reviewer that reads the upstream artifact and issues a concrete instruction for the downstream producer. This preserves division of labor everywhere in the collaboration, not just inside a single pair: each handoff has a separate review role, and each artifact is produced by a worker that receives that review.

The schedule now falls out of the same acyclicity constraint. For an edge `<v_i, v_j>`, the upstream producer must run before the reviewer on that edge, and the reviewer must run before the downstream producer. If `I(.)` is the position in the execution order, then

  I(a_i) < I(a_ij) < I(a_j).

Operationally, I can implement this as a Kahn-style peel. At each round, take the current input layer, meaning the nodes with no remaining earlier inputs. For every outgoing handoff from those nodes, ask the receiving worker to improve the sender's artifact. Once all handoffs out of the layer are done, delete that layer and the visited handoffs. The next input layer is now ready. This is not a separate algorithmic flourish; it is just the executable form of "run a worker only when all earlier required work has arrived."

The next question is what travels along a handoff. The tempting answer is "all prior conversation," but that explodes. I want the arithmetic in front of me so I do not hand-wave the bottleneck. Let `n = |V|` be the number of nodes, `t` the task length, `p` the profile length, `i` the average instruction length, `s` the average artifact length, and `m` the maximum number of local interaction rounds between adjacent workers. The worst pressure is at a sink in the densest acyclic mesh. A directed acyclic mesh has one edge for each ordered-by-index pair, so the internal edge count is

  |E_mesh| = n(n - 1) / 2.

The complexity model also appends a single-source/single-sink completion around the interior nodes, contributing two extra visible handoffs for each of the `n - 2` interior nodes, hence `2(n - 2)`. Without context control, the sink can be forced to absorb the full local dialogue from every visible handoff. A local `m`-round interaction has `2m - 1` generated instruction/artifact turns, each on the scale of `(i + s)`, so the sink-side budget is

  O(n)_{w/o} = t + p + s + (2m - 1)(i + s) * ( n(n - 1)/2 + 2(n - 2) ).

The leading term is from the mesh edge count:

  n(n - 1)/2 = (n^2 - n)/2,

so for large `n`,

  O(n)_{w/o} ~ ((2m - 1)(i + s)/2) n^2 = C n^2,
  C = (2m - 1)(i + s)/2.

That is the wall. The token cost, latency, and context-window pressure grow quadratically before I even learn whether large collaboration is useful.

The only way out is to keep local work local. The review-and-rewrite dialogue inside one handoff can use its short-term context, but when that handoff finishes, the only thing allowed to move onward is the artifact it produced. The analysis, failed suggestions, and intermediate dialogue are not broadcast. This does not throw away the work: the worker has folded the useful information into the artifact. Now the sink does not ingest every full dialogue in the mesh. It sees a linear number of artifacts and only pays local interaction cost for the handoffs that are visible to it. The corresponding budget is

  O(n)_w = t + p + s + m(i + s) * ( (n - 1) + 2(n - 2) ).

The bracket is exactly

  (n - 1) + 2(n - 2) = 3n - 5,

so for large `n`,

  O(n)_w ~ 3m(i + s)n = C_bar n,
  C_bar = 3m(i + s).

The factor changes from `(2m - 1)` times a quadratic edge count to `m` times a linear visible-handoff count. The leading growth changes from `n^2` to `n`. That is the scalability constraint I need the implementation to respect: local dialogue can be rich, but global propagation must be artifact-only.

Now I can write the local handoff itself. If a worker receives no earlier artifact, it produces the first artifact from the task. If it receives an earlier artifact, the review side reads that artifact and the task, produces the highest-priority suggestion, optionally runs code when the suggestion asks for compilation feedback, and then the producing worker emits a complete revised artifact. The downstream worker, not the upstream worker, performs this update. That detail matters: along handoff `(u, v)`, worker `v` is the one that calls `update(task, artifact_from_u)`, matching the idea that the target refines the predecessor's artifact.

Convergence creates one extra case. If several artifacts arrive at the same worker, it cannot pick one arbitrarily. It has to wait until all expected inputs arrive, merge them into a single artifact, and only then continue. But this branch should not fire for a single input; with one incoming artifact there is nothing to merge, so the worker just uses the already refined artifact it received. The practical guard is therefore: aggregate only when the number of received artifacts equals the number of required incoming handoffs and is at least the configured aggregation unit count. With `Aggregate_unit_num = 2`, a single-input worker passes through its one refined artifact unchanged.

The simplest useful topology is now obvious. If I want pure sequential refinement, with no branching and no convergence, I use a path:

  0 -> 1 -> 2 -> ... -> (N - 1).

This path has exactly `N - 1` handoffs. The first real worker produces from the task through the input sentinel. Each later worker receives exactly one artifact, reviews and refines it, and passes the resulting artifact along. No worker ever receives two real inputs, so aggregation never fires. The tradeoff is clean: maximum refinement depth for `N` workers, zero parallel diversity. That makes it the natural first structure to implement and the clean baseline for richer structures.

I can now fill the three empty code slots from the scaffold. The handoff recipe is just the path, the worker update is review-then-produce, and the run loop is the layer-peel traversal with the input and output sentinels `-1` and `-2`.

```python
from typing import Callable, Dict, List, Tuple


NODE_IN = -1
NODE_OUT = -2
AGGREGATE_UNIT_NUM = 2


def choose_handoffs(worker_count: int) -> List[Tuple[int, int]]:
    """Path handoffs: 0 -> 1 -> ... -> worker_count - 1."""
    handoffs = [(i, i + 1) for i in range(worker_count - 1)]
    assert len(handoffs) == worker_count - 1
    return handoffs


class Worker:
    """A foundation-model worker with local state."""

    def __init__(self, worker_id: int, model):
        self.id = worker_id
        self.model = model
        self.received: Dict[int, str] = {}
        self.artifact = ""
        self.incoming_workers: List["Worker"] = []
        self.outgoing_workers: List["Worker"] = []

    def update(self, task: str, incoming: str) -> str:
        """The receiver refines the sender's artifact."""
        suggestions = "None."
        if incoming != "":
            suggestions = self.model.review(task, incoming)
            if suggestions.startswith("<API>compile()</API>"):
                compile_info = self.model.compile(incoming)
                suggestions = self.model.review(
                    task,
                    incoming,
                    extra_feedback=compile_info,
                    previous_suggestions=suggestions,
                )
        return self.model.produce(task, incoming, suggestions)

    def aggregate(self, task: str) -> str:
        """Merge several completed incoming artifacts into one."""
        artifacts = list(self.received.values())
        if len(artifacts) == 1:
            return artifacts[0]
        return self.model.combine(task, artifacts)


class CollaborationRun:
    """Builds the sentinels, applies handoffs, and returns the final artifact."""

    def __init__(
        self,
        handoffs: List[Tuple[int, int]],
        worker_count: int,
        make_worker: Callable[[int], Worker],
        aggregate_unit_num: int = AGGREGATE_UNIT_NUM,
    ):
        self.aggregate_unit_num = aggregate_unit_num
        self.workers: Dict[int, Worker] = {
            NODE_IN: make_worker(NODE_IN),
            NODE_OUT: make_worker(NODE_OUT),
        }
        self.workers.update({i: make_worker(i) for i in range(worker_count)})

        for sender, receiver in handoffs:
            self.add_handoff(sender, receiver)
        for i in range(worker_count):
            if not self.workers[i].incoming_workers:
                self.add_handoff(NODE_IN, i)
        for i in range(worker_count):
            if not self.workers[i].outgoing_workers:
                self.add_handoff(i, NODE_OUT)

    def add_handoff(self, sender_id: int, receiver_id: int) -> None:
        sender = self.workers[sender_id]
        receiver = self.workers[receiver_id]
        sender.outgoing_workers.append(receiver)
        receiver.incoming_workers.append(sender)

    def remove_handoff(self, sender: Worker, receiver: Worker) -> None:
        sender.outgoing_workers.remove(receiver)
        receiver.incoming_workers.remove(sender)

    def execute(self, task: str) -> str:
        self.workers[NODE_IN].artifact = ""
        active = dict(self.workers)
        expected_inputs = {
            worker_id: len(worker.incoming_workers)
            for worker_id, worker in self.workers.items()
        }

        while active:
            input_layer = [w for w in active.values() if not w.incoming_workers]
            if not input_layer:
                raise ValueError("cycle or unsatisfied handoff")

            visited: List[Tuple[Worker, Worker]] = []
            touched: Dict[int, Worker] = {}
            for current in input_layer:
                for receiver in list(current.outgoing_workers):
                    refined = receiver.update(task, current.artifact)
                    receiver.received[current.id] = refined
                    visited.append((current, receiver))
                    touched[receiver.id] = receiver

            for receiver in touched.values():
                all_inputs_ready = len(receiver.received) == expected_inputs[receiver.id]
                enough_to_merge = len(receiver.received) >= self.aggregate_unit_num
                if all_inputs_ready and enough_to_merge:
                    receiver.artifact = receiver.aggregate(task)
                elif receiver.received:
                    receiver.artifact = next(iter(receiver.received.values()))

            for sender, receiver in visited:
                self.remove_handoff(sender, receiver)
            for current in input_layer:
                active.pop(current.id, None)

        return self.workers[NODE_OUT].artifact
```

This code keeps the exact invariants I need. The handoff recipe creates only `(i, i + 1)` pairs, so the path has `N - 1` real handoffs. The input sentinel feeds the first real worker with an empty prior, so that worker produces from the task. Every real handoff is refined by the receiver through `receiver.update(task, current.artifact)`. A single received artifact becomes the worker's artifact directly; aggregation is gated by both "all inputs are present" and "at least two inputs are present." The output sentinel receives the final worker's artifact, so the returned value is the artifact at `-2`. The path is therefore pure sequential refinement: deep, ordered, artifact-only propagation, with the general merge branch present in the runner but never used by the path itself.
