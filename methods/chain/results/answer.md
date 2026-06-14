# MacNet Chain Topology

MacNet chain is the path-shaped instance of the collaboration network: workers are connected
as `0 -> 1 -> ... -> (N - 1)`. It gives maximum sequential refinement depth for `N` workers,
has exactly `N - 1` real handoffs, and has no branching or multi-input merge point.

The canonical structure function is the path edge generator:

```python
def choose_handoffs(worker_count: int):
    handoffs = [(i, i + 1) for i in range(worker_count - 1)]
    assert len(handoffs) == worker_count - 1
    return handoffs
```

The runtime wraps this path with input and output sentinels `-1` and `-2`. The input sentinel
feeds an empty prior to the first real worker, so that worker produces from the task. Along
each real handoff `(u, v)`, the receiver `v` refines `u`'s artifact. A single incoming artifact
passes through as the receiver's solution; merging only occurs when all expected inputs are
present and the count is at least `Aggregate_unit_num = 2`. Therefore the chain never merges:
every real worker has only one real input.

## Token Complexity

Let `n = |V|`, `t` be task length, `p` profile length, `i` average instruction length, `s`
average artifact length, and `m` the maximum local interaction rounds. The densest acyclic
mesh has one ordered-by-index edge for every pair, hence `n(n - 1)/2` internal edges. The
single-source/single-sink completion contributes `2(n - 2)` additional visible handoffs.

Without artifact-only propagation, the sink can absorb each local dialogue, whose size is
`(2m - 1)(i + s)`:

```text
O(n)_w/o = t + p + s + (2m - 1)(i + s) * ( n(n - 1)/2 + 2(n - 2) )
         ~ C n^2,  C = (2m - 1)(i + s)/2.
```

With artifact-only propagation, the visible count is linear:

```text
O(n)_w = t + p + s + m(i + s) * ( (n - 1) + 2(n - 2) )
       = t + p + s + m(i + s)(3n - 5)
       ~ C_bar n,  C_bar = 3m(i + s).
```

The key change is the leading count: `n(n - 1)/2 ~ n^2/2` without control, versus
`(n - 1) + 2(n - 2) = 3n - 5 ~ 3n` with artifact-only propagation.

## Code Artifact

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
    """Foundation-model worker with local state."""

    def __init__(self, worker_id: int, model):
        self.id = worker_id
        self.model = model
        self.received: Dict[int, str] = {}
        self.artifact = ""
        self.incoming_workers: List["Worker"] = []
        self.outgoing_workers: List["Worker"] = []

    def update(self, task: str, incoming: str) -> str:
        """Receiver refines the sender's artifact."""
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
        artifacts = list(self.received.values())
        if len(artifacts) == 1:
            return artifacts[0]
        return self.model.combine(task, artifacts)


class CollaborationRun:
    """Layer-peel execution with sentinel input/output workers."""

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
