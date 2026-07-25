The problem is deterministic consensus in a fully asynchronous message-passing system with at most one crash fault. Each process holds a one-bit input and must eventually write a one-bit output, and all outputs must agree; both 0 and 1 must be possible outcomes for some inputs. The network delivers every message eventually, but delays are unbounded and messages may arrive out of order, and there are no clocks or timeouts. The central difficulty is that a crashed process and a merely slow one look identical from outside: silence carries no information, so no process can safely infer a crash and commit without risking disagreement with a late-arriving message.

The usual synchronous approach relies on rounds and bounded delay, where missing a message in a round is itself a fault signal. In the asynchronous model that signal is gone. Every attempt to design a deterministic protocol stalls because at some moment the decision depends on hearing from a single process, and that process may have crashed or may simply be slow. The only way to close that window is to assume synchrony. This suggests the obstacle is not a design flaw but a fundamental limit, and the right response is to prove impossibility rather than construct yet another protocol.

The method is the FLP impossibility theorem, named after Fischer, Lynch, and Paterson. It proves that no deterministic consensus protocol can be totally correct in this model even when only one process may fail by crashing. The proof introduces the idea of bivalence: a configuration is bivalent if both decision values are still reachable from it, and univalent otherwise. The argument has two parts. First, some initial configuration must be bivalent, because otherwise adjacent initial configurations differing in one process's input would force the rest of the system to decide differently on inputs that look identical when that process crashes, which is impossible. Second, from any bivalent configuration and for any applicable event, there is a bivalent continuation reachable by a schedule ending in that event. If every continuation after the event were univalent, the proof uses the commutativity of steps by different processes to show that the decisive step would have to be a receive by the same process whose event was delayed; freezing that process then leaves the rest of the system able to reach both values, contradicting univalence. Chaining these stages round-robin yields an admissible run in which every configuration is bivalent, so no process ever decides.

The constructive content can be captured by an adversarial scheduler. Given any deterministic protocol, the scheduler starts from a bivalent initial configuration and repeatedly chooses the next process to step and its earliest pending message, then runs a finite schedule ending in that event while preserving bivalence. Because each process steps infinitely often and every message is eventually delivered, the run is fair; because bivalence is preserved at every stage, the run never decides. The theorem is tight: if processes may be dead only at the start and a strict majority are initially alive, consensus is possible.

```python
# Given ANY deterministic consensus protocol P, build an admissible run
# (every process steps infinitely often; every message eventually delivered)
# in which no process ever decides — the constructive content of the theorem.

def flp_nondeciding_run(P, N):
    C = find_bivalent_initial(P)          # Lemma 2: a bivalent initial configuration exists
    queue = list(range(N))                # round-robin => fairness, no starvation
    while True:                           # the run never terminates
        p = queue.pop(0)
        m = earliest_message_for(p, C)    # deliver oldest first => no message delayed forever
        # Lemma 3: a bivalent config is reachable from C by a schedule ending in (p, m).
        # The decisive step would be p's own receive; freezing p (a crash indistinguishable
        # from slowness) leaves the others able to reach both decision values.
        C = reach_bivalent_after(P, C, Event(p, m))
        queue.append(p)
        assert C.decision_value() is None # bivalent => never decides
```
