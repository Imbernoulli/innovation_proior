# Context: deterministic agreement in an asynchronous message-passing system

## Research question

A collection of separate processes, each holding a private bit, must come to a single common decision — one binary value that every healthy process adopts. The motivating instance is the **distributed transaction commit problem**: the data-manager processes that handled a transaction must all agree to *install* its results or all agree to *discard* them, because a split decision corrupts the database. The same shape recurs in replicated-state agreement, leader election, and clock settlement.

When the processes and the network are perfectly reliable, this is trivial: broadcast the bits, apply a fixed rule. The difficulty is that real systems suffer faults — a process can crash, a message can be delayed. We want a protocol that still reaches agreement when faults occur. The precise question here: in a **fully asynchronous** system — no bound on relative process speeds, no bound on message delay, no synchronized clocks, no way to time out — is there a *deterministic* protocol that is guaranteed to make the healthy processes agree, even if just **one** process may crash (stop, silently, at the worst possible moment)?

A solution would have to satisfy three things at once: **agreement** (no two processes decide differently), **validity** (the decision isn't hard-wired — both 0 and 1 are reachable outcomes, so the protocol actually reflects the inputs), and **termination** (every healthy process eventually decides, even when one process has crashed). The pain point that makes this hard, and that we want to either defeat or prove insurmountable, is a recurring observation about deployed commit protocols: each seems to have a "window of vulnerability," an interval in which the inaccessibility of a single process can stall the whole protocol indefinitely. Folklore holds this window is an accident of imperfect design. The question is whether it is fundamental.

## Background

The setting is **asynchronous message passing**. Processes communicate only by messages; the message system delivers correctly and exactly once, but with *arbitrary* delay and *out of order*. There is no shared clock and no upper bound on how long a process takes between steps or how long a message sits in flight. The decisive consequence, observed repeatedly in practice, is **the indistinguishability of a slow process from a dead one**: if a process has sent nothing for a long while, no other process can tell whether it has crashed or is merely running slowly with its messages still in transit. Any mechanism that "detects" a crash — a timeout — is therefore unavailable in the pure asynchronous model, because it would require a known bound on delay.

The fault we care about is the mildest possible: **fail-stop (crash)**. A faulty process simply halts at some point and sends nothing further; it never lies, never sends spurious messages, never deviates from its program before halting. This is far weaker than a malicious or arbitrary ("Byzantine") fault. We deliberately use the weak crash fault and a *reliable* message system, so that any impossibility we find cannot be blamed on adversarial messages or unreliable links.

A formal object for "the state of the whole system" is needed: a **configuration** — the internal state of every process together with the multiset of messages sent but not yet delivered. The system evolves one **step** at a time: a single process attempts to receive a message, and based on what it gets (a particular message, or nothing) deterministically updates its state and sends a finite set of messages. Because each process is deterministic, a step is fully determined by *which* process moves and *which* message event (if any) it receives. A **schedule** is a sequence of such moves; it carries one configuration to another.

Two facts about this model are load-bearing and knowable before any protocol is designed. First, **a process can always take a step**: receiving "nothing" is always a legal move, so the system never gets stuck for purely structural reasons — non-termination, if it happens, is about *deciding*, not about *moving*. Second, **steps by different processes commute**. If two schedules are both applicable from the same configuration and touch *disjoint* sets of processes, then running either one first leaves the other schedule's process states and selected message events still available; the first schedule may add extra messages, but it does not change what the other schedule consumes. Both orders therefore perform the same local transitions, consume the same old messages, add the same new messages, and reach the *same* configuration. This "diamond" property is a direct consequence of the model's locality.

The relevant **diagnostic phenomenon** from prior practice is exactly the window of vulnerability in asynchronous commit protocols: an existing protocol can be driven to wait forever on one tardy participant. This is a fact about the *existing* protocols, established before any new result — the kind of observation a theory should either explain away or prove unavoidable.

## Baselines

The prior art is the family of agreement protocols that *do* work, and understanding precisely *what they rely on* is the point.

**Synchronous interactive consistency — Pease, Shostak & Lamport (1980), "Reaching Agreement in the Presence of Faults."** Here *n* processors, of which at most *m* may be faulty (and arbitrarily so — they may lie), each hold a private value, and the goal is for the nonfaulty processors to compute a common vector with the true value in each nonfaulty slot. They prove this is solvable **if and only if n ≥ 3m + 1** (so four processors for one fault), by a recursive message-relay protocol over a fixed number of rounds; with authentication (faulty processors can withhold but not forge), it works for arbitrary *n ≥ m*. Once the common vector is agreed, a fixed averaging/filtering rule yields exact agreement. The decisive assumption, stated explicitly, is that the communication medium is "of negligible delay" and proceeds in **rounds** — a process knows when a round is over, so *silence is itself information*: a missing message in a round identifies a fault. The gap this leaves: the whole construction is built on synchrony. Remove the round structure and the bounded delay, and "I heard nothing this round" no longer means "the sender is faulty" — it could just be slow.

**The Byzantine Generals — Lamport, Shostak & Pease (1982).** The same synchronous agreement result recast: loyal generals must agree on a plan despite traitors; solvable with oral messages exactly when more than two-thirds are loyal (the *n ≥ 3m+1* bound again), and for any number with signed messages. The contribution is tolerance of *arbitrary* (malicious) faults — strictly harder faults than crashes. Yet it, too, lives entirely in the synchronous, round-based world; its correctness leans on the ability to know that a round has elapsed.

**The Two Generals' problem — Akkoyunlu, Ekanadham & Huber (1975), named the "Two Generals' Paradox" by Jim Gray (1978).** Two parties coordinating an attack over an *unreliable* messenger channel can never reach common certainty of a shared time: every acknowledgement itself needs acknowledging, an infinite regress, so no deterministic protocol works. This is the closest existing impossibility in spirit. But its hardness comes from **message loss** on an unreliable channel. It leaves open the deeper question: what if the channel is *reliable* (every message eventually arrives) but the system is merely asynchronous and one process may crash? Is the difficulty really about lost messages, or about something subtler?

**The synchronous lower-bound line — Fischer & Lynch (1982).** In the synchronous Byzantine model, reaching agreement provably needs *t+1* rounds to tolerate *t* faults. This establishes that even where agreement *is* solvable, fault tolerance has an unavoidable cost measured in rounds — again a result whose very statement presumes a round structure that the asynchronous model lacks.

The common limitation across all of these: **they purchase agreement with synchrony.** Each relies, directly or indirectly, on bounded delay / rounds / the ability to interpret silence as failure. None speaks to the fully asynchronous model where a crashed process and a slow process are indistinguishable.

## Evaluation settings

This is a theoretical question, so the "yardstick" is a model and a specification, not a dataset. The model is the asynchronous message-passing system above: *N ≥ 2* deterministic processes with one-bit inputs and write-once binary outputs; a reliable but arbitrarily-delaying, reordering message system that is *fair* (a message is eventually delivered to any process that keeps trying to receive); configurations and steps as defined; and the fault model of at most one crash. The specification to be met or refuted is the conjunction **agreement + validity + termination-despite-one-crash**, with "termination" weakened as far as possible — it suffices that *some* process eventually decides — so that an impossibility is as strong as possible. The natural notion of a "run that should succeed" is an **admissible** run: at most one process faulty, and every message to a healthy process eventually received. A protocol is judged a solution if and only if *every* admissible run decides. The companion question — under what *restricted* failure pattern agreement *is* achievable — is measured against the same model with a stronger liveness assumption (e.g., processes may be dead at the start but none dies mid-run, with a strict majority initially alive).

## Code framework

A generic simulator for an asynchronous message-passing system separates the deterministic process transition from the scheduler that chooses delivery order and process steps.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Event:
    """A single atomic step: process `p` receives message event `m`
    (or m=None meaning 'received nothing this attempt')."""
    p: int
    m: Optional[object]

@dataclass
class Configuration:
    """Whole-system state: each process's internal state + the message buffer (a multiset)."""
    states: tuple          # internal state per process (input reg, output reg, program counter, storage)
    buffer: tuple          # multiset of in-flight (dest, value) messages

    def decision_value(self):
        # returns 0 or 1 if some process is in a decision state, else None
        pass               # TODO: read the write-once output registers

class Protocol:
    """The per-process deterministic transition function under design."""
    def initial_config(self, inputs) -> Configuration:
        pass               # TODO

    def step(self, config: Configuration, event: Event) -> Configuration:
        # apply receive(p) -> deterministic state update + finite set of sends
        pass               # TODO: the transition function we are trying to design

def applicable(event: Event, config: Configuration) -> bool:
    # (p, None) is always applicable; (p, m) is applicable exactly when that message is in the buffer
    pass                   # TODO

def run(protocol: Protocol, config: Configuration, schedule) -> Configuration:
    for event in schedule:
        assert applicable(event, config)
        config = protocol.step(config, event)
    return config

class Scheduler:
    """Chooses which event happens next — the order of deliveries and which
    process moves. Must keep runs admissible (fair: <=1 crash, every message
    to a live process eventually delivered)."""
    def next_event(self, config: Configuration) -> Event:
        pass               # TODO: the scheduling strategy
```

The open question is whether any filling-in of `Protocol.step` makes *every* admissible schedule produced by *every* `Scheduler` reach a `decision_value`.
