# The FLP Impossibility Theorem

**The result.** In a fully asynchronous message-passing distributed system, there is **no deterministic protocol that solves consensus while tolerating even a single crash (fail-stop) fault** — and this holds even though the message system is reliable and faults are non-malicious. Equivalently: any deterministic protocol that always satisfies agreement and validity has some admissible (fair, ≤ 1 crash) run that never decides.

**Why it is true, in one sentence.** Any alleged commitment can be localized to one process's receive step; because that process may be frozen without anyone else distinguishing crash from delay, a fair scheduler can keep the system in an undecided ("bivalent") state forever.

---

## Model

- *N ≥ 2* deterministic processes. Each has a one-bit input register, a **write-once** output register with values in {b, 0, 1} (b = blank/undecided), and unbounded internal storage. The transition function cannot rewrite the output once a value 0/1 is written (a **decision** is irrevocable).
- **Message system:** a multiset *buffer* of (destination, value) pairs. `send(p,m)` adds (p,m). `receive(p)` deletes and returns some (p,m), **or** returns the null marker ∅ leaving the buffer unchanged (this encodes arbitrary delay). Fairness: if `receive(p)` is performed infinitely often, every (p,m) in the buffer is eventually delivered.
- **Configuration** *C* = internal state of every process + buffer contents. **Initial configuration:** every process in an initial state, empty buffer.
- **Step / event:** a step is one process *p* performing `receive(p)`, obtaining a particular *m ∈ M ∪ {∅}*, then deterministically updating state and sending a finite set of messages. Determinism ⇒ the step is fixed by the **event** *e = (p,m)*; write *e(C)*. Note *(p,∅)* is always applicable, so a process can always move.
- **Schedule** σ = sequence of applicable events; σ(C) = resulting configuration. *Accessible* = reachable from an initial configuration.

**Correctness specification.**
- *Partially correct:* (1) **Agreement** — no accessible configuration has two different decision values; (2) **Validity** — for each *v ∈ {0,1}*, some accessible configuration decides *v*.
- A process is *nonfaulty* in a run if it takes infinitely many steps. A run is *admissible* if at most one process is faulty and every message to a nonfaulty process is eventually received. A run is *deciding* if some process reaches a decision state.
- *Totally correct in spite of one fault:* partially correct **and** every admissible run is deciding.

## Theorem

> **No consensus protocol is totally correct in spite of one fault.**

## Definitions for the proof

For a configuration *C*, let *V(C)* be the set of decision values of configurations reachable from *C*. *C* is **bivalent** if *|V(C)| = 2*, **univalent** (0-valent / 1-valent) if *|V(C)| = 1*. Reachability only shrinks valence: every successor of an *i*-valent configuration is still *i*-valent. A bivalent configuration is "not yet committed"; a decided configuration is univalent.

## Lemma 1 (commutativity / diamond)

If schedules σ₁, σ₂ from *C* lead to *C₁, C₂* and the **sets of processes taking steps are disjoint**, then σ₂ is applicable to *C₁*, σ₁ to *C₂*, and both reach the **same** *C₃*.
*Proof.* Disjoint-process schedules do not interact in the state used by their own events: neither reads the other's local state, neither removes the other's selected received messages, and new messages sent by the first schedule are only extra buffer entries while the second schedule consumes the same messages it consumed from *C*. Both orders make the same local updates, remove the same old messages, and add the same new messages. ∎

## Lemma 2 (a bivalent initial configuration exists)

*Proof.* Suppose not — every initial configuration is univalent. By validity there are 0-valent and 1-valent initial configurations. Initial configurations differing in exactly one process's input are **adjacent**; the input vectors form a connected hypercube, so there exist adjacent *C₀* (0-valent) and *C₁* (1-valent) differing only in the input of one process *p*. Since the protocol tolerates one crash, take an admissible deciding run from *C₀* in which *p* takes **no** steps; let σ be its schedule (it decides 0). σ touches only processes ≠ *p* and reads nothing of *p*'s differing input, so σ is applicable to *C₁* and yields a configuration identical to σ(*C₀*) except for *p*'s untouched state — hence the same decision, **0**. But *C₁* is 1-valent: contradiction. So some initial configuration is bivalent. ∎

## Lemma 3 (a bivalent continuation is always reachable)

Let *C* be bivalent and *e = (p,m)* applicable to *C*. Let 𝒞 be the configurations reachable from *C* **without** applying *e*, and 𝒟 = e(𝒞) = { e(E) : E ∈ 𝒞 }. Then 𝒟 **contains a bivalent configuration.**

*Proof.* First, *e* is applicable throughout 𝒞: its message event can be delayed arbitrarily, so it persists in the buffer along any *e*-free schedule (and (p,∅) is always applicable). Assume for contradiction every *D ∈ 𝒟* is univalent.

For *i = 0,1* let *Eᵢ* be an *i*-valent configuration reachable from *C* (exists since *C* is bivalent). If *Eᵢ ∈ 𝒞*, set *Fᵢ = e(Eᵢ) ∈ 𝒟*; then *Fᵢ* is *i*-valent because successors of *i*-valent configurations remain *i*-valent. Otherwise *e* was already applied en route, so some *Fᵢ ∈ 𝒟* has *Eᵢ* reachable from it; since *Fᵢ* is univalent and cannot be *(1−i)*-valent while reaching the *i*-valent *Eᵢ*, it is *i*-valent. So 𝒟 contains both a 0-valent and a 1-valent configuration.

Call configurations **neighbors** if one is a single step from the other. By an induction along an *e*-free path in 𝒞 between preimages of the two valences, there are neighbors *C₀, C₁ ∈ 𝒞* with *D₀ = e(C₀)* 0-valent and *D₁ = e(C₁)* 1-valent. WLOG *C₁ = e′(C₀)* with *e′ = (p′, m′)*.

- **Case 1: p′ ≠ p.** By Lemma 1, *e* and *e′* commute, so *D₁ = e(e′(C₀)) = e′(e(C₀)) = e′(D₀)*. Then *D₁* is a successor of the 0-valent *D₀*, hence 0-valent — contradicting *D₁* 1-valent.
- **Case 2: p′ = p.** Take a finite deciding continuation from accessible *C₀* in which *p* takes no steps; schedule σ, *A = σ(C₀)*. σ touches only processes ≠ *p*, so it commutes with *e* and *e′*. Then *e(A) = σ(e(C₀)) = σ(D₀)* is reachable from the 0-valent *D₀* ⇒ 0-valent; and *e(e′(A)) = σ(e(e′(C₀))) = σ(e(C₁)) = σ(D₁)* is reachable from the 1-valent *D₁* ⇒ 1-valent. Thus *A* has a 0-valent one-step continuation and a 1-valent finite continuation, so *A* is **bivalent**. But *A* ends a deciding run ⇒ univalent. Contradiction.

Both cases fail, so 𝒟 contains a bivalent configuration. ∎

## Constructing an admissible non-deciding run (proof of the Theorem)

A deciding run from a bivalent initial configuration ends univalent, so some step is the first to go bivalent → univalent (a **decisive step**); Lemma 3 shows that a fair construction can keep finding bivalent continuations after designated events. Build the run in **stages**, maintaining that each stage begins bivalent. Keep a round-robin process queue and order the buffer by send-time (earliest first).

- Start at a bivalent initial configuration *C₀* (Lemma 2).
- **Stage** from bivalent *C* with *p* at the queue head: let *m* be *p*'s earliest pending message (or ∅), *e = (p,m)*. By Lemma 3 there is a bivalent *C′* reachable from *C* by a schedule whose **last** event is *e*; run it. Move *p* to the back of the queue.

Over infinitely many stages every process is head infinitely often ⇒ takes infinitely many steps (no process faulty), and every message eventually becomes earliest-pending for its destination and is delivered ⇒ the run is **admissible**. Every stage begins and ends bivalent; since valence only shrinks along steps, no intermediate configuration in a stage can become univalent and later return to bivalent. Thus every configuration in the run is bivalent, so no process ever decides. *P* has an admissible non-deciding run; it is not totally correct. ∎

## Companion (the boundary is tight)

There **is** a partially-correct protocol in which all nonfaulty processes always decide, **provided** no process dies during execution and a strict majority are alive initially (some may be initially dead, unknown which). With *L = ⌈(N+1)/2⌉*: (1) each process broadcasts its id and listens for *L−1* others, forming digraph *G* with an edge *i→j* when *j* heard from *i*; every finishing process records *L−1* incoming edges. (2) Processes broadcast id + initial value + the *L−1* heard from, waiting transitively for all known ancestors, thus learning their ancestors and the relevant edges of the transitive closure *G⁺*. A node *k* belongs to an initial clique of *G⁺* exactly when *k* is an ancestor of every node that is an ancestor of *k*. In a source clique, each member's *L−1* predecessors must also lie inside the clique, so any initial clique has size at least *L*; two distinct initial cliques would be disjoint source components, and two disjoint majorities cannot both fit inside *N* processes. Hence *G⁺* has **exactly one** initial clique; every finishing process learns precisely that clique and its members' initial values, and decides by a fixed rule on them — so all agree. This shows it is a crash **during** the run, not uncertainty about who is alive, that defeats asynchronous consensus.

## The adversary, as code

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
