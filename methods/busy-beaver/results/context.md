## Scope
Busy Beaver is a method for turning computability limits into a concrete extremal object. It starts with the same formal world as the halting problem: Turing machines with finitely many states, run on a blank tape, either halt or run forever. The shift is that it does not ask for a universal yes/no halting oracle. It asks, among machines of a fixed small size that do halt, which one does the most before stopping.

## Definition
Fix a precise Turing-machine convention, usually two symbols and `n` non-halting states. There are only finitely many such machines. Run each on a blank tape and ignore the ones that never halt. The score version `Sigma(n)` is the maximum number of `1`s left on the tape by any halting `n`-state machine. The step version `S(n)` is the maximum number of steps taken before halting by any halting `n`-state machine.

## Core Insight
The unique insight is that minimal machine size can be used as a measuring stick for maximal finite behavior. Busy Beaver does not merely say that some computations evade prediction. It defines a specific growth rate by asking for the largest output or longest finite run obtainable under a tiny description budget. That growth eventually exceeds every computable function, so it gives a named, concrete curve that lives beyond algorithmic reach.

## From Undecidable To Numeric
The halting problem says there is no algorithm that always decides whether an arbitrary program stops. Busy Beaver adds a numerical contour to that boundary. If `S(n)` were computable, one could decide halting for every `n`-state machine by simulating it for exactly `S(n)` steps: a machine not halted by then never will. Therefore the noncomputability of Busy Beaver is not a separate curiosity; it is the halting barrier expressed as a finite but unknowable bound.

## Why It Matters
Busy Beaver changes the mental image of undecidability. The boundary is not just a blank wall labeled "impossible." It has comparable landmarks: exact values for very small machines, lower bounds from discovered champions, and open gaps where proving non-halting becomes as hard as deep mathematics. This makes computability limits feel less like an abstract prohibition and more like a landscape with distances, cliffs, and partial maps.
