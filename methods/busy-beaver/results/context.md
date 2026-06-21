## Scope
The setting is the formal theory of computation built on Turing machines. A machine has finitely many internal states and operates on a one-way-infinite tape over a fixed small alphabet, usually two symbols. A finite transition table specifies, for each state and scanned symbol, what symbol to write, which way to move the head, and which state to enter next. Started on a blank tape, a machine either eventually reaches a halting state or runs forever.

## The Machines Of A Fixed Size
For a fixed number of states `n` and a fixed alphabet, a transition table has only finitely many entries, each chosen from a finite set of possibilities. So there are only finitely many distinct `n`-state machines, and one can in principle enumerate them all. Each such machine, run on a blank tape, produces a definite trajectory: a sequence of tape configurations that either reaches a halt after some finite number of steps or continues without end.

## The Halting Problem
The central known result about this world is the undecidability of halting. There is no single algorithm that, given the description of an arbitrary Turing machine, always correctly decides whether that machine halts. Simulation gives one direction of evidence: if a machine does halt, running it long enough eventually reaches the halting state and confirms it. For a machine that never halts, plain simulation produces no terminating certificate on its own.

## Computable Functions
A function from naturals to naturals is computable when some Turing machine produces its value on every input. The computable functions are closed under the usual constructions and include functions that grow extremely fast in standard notation. Whether a given function is computable is a separate matter from how large its values are: a function can grow astronomically and still be computable, while computability is about whether an algorithm can produce the values at all.

## Research Question
Within this finite-but-unbounded world, every `n`-state machine that halts does so after some definite finite number of steps and leaves some definite finite output on the tape. The open question is whether the limits of finite computation under a bounded description size can be summarized as a concrete numerical quantity indexed by machine size, and how such a quantity relates to the halting problem and to the class of computable functions.
