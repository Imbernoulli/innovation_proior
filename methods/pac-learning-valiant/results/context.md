## Research Question

There is a mature theory of computation because "mechanical calculation" has precise formal models. Learning lacks the same kind of starting point. People and machines can acquire recognition skills without being handed explicit programs, but that phenomenon is too vague to support theorem-level claims unless it is turned into a computational model.

The target question is: what would count as a feasible, provable account of acquiring a recognizer from information short of explicit programming?

## Background

The intended learned object is a recognizer for a concept: a procedure that says whether a predicate is true of given data. Knowledge-based systems already represent much working expert knowledge with propositional structure, so Boolean variables, Boolean expressions, and circuits are a natural representation class to examine first.

A useful theory must also respect computational complexity. A model that permits eventual convergence after unbounded time cannot explain feasible acquisition.

## Baselines

Prior rigorous work on inductive inference studies how examples can support inference of recursive functions or formal grammars. That supplies useful formality regarding the relationship between examples and hypothesis classes.

Statistical pattern recognition supplies distributions and error rates, framing learning problems in terms of measurable distributional quantities.

AI concept-learning work explores learning by example, analogy, and instruction, connecting symbolic representations to empirical evidence.

## Pre-Method Scaffold

A minimal formal setting has Boolean variables, concept classes, hidden target recognizers, and some source of examples or tests. The open problem is to formalize what it means to acquire a recognizer, how to turn finite observations into a recognizer, and under what conditions such acquisition is computationally feasible.
