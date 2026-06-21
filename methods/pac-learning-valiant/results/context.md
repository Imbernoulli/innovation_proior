## Research Question

There is a mature theory of computation because "mechanical calculation" has precise formal models. Learning lacks the same kind of starting point. People and machines can acquire recognition skills without being handed explicit programs, but that phenomenon is too vague to support theorem-level claims unless it is turned into a computational model.

The target question is: what would count as a feasible, provable account of acquiring a recognizer from information short of explicit programming? The answer must say what information the learner receives, what kind of recognizer it may output, how success is measured, and what resource bounds make the process feasible.

## Background

The intended learned object is a recognizer for a concept: a procedure that says whether a predicate is true of given data. Knowledge-based systems already represent much working expert knowledge with propositional structure, so Boolean variables, Boolean expressions, and circuits are a natural representation class to examine first.

A useful theory must also respect computational complexity. A model that permits eventual convergence after unbounded time cannot explain feasible acquisition. A model that only gives statistical error rates without asking which program classes can be deduced efficiently also misses the central computational question.

## Baselines

Prior rigorous work on inductive inference studies how examples can support inference of recursive functions or formal grammars. That supplies useful formality, but it leaves open the question of when Boolean recognition programs can be deduced with explicit polynomial resource bounds.

Statistical pattern recognition supplies distributions and error rates, but it does not by itself characterize which symbolic concept classes can be learned by a polynomial-time deduction procedure.

AI concept-learning work explores learning by example, analogy, and instruction, but usually without class-wide guarantees or sharp resource boundaries.

## Design Tensions

Exact recovery from a small sample is too strong: unobserved regions can distinguish many candidate recognizers. But a purely heuristic notion of "usually works" is too weak. The future model must decide how much error can be tolerated, what probability of failure is acceptable, and under what distribution the error is measured.

The teacher's power is also delicate. A deliberately chosen example sequence can encode the target program and make the task indistinguishable from programming. With too little access to typical cases, the learner may need exhaustive search just to find rare positives.

## Pre-Method Scaffold

A minimal formal setting has Boolean variables, concept classes, hidden target recognizers, and some source of examples or tests. What remains open is the success criterion and the deduction procedure: how to turn finite observations into a recognizer, how to certify that the recognizer generalizes, and how to prove that both sample count and computation stay polynomial.
