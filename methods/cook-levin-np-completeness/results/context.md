## A Shared Search Wall

Many finite mathematical tasks have an obvious brute-force solution: try all routes, all assignments, all proofs, all subgraphs, or all certificates. The difficulty is not computability. The difficulty is that the search space grows too fast for the known methods, even though a proposed solution can often be checked quickly.

## Machines As Configurations

The standard machine model represents a computation as a sequence of configurations: finite control state, tape contents, and head position. A single step changes only bounded local information. This gives a precise language for talking about arbitrary algorithms and their histories, but by itself it classifies what can be computed, not what can be computed feasibly.

## Feasible Time As A Boundary

Work on combinatorial optimization makes polynomial time the candidate mathematical boundary for a "good" algorithm. This changes the question from whether a process eventually halts to whether every part of a proposed solution method, including any translation between problems, stays within a polynomial bound.

## Nondeterministic Checking

Some problems are naturally phrased as: there exists a short object such that a deterministic check accepts it. This is different from having a deterministic method to find the object. The key common form is a bounded verifier, an input, and a sequence of choices or witness bits whose length is polynomial in the input size.

## Proof Search And Finite Syntax

Propositional formulas, proof procedures, and finite constraint systems already provide languages for yes/no conditions. The open challenge is to compare arbitrary bounded verifiers with one concrete finite syntax without hiding the search or the translation cost inside the comparison itself.
