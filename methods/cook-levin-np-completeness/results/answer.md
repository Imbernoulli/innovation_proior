# Cook-Levin NP-Completeness

Cook and Levin's result identifies a first universal problem for efficient search. In modern form: SAT is in NP, and every language accepted by a nondeterministic Turing machine in polynomial time is reducible to SAT in polynomial time. Cook's 1971 paper states this through P-reducibility to tautology/DNF tautology and related problems; Levin's 1973 paper states the parallel result through universal "perebor" search problems.

The proof is a bounded-computation encoding. Given a nondeterministic machine `M`, input `x`, and time bound `T = p(|x|)`, construct variables for a `T`-step tableau: tape symbol per cell and time, machine state per time, and head position per time. Add clauses saying the first row is the input, each cell/time has exactly one symbol, each time has exactly one state and head position, every adjacent pair of rows respects `M`'s transition rule, and an accepting state appears. The formula is polynomial size because `T` is polynomial and each transition condition is local.

Correctness is exact. An accepting branch of `M` fills the tableau and satisfies the formula. Any satisfying assignment decodes to a valid accepting branch. Therefore solving the target formula problem would solve the original nondeterministic polynomial-time problem after only polynomial overhead.

The distinctive insight is not that machines can be encoded in logic. It is that polynomially bounded nondeterministic computation can be abstracted as locally checkable finite history, and efficient reducibility can turn that encoding into a universal hardness statement. Turing supplied configurations and locality, Edmonds/Cobham supplied polynomial time as the feasible boundary, and proof theory supplied formula syntax; Cook and Levin made these into a complexity-theoretic comparison principle.

Source grounding: Cook 1971 (`refs/primary/cook-1971-theorem-proving-ocr.txt`), Levin 1973 Russian original plus English reproduction (`refs/primary/levin-1973-*`), Turing 1936, Edmonds/Witzgall on good algorithms, Cook's Clay P vs NP account, Levin's BU retrospective pages, and modern lecture-note cross-checks in `refs/explainers/`.
