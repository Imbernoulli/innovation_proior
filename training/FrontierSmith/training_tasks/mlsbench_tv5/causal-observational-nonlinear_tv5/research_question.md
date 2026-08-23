Most nonlinear structure learners pay for accuracy with iteration: neural
fits, augmented-Lagrangian loops, kernel matrices, repeated regressions per
candidate parent set. This variant forbids all of it. Admissible building
blocks are closed-form statistics computable from a constant number of passes
over the data matrix — correlations, rank statistics, higher-order cross
moments, histogram summaries — combined by sorting and simple graph logic.
No optimizer may run, no model may be trained, and total work should scale
like the cost of a handful of matrix products even on the largest setting.

Frugality alone is not the point; the required output discipline is. The
algorithm must assemble its answer as an ordered stream of directed-edge
claims, most trustworthy first, each admitted only if it keeps the graph
acyclic, so that truncating the stream at any prefix yields a coherent
partial DAG whose precision is highest at the front. Where the stream is cut
— the confidence level below which claims stop — is itself a decision to be
made from the data, and making it well is most of the score. The three
generators keep the exercise honest: exponential, Gaussian, and Laplace noise
with GP and mixed function families mean any single cheap statistic will
sometimes point backwards, so the interesting engineering is in which
inexpensive quantities to combine into the ranking and how to price direction
against dependence strength.

How runs are graded does not move: identical generators, identical
directed-edge accounting, and zero credit for wall-clock savings — frugality
here is a design constraint, not a scored quantity. The claim under test:
a rank-and-admit pipeline built purely from single-pass statistics can hold
its own against methods that spend orders of magnitude more, and degrades
gracefully rather than catastrophically when its cheap signals weaken.
