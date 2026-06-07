# Sources retrieved this run

- Dartmouth lecture notes (Deeparnab Chakrabarty), "Greedy Algorithms and Submodularity":
  https://www.cs.dartmouth.edu/~deepc/LecNotes/Appx/1b.%20Greedy%20Algorithms%20and%20Submodularity.pdf
  -> full (1-1/e) proof for cardinality-constrained monotone submodular max; submodular set cover H_N proof.
  Cites: Fisher, Nemhauser, Wolsey (constrained max); Wolsey 1982 (set cover).

- Jeremy Kun blog, "When Greedy Algorithms are Good Enough: Submodularity and the (1-1/e)-Approximation":
  https://www.jeremykun.com/2014/07/07/when-greedy-algorithms-are-good-enough-submodularity-and-the-1-1e-approximation/
  -> diminishing-returns + lattice (union/intersection) equivalent definitions, monotonicity,
     recurrence a_{i+1} <= (1-1/k) a_i, 1-x <= e^{-x}, f(S_l) >= (1-e^{-l/k}) f(S*).

- Marco Tulio Ribeiro blog, "The greedy algorithm for monotone submodular maximization":
  https://homes.cs.washington.edu/~marcotcr/blog/greedy-submodular/  (intuition; pseudocode)

- apricot library maxCoverage.py (jmschrei): real implementation, naive/lazy greedy optimizer,
  marginal gain via fmin(current_values + X, threshold).sum() - current_values_sum.
  https://github.com/jmschrei/apricot/blob/master/apricot/functions/maxCoverage.py

- Matroid generalization context: Fisher, Nemhauser, Wolsey, "An analysis of approximations for
  maximizing submodular set functions—II" (1978): greedy is 1/2-approx under a matroid constraint;
  cardinality is the uniform-matroid special case where it sharpens to 1-1/e.
