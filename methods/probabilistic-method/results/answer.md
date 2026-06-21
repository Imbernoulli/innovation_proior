The Probabilistic Method is a way to prove deterministic existence by studying a random experiment.
Its central insight is simple but radical: to show that an object exists, it can be enough to show
that a randomly chosen object has the desired property with positive probability. The proof does not
need to construct the object first.

For Erdős, this changed what was reachable in extremal combinatorics. Direct constructions usually
bring visible structure, and visible structure often creates unwanted cliques, independent sets,
short cycles, or colorings. A random graph, by contrast, is typically irregular in exactly the useful
way. By bounding the expected number of bad configurations, Erdős could show that some graph avoids
all large homogeneous sets, giving strong Ramsey lower bounds long before comparable explicit
constructions were known.

The method has three reusable engines:

- Expectation: if the expected number of bad objects is less than 1, some outcome has none; if the
  expected score is M, some outcome scores at least M.
- Local lemma: if each bad event is rare and depends on only a small local neighborhood, there can be
  a positive probability that no bad event happens.
- Alteration: if a random object has many useful features and only a controlled number of defects,
  delete or edit the defects while preserving the useful bulk.

So the distinctive lesson is not "randomness builds the final object." The lesson is that randomness
can certify the existence of a deterministic object whose structure is too diffuse or patternless to
find directly.
