## Research question

The central question is what it means for one concrete object, such as a finite binary string, to be random. The usual probabilistic answer describes a process: a fair coin, an independent source, or a distribution over possible strings. That is powerful, but it says randomness belongs first to the generating distribution and only indirectly to the observed object.

Kolmogorov complexity asks for a different kind of explanation. Given the string already in hand, how short can a complete generating description of it be? If a string has a short program that prints it, the string has a pattern. If no program much shorter than the string itself prints it, then the string is random in the algorithmic sense.

## Probability baseline

Classical information theory measures surprise relative to a probability model. An outcome with probability `p` carries about `-log p` bits of information. This makes randomness a relation among an object, a model, and an ensemble of alternatives. The same string can look typical under one distribution and exceptional under another.

This external viewpoint is not a defect. It is exactly what communication theory, statistics, and coding need: if a source distribution is known, optimal codes exploit that distribution. But it leaves a philosophical and mathematical gap for individual objects. A single string by itself has no frequency table, no ensemble, and no repeated trials. Probability can say how likely the string was under a model; it does not directly say how much structure the string itself contains.

## Kolmogorov move

Kolmogorov's move is to replace probability-model description with program-length description. Fix a universal computing language. The Kolmogorov complexity `K(x)` of a finite string `x` is, up to standard variants, the length of the shortest program that outputs `x` and halts.

This makes regularity operational. A regularity is not a vague visual pattern; it is any effective procedure that generates the object more concisely than listing it. The string's pattern is its shortest generating program. A long repetition, the first million digits of `pi`, or a computable fractal may look large as raw data, but each has low algorithmic information if a compact program plus parameters generate it.

## Fundamental shift

The shift is from describing randomness outside the object to measuring description length inside the object. In the probabilistic view, randomness is attached to a source: the object is random because it was sampled from a high-entropy distribution or passes tests expected under that distribution. In the Kolmogorov view, randomness is attached to irreducibility: the object is random if it cannot be effectively compressed.

This matters because it turns randomness into a property of individual finite strings, modulo the fixed universal machine and additive constants. The random-looking string is not random because we failed to notice a pattern; it is random when every pattern powerful enough to generate it is essentially as long as the string. The absence of a shorter program is the formal version of "no exploitable regularity."

## Artifact scope

The final answer should emphasize the conceptual innovation rather than a full technical survey. It should explain `K(x)` as shortest program length, incompressibility as algorithmic randomness, and the contrast with probability distributions. It should also mention the main caveats: exact Kolmogorov complexity is uncomputable, machine choices change values by an additive constant, and finite-string randomness is best understood asymptotically or with a fixed reference machine.

The source base is intentionally narrow. I used a modern overview by Grunwald and Vitanyi for the Shannon-versus-Kolmogorov contrast, shortest-program definition, invariance, incompressibility, and uncomputability, plus a short Vitanyi note for the formal "shortest program reconstructs the object" phrasing. I did not perform broad historical source collection.
