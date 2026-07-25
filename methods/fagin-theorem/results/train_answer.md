The problem is to describe the complexity class NP without starting from Turing machines, tape heads, or polynomial clocks. A class defined by machines is useful for algorithms, but it hides the logical shape of the properties being recognized. A graph, a database instance, or any finite relational structure should be the real object of study, and the answer should not depend on how the elements happen to be named.

The natural first attempt is to use classical spectra: the set of cardinalities of finite models of a first-order sentence. But spectra recognize a set of numbers, not a class of structures. Because the binary notation for a cardinality n has length only log n, a brute-force search over relations on n elements looks exponential in the input length. That mismatch pushes spectra toward nondeterministic exponential time, not ordinary NP. What is needed is a setting where the input already carries the polynomial-sized relation tables of a finite structure, so that guessing an extra finite relation costs only polynomially many bits.

The right bridge is Fagin's theorem. It says that, for every nonempty finite vocabulary and every isomorphism-closed class of finite structures over that vocabulary, the class is a generalized spectrum if and only if its standard string encodings are in NP. In short, NP equals existential second-order definability: NP = SO∃.

A defining sentence has the form ∃R1 ... ∃Rk φ, where R1 through Rk are auxiliary relation variables of fixed arity and φ is a first-order sentence over the original vocabulary together with these new relations. The existential second-order quantifiers are the logical counterpart of nondeterministic guessing: the machine's certificate is replaced by a finite relational witness living on the same universe as the input structure. The first-order part φ is the logical counterpart of polynomial-time local verification: it can express that the guessed relations satisfy the required constraints.

The forward direction is immediate. Given such a sentence, a nondeterministic polynomial-time algorithm guesses the interpretation tables for R1,...,Rk and then evaluates φ on the expanded structure. Because the arities and the formula are fixed, the guessed tables have size polynomial in the input structure, and first-order evaluation is polynomial.

The reverse direction is the deeper half. Take a nondeterministic Turing machine running in time n^k on inputs of size n. Represent the accepting computation as finite relations over the input universe. Use k-tuples of elements to index time instants and tape positions, so the tableau has polynomially many cells. Existentially quantify relations that record, for each cell at each time, the symbol written there and, when the head is present, the current state. Another relation can record the nondeterministic choices if that simplifies the transition check. The first-order formula then asserts that the first row encodes the input, that every cell has a unique content, that each row follows from the previous one by the machine's local transition rules, and that an accepting state appears at the final time. Because an arbitrary finite structure need not come with an order, the auxiliary witness also guesses a linear order on the universe; this is just bookkeeping for tuple indexing and does not add computational power beyond the second-order existential guess.

The theorem therefore identifies the certificate of an NP problem with finite relational structure. Nondeterminism becomes existential quantification over relations, and polynomial verification becomes first-order local checking.

The result I land on is this. Fix any nonempty finite relational vocabulary $\sigma$, and let $\mathcal{K}$ be any class of finite $\sigma$-structures closed under isomorphism. Write $E(\mathcal{K})$ for the set of standard string encodings of the structures in $\mathcal{K}$. Then

$$\mathcal{K}\text{ is a generalized spectrum} \iff E(\mathcal{K}) \in \mathrm{NP},$$

and $\mathcal{K}$ is a generalized spectrum exactly when it is definable by a sentence of the shape

$$\exists R_1 \cdots \exists R_k\, \varphi,$$

where $R_1,\dots,R_k$ are relation symbols of fixed arity not already in $\sigma$, and $\varphi$ is an ordinary first-order sentence over the expanded vocabulary $\sigma \cup \{R_1,\dots,R_k\}$. Under this correspondence the existential relation quantifiers $\exists R_1 \cdots \exists R_k$ are the guess — a finite relational witness on the same universe as the input structure, whose tables have size polynomial in the encoding — and $\varphi$ is the check, since a fixed first-order formula is evaluated on the expanded structure in time polynomial in the size of its universe. Collapsing the equivalence to a single identity between a complexity class and a logic gives the statement I take as the theorem:

$$\mathrm{NP} = \mathrm{SO}\text{-}\exists.$$

This holds for every nonempty vocabulary $\sigma$; the one exception is the empty vocabulary, where the input carries no relations at all, $E(\mathcal{K})$ degenerates to a set of cardinalities, and the corresponding logical/complexity scale is the older spectrum problem at nondeterministic exponential time rather than $\mathrm{NP}$.
