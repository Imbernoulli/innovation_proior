## Research question

The question is not whether some particular concrete program property is decidable, but rather: when a property is truly about the semantics a program computes, when can it possibly have a general decision procedure?

The halting problem has already shown that there is no algorithm that is always correct and always halts for "given a program and an input, does the program halt". This naturally raises the question: for the partial function a program computes, the language it recognizes, or its input-output behavior, which judgments can be carried out by an algorithm for arbitrary programs?

The key word here is extensional. Such a property depends only on the function or language the program implements, not on what the source code looks like, how many steps it takes, what the variable names are, or whether it contains some particular syntactic fragment. The question is placed at the level of "what the program is" rather than "how the program is written": what is the relationship between a nontrivial semantic-level judgment and the halting problem?

## Background

A program can be viewed as a partial computable function: for some inputs it returns an output, for others it may never return. Many natural questions are semantic properties of this kind: whether a program halts on all inputs, whether it accepts some string, whether it computes a constant function, whether it never outputs some particular value, whether it recognizes the empty language, whether it recognizes a finite language.

"Nontrivial" means the property is neither satisfied by all program semantics nor satisfied by none. If the property is true for all partial functions, or false for all partial functions, a decider can simply output a fixed answer.

A common formulation of a decision problem is: let `P` be a property on the set of partial computable functions, and consider whether the set `{e | phi_e has P}` is decidable. This question depends only on the input-output semantics of the program, and some programs satisfy it while others do not.

## Baselines

Proving undecidability property by property is the most direct approach. For example one could separately prove "whether it accepts the empty string" is undecidable, then prove "whether it recognizes the empty language" is undecidable, then prove "whether it computes a total function" is undecidable. Each property is handled with its own tailored construction.

Reduction from the halting problem gives a common template. If some semantic property has a decider, encode an arbitrary halting instance into a new program: when the original program halts, the new program exhibits one known semantics; when the original program does not halt, the new program exhibits another known semantics. Deciding that property would then decide the original halting instance.

## Evaluation setting

The objects are program indices, Turing machine indices, or executable descriptions in any equivalent model of computation. The decider must halt on every input program and output whether the partial function the program computes has property `P`.

Positive and negative examples must be distinguished by semantics. If two programs compute the same partial function, an extensional property must give them the same answer. Purely syntactic properties and properties of execution with a fixed time bound belong to a different category: for example "whether the program text contains a certain instruction" is decidable, and "whether it halts within 100 steps" is also decidable, because these are not properties of the partial function computed itself.

The core of the evaluation is not a complexity upper bound but the boundary of decidability. For any algorithm that claims to make a general semantic judgment, one should examine its reduction relationship with the halting problem: if it can decide a nontrivial semantic property, can it also solve the halting problem.
