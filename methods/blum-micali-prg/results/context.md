## Computational Randomness

A deterministic string cannot be random in the information-theoretic sense once its short description is known. Cryptography uses a more operational notion: bits that remain unusable to any adversary with a bounded amount of computation. A coin toss may already be determined while the coin is in the air; the relevant question is whether the observer has the time and machinery to compute the outcome.

This phrases randomness as "can a feasible computation exploit the description?" rather than "is there a short description?" The model of computation and the allowed resources become part of the definition.

## Short Seeds

Private-key cryptography has a simple ideal object: the one-time pad, whose shared secret is as long as the message. A short secret seed that expands into many usable pad bits would let secure communication depend less on moving long secrets through protected channels.

The expansion requirement has three parts. The number of produced bits is polynomially larger than the seed, each bit is efficient to compute, and the public description of the process does not help a feasible adversary forecast the next unseen bit.

## Earlier Generators

Classical numerical generators are judged by statistical behavior. Linear congruential sequences, for example, are inferable from enough observed values. Generators are routinely checked with frequency counts, runs tests, and inspection for visual irregularity.

Earlier theoretical proposals frame security at the level of recovering a full next number rather than at the level of individual emitted bits. A stream cipher emits bits.

## Hard Forward Motion

Public-key cryptography supplies functions that are easy to compute forward and apparently hard to reverse. Discrete exponentiation modulo a prime is a central example, and trapdoor permutations such as RSA add private information that makes inversion easy for the holder of the trapdoor.

A deterministic generator can advance a secret state by applying such a function, computing forward while reversing the step is tied to inverting the function. Inversion hardness is a statement about recovering the whole state.

## Prediction Test

The natural adversarial experiment is sequential. The adversary sees the public generator and a prefix of the output, but not the seed, and then tries to predict the next bit. Success noticeably above one half counts as a break.

This next-bit view connects a hardness assumption about the underlying computational problem to a guarantee about the emitted stream.
