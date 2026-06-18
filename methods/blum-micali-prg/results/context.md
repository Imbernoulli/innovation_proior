## Computational Randomness

A deterministic string cannot be random in the information-theoretic sense once its short description is known. Yet cryptography needs something more operational than incompressibility: it needs bits that remain unusable to any adversary with a bounded amount of computation. A coin toss may already be determined while the coin is in the air, but the relevant question is whether the observer has enough time and machinery to compute the outcome.

This shifts the randomness question from "is there a short description?" to "can a feasible computation exploit the description?" The model of computation and the allowed resources become part of the definition.

## Short Seeds

Private-key cryptography has a simple ideal object: a one-time pad. Its weakness is not security but logistics, because the shared secret must be as long as the message. If a short secret seed could safely expand into many usable pad bits, secure communication would become far less dependent on moving long secrets through protected channels.

The expansion requirement has three parts. The number of produced bits must be polynomially larger than the seed, each bit must be efficient to compute, and the public description of the process must not help a feasible adversary forecast the next unseen bit.

## Earlier Generators

Classical numerical generators can look statistically plausible while remaining algebraically vulnerable. Linear congruential sequences, for example, may pass many routine tests yet still be inferable from enough observed values. A generator for cryptography cannot be certified merely by frequency counts, runs tests, or visual irregularity.

Earlier theoretical proposals also leave a bit-level concern. A next number might be hard to recover as a whole while some property or bit of that number is easy to predict. A stream cipher needs bits, so the security target must reach down to the information actually emitted.

## Hard Forward Motion

Public-key cryptography introduces a different kind of raw material: functions that are easy to compute forward and apparently hard to reverse. Discrete exponentiation modulo a prime is a central example, while trapdoor permutations such as RSA add private information that makes inversion easy for the holder of the trapdoor.

For a deterministic generator, hard forward motion is attractive because it can advance a secret state without exposing how to go backward. But inversion hardness alone does not say that every candidate output bit is hidden. A hard-to-invert state may still leak simple, predictable information.

## Prediction Test

The natural adversarial experiment is sequential. The adversary sees the public generator and a prefix of the output, but not the seed, and then tries to predict the next bit. Success noticeably above one half is a failure, even if no full state is recovered.

This next-bit view asks for a bridge between a local hardness assumption about hidden state information and a global stream guarantee. The proof must explain why any successful forecast of a future output bit would contradict the assumed hardness of the underlying computational problem.
