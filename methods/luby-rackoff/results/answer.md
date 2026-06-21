# The Luby-Rackoff Theorem

Luby and Rackoff show how to construct pseudorandom permutations from pseudorandom functions using Feistel composition.

For an `n`-bit round function `f`, define the Feistel permutation on two `n`-bit halves by

```text
D_f(L, R) = (R, L xor f(R)).
```

This is invertible even if `f` is not. Given a pseudorandom function generator, build independent keyed copies of this Feistel round and compose them.

The main results are:

- Two Feistel rounds are distinguishable.
- Three independent Feistel rounds give a pseudorandom invertible permutation generator against forward oracle queries.
- Four independent Feistel rounds give a super, or strong, pseudorandom invertible permutation generator against both forward and inverse oracle queries.

The proof first analyzes the construction with truly random round functions. In that ideal setting, an efficient oracle transcript differs from a random-object transcript only when internal half-block values collide; the primary paper's main lemma bounds this by a birthday-style term, `m^2 / 2^n`, for an `m`-query oracle circuit in the three-round forward-query analysis. Trevisan's notes give the corresponding four-round strong-PRP bound as `4 * epsilon + t^2 * 2^-m + t^2 * 2^-2m` when the round function is `(O(t r), epsilon)`-secure and computable in time `r`.

The proof then replaces truly random round functions with PRFs by a hybrid argument. If the Feistel permutation changes detectably during one replacement, the same distinguisher breaks the underlying PRF. Thus the construction converts local oracle pseudorandomness into block-cipher-style permutation indistinguishability, with only the quantified collision terms left over.

The theorem is not a blanket proof of security for every practical Feistel cipher. It is a reductionist sufficient condition: independent secure PRF round functions, arranged in enough Feistel rounds, yield a provable PRP or strong PRP.

