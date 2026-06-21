The canonical method I want to present is the GGM pseudorandom function family, built on top of the Blum–Micali–Yao pseudorandom generator. Together they turn a tiny amount of true randomness into an exponentially large object that no efficient observer can tell from genuinely random. The problem they solve is how to manufacture randomness cheaply. A one-time pad needs one random bit per message bit; a truly random function on k-bit inputs needs about k times two to the k bits just to write down. Both are impractical at scale. The GGM construction replaces them with deterministic algorithms that take a short secret key and produce outputs which, for every efficient test, behave as if they were uniform.

The crucial move is to redefine "random enough." Kolmogorov complexity asks whether an individual string has a short description, but any generator that reads a short seed and outputs a long string has automatically given the output a short description, so this notion rules out stretching entirely. Fixed statistical batteries are no better: a linear congruential generator passes many standard tests yet is completely predictable from a few outputs. The right notion is computational indistinguishability. Two distributions are computationally indistinguishable if every probabilistic polynomial-time algorithm accepts samples from the first distribution with almost exactly the same probability as samples from the second, where "almost" means the difference is negligible in the security parameter. A generator is pseudorandom if its output on a random seed is computationally indistinguishable from a uniform string of the same length.

This definition is powerful because it says the fake randomness can be substituted for real randomness inside any efficient computation without changing the probability that any efficient observer notices. The difficulty is proving it, because "every efficient test" is an uncountable quantifier. The Blum–Micali–Yao generator solves this by reducing everything to the next-bit test: no efficient predictor, having seen any prefix of the output, should guess the next bit better than a coin. Yao proved that passing the next-bit test is equivalent to passing all efficient statistical tests. The proof is a hybrid argument. If some test distinguished the generator's output from uniform with advantage epsilon, one would build intermediate distributions that replace the output with uniform bits one position at a time. The total advantage epsilon must telescope across the positions, so some adjacent pair differs by at least epsilon over the length. A predictor that sees the prefix can plug in a random guess for the next bit, append a random tail, run the test, and let the test's verdict vote on whether the guess was right. That gives a next-bit predictor with advantage epsilon over the length, contradicting unpredictability.

So to build a pseudorandom generator it is enough to make the next bit unpredictable. The Blum–Micali–Yao construction does this from a one-way permutation equipped with a hard-core predicate. A one-way permutation is a bijection that is easy to compute but hard to invert. A hard-core predicate is an efficiently computable bit of the preimage that remains unpredictable even when the image is known. The discrete-logarithm permutation, mapping an exponent to a group element, has such a predicate in the principal-square-root bit. The generator starts from a uniform seed and walks the orbit of the permutation, emitting one hard-core bit at each step. Because the permutation is a bijection, every state on the orbit is uniform, so the hard-core property applies at each step. Reversing the emitted bits makes the next-bit reduction work cleanly, and since bit-reversal preserves indistinguishability, the bits can be output in forward order in practice. The result stretches a short seed into any polynomial number of pseudorandom bits.

The GGM construction then bootstraps a pseudorandom generator into a pseudorandom function family. The key idea is to read a length-doubling generator not as "twice as many bits" but as branching: one seed becomes two seeds. Let G map k bits to 2k bits and split it into a left half G0 and a right half G1. Place a random k-bit key at the root of a binary tree of depth k. A node labeled v gets left child G0(v) and right child G1(v). An input x is interpreted as a path from the root to a leaf; the leaf label is the function value. Choosing a function is just choosing the root key. Evaluating the function at any single input takes only k applications of G, even though the full tree would have exponentially many leaves.

Security is again proved by hybrids, this time over the levels of the tree. In the real world, level zero is random and every deeper level is derived by G. In the ideal world, every leaf is an independent uniform string, which is exactly a random function. Intermediate hybrids replace one level of G-derived labels by truly random labels at a time. A distinguisher that separates real from ideal must separate some adjacent pair of hybrids. Between two adjacent levels, the difference is just "two children of a random parent are produced by G" versus "two children are independent random strings," which is exactly a pseudorandom-generator challenge. The tree has exponentially many nodes, but a polynomial-time distinguisher makes only polynomially many queries, so only the nodes on those query paths ever need to be materialized. The lazy simulation keeps the reduction efficient. A similar argument per individual G-invocation gives the same conclusion: swapping any one doubling for a fresh random doubling is indistinguishable under the generator assumption.

The same hybrid idea also shows that a pseudorandom function family cannot be polynomially inferred. An inferrer adaptively queries the function and then, at a fresh point, tries to recognize the real function value among random alternatives. If some test distinguished the family from a random function, one would choose a random query at which to turn the test into an exam, feed the test either the real value or a random value, and use its final output to decide. The telescoping advantage again gives a non-negligible inference edge, which is impossible for a random function. So non-inferability, indistinguishability from a random function, and failure of every efficient statistical test are all equivalent for function families.

The concrete code below is not a provably secure instantiation, because a true hard-core predicate would need a carefully chosen one-way permutation, but it gives a runnable simulation of the same architecture. It uses SHA-256 as a heuristic length-doubling generator and builds the GGM tree. It shows that many evaluations look balanced and that any single leaf can be computed on demand without expanding the whole tree.

```python
import hashlib, secrets, statistics

def prg_double(seed: bytes, k: int) -> bytes:
    """Length-doubling generator: k bytes -> 2k bytes using SHA-256."""
    left = hashlib.sha256(seed + b'\x00').digest()[:k]
    right = hashlib.sha256(seed + b'\x01').digest()[:k]
    return left + right

def ggm_prf(key: bytes, x: bytes, k: int) -> bytes:
    """GGM pseudorandom function: key and x are k-byte strings."""
    label = key[:k]
    for byte in x:
        for shift in range(7, -1, -1):
            out = prg_double(label, k)
            label = out[:k] if ((byte >> shift) & 1) == 0 else out[k:]
    return label

k = 16
key = secrets.token_bytes(k)
samples = 200
outputs = [ggm_prf(key, secrets.token_bytes(k), k) for _ in range(samples)]
all_bits = ''.join(format(int.from_bytes(o, 'big'), '0128b') for o in outputs)
print('samples:', samples)
print('mean bit:', statistics.mean(map(int, all_bits)))
print('first output hex:', outputs[0].hex())
```
