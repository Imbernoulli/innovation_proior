I will describe the Luby-Rackoff theorem, which gives a clean way to build a pseudorandom permutation out of pseudorandom functions. The starting point is a familiar gap. A pseudorandom function family lets me evaluate a keyed function that cannot be distinguished from a truly random function by any efficient oracle algorithm, but it does not promise that the function is a permutation. Two distinct inputs may collide, and there is no efficient procedure for inversion. A block cipher, by contrast, must be a keyed permutation: every key should define a reversible map from plaintext blocks to ciphertext blocks of the same length, and decryption should recover the plaintext. The Luby-Rackoff result closes this gap by showing that a small number of Feistel rounds turn pseudorandom functions into a pseudorandom permutation.

The Feistel round is the central gadget. Given an n-bit round function f, I define a permutation D_f on 2n-bit strings by splitting the input into a left half L and a right half R, each n bits long, and returning D_f(L, R) = (R, L xor f(R)). The key observation is that this map is invertible even when f is not. Given the output (L', R') = (R, L xor f(R)), I recover R = L' and then L = R' xor f(L'). Therefore the round function itself need not be bijective, which is a huge convenience for a designer. The Feistel construction converts arbitrary functions into permutations at the cost of doubling the block width.

A single Feistel round, however, is obviously insecure as a pseudorandom permutation. The output left half is exactly the input right half, so an adversary can immediately detect structure after one query. Two rounds hide that obvious leakage, but Luby and Rackoff showed that two rounds are still distinguishable from a random permutation with non-negligible advantage. The reason is that algebraic relations remain between queries that share a right half. Even three rounds, which suffice in the forward-query setting, fail against an adversary that can also make inverse queries. The full strength of the theorem requires four rounds.

The main result has two parts. First, three independent Feistel rounds with independent pseudorandom functions yield a pseudorandom invertible permutation generator against adversaries that may only make forward oracle queries. The security reduction says that any efficient distinguisher either breaks one of the underlying pseudorandom functions or causes an internal collision in the half-block space. Second, four independent Feistel rounds yield a strong pseudorandom invertible permutation generator, meaning the construction remains indistinguishable from a random permutation even when the adversary can submit both forward and inverse queries. The extra round in the strong case protects the inverse side of the transcript.

The proof strategy is modular. I first analyze the construction assuming the round functions are truly random rather than pseudorandom. In that ideal world, I compare the transcript an adversary sees from the Feistel permutation with the transcript it would see from a uniformly random permutation. The two transcripts agree perfectly unless some internal half-block value repeats. When all relevant internal values are fresh, the middle random functions return fresh random values, and the resulting output pair has exactly the distribution expected from a random permutation. The bad event is therefore a birthday collision among n-bit half-blocks, and for m queries its probability is bounded by a term like m^2 / 2^n in the three-round forward case. Once the ideal analysis is in place, I return to pseudorandom functions through a hybrid argument. I replace the truly random round functions with pseudorandom ones one at a time. If the adversary can distinguish any two adjacent hybrids, then I can build an efficient distinguisher that violates the pseudorandom function assumption. Combining the two steps gives the final bound: the Feistel permutation is pseudorandom up to the PRF advantage plus the collision probability.

It is worth emphasizing what the theorem does not say. It does not claim that every Feistel cipher with many rounds is secure, nor does it specify concrete block sizes or key schedules. It is a reductionist statement: if I have independent secure pseudorandom functions and compose them in three or four Feistel rounds, then I obtain a provably secure pseudorandom permutation or strong pseudorandom permutation. The practical relevance is that a block-cipher designer can focus on building good round functions, and the Feistel wiring automatically supplies invertibility and a rigorous security guarantee.

To make the construction concrete, I will implement a tiny Feistel permutation with random round functions and verify both invertibility and a basic statistical property. The code below is not intended as a secure cipher; it uses short blocks and a toy random-function model so that the mechanics can be inspected directly. It defines the Feistel round, composes three or four rounds, checks that encryption and decryption are exact inverses, and performs a quick frequency test to confirm that repeated inputs produce balanced-looking outputs.

```python
import random
from collections import Counter

def make_random_function(n):
    """Return a random function from n-bit strings to n-bit strings."""
    table = {}
    for x in range(1 << n):
        table[x] = random.randrange(1 << n)
    return lambda x: table[x]

def feistel_round(left, right, f, n):
    """One Feistel round: (L, R) -> (R, L xor f(R))."""
    return right, left ^ f(right)

def feistel_encrypt(block, round_funcs, n):
    left = block >> n
    right = block & ((1 << n) - 1)
    for f in round_funcs:
        left, right = feistel_round(left, right, f, n)
    return (left << n) | right

def feistel_decrypt(block, round_funcs, n):
    left = block >> n
    right = block & ((1 << n) - 1)
    for f in reversed(round_funcs):
        # Invert one round: from (L', R') = (R, L xor f(R)),
        # recover R = L' and L = R' xor f(R).
        right, left = left, right ^ f(left)
    return (left << n) | right

def test():
    n = 4
    num_rounds = 4
    round_funcs = [make_random_function(n) for _ in range(num_rounds)]

    # Verify invertibility on all 2n-bit inputs.
    all_ok = True
    for x in range(1 << (2 * n)):
        c = feistel_encrypt(x, round_funcs, n)
        p = feistel_decrypt(c, round_funcs, n)
        if p != x:
            all_ok = False
            break
    print("Invertibility check passed:", all_ok)

    # Quick frequency test: encrypt a few inputs and inspect output distribution.
    outputs = [feistel_encrypt(x, round_funcs, n) for x in range(1 << (2 * n))]
    counts = Counter(outputs)
    print("Distinct outputs:", len(counts))
    print("First few ciphertexts:", outputs[:8])

if __name__ == "__main__":
    test()
```

Running this script confirms that the Feistel network is a permutation of the small block space, because decryption exactly reverses encryption. It also shows that distinct plaintexts map to distinct ciphertexts, which is the first structural requirement a block cipher must satisfy. The Luby-Rackoff theorem lifts this simple observation into a cryptographic guarantee: when the round functions are pseudorandom, the resulting permutation is computationally indistinguishable from random.

This result is important because it separates the job of a block-cipher designer into two clean pieces. The round functions can be optimized for confusion and diffusion without worrying about invertibility, because the Feistel structure handles reversibility automatically. The theorem then tells us that enough independent rounds convert the local pseudorandomness of the round functions into the global pseudorandomness of a permutation. Although modern block ciphers such as AES are not pure Luby-Rackoff constructions, the theorem remains a foundational explanation for why Feistel networks have been so influential in symmetric cryptography.
