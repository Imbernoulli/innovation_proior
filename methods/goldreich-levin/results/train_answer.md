The problem is to find a hard-core predicate for a general one-way function. A one-way function guarantees only that the whole input is hard to recover from its image; it does not automatically hide any particular bit of the input. Early constructions showed that specific algebraic functions, such as discrete exponentiation or RSA, do have individual hard bits, but those proofs rely on special number-theoretic structure and do not apply to an arbitrary one-way function. A generic transformation existed, but it worked by splitting the input into many small pieces and applying the one-way function to each piece, which degrades security and is impractical for realistic input sizes. What is missing is a security-preserving, general way to extract a single unpredictable bit from any one-way function.

The solution is the Goldreich-Levin hard-core predicate. Instead of fixing a coordinate of the input, which a one-way function could simply leak, the predicate is defined on a randomized encoding of the input. For a one-way function f and an auxiliary random mask r of the same length as the input x, define g(x, r) = (f(x), r) and the predicate b(x, r) = <x, r> mod 2, the inner product modulo 2. This bit is easy to compute given x and r, and the theorem is that it is hard to predict from (f(x), r) alone if f is one-way.

The proof works by reduction. Suppose an efficient algorithm G predicts b(x, r) given (f(x), r) with probability 1/2 + epsilon over random x and r. By averaging, there is a noticeable set of inputs x for which G(y, ·) is correlated with the Hadamard codeword r -> <x, r>. Fix such an x and set y = f(x). To recover a coordinate x_i, one would like to use the identity <x, r> xor <x, r xor e_i> = x_i, but asking G for both parity values doubles the error and can erase a weak signal. The key idea is to generate many random masks r_J and know their parities <x, r_J> without calling G.

Choose m = poly(n, 1/epsilon) and ell = ceil(log_2(m + 1)). Sample ell independent masks s_1, ..., s_ell and guess the ell bits <x, s_j>. Although the full guess is correct with only inverse-polynomial probability, the enumeration is small enough to repeat. For each nonempty subset J of {1, ..., ell}, define r_J = xor_{j in J} s_j and alpha_J = xor_{j in J} <x, s_j>, which is the guessed parity for r_J. By linearity of the inner product, if the seed guesses are correct, every alpha_J equals <x, r_J>. The collection {r_J} is pairwise independent and uniformly distributed. To vote for x_i, compute alpha_J xor G(y, r_J xor e_i). When the seed labels are correct, this equals x_i exactly when G is correct on the shifted mask. Taking a majority over enough subsets concentrates around x_i by pairwise-independent sampling, and a union bound over all n coordinates recovers the whole string with constant probability.

Because the seed-label guesses might be wrong, the inverter enumerates all 2^ell possible assignments, producing a polynomial-size list of candidate strings. For each candidate x', it checks whether f(x') = y. When the correct seed labels were used, the recovered string is in the list and passes the test, so the inverter succeeds with noticeable probability. This contradicts the one-wayness of f, so no efficient predictor G can exist. Thus the random inner product bit is hard-core for g.

```python
import random
from typing import Callable, List, Tuple


def inner_product_mod2(x: int, r: int, n: int) -> int:
    """Compute <x, r> mod 2 over n-bit vectors."""
    return bin(x & r).count("1") & 1


def xor_vectors(vectors: List[int]) -> int:
    """XOR together a list of bit vectors."""
    result = 0
    for v in vectors:
        result ^= v
    return result


def goldreich_levin_inverter(
    f: Callable[[int], int],
    y: int,
    n: int,
    epsilon: float,
    predictor: Callable[[int, int], int],
) -> int:
    """
    Invert a one-way function f using a predictor for the Goldreich-Levin
    hard-core predicate. Returns a candidate preimage x such that f(x) == y.
    """
    m = int((4 * n) / (epsilon ** 2)) + 1  # number of subset masks needed
    ell = max(1, (m.bit_length()))          # ceil(log2(m + 1))

    # Repeat with fresh random seed masks until a candidate passes the test.
    for _ in range(10 * (1 << ell)):
        seeds = [random.randint(0, (1 << n) - 1) for _ in range(ell)]

        # Enumerate all possible seed-label assignments.
        for assignment in range(1 << ell):
            seed_labels = [(assignment >> j) & 1 for j in range(ell)]

            # Build derived subset masks and their guessed labels.
            subset_masks = []
            subset_labels = []
            for J in range(1, 1 << ell):
                masks = [seeds[j] for j in range(ell) if (J >> j) & 1]
                subset_masks.append(xor_vectors(masks))
                label = 0
                for j in range(ell):
                    if (J >> j) & 1:
                        label ^= seed_labels[j]
                subset_labels.append(label)

            # Recover each coordinate by majority vote.
            x_recovered = 0
            for i in range(n):
                e_i = 1 << i
                votes = 0
                for r_J, alpha_J in zip(subset_masks, subset_labels):
                    vote = alpha_J ^ predictor(y, r_J ^ e_i)
                    if vote == 1:
                        votes += 1
                    else:
                        votes -= 1
                if votes > 0:
                    x_recovered |= e_i

            if f(x_recovered) == y:
                return x_recovered

    raise RuntimeError("Failed to invert f with the given predictor.")


def make_one_way_function(n: int) -> Callable[[int], int]:
    """A toy one-way function for demonstration: modular exponentiation."""
    p = (1 << n) + 1  # not necessarily prime; used only for illustration
    g = 3

    def f(x: int) -> int:
        return pow(g, x, p)

    return f


def make_predictor(secret_x: int, n: int, noise: float = 0.0) -> Callable[[int, int], int]:
    """A predictor that knows the secret but flips answers with probability noise."""
    def predict(y: int, r: int) -> int:
        bit = inner_product_mod2(secret_x, r, n)
        if random.random() < noise:
            bit ^= 1
        return bit
    return predict


if __name__ == "__main__":
    n = 8
    x_secret = random.randint(0, (1 << n) - 1)
    f = make_one_way_function(n)
    y = f(x_secret)

    epsilon = 0.3
    predictor = make_predictor(x_secret, n, noise=(1 - epsilon) / 2)

    x_found = goldreich_levin_inverter(f, y, n, epsilon, predictor)
    assert f(x_found) == y
    print(f"Recovered preimage: {x_found} (secret was {x_secret})")
```
