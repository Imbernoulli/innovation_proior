The problem is to expand a short secret seed into a long stream of bits that looks random to any feasible observer. A purely statistical generator is not enough: a deterministic recurrence can pass frequency tests while still being easy to invert once the rule is known. The right target is computational unpredictability. We ask the adversary to predict the next bit after seeing a prefix of the output, and we require that no efficient algorithm can do noticeably better than random guessing.

The difficulty is connecting this next-bit test to a concrete hardness assumption. Simply iterating a one-way function is insufficient, because a function can be hard to invert as a whole while still leaking an easy-to-predict predicate of its output. We need a specific bit that is easy to compute during generation but hard to compute on a hidden state that the adversary must infer. That bit should also be extractable from a random state without changing the distribution of challenges.

The method that satisfies these requirements is the Blum-Micali pseudorandom bit generator. It combines an efficiently computable permutation with a hard predicate. The state is advanced by repeatedly applying the permutation, and the output bits are emitted from the far end of the orbit back toward the seed. Because the permutation is a bijection, every shifted state remains uniformly distributed, so a predictor that succeeds on the stream can be turned into a predictor for the hard predicate on a fresh random state. This gives next-bit unpredictability under the hardness assumption on the predicate.

Concretely, the generator can be instantiated using discrete exponentiation modulo a prime. Let p be a large odd prime such that p − 1 has a large prime factor, and let g be a generator of the multiplicative group modulo p. The state update is f(x) = g^x mod p, and the hard predicate is whether the discrete logarithm of the state lies in the lower half of the exponent range. Since the generator always knows the current exponent, this bit is trivial to compute during generation; an adversary that sees only the group element would have to break the discrete logarithm problem to predict it.

The reduction works as follows. Given a challenge group element y whose predicate we want to learn, compute the forward prefix B(f(y)), B(f^2(y)), ..., feed it to the stream predictor, and return the predictor’s answer as the guess for B(y). If the predictor has non-negligible advantage, we obtain a non-negligible advantage on the hard predicate, which can be amplified into a discrete-log algorithm. Therefore, under the discrete-log hardness assumption, the Blum-Micali generator is next-bit unpredictable.

```python
import secrets
from typing import List

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(64):
        a = secrets.randbelow(n - 2) + 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def find_generator(p: int) -> int:
    # Assumes p is a safe prime of the form p = 2*q + 1.
    q = (p - 1) // 2
    for g in range(2, p):
        if pow(g, q, p) != 1 and pow(g, 2, p) != 1:
            return g
    raise ValueError("no generator found")

def generate_safe_prime(bits: int) -> int:
    while True:
        q = secrets.randbits(bits - 1) | (1 << (bits - 2)) | 1
        p = 2 * q + 1
        if is_prime(q) and is_prime(p):
            return p

def blum_micali_prg(p: int, g: int, seed: int, length: int) -> List[int]:
    if not (1 <= seed < p):
        raise ValueError("seed must lie in [1, p-1]")
    state = seed
    bits = []
    for _ in range(length):
        # Output the principal-square-root predicate for the current state.
        # Since state = g**exponent mod p, exponent is tracked implicitly.
        # Here we compute the bit directly from the current exponent for clarity.
        bits.append(1 if state <= (p - 1) // 2 else 0)
        state = pow(g, state, p)
    return bits

if __name__ == "__main__":
    p = generate_safe_prime(128)
    g = find_generator(p)
    seed = secrets.randbelow(p - 1) + 1
    out = blum_micali_prg(p, g, seed, length=128)
    print("p =", p)
    print("g =", g)
    print("seed =", seed)
    print("bits =", "".join(map(str, out)))
```
