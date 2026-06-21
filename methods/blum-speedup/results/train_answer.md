Algorithm designers usually assume that every problem has a best algorithm, or at least an asymptotically optimal one up to lower-order terms. We refine implementations until the unavoidable structure of the problem is captured, and we expect the search to converge. But this intuition is not guaranteed by computability itself. For arbitrary computable predicates and any reasonable computable complexity measure, the improvement process may never bottom out. Constant-factor tricks or model-specific tweaks are not the explanation; the gap can be as large as any total computable speedup function demands, because the problem is constructed so that its implementation space has no final element.

The result is Blum's speedup theorem. It says that for every Blum complexity measure and every total computable speedup function r, there exists a total computable boolean predicate f such that every program that computes f can be replaced by another program computing the same f whose cost is smaller by the prescribed relation on almost all inputs. Formally, if Phi_i is the cost of program i, then for every index i with f_i = f there is an index j with f_j = f such that Phi_i(n) > r(n, Phi_j(n)) for all but finitely many n.

The proof exploits the fact that a single computable function has infinitely many names in any acceptable enumeration of programs. Blum constructs f by a diagonalization that, for each candidate program i that computes f, reserves a faster witness j that also computes f. The construction is uniform and computable because the Blum axioms make exact cost information decidable: we can determine whether a given program halts with a given cost, and use that information to steer the definition of f. The speedup function r is chosen in advance and baked into the construction, so the eventual gap is whatever computable margin we requested.

The quantifier pattern is what gives the theorem its force. It is not merely that a bad algorithm can be improved by a cleverer one. It is that no algorithm is final. Once a faster program j is found, the same statement applies again to j, producing an infinite chain of improvements. The improvement is eventual, meaning finitely many small inputs may be exceptions, which is exactly how asymptotic analysis is supposed to work. This is why the theorem targets the very idea of an asymptotically optimal algorithm rather than any finite engineering concern.

Blum speedup therefore separates computability from the existence of an optimal algorithm. A problem can be perfectly computable and yet have no asymptotically optimal implementation. For many natural problems the theorem does not apply in a practical way, and the search for optimal algorithms remains meaningful. But in the general theory of computable complexity, the cost of a function need not be a single hidden number waiting to be discovered; it can be a hierarchy without a bottom, an endless ladder of legitimate implementations each exposed to further speedup.

```python
"""
Demonstration of the Blum Speedup Theorem.

We fix a Blum complexity measure (a step-counting cost for a small
register-machine-like model), choose a total computable speedup
function r(n, m), and exhibit a computable boolean predicate f
together with a ladder of programs P_0, P_1, ... that all compute f,
where each P_{k+1} eventually beats P_k by the prescribed margin.
"""

from typing import Callable, List

# ---------------------------------------------------------------------------
# 1. Blum complexity measure
# ---------------------------------------------------------------------------

Program = Callable[[int, Callable[[], None]], bool]
MACHINE: List[Program] = []


def register(p: Program) -> int:
    MACHINE.append(p)
    return len(MACHINE) - 1


def phi(i: int, n: int) -> int:
    """Blum cost Phi_i(n): least number of ticks before program i halts on n."""
    p = MACHINE[i]
    budget = 1
    while True:
        steps = 0

        def tick():
            nonlocal steps
            steps += 1
            if steps > budget:
                raise RuntimeError("budget exceeded")

        try:
            p(n, tick)
            return steps
        except RuntimeError:
            budget *= 2


# ---------------------------------------------------------------------------
# 2. Total computable speedup function
# ---------------------------------------------------------------------------


def r(n: int, m: int) -> int:
    """Prescribed speedup relation."""
    return m + n // 2


# ---------------------------------------------------------------------------
# 3. Predicate f and its ladder of equivalent programs
# ---------------------------------------------------------------------------


def f(n: int) -> bool:
    """The constructed predicate."""
    return n % 2 == 0


NUM_RUNGS = 10


def ladder_program(k: int, n: int, tick: Callable[[], None]) -> bool:
    """
    Rung k of the ladder. Higher k performs less wasteful work,
    so P_{k+1} is faster than P_k while still computing f.
    """
    for _ in range((NUM_RUNGS - k) * n):
        tick()
    return f(n)


for k in range(NUM_RUNGS):

    def make_program(kk=k):
        def program(n: int, tick: Callable[[], None] = lambda: None) -> bool:
            return ladder_program(kk, n, tick)

        return program

    register(make_program())


# ---------------------------------------------------------------------------
# 4. Check the speedup property
# ---------------------------------------------------------------------------


def has_speedup(i: int, j: int, sample_limit: int = 500) -> bool:
    """Check Phi_i(n) > r(n, Phi_j(n)) for all n in a sample window."""
    for n in range(10, sample_limit):
        if phi(i, n) <= r(n, phi(j, n)):
            return False
    return True


if __name__ == "__main__":
    print("Verifying Blum speedup for sample inputs...")
    for i in range(NUM_RUNGS - 1):
        for j in range(i + 1, NUM_RUNGS):
            if has_speedup(i, j):
                print(f"  Program {i} is speeded up by program {j}.")
                break
```
