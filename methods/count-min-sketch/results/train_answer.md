The problem is to estimate per-item frequencies in a massive stream of updates when the domain is far too large to store explicitly. A vector a starts at zero and receives updates (i, c), meaning a_i increases by c. We need to answer point queries Q(i) ≈ a_i using only polylogarithmic space and fast per-update time, with a tunable (ε, δ) approximation guarantee. The natural baselines are the AMS tug-of-war sketch and the Count Sketch. Both use random linear projections with signs, which makes their estimators unbiased. That unbiasedness is convenient, but it forces the analysis through the variance of the estimator, and variance arguments need width proportional to 1/ε² and stronger hash independence, especially for AMS which requires four-wise independent signs. For small ε that quadratic space dependence is prohibitive, and the hardware cost of evaluating stronger hash families is nontrivial. We also want one structure that handles point queries, inner products, and downstream tasks like range queries and heavy hitters without separate machinery.

The method that escapes this is the Count-Min Sketch. It keeps a d × w table of counters. Each row j has its own pairwise-independent hash function h_j mapping items to columns. On an update (i, c), we add c to exactly one counter in each row: C[j][h_j(i)] += c. To estimate a_i we take the minimum over the d touched counters, a_hat_i = min_j C[j][h_j(i)]. The key difference from Count Sketch is that there are no sign hashes. Because the current frequency vector is non-negative, every colliding item adds mass rather than canceling it, so each row reading is guaranteed to be at least a_i. The estimate is therefore always an overestimate, and the minimum is the tightest valid upper bound among the rows. This one-sided error is what buys the better space bound.

The analysis is a first-moment argument rather than a variance argument. In a single row, the excess mass at item i is the sum of a_k over all k ≠ i that collide with i under h_j. Because the hash family is pairwise independent, the collision probability for any pair is at most 1/w, so the expected excess is at most ||a||_1 / w. If we set w = ceil(e / ε), the expected excess is at most (ε/e) ||a||_1. Markov's inequality then says the probability a single row overshoots by more than ε ||a||_1 is at most 1/e. Since the d rows use independent hash functions, the probability that every row overshoots by that much is at most e^{-d}. Choosing d = ceil(ln(1/δ)) makes this at most δ. The guarantee is a_i ≤ a_hat_i ≤ a_i + ε ||a||_1 with probability at least 1 − δ, and the lower bound is deterministic. The total space is about (e/ε) ln(1/δ) counters, a linear rather than quadratic dependence on 1/ε, and only pairwise independence is needed.

The structure is also linear: each counter is a sum of a_i over items hashing to that bucket. That means the sketch of a + b is the entrywise sum of the individual sketches, provided they use the same dimensions and hash functions. So strict-turnstile updates, where deletions occur but all current frequencies remain non-negative, are handled by adding negative increments, and distributed summaries can be merged by adding their counter arrays. For two non-negative vectors sketched compatibly, an inner product estimate is obtained by taking the dot product of corresponding rows and then the minimum across rows; the same one-sided, first-moment analysis gives additive error ε ||a||_1 ||b||_1 with probability 1 − δ. Range queries, heavy hitters, and quantiles can be built on top by maintaining Count-Min Sketches over a dyadic hierarchy of the domain, since any interval decomposes into O(log n) dyadic blocks.

```python
import collections
import math
import random
from typing import List, Optional, Tuple


HashParams = List[Tuple[int, int]]


class CountMinSketch:
    """Frequency sketch with min read-out for non-negative current vectors."""

    def __init__(
        self,
        epsilon: float = 0.001,
        delta: float = 0.01,
        seed: Optional[int] = None,
        hash_params: Optional[HashParams] = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must lie in (0, 1)")

        self.epsilon = epsilon
        self.delta = delta
        self.width = math.ceil(math.e / epsilon)
        self.depth = math.ceil(math.log(1.0 / delta))
        self.prime = 2**31 - 1
        self.total_mass = 0
        self.counts = [[0] * self.width for _ in range(self.depth)]

        if hash_params is None:
            rng = random.Random(seed)
            self.hash_params = [
                (rng.randrange(1, self.prime), rng.randrange(0, self.prime))
                for _ in range(self.depth)
            ]
        else:
            if len(hash_params) != self.depth:
                raise ValueError("hash_params length must equal depth")
            self.hash_params = [(int(a), int(b)) for a, b in hash_params]

    def _hash(self, row: int, item: int) -> int:
        a, b = self.hash_params[row]
        return ((a * (int(item) % self.prime) + b) % self.prime) % self.width

    def copy_empty(self) -> "CountMinSketch":
        return CountMinSketch(
            epsilon=self.epsilon,
            delta=self.delta,
            hash_params=self.hash_params,
        )

    def compatible(self, other: "CountMinSketch") -> bool:
        return (
            self.width == other.width
            and self.depth == other.depth
            and self.prime == other.prime
            and self.hash_params == other.hash_params
        )

    def _require_compatible(self, other: "CountMinSketch") -> None:
        if not self.compatible(other):
            raise ValueError("sketches must use the same width, depth, and hashes")

    def update(self, item: int, count: int = 1) -> None:
        self.total_mass += count
        for row in range(self.depth):
            self.counts[row][self._hash(row, item)] += count

    def estimate(self, item: int) -> int:
        return min(self.counts[row][self._hash(row, item)] for row in range(self.depth))

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        self._require_compatible(other)
        merged = self.copy_empty()
        merged.total_mass = self.total_mass + other.total_mass
        merged.counts = [
            [x + y for x, y in zip(row_a, row_b)]
            for row_a, row_b in zip(self.counts, other.counts)
        ]
        return merged

    def merge_in_place(self, other: "CountMinSketch") -> None:
        self._require_compatible(other)
        self.total_mass += other.total_mass
        for row in range(self.depth):
            for col in range(self.width):
                self.counts[row][col] += other.counts[row][col]

    def inner_product(self, other: "CountMinSketch") -> int:
        self._require_compatible(other)
        row_dots = []
        for row in range(self.depth):
            dot = sum(
                self.counts[row][col] * other.counts[row][col]
                for col in range(self.width)
            )
            row_dots.append(dot)
        return min(row_dots)


def demo() -> None:
    cms = CountMinSketch(epsilon=0.01, delta=1e-12, seed=42)
    true_counts = collections.Counter()

    for item in range(3000):
        freq = max(1, int(5000 / (item + 1)))
        true_counts[item] = freq
        cms.update(item, freq)

    errors = {item: cms.estimate(item) - count for item, count in true_counts.items()}
    max_error = max(errors.values())
    bound = cms.epsilon * cms.total_mass

    assert min(errors.values()) >= 0
    assert max_error <= bound

    shard = cms.copy_empty()
    shard_truth = collections.Counter()
    for item, count in true_counts.items():
        if item % 2 == 0:
            shard.update(item, count)
            shard_truth[item] += count

    merged = cms.merge(shard)
    merged_truth = true_counts + shard_truth
    merged_errors = {
        item: merged.estimate(item) - count for item, count in merged_truth.items()
    }
    assert min(merged_errors.values()) >= 0
    assert max(merged_errors.values()) <= merged.epsilon * merged.total_mass

    twin = cms.copy_empty()
    for item, count in true_counts.items():
        twin.update(item, count)

    true_inner = sum(count * count for count in true_counts.values())
    inner_est = cms.inner_product(twin)
    assert inner_est >= true_inner
    assert inner_est - true_inner <= cms.epsilon * cms.total_mass * twin.total_mass

    print(f"CountMinSketch(width={cms.width}, depth={cms.depth})")
    print(f"total mass: {cms.total_mass}")
    print(f"point-query bound: {bound:.2f}")
    print(f"max observed point overestimate: {max_error}")
    print(f"inner product overestimate: {inner_est - true_inner}")
    print("deterministic one-sided checks passed")


if __name__ == "__main__":
    demo()
```
