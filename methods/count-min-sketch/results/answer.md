# Count-Min Sketch: Algorithm and Guarantees

## Data Structure

A Count-Min Sketch stores a `d x w` table of counters. Row `j` has its own hash function `h_j` from items to columns `{0, ..., w-1}`. Each update touches one counter per row.

For accuracy `epsilon` and failure probability `delta`, choose

- `w = ceil(e / epsilon)`
- `d = ceil(ln(1 / delta))`

The hash functions are independent across rows, and each row's hash family must satisfy `Pr[h_j(i) = h_j(k)] <= 1/w` for `i != k`.

## Operations

**Update item `x` by `c`:**

```text
for j in 0..d-1:
    C[j][h_j(x)] += c
```

**Point query for item `x`:**

```text
return min_j C[j][h_j(x)]
```

The minimum is the right read-out because, for a non-negative current frequency vector, every collision only adds extra mass.

## Error Bound

Let `a` be the current non-negative frequency vector and let `N = ||a||_1 = sum_i a_i`. For any fixed item `i`, the estimate `a_hat_i` satisfies

```text
a_i <= a_hat_i <= a_i + epsilon * N
```

with probability at least `1 - delta`. The lower bound is deterministic.

For one row, write the collision noise as

```text
X_{i,j} = sum_{k != i} 1[h_j(k) = h_j(i)] * a_k.
```

Then

```text
E[X_{i,j}]
  = sum_{k != i} a_k * Pr[h_j(k) = h_j(i)]
  <= N / w
  <= (epsilon / e) * N.
```

Since `X_{i,j} >= 0`, Markov's inequality gives

```text
Pr[X_{i,j} > epsilon * N]
  <= E[X_{i,j}] / (epsilon * N)
  <= 1/e.
```

The minimum is too large only when every independent row has collision noise above `epsilon * N`, so

```text
Pr[a_hat_i > a_i + epsilon * N]
  <= (1/e)^d
  = e^{-d}
  <= delta.
```

For `m` fixed queries that must all satisfy the guarantee, run the same theorem with per-query failure probability `delta / m` and union-bound the `m` bad events.

The constant `e` minimizes the counter count in this Markov amplification scheme. If `w = b / epsilon`, then one row fails with probability at most `1/b`, so `d = log_b(1/delta)` rows are needed. The table size is proportional to

```text
(b / epsilon) * (ln(1/delta) / ln b),
```

and `b / ln b` is minimized at `b = e`.

## Linearity and Merge

Each counter is a linear projection:

```text
C[j,k] = sum_{i: h_j(i)=k} a_i.
```

With the same hash functions, sketches add exactly:

```text
sketch(a + b) = sketch(a) + sketch(b).
```

That is why strict-turnstile deletions work when the current vector remains non-negative, and why distributed summaries can be merged by adding corresponding counters. The one-sided minimum guarantee is still a non-negative-vector guarantee; a general signed vector loses the deterministic overestimate property.

For non-negative vectors `a` and `b` sketched with identical width, depth, and hash functions, an inner product estimate is obtained by taking the dot product of corresponding rows and then the minimum over rows:

```text
row_j = sum_k C_a[j,k] * C_b[j,k]
estimate = min_j row_j.
```

The row value equals `a . b` plus non-negative collision cross-talk, and the same first-moment/Markov argument gives additive error at most `epsilon * ||a||_1 * ||b||_1` with probability at least `1 - delta`.

## Comparison to Count Sketch

| Feature | Count-Min Sketch | Count Sketch |
|---|---|---|
| Update | `C[j][h_j(x)] += c` | `C[j][h_j(x)] += g_j(x) * c` |
| Query | minimum over rows | median over signed row estimates |
| Error | one-sided overestimate | two-sided |
| Norm | `epsilon * ||a||_1` | `epsilon * ||a||_2` |
| Width dependence | `1/epsilon` | `1/epsilon^2` |
| Main proof tool | Markov on first moment | variance/concentration |

Count Sketch uses signs to make each row unbiased. Count-Min drops the signs, accepts one-sided bias on non-negative data, and avoids the variance route that creates the quadratic `1/epsilon^2` dependence.

## Working Python Implementation

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
        """Carter-Wegman-style affine hash, then reduce to the table width."""
        a, b = self.hash_params[row]
        return ((a * (int(item) % self.prime) + b) % self.prime) % self.width

    def copy_empty(self) -> "CountMinSketch":
        """Create an empty sketch with identical dimensions and hash functions."""
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
        """Apply a stream update. When current counts are non-negative, total_mass is ||a||_1."""
        self.total_mass += count
        for row in range(self.depth):
            self.counts[row][self._hash(row, item)] += count

    def estimate(self, item: int) -> int:
        """Return the minimum row reading for item."""
        return min(self.counts[row][self._hash(row, item)] for row in range(self.depth))

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        """Return the sketch of the summed streams."""
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
        """Estimate a . b from compatible sketches of non-negative vectors."""
        self._require_compatible(other)
        row_dots = []
        for row in range(self.depth):
            dot = sum(
                self.counts[row][col] * other.counts[row][col]
                for col in range(self.width)
            )
            row_dots.append(dot)
        return min(row_dots)

    def __repr__(self) -> str:
        return (
            f"CountMinSketch(width={self.width}, depth={self.depth}, "
            f"epsilon={self.epsilon}, delta={self.delta})"
        )


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
    observed_point_ok = max_error <= bound

    assert min(errors.values()) >= 0
    assert observed_point_ok

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
    observed_merge_ok = max(merged_errors.values()) <= merged.epsilon * merged.total_mass

    assert min(merged_errors.values()) >= 0
    assert observed_merge_ok

    twin = cms.copy_empty()
    for item, count in true_counts.items():
        twin.update(item, count)

    true_inner = sum(count * count for count in true_counts.values())
    inner_est = cms.inner_product(twin)
    observed_inner_ok = (
        inner_est - true_inner <= cms.epsilon * cms.total_mass * twin.total_mass
    )

    assert inner_est >= true_inner
    assert observed_inner_ok

    print(cms)
    print(f"total mass: {cms.total_mass}")
    print(f"point-query bound: {bound:.2f}")
    print(f"max observed point overestimate: {max_error}")
    print(f"inner product overestimate: {inner_est - true_inner}")
    print(f"observed point upper-bound event: {observed_point_ok}")
    print(f"observed merge upper-bound event: {observed_merge_ok}")
    print(f"observed inner-product upper-bound event: {observed_inner_ok}")
    print("deterministic one-sided checks passed")


if __name__ == "__main__":
    demo()
```

## Key Takeaways

1. The sketch uses `d` independent pairwise-independent hashes into a `d x w` counter table.
2. The point estimate is the minimum row reading.
3. Non-negative collision noise gives deterministic no-underestimate and a Markov first-moment upper tail.
4. `w = ceil(e/epsilon)` and `d = ceil(ln(1/delta))` give additive error `epsilon * ||a||_1` with probability at least `1 - delta`.
5. Linearity gives merge and inner products only for sketches with identical width, depth, and hash functions.
