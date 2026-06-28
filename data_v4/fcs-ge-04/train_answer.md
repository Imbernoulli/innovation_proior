**Problem.** Given a simple (non-self-intersecting) polygon with `n` integer-coordinate vertices in
boundary order, count the lattice points strictly inside it (boundary points excluded). Read `n` and
the `n` vertices from stdin, print the interior count. Constraints: `3 <= n <= 10^5`,
`|x_i|, |y_i| <= 10^9`.

**Why the obvious box scan is wrong (too slow).** Enumerating every lattice point in the bounding
rectangle and testing containment is exact but its cost scales with the *enclosed area*, not with `n`.
The box can be `2*10^9` by `2*10^9` = `4*10^18` cells — centuries against a 1-second limit. It only
survives on tiny coordinates, so it is useful purely as a test oracle, not as the solution.

**Key idea — Pick's theorem.** For a simple polygon with integer vertices,
`A = I + B/2 - 1`, where `A` is the area, `I` the interior lattice points, and `B` the boundary
lattice points. Rearranged for what we want, and scaled by 2 to stay in exact integers:

```
I = A - B/2 + 1   =>   I = (2A - B) / 2 + 1.
```

Both ingredients are `O(n)` and exact:

- `2A` = `|sum_i (x_i*y_{i+1} - x_{i+1}*y_i)|` via the shoelace formula (doubled area is an exact
  integer; take the absolute value so clockwise input works).
- `B` = `sum_i gcd(|dx_i|, |dy_i|)` over edges, where `(dx_i, dy_i)` is edge `i`'s delta. One edge
  carries exactly `gcd(|dx|,|dy|)` lattice points when counting a single endpoint per edge, and summed
  around the closed boundary this counts every vertex once.

`2A - B` is always even for an integer polygon, so the division is exact. This turns an
area-proportional scan into a single linear pass over the vertices.

**Pitfalls to get right.**
1. *Shoelace overflow.* Each cross term `x_i*y_j` is up to `10^18` (fits in 64-bit), but the **sum**
   over `10^5` edges reaches `~10^23`, which overflows `long long` by orders of magnitude — a silent
   wrong answer on exactly the large hidden tests. Accumulate the doubled area in `__int128`, and cast
   each product to `__int128` before multiplying. The boundary sum `B` (`<= ~1.4*10^14`) is safe in
   `long long`.
2. *Orientation.* The signed shoelace area is negative for clockwise polygons. Take the absolute value
   of the **128-bit** accumulator with a hand-written `if (twiceArea < 0) twiceArea = -twiceArea;` —
   `llabs` would truncate to 64 bits and reintroduce the overflow.
3. *Printing `__int128`.* `cout` cannot print `__int128`; extract decimal digits by hand.

**Edge cases.** Clockwise and counter-clockwise input (absolute area handles both); triangles whose
only lattice points are on the boundary, e.g. `(0,0),(2,0),(0,2)` -> `I = 0`; thin slivers with
positive area but no interior point -> `I = 0`; redundant collinear vertices along an edge (contribute
0 area, leave `B` and hence `I` unchanged); large/negative coordinates near `10^9` (the `__int128`
accumulator absorbs them). The problem guarantees a non-degenerate simple polygon, so zero-area
(all-collinear) inputs do not occur.

**Complexity.** `O(n)` time, `O(n)` space for the vertices (`O(1)` beyond input).

**Worked example.** L-shaped hexagon `(0,0),(4,0),(4,2),(2,2),(2,4),(0,4)`: `2A = 24` so `A = 12`;
edge deltas have gcds `4,2,2,2,2,4` so `B = 16`; `I = (24 - 16)/2 + 1 = 5`.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Count lattice points strictly interior to a simple polygon.
// Pick's theorem: A = I + B/2 - 1  =>  I = A - B/2 + 1.
// With doubled area  S = 2A  (exact integer via shoelace) and boundary count B,
// I = (S - B) / 2 + 1.  S can reach ~2e23, so accumulate it in __int128.
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    __int128 twiceArea = 0; // signed doubled area via shoelace
    long long boundary = 0; // total boundary lattice points

    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        // shoelace cross term  x_i*y_j - x_j*y_i  (fits in __int128)
        twiceArea += (__int128)x[i] * y[j] - (__int128)x[j] * y[i];
        // boundary lattice points on edge i->j = gcd(|dx|, |dy|)
        long long dx = llabs(x[j] - x[i]);
        long long dy = llabs(y[j] - y[i]);
        boundary += std::__gcd(dx, dy);
    }

    if (twiceArea < 0) twiceArea = -twiceArea; // S = |2A|

    // interior = (S - B) / 2 + 1   (S - B is always even for integer polygons)
    __int128 interior = (twiceArea - (__int128)boundary) / 2 + 1;

    // print the __int128 result
    if (interior == 0) {
        cout << 0 << "\n";
        return 0;
    }
    bool neg = interior < 0;
    if (neg) interior = -interior;
    string s;
    while (interior > 0) {
        int d = (int)(interior % 10);
        s.push_back((char)('0' + d));
        interior /= 10;
    }
    if (neg) s.push_back('-');
    reverse(s.begin(), s.end());
    cout << s << "\n";
    return 0;
}
```
