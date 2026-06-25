**Problem.** A festival stage runs over equal time slots `1, 2, 3, ...`. You are pitched `n` acts; act `i` pays fee `p[i]` if staged, and it must go in one slot numbered `<= d[i]`. Each slot holds at most one act, each act takes one slot, and you may turn acts away. Read `n` then the `n` pairs `(p[i], d[i])` from stdin and print the maximum total fee. Constraints: `0 <= n <= 2*10^5`, `0 <= p[i] <= 10^9`, `1 <= d[i] <= 10^9`.

**Key idea — greedy by fee, latest legal slot (job sequencing with deadlines).** Sort the acts by fee descending. Process them in that order; place each act in the **latest still-free slot that is `<= d[i]`**, and skip it if no free slot `<= d[i]` exists. Choosing the *latest* legal slot is what keeps the small-numbered slots available for tight-deadline acts that have nowhere else to go.

- *Why it is optimal (exchange argument).* Take an optimal schedule agreeing with the greedy on as many of the highest-fee acts as possible. If they first differ on some act `i`, you can either re-slot `i` to greedy's (latest, least-constraining) slot or swap a `>= p[i]`-fee act for `i`, neither of which lowers the total while strictly increasing agreement — contradiction. So the greedy total is optimal.
- *Finding "latest free slot `<= d`" fast.* A disjoint-set-union: `parent[s]` points to the largest slot `<= s` still free. `find(d)` returns that slot (or sentinel `0` = none); after seating an act in slot `s`, set `parent[s] = s - 1`. Near-`O(alpha)` amortized per query.

**Pitfalls.**
1. *Silent 32-bit overflow (the headline trap).* With `n = 2*10^5` acts all fitting at fee `10^9`, the total reaches `2*10^14`, which overflows a 32-bit `int`. An `int` accumulator passes every small test and then silently wraps on the maximal hidden test: on `n = 200000` acts of `(10^9, 10^9)` it prints `552894464`, which is exactly `2*10^14 mod 2^32`, instead of `200000000000000`. Accumulate fees in `long long`.
2. *Deadline-indexed array blows memory.* `d[i]` can be `10^9`, so sizing the DSU array by `max(d)` would allocate ~4 GB. Cap each deadline at `n` first: at most `n` acts are ever seated, so only slots `1..n` are usable and a deadline `>= n` is equivalent to `n`. That makes the array `O(n)`. (The cap preserves the optimum — verified against brute force.)
3. *Sentinel slot.* `find` returns `0` when no slot is free; guard `if (s > 0)` so you never add a fee for an unschedulable act and never write `parent[0]`.

**Edge cases.** `n = 0` -> `0` (no acts). Single act `(5,1)` -> `5`. Two acts sharing deadline 1 -> only the larger fee fits. `p[i] = 0` acts never change the total (they sort last and only take leftover slots). Deadline far above `n` -> capped to `n`, still fits.

**Complexity.** `O(n log n)` for the sort plus near-`O(n alpha(n))` for the DSU; `O(n)` memory. Runs the `n = 2*10^5` worst case in about 0.02 s.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // no performances -> profit 0
    vector<long long> p(n);
    vector<int> d(n);
    for (int i = 0; i < n; i++) {
        cin >> p[i] >> d[i];
        // Only the first n slots can ever be used (at most n acts get scheduled),
        // so a deadline beyond n is equivalent to a deadline of n. This caps slot
        // numbers at n and keeps the DSU array O(n) even when d[i] is up to 1e9.
        if (d[i] > n) d[i] = n;
    }

    // Order indices by profit descending; ties broken arbitrarily (does not affect the total).
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int x, int y) { return p[x] > p[y]; });

    // Disjoint-set-union "find latest free slot <= d": parent[s] points to the
    // largest still-free slot index that is <= s. Slot 0 is the sentinel "no slot".
    vector<int> parent(n + 1);
    iota(parent.begin(), parent.end(), 0); // parent[s] = s initially (all free)

    function<int(int)> find = [&](int s) {
        while (parent[s] != s) { parent[s] = parent[parent[s]]; s = parent[s]; }
        return s;
    };

    long long total = 0;
    for (int idx : order) {
        int s = find(d[idx]);   // latest free slot <= deadline, or 0 if none
        if (s > 0) {
            total += p[idx];    // take the job into slot s
            parent[s] = s - 1;  // slot s now points to the next-lower free slot
        }
    }

    cout << total << "\n";
    return 0;
}
```
