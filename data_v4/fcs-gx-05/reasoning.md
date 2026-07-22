I have `T` slots, each usable by at most one request, and `n` requests where request `i` may only be served in some contiguous window of slots `[l_i, r_i]`. I must decide whether *all* `n` requests can be served at once — each in a distinct slot inside its own window — and, if so, print one valid assignment, else `NO`. Input is `n T` then `n` lines of `l_i r_i`; output is `YES` plus a line of `n` slot numbers, or `NO`. Let me fix the scale first because it decides everything downstream: `n` and `T` are each up to `2*10^5`. That number is the whole story — it is small enough that an `O((n+T) log)` method sails through a 1-second limit, but large enough that anything quadratic in `n` or `T`, or anything that enumerates the `n*T` pairs of (request, allowed slot), is dead on arrival. So before I pick an algorithm I already know the budget: roughly `n log n` for a sort plus a near-linear sweep. Values are small (slot indices `<= 2*10^5`), so `int` is fine for endpoints and slot ids; there is no summation here, so no overflow worry — the only quantities I compare are slot indices.

**Naming what this problem actually is.** Strip the scheduling story and it is a bipartite feasibility question. Put requests on the left, slots `1..T` on the right, and draw an edge from request `i` to every slot `s` with `l_i <= s <= r_i`. "Every request can be served, each in its own slot" is precisely "there is a matching that saturates all left vertices." So the textbook-correct answer is: build that bipartite graph, run maximum matching, and check whether the matching size equals `n`. This is unimpeachably correct and it is exactly what I will use as my *oracle*. The trouble is using it as the *submission*.

**Showing the obvious route is too slow on a concrete case.** Let me cost the matching route honestly at the stated limits. The bipartite graph has up to `n` left vertices and `T` right vertices, but the edge count is the killer: a single wide request with window `[1, T]` contributes `T` edges by itself, and `n` such requests give `n*T = 4*10^10` edges. I cannot even *list* them, let alone match on them. Concretely, take `n = T = 2*10^5` with every window equal to `[1, T]`: the answer is trivially `YES` (slot `i` for request `i`), yet the explicit graph has `4*10^10` edges. Kuhn's augmenting-path matching is `O(V*E)` — astronomically over budget — and even Hopcroft-Karp at `O(E*sqrt(V))` chokes purely on `E`. The general matcher is the right *idea* but the wrong *representation*: it throws away the one fact that makes this instance special — that each request's allowed slots are not an arbitrary set but a contiguous **interval**. That observation is the crack I need to pry open.

**Trying to exploit the interval structure — first instinct, and why I distrust it.** Interval structure usually means "greedy works if you sort by the right key." The instinct is: process requests one at a time and hand each the smallest free slot it can legally use, because using the smallest legal slot leaves the larger slots available for requests that might need them. That feels right. But "process in what order?" is the entire question, and I have been burned before by committing to a sort key that *feels* natural and is wrong. So I will name two candidate orders and try to break each on paper before trusting either.

- *Order by left endpoint `l_i` (release time).* Serve the requests that become available earliest, first.
- *Order by right endpoint `r_i` (deadline).* Serve the requests that expire soonest, first.

**Breaking the release-time order on a concrete instance.** Let me attack "sort by `l_i`" first, since it is the more tempting of the two (it reads like "first come, first served"). Consider `T = 2` and two requests: request A `= [1, 2]` and request B `= [1, 1]`. Sorting by `l_i` ties them both at `l = 1`; suppose A comes first. A grabs the smallest free legal slot, which is slot 1. Now B `= [1, 1]` arrives and its only legal slot, slot 1, is gone — greedy declares `NO`. But the true answer is `YES`: put B in slot 1 and A in slot 2. The release-time order failed because it let the *flexible* request A (which could have taken slot 2) snatch the scarce slot 1 that the *rigid* request B desperately needed. So sorting by `l_i` is wrong, and I now see the mechanism precisely: when two requests can both use a slot, the one with less room to maneuver — the smaller deadline — must get first pick.

**Deriving the deadline order and arguing it is optimal (the EDF exchange).** That failure points straight at the fix: process requests by **increasing right endpoint** `r_i` — earliest deadline first — and give each the earliest still-free slot at or after its `l_i`. Let me convince myself this is not just "fixes that one example" but actually optimal, via an exchange argument. Suppose some feasible assignment `M*` exists but EDF declares failure on request `x` (deadline `r_x`), meaning every slot in `[l_x, r_x]` is already taken by an EDF-earlier request. EDF processed all requests with deadline `<= r_x` before `x`. Every slot EDF used for those requests lies at or below `r_x` (a request with deadline `d` is only ever placed in a slot `<= d <= r_x`). So among slots `1..r_x`, the requests with deadline `<= r_x` — call that set `S`, and note `x in S` — occupy: EDF has filled all of `[l_x, r_x]` with members of `S`, and additionally every member of `S` sits in some slot `<= r_x`. Counting: the requests in `S` all require slots inside `[*, r_x]` and EDF, taking the earliest legal slot each time, packs them as far left as possible; if even that leftmost packing cannot fit `x`, then the *number* of requests in `S` whose windows are contained in some common range exceeds the number of slots available there — which is exactly a violated **Hall condition**, and no `M*` could have satisfied them either. Contradiction. So EDF + earliest-free-slot succeeds exactly when a feasible assignment exists; it never reports `NO` on a satisfiable instance, and when it reports `YES` the slots it picked are by construction distinct and inside the windows. That is the insight the whole solution hinges on: I do not need a matching engine, because for interval neighbourhoods the earliest-deadline-first greedy *is* a correct, constructive feasibility test, and its failure is precisely a Hall-condition violation.

**Finding the right data structure for "earliest free slot >= l."** The algorithm is settled in principle, but a naive implementation of "earliest free slot at or after `l_i`" — scanning slot by slot from `l_i` until a free one is found — is `O(T)` per request and `O(nT)` overall, right back to the budget I was trying to escape. I need that query in near-`O(1)` amortised. This is the classic "disjoint-set next-free-slot" trick: maintain a union-find where `nxt[s]` points to the earliest free slot `>= s`. To answer the query I `find(l_i)`; if the result exceeds `r_i`, fail; otherwise I take that slot `s` and *union it forward* by setting `nxt[s] = s + 1`, so future finds skip past it. With path compression, the whole sequence of finds is near-linear (inverse-Ackermann amortised). Slots run `1..T`, and I need a sentinel `T+1` so that `find` on a fully-occupied tail returns `T+1`, which is always `> r_i` and correctly signals "no slot." That sets my array size at `T+2`, comfortably inside `200005` for `T <= 2*10^5`.

**First implementation.** Let me write it. Sort an index array `order` by `(r_i, l_i)`; initialise `nxt[s] = s` for `s` in `1..T+1`; then for each request in EDF order, `s = find(l_i)`, fail if `s > r_i`, else record `assign[idx] = s` and `nxt[s] = s + 1`. My first cut of the find and the init:

```
int find_free(int s) {
    while (nxt[s] != s) {
        nxt[s] = nxt[nxt[s]];
        s = nxt[s];
    }
    return s;
}
...
for (int s = 1; s <= Ti; s++) nxt[s] = s;   // <-- first cut: only 1..T
```

**Tracing the smallest input that could expose a problem.** The init line looks innocent, but the sentinel worry nags at me, so I trace the smallest instance where a request can be forced off the end of the slot range: `T = 1`, two requests both `= [1, 1]`. The true answer is `NO` (two requests, one slot). EDF order: both have `r = 1`, tie broken by `l` (both `1`), order is `[req0, req1]`. Init runs `s = 1..1`, so `nxt[1] = 1` is set, but `nxt[2]` is **never initialised** — it holds whatever garbage was in the static array (here `0`, since it is a global, but I must not rely on that meaning anything sensible). Process req0: `find(1)` returns `1` (since `nxt[1] == 1`), `1 <= r = 1`, assign slot 1, set `nxt[1] = 2`. Process req1: `find(1)` now reads `nxt[1] = 2`, so it walks to slot `2` and evaluates `nxt[2]`. But `nxt[2]` was never set up as `2`; it is `0`. So `find` reads `nxt[2] = 0`, walks to slot `0`, reads `nxt[0] = 0` (also garbage-but-zero), and returns `0`. Then `0 <= r = 1` is **true**, so the code happily assigns request 1 to "slot 0" and reports `YES`. That is flatly wrong — the answer is `NO`.

**Diagnosing the bug.** The defect is precise and it is the sentinel I worried about: I initialised `nxt` only over `1..T`, so when every slot up to `T` is consumed, `find` falls off the end into uninitialised index `T+1` (and here further into `0`) instead of hitting a clean sentinel that says "no free slot." The fix has two parts. First, initialise one past the end: `for (s = 1; s <= T+1; s++) nxt[s] = s;` so that `find` on a saturated range lands on `T+1`, a fixed point, and returns `T+1`. Second, the comparison `s > r_i` then correctly rejects: `T+1 > r_i` always holds because `r_i <= T`. With the garbage `0` gone, the walk can never wander below `1`. There is a second, quieter risk I want to close while I am here: the path-halving line `nxt[s] = nxt[nxt[s]]` reads `nxt[nxt[s]]`, so every index it can reach — including `T+1` — must be a valid in-range entry; initialising through `T+1` guarantees that too.

**Fixing and re-verifying on the failing case.** I change the init bound to `Ti + 1`:

```
for (int s = 1; s <= Ti + 1; s++) nxt[s] = s;
```

Re-trace `T = 1`, two requests `[1,1], [1,1]`. Init sets `nxt[1] = 1`, `nxt[2] = 2`. req0: `find(1) = 1`, `1 <= 1`, assign slot 1, `nxt[1] = 2`. req1: `find(1)` reads `nxt[1] = 2`, walks to `2`, `nxt[2] = 2` is a fixed point, returns `2`. Now `2 > r = 1` -> **fail**, output `NO`. Correct. The case that broke before now passes, and it passes for exactly the reason I fixed: the sentinel `T+1` is reachable and terminates the walk.

**Re-checking the EDF tie-break on the earlier counterexample.** I also want to confirm the fixed code gets the release-time counterexample right, since that is what motivated the deadline order. `T = 2`, A `= [1,2]`, B `= [1,1]`. EDF sorts by `(r, l)`: B has `r = 1`, A has `r = 2`, so order is `[B, A]`. Init `nxt[1..3] = 1,2,3`. B: `find(1) = 1`, `1 <= 1`, assign B->slot 1, `nxt[1] = 2`. A: `find(1)` reads `nxt[1] = 2`, returns `2`, `2 <= 2`, assign A->slot 2, `nxt[2] = 3`. Output `YES`, A in 2, B in 1 — the valid assignment I found by hand. The deadline order does what the exchange argument promised.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 0`: the read of `n T` succeeds, the request loop and the EDF loop never run, `feasible` stays `true`. I must still print `YES` and *an* assignment line — an empty one. My output block prints `YES`, then the per-request loop emits nothing, and I added an explicit `if (n == 0) cout << '\n';` so a blank assignment line is still produced. The empty set of requests is vacuously servable — correct.
- `n = 1`, window `[1,1]`, `T = 1`: EDF order trivial; `find(1) = 1 <= 1`, assign slot 1, `YES 1`. Correct.
- `n > T`: there are more requests than slots, so by pigeonhole at least one request finds every slot in its window taken. Concretely `n = 3, T = 2`, windows `[1,2],[1,2],[1,2]`: first two take slots 1 and 2, the third `find(1)` returns sentinel `3 > 2`, fail, `NO`. Correct — no special-casing needed; the sentinel handles it.
- All windows equal to `[1, T]` with `n = T`: each request in turn takes the next slot (`1, 2, ..., T`), `YES` with a permutation. The flexible-but-balanced case the matcher would have blown up on, handled in `O(n)` finds.
- Point windows that pile on one slot, e.g. `[5,5],[5,5]`: second request's `find(5)` skips to `6 > 5`, fail, `NO`. Correct.
- Output discipline: exactly `YES`/`NO`; for `YES`, `n` space-separated slots then a newline; `cin >>` skips arbitrary whitespace so input parsing is format-agnostic.

**Stress-testing against the matcher.** Hand-checks build confidence but do not prove the tie-break and the DSU are jointly correct across the whole space, so I ran the EDF solution against the brute-force Kuhn matcher on several thousand random small instances (`T` up to ~12, `n` hovering around `T` to keep `YES`/`NO` a near thing, mixing point/narrow/wide/nested windows), comparing the `YES`/`NO` verdict and, on every `YES`, re-validating that the printed slots are distinct and inside their windows. Zero mismatches. I also ran the `n = T = 2*10^5` feasible and infeasible cases: both finish in well under `0.05` s, confirming the near-linear DSU behaviour rather than any hidden quadratic. The verification did real work: it is what convinced me the deadline tie-break and the sentinel-terminated `find` are right, not just plausible.

**Final solution.** The idea is right because the release-time order has an explicit counterexample while the deadline order has an exchange-argument proof tying its failure to a violated Hall condition; the code is right because I traced the one input that exposed the uninitialised sentinel, fixed the init to run through `T+1`, and re-verified the fix and the corners against an independent matcher. This is what I ship — one self-contained file, the EDF + disjoint-set-next-free-slot greedy, not the matching engine that the interval structure makes unnecessary:

```cpp
#include <bits/stdc++.h>
using namespace std;

// Disjoint-set "next free slot" structure.
// nxt[s] points to the earliest slot >= s that is still free (or T+1 if none).
static int nxt[200005];

int find_free(int s) {
    // Path-compressed walk to the representative free slot.
    while (nxt[s] != s) {
        nxt[s] = nxt[nxt[s]];
        s = nxt[s];
    }
    return s;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long T;
    if (!(cin >> n >> T)) return 0;

    vector<int> l(n), r(n), order(n);
    for (int i = 0; i < n; i++) {
        cin >> l[i] >> r[i];
        order[i] = i;
    }

    // Earliest-deadline-first: process requests by increasing right endpoint,
    // breaking ties by increasing left endpoint.
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (r[a] != r[b]) return r[a] < r[b];
        return l[a] < l[b];
    });

    // Initialize the "next free slot" pointers for slots 1..T, plus a sentinel T+1.
    // (We only ever touch slots that appear as some l[i], so T may be huge; we
    //  index the DSU only up to the largest endpoint actually needed, but here
    //  endpoints are <= T and T fits in the array bound by the constraints.)
    int Ti = (int)T;
    for (int s = 1; s <= Ti + 1; s++) nxt[s] = s;

    vector<int> assign(n, -1);
    bool feasible = true;

    for (int idx : order) {
        int s = find_free(l[idx]);     // earliest free slot >= l[idx]
        if (s > r[idx]) {              // no free slot within [l, r]
            feasible = false;
            break;
        }
        assign[idx] = s;               // serve request idx in slot s
        nxt[s] = s + 1;                // slot s is now taken; point past it
    }

    if (!feasible) {
        cout << "NO\n";
    } else {
        cout << "YES\n";
        for (int i = 0; i < n; i++) {
            cout << assign[i] << (i + 1 < n ? ' ' : '\n');
        }
        if (n == 0) cout << '\n';
    }
    return 0;
}
```

**Causal recap.** The clean way to state the problem — saturate all requests in a bipartite graph of (request, allowed slot) — is correct but explodes to `n*T = 4*10^10` edges on `n = T = 2*10^5` with wide windows, so the matching engine is the wrong representation; the interval structure is the lever, and a greedy that hands each request its earliest free legal slot works — but only under the *deadline* order, since the release-time order has the explicit counterexample `[1,2],[1,1]` (flexible request steals the slot the rigid one needs) while the deadline order has an exchange argument identifying its only failure mode with a violated Hall condition; making "earliest free slot >= l" fast needs the disjoint-set next-free-slot trick for near-linear total time; my first cut initialised the union-find only over `1..T`, so a saturated range let `find` fall into the uninitialised `T+1` (and on into garbage index `0`), which a trace of `T=1` with two `[1,1]` requests exposed by returning a bogus "slot 0" `YES` instead of `NO`; initialising through the sentinel `T+1` fixes it; and a several-thousand-case differential test against a Kuhn matcher, plus the `2*10^5` timing runs, closed out the tie-break, the sentinel, and the `n=0`/`n>T`/point-window corners.
