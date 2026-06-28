The problem is online one-dimensional bin packing: a stream of items arrives one at a time, each with
a size, and each must be placed immediately and irrevocably — before the next item is seen — into an
open bin of fixed capacity $C$ that still has room, or into a brand-new bin. Once an item is placed it
never moves. I want to finish the stream having used as few bins as possible. What makes this hard is
the blindness: offline, with the whole multiset in hand, I could sort and pack cleverly, but here
every placement is a bet against items I have not yet seen. The interesting decision is therefore not
*whether* to open bins but *which* already-open bin to drop the current item into, because every
forced new bin is an admission that the open bins could not absorb this item — and the leftover room
in open bins is the only thing standing between me and the trivial one-bin-per-item policy. Since the
true online optimum is order-dependent and intractable to pin down, I measure against the L1 lower
bound $\mathrm{LB} = \lceil \sum_i s_i / C \rceil$, the count I would need if every bin were filled to
the brim with zero waste, and report performance as percent excess over it, $100\,(\text{mean
bins} - \mathrm{LB})/\mathrm{LB}$.

Before reaching for anything clever I want the most conservative, most obviously-correct rule I can
state — one whose behaviour I can fully predict and which gives a floor every later method must beat.
I propose First-Fit: keep the open bins in a fixed order and place each item into the *first* bin in
that order that still has room for it, opening a new bin only when none do. Its whole appeal is that
it never does anything clever and never does anything stupid — it greedily reuses capacity in a
deterministic sweep, so it can never do worse than one bin per item, and it reuses a bin whenever
reuse is possible at all. That is precisely why it is the right baseline: any heuristic worth
proposing must beat the policy that simply reuses the earliest bin it can.

I want to be honest about why First-Fit will not be very good, because that diagnosis is what later
methods have to attack. The weakness is the word "first." When several open bins could all fit the
item, First-Fit takes the earliest one purely by accident of bin-creation order, with no regard for
how much room it leaves behind. Picture a bin with $90$ units of slack and a bin with $12$, and an
incoming item of size $10$: if the roomy bin comes earlier in the order, First-Fit drops the $10$
there, leaving $80$, and has just spent a big, valuable pocket of capacity on a small item that the
tight bin could have swallowed almost perfectly. Tight bins are exactly the ones I most want to
finish off — a bin at $99/100$ is essentially retired, while a bin left at $80$ is a half-used
resource I must hope future items fit. First-Fit systematically spreads items across the early bins
and leaves a long tail of partially-filled bins that never get topped off, and that wasted capacity
becomes extra bins.

The implementation has to fit the harness, which is slightly awkward because the harness does not hand
me the open bins in creation order with their fill levels. For the current item it hands me only the
sub-array of remaining capacities of the bins that *can* fit the item — the valid bins — and asks me
to score them; it then places the item in the highest-scoring bin, breaking ties toward the lowest
index. Crucially the bins array is in a stable positional order (index $0$ is the first bin ever
opened, and so on), so "first bin that fits" is exactly "the valid bin with the smallest index." I
realize First-Fit by handing back a score that strictly decreases with the bin's position in the
valid sub-array, so the earliest valid bin is the unambiguous maximum — the intent is unmistakable and
does not lean on the tie-break convention. The new-bin behaviour then takes care of itself: the
harness pre-fills a large bins array all sitting at full capacity, so if no already-used bin fits the
item, the only valid bins left are still-full ones, and First-Fit takes the earliest of those, which
*is* opening a new bin. There is nothing special to code — the rule is genuinely just "earliest valid
bin," and the same one line both reuses and opens.

As a single-file program the policy is even more direct than the priority-function form: I keep the
open bins' remaining capacities in creation order, and for each item I sweep left to right and drop it
into the first bin that still fits, opening a fresh bin only when the sweep finds none. The program
reads the instance from stdin — capacity `C`, the item count `n`, then the `n` item sizes — and prints
the number of bins First-Fit uses followed by the L1 lower bound `ceil(Σ items / C)` for reference.
Capacities and the running total are `long long` so a long stream of large sizes cannot overflow.

```cpp
// Online 1-D bin packing, First-Fit policy.
// Reads from stdin: capacity C, item count n, then n item sizes.
// Prints the number of bins used (and the L1 lower bound) to stdout.
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long C;
    int n;
    if (!(cin >> C >> n)) return 0;

    vector<long long> items(n);
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        cin >> items[i];
        total += items[i];
    }

    // remaining[b] = leftover capacity of bin b, in bin-creation order.
    // First-Fit: place each item in the earliest (lowest-index) bin that still
    // fits it; if none fit, open a new bin at the end. Equivalent to the
    // priority rule "score strictly decreasing in bin index, take the argmax".
    vector<long long> remaining;
    remaining.reserve(n);
    for (int i = 0; i < n; ++i) {
        long long item = items[i];
        int chosen = -1;
        for (int b = 0; b < (int)remaining.size(); ++b) {
            if (remaining[b] >= item) { chosen = b; break; }  // earliest valid bin
        }
        if (chosen == -1) {                 // no open bin fits -> open a fresh bin
            remaining.push_back(C - item);
        } else {
            remaining[chosen] -= item;
        }
    }

    long long used = (long long)remaining.size();
    long long lb = (total + C - 1) / C;     // L1 lower bound ceil(sum/C)

    cout << used << "\n";
    cout << lb << "\n";
    return 0;
}
```
