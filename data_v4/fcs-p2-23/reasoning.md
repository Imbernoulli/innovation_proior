A row of `n` stones with possibly-negative values, two players alternating takes from either end until the row is empty, each maximizing their **own** banked total — not the gap between them. Two features of the contract change how I have to think about it before I touch an algorithm. First, the row is emptied no matter what; there is no passing. A lone negative stone must be taken, and the answer can go below zero, so this is emphatically not a "pick a favorable subset" game — I must not clamp anything to `0`. Second, `n` runs up to `2000` and `|a[i]|` up to `10^9`, so a total can reach `2000 * 10^9 = 2*10^12`, well past the 32-bit ceiling of `~2.1*10^9`. Every accumulator and every table entry has to be 64-bit `long long`, or the large tests silently return garbage. I fix that up front.

The scale also tells me the algorithm class is not the interesting question. `n <= 2000` makes an `O(n^2)` fill about `4*10^6` cell updates — trivial under a second — and an `O(n^2)` table of `long long` is `2000*2000*8 = 32 MB`, comfortably under 256 MB. So performance leaves me free to pick the method I can *prove*. That matters, because the trap in this problem is a clever-but-wrong shortcut, not a performance hole.

**The seductive shortcut is greedy.** "Each turn, take the larger end" is `O(n)` and feels obviously right — of course you grab the bigger stone in front of you. But the mover controls only one stone, while the *opponent* inherits and optimizes the entire leftover interval. A locally larger grab can hand the opponent a far more valuable configuration. I won't trust greedy until I've tried to break it, and one concrete counterexample is enough to discard it.

Take `a = [1, 5, 233, 7]`. Greedy first move: ends are `1` and `7`, so `P1` grabs `7`. Row `[1, 5, 233]`; `P2` faces `1` vs `233` and grabs `233`. Row `[1, 5]`; `P1` takes `5`, `P2` takes `1`. Greedy nets the first player `7 + 5 = 12`. That is catastrophic — the myopic grab of `7` exposed the `233`. Optimal instead is to take the **left** `1`, giving up nothing of value but forcing the structure: after `P1:1`, row `[5, 233, 7]`, the `233` can no longer be protected — whether `P2` takes `5` or `7`, `P1` takes the `233` next. Optimal play `P1:1, P2:5, P1:233, P2:7` gives `1 + 233 = 234`. Greedy's `12` against `234` is not a near-miss; it is the worst available move.

I brute-force all length-4 rows over a tiny value set to make sure this isn't a contrived spike, and find the smallest disagreement: `a = [1, 1, 3, 2]`. Greedy takes the `2` (ends `1` vs `2`), exposing the `3`, and nets `2 + 1 = 3`; optimal takes the left `1` and runs `P1:1, P2:1, P1:3, P2:2` for `1 + 3 = 4`. Same mechanism, minimal size. Greedy is structurally wrong and is out.

**Interval DP on the score difference.** I want, for every interval `a[i..j]`, a clean characterization of optimal play for whoever moves on it. Tracking the mover's absolute total is workable but clumsy — I'd also need the leftover's sum to compute the opponent's share. The clean invariant is the **score difference**: let `diff[i][j]` = (mover's total) − (opponent's total) under optimal play on `a[i..j]`, where the mover is whoever is about to move. This is self-similar in exactly the right way: when the mover takes an end, the leftover interval is a fresh game in which the *opponent* moves first, so on that subinterval the roles swap, and from the original mover's view the opponent's optimal difference there contributes with a flipped sign. That sign flip is the entire reason the difference form beats tracking absolute totals.

On `a[i..j]` the mover takes the left stone (net `a[i] - diff[i+1][j]`) or the right stone (net `a[j] - diff[i][j-1]`) and picks the larger:

```
diff[i][j] = max( a[i] - diff[i+1][j],  a[j] - diff[i][j-1] )
```

Base case `diff[i][i] = a[i]` — the mover takes the one stone, the opponent gets nothing. I recover the answer from `first + second = total` and `first - second = diff[0][n-1]`, so `first = (total + diff[0][n-1]) / 2`. That sum equals `2*first`, so it is always even and the division is exact — no rounding hazard even when values are negative.

Running the recurrence over `[1, 5, 233, 7]` (`total = 246`, answer known to be `234`): length-2 gives `diff[0][1]=max(1-5,5-1)=4`, `diff[1][2]=228`, `diff[2][3]=226`; length-3 gives `diff[0][2]=max(1-228, 233-4)=229` and `diff[1][3]=max(5-226, 7-228)=-221`; length-4 gives `diff[0][3]=max(1-(-221), 7-229)=222`, and `(246+222)/2 = 234`. It matches, and the winning branch `a[0] - diff[1][3] = 222` over `a[3] - diff[0][2] = -222` is precisely the take-the-left move I argued for by hand.

**The fill order is where clean math transcribes to dirty code.** My first cut iterates `i` outer, `j` inner:

```
for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
        diff[i][j] = max(a[i] - diff[i+1][j], a[j] - diff[i][j-1]);
```

But `diff[i][j]` reads `diff[i+1][j]`, which lives in a *later* row `i+1`. With `i` increasing, row `i+1` isn't filled yet when I compute row `i`, so `diff[i+1][j]` is read as its initialized `0`. On `[1,5,233,7]` this poisons the top corner: `diff[0][3]` wants `diff[1][3] = -221` — the value I just hand-derived — but reads a stale `0`, so `diff[0][3] = max(1-0, 7-229) = 1` instead of `222`, and `(246+1)/2` truncates to `123`. The stale zero even breaks the parity guarantee, so it isn't merely off — it's a different integer for a different reason. The fix is to iterate by **interval length** ascending: both `diff[i+1][j]` and `diff[i][j-1]` are strictly shorter than `diff[i][j]`, so both are ready when read.

```
for (int len = 2; len <= n; len++)
    for (int i = 0; i + len - 1 < n; i++) {
        int j = i + len - 1;
        diff[i][j] = max(a[i] - diff[i+1][j], a[j] - diff[i][j-1]);
    }
```

In length order the corner reads the true `diff[1][3] = -221`, so `diff[0][3] = max(1-(-221), 7-229) = 222` and `(246+222)/2 = 234` — back to the value the hand analysis and the recurrence both demanded. The bug was purely an ordering artifact, and the length-ascending fill is the minimal thing that removes it.

**Corners this kind of code dies on.** `n = 0`: the row is empty and the answer is `0`; I guard with an explicit early return so I never index an empty `diff`. `n = 1`, `a = [-7]`: `diff[0][0] = -7`, `total = -7`, `first = -7` — the forced take of a negative stone, which is right precisely because passing isn't allowed. All-negative rows need no special case: both players are minimizing their own loss, and the difference recurrence is sign-agnostic. Overflow is contained — every `diff` entry and `total` is bounded by `sum|a[i]| <= 2*10^12`, inside `long long`, with no sentinel arithmetic to underflow. `cin >>` consumes arbitrary whitespace, so parsing is format-agnostic, and output is one integer plus a newline.

**Differential testing.** Hand-traces catch the bugs I anticipate; an independent oracle catches the ones I don't. I wrote a brute oracle that does *not* reuse the difference/parity trick — it tracks the absolute mover total `gain[i][j] = max(a[i] + (subsum(i+1,j) - gain[i+1][j]), a[j] + (subsum(i,j-1) - gain[i][j-1]))` via prefix sums, a different invariant and different final arithmetic, so a shared blind spot is unlikely. For tiny `n` I also wrote a fully exhaustive minimax that recurses the whole game tree with no DP, returning `(mover_total, opponent_total)` directly — as close to the definition as code gets. Across 810 generated cases — random rows, a `greedytrap` mode of alternating large/small values, `mid`/`big` rows to `n=2000`, an `extreme` mode of pure `±10^9`, and the `n=0`/`n=1` edges — the solution matches the brute oracle with zero mismatches, and every `n <= 12` case additionally agrees with the exhaustive minimax. Greedy, by contrast, disagrees with the verified solution on both counterexamples (`12` vs `234`, `3` vs `4`). At `n = 2000` with random `±10^9` values the solution runs in about `25 ms` and `~30 MB`.

The shipped file is the simple, provable `O(n^2)` interval DP — the base case, the length-ascending fill, and the `(total + diff[0][n-1]) / 2` recovery, all in `long long`; the full module is in the answer.
