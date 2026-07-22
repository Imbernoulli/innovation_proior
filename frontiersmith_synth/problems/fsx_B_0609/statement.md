# Gap-Seeded Shelf

A librarian keeps books on one long shelf of **C** slots, sorted by call number. Some
slots hold a book (a **key**); the rest are deliberate **gaps**. A big donation is coming
and its whole schedule is already known: exactly which books will be shelved and which
range look-ups readers will run, in order. You place the gaps **once, up front**; then the
schedule replays and you pay for the disruption.

## Input (stdin)
```
N M Q C VMAX
k_1 k_2 ... k_N            (the N initial keys, strictly increasing, in [0,VMAX))
```
followed by `Q` operations, one per line, in replay order:
```
I v        insert key v          (v distinct from every key ever present)
S l r      range scan over [l,r]  (0 <= l <= r < VMAX)
```
`M` of the operations are inserts. `C = 2*(N+M)` is the shelf length.

## Output (stdout)
```
N R
p_1 p_2 ... p_N            (physical slot of each initial key, in value order)
b_1 ... b_R               (optional rebuild op-indices; omit the line if R = 0)
```
`p_1 < p_2 < ... < p_N`, each in `[0, C)` — key *i* (the *i*-th smallest value) sits in
slot `p_i`; every other slot is a gap. The `R` rebuild indices are strictly increasing in
`[0, Q)`.

## Replay & cost (minimize)
The shelf always keeps keys in value order left-to-right. Processing operation *j*:

- If *j* is a rebuild index, first **re-spread** all current keys evenly across the C
  slots (`slot(i) = floor(i*C/K)`). Cost **+C**.
- **`I v`** — locate v's value-neighbours. If a gap already sits between them, v drops into
  it for **free**. Otherwise the block of keys between v's landing slot and the **nearest**
  gap is shifted one slot toward that gap; cost **+ (number of keys moved)**.
- **`S l r`** — let the matched keys span physical slots `a..b` (first key `>= l`, last key
  `<= r`). Cost **+ (b - a + 1)** — every slot touched, **gaps included**. Empty match costs 0.

Your score is the total replayed cost `F`. Lower is better.

## Scoring
The checker replays your layout to get `F`, and replays a naive **packed-at-the-front**
layout (keys in slots `0..N-1`, no rebuilds) to get baseline `B`. Then
```
Ratio = min(1000, 100 * B / F) / 1000
```
so the front-packed baseline scores ~0.10 and a 10x-better layout caps at 1.0.

## What makes it hard
Two forces pull opposite ways. **Inserts** want slack sitting exactly where the next key
lands — otherwise a long block shifts. **Scans** want density — every gap inside a scanned
range is dead cost. Uniform gaps (the classic packed-memory hedge) give cheap inserts
everywhere but tax every scan and squander slack on cold regions no insert visits. Because
the entire future is visible, gap placement is really a **transport** problem: move capacity
to the intervals that will actually receive inserts, and keep the scan-heavy stretches packed
solid. The inserts cluster around a few drifting hotspots; the scans mostly sweep the cold
majority.

## Constraints
`N, M, S` up to a few thousand; `VMAX = 10^7`; each input < 5 MB. Deterministic scoring;
time limit 5 s.

## Example (illustrative, not to scale)
`N=3, C=10`, keys `10 20 30`, ops: `S 0 40`, `I 25`, `S 20 40`.

Front-packed `p = 0 1 2`: scan[0,40] touches slots 0..2 (cost 3); insert 25 between slots
1,2 (adjacent) shifts one key to reach the gap at slot 3 (cost 1); scan[20,40] now spans
slots 1..3 (cost 3). Total **7**.

Reservoir `p = 0 1 5` (gap parked between 20 and 30, where 25 will land): scan[0,40] spans
slots 0..5 (cost 6); insert 25 drops into a gap between slots 1 and 5 for free (cost 0);
scan[20,40] spans the 20/25/30 slots (cost `≈4`). A layout tuned to the real trace — dense
where scans sweep, slack only where inserts land — beats both.
