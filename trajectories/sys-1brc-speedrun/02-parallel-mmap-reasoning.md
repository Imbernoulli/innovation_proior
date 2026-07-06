The baseline came back at 04:49.679, and the profile is exactly what the code predicted: one core busy,
seven idle, and a torrent of short-lived allocation underneath. Before I change anything I want to read that
number for what it mechanically *is*, because the decomposition tells me where the multipliers hide.
04:49.679 is 289.679 seconds, and the job is one billion rows on a single core, so I am spending about
290 nanoseconds per row. On a 2.5 GHz core that is on the order of 700 clock cycles per row. The actual
work a row demands — one min compare, one max compare, one add, one increment — is a handful of cycles. So
roughly 99% of those 700 cycles are overhead: allocating and later collecting four objects, UTF-8-decoding
the line, running a regex-style `split`, running a general float parser, and hashing a `String` into a
`HashMap`. That is not a job that is slow because the problem is hard; it is a job that is slow because every
row is dragging four objects and several redundant passes over its own bytes behind it. Two large
multipliers are therefore sitting untouched, and because they attack independent costs they should compound
rather than merely add, so I take both at once.

The first multiplier is the seven idle cores. The evaluation machine pins me to eight, and I am using one,
so there is an up-to-8× ceiling that the associative combiner I wrote into the baseline was already designed
to unlock. The second is the per-row overhead itself, which is a large constant factor independent of the
core count: kill the allocation, the decode, the split, and the float parse, and each row's 700 cycles
should collapse toward the handful the arithmetic actually needs. Threading sets the shape of everything
else, so I design it first and fit the overhead-removal inside it.

To use eight cores I give each one an independent slice of the file, let it build its own
`(min, max, sum, count)` aggregate per station, and merge the slices at the end. This is sound precisely
because that quadruple merges associatively — the property I checked numerically in the baseline — so
whichever slice sees which rows, the final elementwise merge lands on the same answer as a single fold. I
split into `numberOfSegments = availableProcessors()` segments, which on this machine is eight, one per
core. The subtlety is that I cannot cut on raw byte offsets: the nominal cut `i * segmentSize` lands in the
middle of some line, so one segment would see a truncated `station;temp` while its neighbour sees the other
half, and both partial rows would be garbage. The fix is to make each boundary *newline-aligned*: from the
nominal offset I scan forward to the next `'\n'` and take the byte after it as the true start; the previous
segment's true end is this segment's true start, so there is exactly no gap and no overlap. Let me sanity
this against the numbers. The file is ~12 GB; over eight segments the nominal `segmentSize = fileSize / 8`
is ~1.5 GB each. A station name is at most 100 bytes and a whole row at most ~110 bytes, so scanning from a
nominal offset to the next newline reads at most ~110 bytes before it finds one. Eight boundaries times
~110 bytes is under a kilobyte of scanning to align the entire 12 GB split — utterly negligible against the
1.5 GB each thread then processes. And there is a guard for the degenerate case: below a 1 MB segment size
the split is pointless overhead, so I fall back to a single segment covering the whole file. A `parallel()`
stream over the segment list fans them across the fork-join pool; with only eight coarse tasks the
scheduling overhead is a rounding error, which is why I do not reach for a hand-managed `ExecutorService` —
the granularity is already right.

Let me trace the alignment on a miniature to be sure the boundary logic leaves no line double-counted or
dropped. Imagine a tiny file `A;1.0\nB;2.0\nC;3.0\n` with two segments and a nominal cut landing inside the
second line, say at the `2` of `B;2.0`. Segment 0's true end is found by scanning forward from that offset
to the next `'\n'`, which is the newline after `B;2.0`; segment 1's true start is the byte after that same
newline, i.e. the `C`. So `B;2.0` belongs wholly to segment 0 and `C;3.0` wholly to segment 1 — the shared
newline is the seam, counted by neither side twice nor skipped. Because segment 1's start is *defined* as
segment 0's end, the invariant holds for any nominal cut. Two more small correctness details fall out of the
byte loop: the scratch buffer is `byte[100]` because a station name is at most 100 bytes, so the name always
fits with no bounds worry; and after parsing the value I check for a `'\r'` before the `'\n'` and skip it if
present, so a file written with CRLF line endings parses identically to one with bare LF. The inner loop also
guards `currentPosition != segmentEnd` while scanning the name, so a thread never reads past its own segment
into a neighbour's bytes.

The bigger structural change lives inside each segment: stop decoding, and work on raw bytes. The baseline
read *decoded lines*, paying a full 12 GB UTF-8 decode to recover ASCII I could have read directly.
`FileChannel.map(READ_ONLY, start, len)` maps a segment straight into memory as a `MappedByteBuffer`, so I
read bytes out of the page cache with no `read()` syscall per buffer, no intermediate copy, and no per-line
`String`. I considered the alternative of a big `FileChannel.read` into a reused `ByteBuffer`, but that
still copies kernel page-cache bytes into my buffer and still costs a syscall per refill; `mmap` lets the
page cache *be* my buffer, and with 128 GB of RAM against a 12 GB file the whole thing stays resident after
first touch. So the inner loop becomes a pointer marching through mapped memory: from the current position,
walk byte by byte accumulating the station name into a scratch `byte[100]` until I hit `';'`, then parse the
temperature, then step past the newline. No `split`, no intermediate objects — just an index advancing
through a mapped buffer.

Three pieces of that loop carry the constant-factor win, and each is worth deriving rather than asserting.

First, the number parse. `Double.parseDouble` is a general float parser — exponents, arbitrary mantissa
length, special values — but the data is only ever one of four shapes: `D.D`, `DD.D`, `-D.D`, `-DD.D`. One
optional minus, one or two integer digits, a dot, exactly one fractional digit. I never need a `double`
during parsing at all; I read the value as a *scaled integer*, its true value times ten, which is exact
because there is exactly one decimal place. So I peek at the bytes directly. If the first byte is `'-'` I set
`negative = -1` and step past it. Then I look at the byte two positions ahead of the first digit: if it is
`'.'`, the integer part is a single digit and the value is `d0*10 + d1`; otherwise the integer part is two
digits and the value is `d0*100 + d1*10 + d2`, where each digit is just `byte - '0'`. Let me trace both
shapes to be sure the index arithmetic is right. For `"5.7\n"` the bytes are `'5' '.' '7' '\n'`; there is no
minus, the byte at position 1 is `'.'`, so the one-digit branch fires: `(5)*10 + (7) = 57`, and I advance
three bytes to sit on the `'\n'`. For `"-12.3\n"` the bytes are `'-' '1' '2' '.' '3' '\n'`; the minus sets
`negative = -1` and steps to `'1'`; the byte one past `'1'` is `'2'`, not `'.'`, so the two-digit branch
fires: `-1 * ((1)*100 + (2)*10 + (3)) = -123`, and I advance four bytes to the `'\n'`. Both land the scaled
integer exactly — `57` and `-123`, i.e. `5.7` and `-12.3` times ten — with a handful of integer adds and
multiplies, no function call, and no floating point. I divide by ten only when I finally store `temp / 10.0`.
This is precisely the kind of unroll the leaderboard shares; the idea of peeking at the `'.'` position to
branch between the one- and two-integer-digit cases I am borrowing from another entry's trick, but it falls
straight out of the fixed data shape.

I should be honest about how far I take the integer trick in *this* rung, because it is only half a step.
The parse produces a scaled integer, but I immediately convert it back with `temp / 10.0` and store a
`double` in the map, and the aggregate's `sum` accumulates as a `double`. So I have removed the general
float *parser* — the expensive part — but I am still summing doubles, which means the floating-point drift I
flagged in the baseline (a billion tenths added into a running sum climbing toward ~10^11, low bits falling
off the mantissa) is still present, and the exact-integer *aggregate* is still an unspent lever. I keep the
double aggregate here because it is the smaller of the two wins and because the map plumbing reads more
simply with a single numeric type; carrying the sum as a `long` of scaled integers is a clean later
tightening that removes the drift entirely and shaves the divide. I note it and move on.

Second, and this is where the real allocation savings live, the per-station lookup. The baseline hashes a
`String` key into a `HashMap` per row; I never want to materialize a `String` per row at all, and I noted in
the baseline that the name bytes were being walked four-plus times per row. Here the station name is just a
range of bytes in my scratch buffer, so I look it up *as bytes* in a small custom map specialized to this
task: a flat open-addressing table. I accumulate the key's hash *in the same pass* that scans it to the
delimiter — `hash = 31*hash + b`, the same polynomial `String.hashCode` uses, computed for free during the
walk I am already doing — so the "hash pass" and the "find `;` pass" become one pass. Then `hash &
(MAPSIZE - 1)` picks a slot in a power-of-two array, and on a hit I update that slot's `(min, max, sum,
count)` in place. On a miss — a `null` slot, or a slot whose stored key bytes do not match — I linear-probe
forward, `slot = (slot + 1) & (MAPSIZE - 1)`, until I find the match or an empty slot, and only then create
the `Result` and copy the name bytes in *once*. Key comparison is `Arrays.equals` over the byte ranges,
comparing bytes directly with no decode.

The sizing of that table is not arbitrary, so let me do the load-factor arithmetic that justifies open
addressing over chaining. `MAPSIZE = 1 << 17 = 131072` slots. The primary dataset has 413 distinct
stations, so the load factor is `413 / 131072 ≈ 0.3%`. Even the stress dataset's ceiling of 10,000 distinct
stations gives `10000 / 131072 ≈ 7.6%`. The expected number of probes for a successful lookup under linear
probing at load factor α is about `(1 + 1/(1 - α)) / 2`; at α = 0.076 that is `(1 + 1.082) / 2 ≈ 1.04`
probes, and at α = 0.003 it is essentially `1.0`. So in the common case every lookup is a single slot touch
with no probing at all — the table behaves like near-perfect hashing on this key set. That tiny load factor
is exactly why open addressing beats a chained `HashMap` here: there are no per-node list objects to
allocate and chase, everything lives in two flat arrays (`Result[] slots` and `byte[][] keys`) that stay in
cache, and with the table two orders of magnitude larger than the key set, probe chains essentially never
form. A trie or radix structure would also key on bytes but would allocate interior nodes and chase pointers
for a key set this small — more machinery than 413-to-10,000 short keys warrant. Flat open addressing is the
right shape.

Let me trace `putOrMerge` through a collision to confirm the probe logic is correct, not just fast. Suppose
two station names hash into the same starting slot `s`. The first name arrives, finds `slots[s] == null`,
so it creates a `Result`, copies its bytes into `keys[s]`, and stops. The second name arrives, computes the
same `s`, finds `slots[s] != null`, and checks whether the stored key matches: `keys[s].length != size ||
!Arrays.equals(keys[s], 0, size, key, offset, size)`. Since the names differ, the comparison fails, so it
linear-probes to `s+1`, finds it `null`, and lands there. Now the first name arrives *again* on a later row:
same `s`, `slots[s]` non-null, key matches, so it updates `min/max/sum/count` in place — no probing past
`s`. And the second name on a later row: `s` occupied by the first (key mismatch), probe to `s+1`, key
matches, update in place. Every distinct name settles into a stable slot and every repeat lands on it, so
the aggregate for each station is exactly the fold of its own rows. The correctness does not depend on the
hash being collision-free — it depends only on the probe eventually finding either the matching key or an
empty slot, which it always does while the table has free slots, and at 0.3–7.6% occupancy it always has
many.

There is one cost of `mmap` I should name because it caps the threading multiplier: page faults on first
touch. The mapping is lazy, so the first time any thread reads a given page the kernel takes a minor fault
to wire it into the page cache. At 4 KB pages, 12 GB is on the order of three million pages and therefore
three million first-touch faults spread across the run; readahead and larger pages fold many of those
together, but the warmup is real and it is front-loaded, so early in the run the eight threads are partly
waiting on I/O rather than computing. That, together with the imperfect lockstep of eight fixed segments, is
why I expect somewhat less than a clean 8× from the cores — the effective threading multiplier is bounded by
how quickly the file's pages become resident, not just by the core count.

The payoff of the byte-keyed map is the number that dominated the baseline: a `String` is now created
exactly once per *distinct* station — a few hundred of them — at merge time, instead of once per *row*. That
turns roughly four billion per-row allocations into a few hundred, which is the single move that takes the
garbage collector from running the whole job to doing essentially nothing. Each segment-thread gets its own
`ByteArrayToResultMap`, so there is zero contention in the hot loop — no locks, no shared state, no
cache-line ping-pong. Only at the very end do I drain each thread's map into a list of `(byte[] key,
Result)` entries, decode each key to a `String` exactly once, and `Collectors.toMap` them into a `TreeMap`
with a merge function that combines any station that appears in more than one thread's map. The `TreeMap`
keeps the output alphabetized, exactly as the baseline did, so the final format is unchanged.

The merge is worth sizing so I can confirm it is not secretly serial-dominant. Each of the eight threads
finishes holding a map with at most a few hundred (up to ten thousand) live entries; draining all eight
produces on the order of a few thousand `(byte[], Result)` pairs, and `Collectors.toMap` walks them once,
decoding each distinct key to a `String` exactly once and applying the merge function when two threads saw
the same station. That is a few thousand hash inserts and a few hundred `String` decodes — microseconds to
low-milliseconds of work against a multi-second parse. So the merge does not threaten the wall-clock; the
serial tail here is genuinely negligible, which is what lets the eight-way parallel parse translate almost
directly into wall-clock speedup rather than being clawed back at the join. The only place the merge matters
is correctness: the merge function must combine colliding stations with the same associative elementwise
rule as the in-thread accumulator, and it does — `min` of mins, `max` of maxes, sums added, counts added —
so a station split across several segments reassembles to exactly the aggregate a single-threaded pass would
have produced.

So the whole structure is: split the file at newline-aligned boundaries into one segment per core;
`parallel()` over the segments; each thread memory-maps its segment and runs a byte-pointer loop that scans
the name to `';'` while folding the hash, parses the temperature as a scaled integer, and updates its own
open-addressed byte-keyed map in place; then merge the per-thread maps into a sorted `TreeMap` and print.
Every layer the baseline paid for per row — the decoded `String`, the `split` array, the boxed value, the
`String` hash, the four-plus redundant passes over the name, the single core — is gone or amortized to
once-per-distinct-station.

Now let me set the bar honestly by projecting the two multipliers rather than guessing. The baseline is
289.679 seconds. The threading multiplier is at most 8× and realistically somewhat less — the final merge is
serial (a few hundred keys, so tiny), the mmap warms via page faults on first touch, and the eight segments
will not finish in perfect lockstep, so call it 6–7× effective. The constant-factor multiplier from removing
per-row overhead is harder to pin but large: eliminating the ~140 GB of allocation churn and its GC, the
12 GB UTF-8 decode, the regex split, the float parse, and the redundant name passes should be well into the
single-digit-times range on its own, plausibly 6–10×. Those two compound: even a conservative 6× × 6× ≈ 36×
puts me near 8 seconds, and a more optimistic 7× × 9× ≈ 63× puts me near 4.5 seconds. So the honest target
is not "a bit under five minutes" — it is *single-digit seconds*, roughly two orders of magnitude below the
baseline. The falsifiable version of that claim, in the wall-clock metric, is that if the diagnosis is right
that the baseline was almost entirely idle cores plus per-row overhead, this rung lands in the low-single-digit
to high-single-digit seconds; if it lands far above that, then some cost I have not named — page-fault I/O
as the mmap warms, or worse-than-expected thread imbalance — is dominating and becomes the next thing to
attack. The hedges I hold going in: I will not get the full 8× from threads, and the custom map's
constant-factor win only materializes if the probe chains actually stay as short as the load-factor
arithmetic predicts, which they should at 0.3–7.6% occupancy. The full module is in the answer.
