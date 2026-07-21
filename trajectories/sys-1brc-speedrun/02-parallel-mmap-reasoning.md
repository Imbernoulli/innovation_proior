The baseline came back at 04:49.679, and the profile is exactly what the code predicted: one core busy,
seven idle, and a torrent of short-lived allocation underneath. Read that number for what it mechanically
*is*. 04:49.679 is 289.679 seconds for a billion rows on one core, about 290 nanoseconds per row, or ~700
clock cycles per row on a 2.5 GHz core. The work a row demands — one min compare, one max compare, one add,
one increment — is a handful of cycles. So roughly 99% of those 700 cycles are overhead: allocating and
later collecting four objects, UTF-8-decoding the line, running a regex-style `split`, running a general
float parser, hashing a `String` into a `HashMap`. Not a job slow because the problem is hard; a job slow
because every row drags four objects and several redundant passes over its own bytes behind it. Two large
multipliers sit untouched — the seven idle cores (an up-to-8× ceiling the associative combiner was built to
unlock) and the per-row overhead itself (a large constant factor independent of core count) — and because
they attack independent costs they should compound rather than merely add, so I take both at once. Threading
sets the shape of everything else, so I design it first and fit the overhead-removal inside it.

To use eight cores I give each an independent slice of the file, let it build its own `(min, max, sum,
count)` aggregate per station, and merge the slices at the end — sound precisely because that quadruple
merges associatively, so whichever slice sees which rows, the final elementwise merge lands on the same
answer as a single fold. I split into `numberOfSegments = availableProcessors()` segments, eight on this
machine. The subtlety is that I cannot cut on raw byte offsets: the nominal cut `i * segmentSize` lands in
the middle of some line, so one segment would see a truncated `station;temp` while its neighbour sees the
other half. The fix is to make each boundary *newline-aligned*: from the nominal offset I scan forward to the
next `'\n'` and take the byte after it as the true start, and the previous segment's true end is this
segment's true start — exactly no gap and no overlap, because segment `i+1`'s start is *defined* as segment
`i`'s end. That alignment scan is negligible: a whole row is at most ~110 bytes, so eight boundaries cost
under a kilobyte of scanning against 1.5 GB per thread. Below a 1 MB segment size the split is pointless
overhead, so I fall back to a single segment. A `parallel()` stream over the segment list fans the eight
coarse tasks across the fork-join pool with rounding-error scheduling overhead, so I do not reach for a
hand-managed `ExecutorService` — the granularity is already right. Two small correctness details fall out of
the byte loop: the scratch buffer is `byte[100]` because a name is at most 100 bytes; after the value I skip
a `'\r'` before the `'\n'` so CRLF files parse identically; and the inner scan guards `currentPosition !=
segmentEnd` so a thread never reads past its own segment.

The bigger structural change lives inside each segment: stop decoding, work on raw bytes. The baseline paid a
full 12 GB UTF-8 decode to recover ASCII it could have read directly. `FileChannel.map(READ_ONLY, start,
len)` maps a segment into memory as a `MappedByteBuffer`, so I read bytes out of the page cache with no
`read()` syscall and no intermediate copy. A big `FileChannel.read` into a reused `ByteBuffer` would still
copy page-cache bytes into my buffer and still cost a syscall per refill; `mmap` lets the page cache *be* my
buffer, and with 128 GB of RAM against a 12 GB file the whole thing stays resident after first touch. So the
inner loop becomes a pointer marching through mapped memory: from the current position, walk byte by byte
accumulating the name into a scratch `byte[100]` until I hit `';'`, then parse the temperature, then step
past the newline. No `split`, no intermediate objects.

Three pieces of that loop carry the constant-factor win. First, the number parse. `Double.parseDouble` is a
general float parser, but the data is only ever `D.D`, `DD.D`, `-D.D`, `-DD.D`: one optional minus, one or
two integer digits, a dot, exactly one fractional digit. I never need a `double` during parsing — I read the
value as a *scaled integer*, its true value times ten, exact because there is exactly one decimal place. If
the first byte is `'-'` I set `negative = -1` and step past it; then the byte two past the first digit tells
me the shape — if it is `'.'` the integer part is one digit and the value is `d0*10 + d1`, otherwise two
digits and `d0*100 + d1*10 + d2`, each digit just `byte - '0'`. So `"5.7"` gives `5*10 + 7 = 57` and
`"-12.3"` gives `-(1*100 + 2*10 + 3) = -123` — the scaled integer exactly, a handful of integer ops, no
function call, no float. I divide by ten only when I store `temp / 10.0`. I am honest that this is only half
a step: I convert straight back to a `double` and the aggregate's `sum` still accumulates as a `double`, so
I have removed the general float *parser* — the expensive part — but the floating-point drift from summing a
billion tenths is still present. Carrying the sum as a `long` of scaled integers is a clean later tightening;
I keep the double aggregate here because the map plumbing reads simpler with one numeric type.

Second, and where the real allocation savings live, the per-station lookup. The baseline hashed a `String`
key per row and walked the name bytes four-plus times; here the station name is just a range of bytes in my
scratch buffer, so I look it up *as bytes* in a flat open-addressing table. I accumulate the key's hash *in
the same pass* that scans it to the delimiter — `hash = 31*hash + b`, the same polynomial `String.hashCode`
uses, computed for free during the walk I am already doing — so the "hash pass" and the "find `;` pass"
become one pass. Then `hash & (MAPSIZE - 1)` picks a slot in a power-of-two array; on a hit I update its
`(min, max, sum, count)` in place, and on a miss I linear-probe forward until I find the match or an empty
slot, creating the `Result` and copying the name bytes in *once*. Key comparison is `Arrays.equals` over the
byte ranges, no decode. The sizing justifies open addressing over chaining: `MAPSIZE = 1 << 17 = 131072`
slots against 413 distinct stations is a 0.3% load factor, and even the 10,000-station stress ceiling is
7.6%. The expected successful-lookup probe count under linear probing is about `(1 + 1/(1-α))/2`, which at
α = 0.076 is ~1.04 and at α = 0.003 is essentially 1.0 — every lookup is a single slot touch, the table
behaves like near-perfect hashing on this key set. That tiny load factor is exactly why open addressing
beats a chained `HashMap`: no per-node list objects to allocate and chase, everything in two flat arrays
that stay in cache, probe chains essentially never forming. A trie would key on bytes too but allocate
interior nodes for a key set this small — more machinery than a few hundred short keys warrant. Correctness
does not depend on the hash being collision-free, only on the probe eventually finding either the matching
key or an empty slot, which it always does at 0.3–7.6% occupancy.

There is one cost of `mmap` I name because it caps the threading multiplier: page faults on first touch. The
mapping is lazy, so the first read of a given page takes a minor fault to wire it into the page cache. At
4 KB pages, 12 GB is ~3 million pages and therefore ~3 million front-loaded first-touch faults; readahead and
larger pages fold many together, but early in the run the eight threads are partly waiting on I/O rather than
computing. That, together with eight fixed segments not finishing in perfect lockstep, is why I expect
somewhat less than a clean 8× — the effective threading multiplier is bounded by how quickly the file's pages
become resident.

The payoff of the byte-keyed map is the number that dominated the baseline: a `String` is now created once
per *distinct* station — a few hundred — at merge time, instead of once per *row*, turning roughly four
billion per-row allocations into a few hundred. That is the single move that takes the garbage collector from
running the whole job to doing essentially nothing. Each thread gets its own `ByteArrayToResultMap`, so zero
contention in the hot loop. Only at the end do I drain each thread's map, decode each key to a `String` once,
and `Collectors.toMap` them into a `TreeMap` with a merge function combining any station that appears in
more than one thread's map. Draining eight maps of a few hundred (up to ten thousand) entries is a few
thousand hash inserts and a few hundred decodes — microseconds against a multi-second parse, so the serial
tail is genuinely negligible and the eight-way parallel parse translates almost directly into wall-clock
speedup. The one place the merge matters is correctness: it must combine colliding stations with the same
associative elementwise rule as the in-thread accumulator, and it does, so a station split across segments
reassembles to exactly the single-threaded aggregate. The `TreeMap` keeps the output alphabetized, so the
final format is unchanged from the baseline.

Now set the bar by projecting the two multipliers rather than guessing. The threading multiplier is at most
8× and realistically 6–7× effective — serial merge tiny, mmap warming via page faults, imperfect lockstep.
The constant-factor multiplier from removing the ~140 GB of allocation churn and its GC, the 12 GB decode,
the regex split, the float parse, and the redundant name passes is harder to pin but large, plausibly 6–10×.
Those compound: even a conservative 6× × 6× ≈ 36× puts me near 8 seconds, and 7× × 9× ≈ 63× near 4.5. So the
honest target is not "a bit under five minutes" but *single-digit seconds*, roughly two orders of magnitude
below the baseline. The falsifiable form: if the diagnosis is right that the baseline was almost entirely
idle cores plus per-row overhead, this rung lands in the low-single-digit to high-single-digit seconds; if it
lands far above that, then some cost I have not named — page-fault I/O as the mmap warms, or worse-than-
expected thread imbalance — is dominating and becomes the next thing to attack. The full module is in the
answer.
