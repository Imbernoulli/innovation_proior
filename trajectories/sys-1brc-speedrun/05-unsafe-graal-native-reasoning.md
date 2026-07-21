02.157, down from 03.210 — a 1.489× drop, and the shape of how it was won matters as much as the number.
The branchless-SWAR-over-Unsafe bet came in through a string of small steady cuts rather than one dramatic
trick, the signature of an algorithm that is essentially *right*: parallel, byte-keyed open addressing,
scaled-integer aggregation, branchless register-width parsing over raw addresses. 2.157 s across eight cores
is ~17.3 core-seconds for a billion rows, ~17.3 ns/row, down from ~25.7 — the SWAR-and-Unsafe move shaved
another ~8.4 ns/row. There is no obvious *new* algorithm to reach for; I am not going to find a cleverer way
to compute a min. What remains is two kinds of *overhead*: the fixed cost the JVM imposes around the work,
and the last branch-misprediction and load-stall slack inside the hot loop. To get under two seconds I attack
both.

Start with the fixed cost, because on a ~2-second job it is a real fraction in a way it never was at four
minutes. A JVM run pays to start the VM, load and link classes, and let the JIT warm up — the hot loop runs
interpreted for its first iterations and is only recompiled to optimized machine code once seen enough times,
all while the clock runs. On a four-minute baseline a few hundred milliseconds of startup and warmup
amortized to nothing; on a two-second run the same is 10–25% of the whole. The fix in kind is ahead-of-time
compilation: build with GraalVM `native-image` into a standalone binary, so there is no VM to start, no class
loading, and no JIT warmup — the hot loop is optimized machine code from the first instruction. The rules
forbid computing the answer at build time, so the native image still reads the file and aggregates at
runtime; I remove only startup and warmup.

I am honest that this is not a brand-new lever so much as one I lean on fully. The previous rung already ran
as a GraalVM native binary, so the fixed-cost savings from *compilation itself* are largely already spent,
and the real distance to sub-two-seconds has to come from the hot-loop refinements below, with the native
binary as the shared substrate that makes their machine code predictable rather than as the headline. I keep
the subprocess unmap trick exactly as before: `spawnWorker` re-launches with `--worker`, the worker does the
work and streams its output, and the slow ~12 GB unmap happens after the answer is out.

The first hot-loop refinement is where the last real scheduling idea lives: **work stealing with small
segments**. The previous rung split the file into one equal slice per core, and the trouble with an equal
split is imbalance — page faults as the mmap warms, memory-controller and NUMA effects, and the luck of which
slice holds more distinct keys or more cache-hostile access mean the eight fixed slices do not finish in
lockstep. Total wall-clock is the time of the *slowest* thread, so if seven finish and one straggles, seven
cores burn wall-clock idle. The fix is to carve the file into many small segments, `SEGMENT_SIZE = 1 << 21 =
2 MB`, handed out from a single shared `AtomicLong` cursor that every thread `addAndGet`s: a thread that
finishes a 2 MB segment immediately grabs the next, so no thread is left holding a long tail. The 2 MB
granularity is arithmetic, not taste: ~12 GB in 2 MB segments is ~6000 segments, ~750 per thread, and the
only shared state is the `AtomicLong` touched ~6000 times over the run — negligible against a billion rows.
Tiny segments (4 KB → ~3 million) would make the atomic and the per-segment newline setup stop amortizing;
huge ones (1 GB → ~12) would bring back the straggler imbalance. 2 MB sits between: ~146,000 rows per
segment, enough to amortize the setup, fine enough that the worst-case end-of-run imbalance is bounded by a
single sub-millisecond segment. The stealing keeps the newline-alignment invariant by the same seam
construction as the outer split, recomputed locally: each stolen block realigns its own start and end to real
newlines, one taking a newline as its end and the next taking the byte after as its start, so the line
straddling a nominal 2 MB boundary belongs wholly to the earlier block and none is dropped, regardless of
which thread grabs which block in which order.

The second refinement is subtler: run **three cursors in the same thread**. I split each 2 MB segment into
three parts and interleave their work in the loop body — read `w1, w2, w3`; find three delimiters; do three
hash lookups; parse three numbers; record three results. The reason is instruction-level parallelism and
memory-latency hiding. Each line's processing is a *dependent chain*: load the name bytes, find the
delimiter, compute the hash, probe the table, load the matched entry, update it — roughly six steps each
waiting on the previous. A single cursor stalls the whole core whenever one of those loads misses cache, and
a probe missing into main memory is ~100+ cycles, ~40 ns, of nothing. With three independent cursors in
flight, while cursor 1 waits on its entry load the out-of-order engine has cursors 2 and 3's independent work
to execute, so the core stays fed. Three and not two or four because two may not have enough independent work
to fully cover a main-memory miss, while four starts to blow the working set — three `Scanner`s, their words,
three live `Result`s — out of the registers and L1 the whole point was to keep it in. I hold "three is the
sweet spot" as a claim to check against the record rather than a proven fact. The three cursors tile the
stolen segment by the identical newline-alignment seam applied one level down — two internal cut points
pulled forward to real newlines, each cursor's start the byte after the previous cursor's newline end — so
the sub-ranges partition the block's lines exactly. The interleaved loop runs while all three cursors have
data, and the ragged tails are finished one cursor at a time afterward.

The per-line work is the same SWAR family, tuned to shave the very last branches. The delimiter search is
`(input − 0x01..01) & ~input & 0x80..80` after XOR with `0x3B..3B` — the same high-bit zero-hunt as before.
What is new is reading the first **16 bytes** of every name *unconditionally* as two words with their two
delimiter masks, building the key from both with no branch between the "name fits in 8 bytes" and "name is
9–16 bytes" cases. The justification is the misprediction arithmetic from last rung: a branch on
data-dependent name length mispredicts a large fraction of a billion times, and at even 10% that is `1e8 ×
~17 / 2.5e9 ≈ 0.68` seconds of pure penalty — so doing the wasted work of always reading and masking both
words is cheaper than paying that branch. The masking uses precomputed `MASK1`/`MASK2` lookup tables rather
than variable bit-shifts (a table load avoids a data-dependent shift), and the tables encode the unification:
`letterCount1 = numberOfTrailingZeros(delimiterMask) >>> 3` is the byte index of the `';'` in the first word
(0..7), or 8 when the word has no delimiter (since `numberOfTrailingZeros(0) = 64`, `>>> 3 = 8`). `MASK2` is
all-zero for indices 0..7 and all-ones only at 8, so `MASK2[letterCount1]` is zero exactly when the `';'` was
in the first word — a 1–8-byte name masks the second word entirely to zero — and all-ones exactly when the
first word had no delimiter, letting a 9–16-byte name's second word contribute, itself trimmed by
`MASK1[letterCount2]`. So the short-name and medium-name cases run the same straight-line loads, masks, and
XOR into the hash, the table entries doing the case selection a branch would otherwise do; only a name longer
than 16 bytes falls into the explicit scanning loop. The number parse is the same branchless magic-constant
SWAR (`convertIntoNumber`): find the `'.'` with `~numberWord & 0x10101000`, recover the sign, align by
`28 − decimalSepPos`, multiply by `0x640a0001`, mask `& 0x3FF`, sign-apply — temperature times ten, no
branches.

The map is open addressing again, sized `1 << 17`, but the entry is now a real `Result` whose first two
fields cache the name's two `long` words alongside the raw `nameAddress`. On a probe the cheap check compares
those cached words directly — `firstNameWord == word && secondNameWord == word2` — resolving the overwhelming
majority of hits in two register comparisons with no memory touch of the name; only on a rare long-name
collision do I walk the name against the stored `nameAddress`, probing `(tableIndex + 31) & (len − 1)`. The
`+31` stride is deliberate: 31 is odd and coprime with the power-of-two table size, so the probe sequence
eventually visits every slot, and a stride larger than one spreads colliding keys apart instead of piling
them into the primary cluster plain `+1` linear probing builds. Aggregation keeps `min`/`max` as `short`
(a scaled temperature in `[−999, 999]` fits with room to spare) and `sum` as `long` (a billion scaled values
summing to ~10^12 needs it), updated branch-lean. Everything reads through `Scanner`'s `Unsafe.getLong` on
raw mmap addresses; the per-thread `Result[]` means no locks in the loop; and a final `TreeMap` accumulation
merges the per-thread results and sorts for output, decoding each name to a `String` only at the end.

This is an assembly of contributor ideas composed into one program — the subprocess and Unsafe-mmap approach
(Alfonso Peterssen), the work-stealing 2 MB segments (Artsiom Korzun), the branch-free `<8`/`8–16`
unification (Jaromir Hamala), the mask-table lookups (Van Phu DO), and the branchless number parse (Quan Anh
Mai) — which is why the reasoning here is one of assembly rather than a single new invention. What makes me
believe it clears the bar is that the three refinements attack *disjoint* parts of the 02.157 budget. Work
stealing does not touch per-row work at all; it recovers the *tail* — if an equal split leaves the slowest
thread running even 8% longer than the mean, that is ~`0.08 × 2.157 ≈ 0.17` s of seven-core-idle wall-clock
that fine-grained stealing collapses toward one 2 MB segment. Three cursors attack the per-row *stall*
component — overlapping three independent chains turns the ~40 ns main-memory probe misses into throughput.
And the unconditional 16-byte read attacks the *branch* component, the length-dependent mispredicts priced
near two-thirds of a second in the worst case. Because those three costs — end-of-run idle, mid-loop memory
stalls, length-branch mispredicts — are largely disjoint, removing each keeps what the others removed, and
their sum is plausibly the headroom between 02.157 and sub-two-seconds.

This is the closing move. The standing record is 02.157 s, ~17.3 ns/row; the bet is that the algorithm is
already right and what remains is overhead of three kinds — surviving startup/warmup tax, the idle-core tail
of an equal split, and the per-line load stalls and mispredicted length branches. Removing the tail with
fine-grained work stealing, hiding the stalls with three interleaved cursors, and deleting the length
branches with unconditional 16-byte key handling should each shave a slice of the 17.3 ns/row; if they
compound to roughly a fifth-to-quarter off, the run lands in sub-two-second territory — the fastest known way
to aggregate a billion rows on this machine, the rank-1 entry. The honest things I would validate rather than
assume: that the atomic cursor's contention stays negligible at 2 MB granularity (too small and the
`AtomicLong` becomes the bottleneck, too large and imbalance returns); that three cursors genuinely beats two
and four on this exact core count and cache; and that the native image's runtime-only computation still
satisfies the no-build-time-precompute rule while beating the JVM build on a job this short. If those hold,
the composition lands under 02.157 and the ladder ends here. The full module is in the answer.
