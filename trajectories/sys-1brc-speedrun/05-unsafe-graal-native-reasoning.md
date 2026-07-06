02.157, down from 03.210 — a 1.489× drop, and the shape of how it was won matters as much as the number.
The branchless-SWAR-over-Unsafe bet came in through a string of small, steady cuts in the optimization log
rather than one dramatic trick, which is the signature of an algorithm that is essentially *right*: parallel,
byte-keyed open addressing, scaled-integer aggregation, branchless register-width parsing over raw
addresses. Let me read the number to see what is left. 2.157 seconds across eight cores is `2.157 × 8 ≈
17.3` core-seconds for a billion rows, ~17.3 ns/row, down from ~25.7. So the SWAR-and-Unsafe move shaved
another ~8.4 ns/row. There is no obvious *new* algorithm to reach for — I am not going to find a cleverer way
to compute a min. What remains is two kinds of *overhead* that the previous rung still pays, and to get under
two seconds I have to attack both at once: the fixed cost the JVM imposes before and around the actual work,
and the last branch-misprediction and load-stall slack inside the hot loop.

Start with the fixed cost, because on a job that finishes in ~2 seconds it is a real fraction of the total in
a way it never was at four minutes. A JVM run pays to start the VM, load and link classes, and let the JIT
warm up — the hot loop runs interpreted for its first iterations and is only recompiled to optimized machine
code once it has been seen enough times, all while the clock is already running. On a four-minute baseline a
few hundred milliseconds of startup and warmup amortized to nothing; on a two-second run the same few
hundred milliseconds is 10–25% of the whole thing. The fix in kind is ahead-of-time compilation: build the
program with GraalVM `native-image` into a standalone native binary, so there is no VM to start, no class
loading, and no JIT warmup — the hot loop is already optimized machine code from the first instruction. The
challenge rules forbid computing the answer at build time, so the native image must still read the file and
do all the aggregation at runtime; I am removing only the *startup and warmup*, not the computation.

I should be honest that this is not a brand-new lever so much as one I lean on fully. The previous rung
already ran as a GraalVM native binary on the board, so ahead-of-time compilation is not new in kind here —
which means I cannot bank the whole JVM-startup story as this rung's win. If AOT were already in play at
02.157, then the fixed-cost savings from *compilation itself* are largely already spent, and the real
distance to sub-two-seconds has to come from the hot-loop refinements below, with the native binary as the
shared substrate that makes those refinements' machine code predictable rather than as the headline. So I
treat AOT as the floor everything else stands on, and I go looking for the actual new time in the loop and
the scheduler. I do keep the subprocess unmap trick exactly as before: `spawnWorker` re-launches the process
with `--worker`, the worker does the work and streams its output, and the slow ~12 GB unmap happens after
the answer is already out, off the measured clock.

The first hot-loop refinement is where the last real scheduling idea lives: **work stealing with small
segments**. The previous rung split the file into one equal slice per core. The trouble with an equal split
is *imbalance*, and it is worth being precise about why, because the fix follows from the mechanism. The OS
schedules eight threads on eight cores, but page faults as the mmap warms, memory-controller and NUMA
effects, and the plain luck of which slice contains more distinct keys or more cache-hostile access mean the
eight fixed slices do not finish in lockstep. And the total wall-clock is the time of the *slowest* thread —
so if seven threads finish and one straggles, seven cores sit idle burning wall-clock while one core
finishes a long tail. The fix is to stop handing out eight big fixed slices and instead carve the file into
many small segments, `SEGMENT_SIZE = 1 << 21 = 2 MB`, handed out from a single shared `AtomicLong` cursor
that every thread `addAndGet`s. A thread that finishes a 2 MB segment immediately grabs the next one; no
thread can be left holding a long tail, because the granularity is fine and the work flows to whichever core
is free.

The 2 MB granularity is the tuning knob, and I want to justify it with arithmetic rather than taste. The
file is ~12 GB, so 2 MB segments make about `12e9 / 2e6 = 6000` segments, roughly 750 per thread. The only
shared state is the `AtomicLong`, touched once per segment: ~6000 `addAndGet`s over the whole run, utterly
negligible against a billion rows of parsing, so the atomic is not a contention bottleneck at this size. Now
consider the two ways to get it wrong. If I made segments tiny — say 4 KB — I would have ~3 million segments,
3 million atomic operations, and, worse, I would pay the per-segment setup (finding the newline-aligned
boundaries and splitting into cursors) over only a few hundred rows each, so the fixed setup cost would stop
amortizing. If I made them huge — say 1 GB — I would be back to ~12 segments and the straggler imbalance of
the equal split. 2 MB sits in between: ~146,000 rows per segment (2e6 bytes over ~14 bytes/row), enough to
amortize the setup, and fine enough that the worst-case end-of-run imbalance is bounded by a single
sub-millisecond segment rather than a whole slice. That is the credited sweet spot, and the arithmetic is
why.

The stealing has to keep the newline-alignment invariant across independently-stolen blocks, so let me check
the seam logic the same way I checked the outer split. A thread steals the block starting at `current =
counter.addAndGet(SEGMENT_SIZE) − SEGMENT_SIZE`, then sets `segmentEnd = nextNewLine(min(fileEnd − 1,
current + SEGMENT_SIZE))` and `segmentStart = current == fileStart ? current : nextNewLine(current) + 1`. So
each block realigns its own start and end to newlines. Consider two adjacent blocks: block N ends at
`nextNewLine(current_N + 2MB)`, and block N+1 begins at `nextNewLine(current_{N+1}) + 1` where
`current_{N+1} = current_N + 2MB`. Both expressions resolve to the *same* physical newline — one takes it as
the end, the other takes the byte after it as the start — so the line straddling the nominal 2 MB boundary
belongs wholly to block N, and block N+1 starts cleanly on the next line. No line is split across two stolen
blocks and none is dropped, regardless of which threads happen to grab which blocks in which order. The
work-stealing is safe precisely because the alignment is recomputed locally from the same newline both
sides agree on.

The second refinement is subtler: run **three cursors in the same thread**. I split each 2 MB segment into
three parts and advance a `Scanner` over each, interleaving their work in the loop body — read `w1`, `w2`,
`w3`; find three delimiters; do three hash lookups; parse three numbers; record three results. The reason
this helps is instruction-level parallelism and memory-latency hiding. Each line's processing is a
*dependent chain*: load the name bytes, find the delimiter, compute the hash, probe the table, load the
matched entry, update it — roughly six steps where each waits on the previous. A single cursor stalls the
whole core whenever one of those loads misses cache, and a probe that misses into main memory is on the
order of a hundred-plus cycles of nothing. With three independent cursors in flight, while cursor 1 is
waiting on its entry load the CPU's out-of-order engine has cursors 2 and 3's independent work to execute,
so the core stays fed instead of stalling. Why three and not two or four? Two may not have enough
independent work to fully cover a main-memory miss; four (or more) starts to blow the working set — three
`Scanner`s, their words, and three live `Result`s — out of the registers and L1 the whole point was to keep
it in. Three is the sweet spot in the sense that it is the smallest number that hides the latency without
spilling the state, and I hold that as a claim to check against the record rather than a proven fact: I would
want to confirm three genuinely beats two and four on this exact core count and cache.

The three cursors have to tile the stolen segment without overlap, and the code builds them by the same
newline-alignment trick applied one level down. Given a block from `segmentStart` to `segmentEnd`, it takes
`dist = (segmentEnd − segmentStart) / 3`, then `midPoint1 = nextNewLine(segmentStart + dist)` and `midPoint2
= nextNewLine(segmentStart + dist + dist)`, and constructs `s1 = [segmentStart, midPoint1]`, `s2 =
[midPoint1 + 1, midPoint2]`, `s3 = [midPoint2 + 1, segmentEnd]`. So the two internal cut points are pulled
forward to real newlines, and each cursor's start is defined as the byte after the previous cursor's
newline end — the identical seam construction as the outer split, so the three sub-ranges partition the
block's lines exactly, no line shared and none missed. The interleaved loop then runs while all three
cursors have data (`s1.hasNext() && s2.hasNext() && s3.hasNext()`), and the ragged tails — where one cursor
runs a line or two longer than the others — are finished one cursor at a time afterward. That the split is
newline-clean is what lets me treat the three cursors as three fully independent streams whose only shared
state is the per-thread `Result[]`, which is exactly the independence the out-of-order engine needs to
overlap their stalls.

The per-line work itself is the same SWAR family as the previous rung, tuned to shave the very last
branches. The delimiter search is `findDelimiter(w) = (input − 0x01..01) & ~input & 0x80..80` after XOR with
`0x3B..3B` — the same high-bit zero-hunt I verified before. What is new is the credited insight to read the
first **16 bytes** of every name *unconditionally* as two words, `word` and `wordB`, with their two
delimiter masks, and to build the key from both with no branch between the "name fits in 8 bytes" and "name
is 9–16 bytes" cases. The justification is exactly the misprediction arithmetic I did last rung: a branch on
data-dependent name length mispredicts a large fraction of a billion times, and at even 10% mispredict rate
that is `1e8 × ~17 / 2.5e9 ≈ 0.68` seconds of pure penalty — so doing the wasted work of always reading and
masking both words is cheaper than paying that branch. The masking is done by precomputed `MASK1`/`MASK2`
lookup tables rather than variable bit-shifts (another credited idea, because a table load avoids a
data-dependent shift), and I want to verify the tables actually implement the unification correctly, because
this is the load-bearing trick.

Trace the length logic. `letterCount1 = Long.numberOfTrailingZeros(delimiterMask) >>> 3`. If the `';'` sits
in `word` at byte `k`, its high-bit mark is at bit `8k + 7`, and `(8k + 7) >>> 3 = k`, so `letterCount1 = k
∈ 0..7`. If there is no `';'` in `word` at all, `delimiterMask = 0`, and `numberOfTrailingZeros(0) = 64`,
so `letterCount1 = 8`. Now look at `MASK2`, which is indexed by `letterCount1` and is all-zero for indices
0..7 and all-ones only at index 8. So `MASK2[letterCount1]` is zero exactly when the `';'` was found in the
first word — meaning a name of 1–8 bytes masks `word2` entirely to zero, and the key is just the first
word's name bytes. And `MASK2[letterCount1]` is all-ones exactly when the first word had no delimiter —
meaning a name of 9–16 bytes lets `word2` contribute, itself trimmed by `MASK1[letterCount2]` down to the
part before *its* delimiter. So the two cases — short name and medium name — are handled by the same
straight-line sequence of loads, masks, and an XOR into the hash, with the table entries doing the case
selection that a branch would otherwise do. Only a name longer than 16 bytes falls into the explicit
scanning loop that keeps reading 8 bytes at a time until it finds the delimiter. That is the branch-free
`<8`/`8–16` unification, and the tables verifiably encode it.

The number parse is the same branchless magic-constant SWAR (`convertIntoNumber`, credited to Quan Anh Mai):
find the `'.'` with `~numberWord & 0x10101000`, recover the sign, align with a shift of `28 − decimalSepPos`
and multiply by `0x640a0001` to sum the digits into bits 32–41, mask `& 0x3FF`, apply sign — temperature
times ten, no branches. It was the right tool three rungs ago and it is still the right tool; the whole line
is now uniform branchless `long` arithmetic over raw addresses.

The map is open addressing again, sized `1 << 17`, but the entry is now a real `Result` object whose first
two fields cache the name's two `long` words (`firstNameWord`, `secondNameWord`) alongside the raw
`nameAddress`. On a probe the cheap check compares those two cached words directly —
`existingResult.firstNameWord == word && existingResult.secondNameWord == word2` — which resolves the
overwhelming majority of hits in two register comparisons with no memory touch of the name at all; only on a
rare long-name collision do I walk the name 8 bytes at a time against the stored `nameAddress`, probing
`(tableIndex + 31) & (len − 1)` on collision. The `+31` stride is deliberate: 31 is odd and therefore
coprime with the power-of-two table size, so the probe sequence eventually visits every slot, and a stride
larger than one spreads colliding keys apart instead of piling them into the primary cluster that plain
`+1` linear probing builds. Aggregation keeps `min` and `max` as `short` and `sum` as `long`, and the ranges
justify the widths exactly: a scaled temperature lives in `[−999, 999]`, which fits a `short`'s
`[−32768, 32767]` with room to spare, while a billion scaled values summing to as much as ~10^12 needs the
`long`. `record` updates them branch-lean. Everything reads through `Scanner`'s `Unsafe.getLong(pos)` on raw
mmap addresses; the per-thread `Result[]` means no locks in the loop; and a final `TreeMap` accumulation
(`putIfAbsent` + `accumulate`) merges the per-thread results and sorts for output, decoding each name to a
`String` only at the very end, once per distinct station, in `calcName`.

So the finale is the same parallel byte-keyed scaled-integer SWAR algorithm brought to its limit on three
axes at once: an ahead-of-time GraalVM native binary that erases whatever JVM startup and JIT warmup remain
and makes the rest predictable machine code; a work-stealing scheduler over 2 MB segments from a shared
atomic cursor that keeps all eight cores busy to the last byte; and a three-cursors-per-thread inner loop
with unconditional 16-byte name reads and table-mask key building that hides memory latency and strips the
last data-dependent branches. Each piece is one contributor's idea composed into a single program — the
subprocess and Unsafe-mmap approach (Alfonso Peterssen), the work-stealing 2 MB segments (Artsiom Korzun),
the branch-free `<8`/`8–16` unification (Jaromir Hamala), the mask-table lookups (Van Phu DO), and the
branchless number parse (Quan Anh Mai) — which is why the reasoning here is one of assembly rather than a
single new invention.

Before I commit I want to see that the three refinements attack *different* parts of the 02.157 budget, so
that they compound rather than fight over the same cost. Work stealing does not touch the per-row work at
all; it recovers the *tail*. If an equal split leaves the slowest of eight threads running even 8% longer
than the mean, that is roughly `0.08 × 2.157 ≈ 0.17` seconds of wall-clock during which seven cores are
idle — and fine-grained stealing collapses that tail toward the cost of one 2 MB segment, so most of that
0.17 s comes back regardless of what the inner loop does. Three cursors, by contrast, attack the per-row
work directly, specifically its *stall* component: a map probe that misses into main memory is ~100+ cycles,
~40 ns, and even a few percent of a billion rows missing that way is a serial cost measured in tenths of a
second on a single cursor — overlapping three independent cursors is what turns those stalls into
throughput. And the unconditional 16-byte read attacks the *branch* component, the length-dependent
mispredicts I priced near two-thirds of a second in the worst case. Because those three costs — end-of-run
idle, mid-loop memory stalls, and length-branch mispredicts — are largely disjoint, removing each one keeps
what the others removed, and their sum is plausibly the headroom between 02.157 and sub-two-seconds. That
disjointness is the real argument that the composition clears the bar rather than each piece stepping on the
next.

This is the closing move, so let me set the bar and say honestly why I believe it clears it. The standing
record is the branchless-SWAR rung at 02.157 seconds, ~17.3 ns/row. The bet is that the algorithm is already
right and what remains is overhead of three specific kinds: whatever fixed startup/warmup tax survives, the
idle-core tail of an equal file split, and the per-line load stalls and mispredicted length branches.
Removing the tail with fine-grained work stealing, hiding the stalls with three interleaved cursors, and
deleting the length branches with unconditional 16-byte key handling should each shave a slice of the 17.3
ns/row, and if they compound to roughly a fifth-to-quarter off, the run lands in sub-two-second territory —
which would make this the fastest known way to aggregate a billion rows on this machine, the rank-1 entry.
The honest things I would want to validate against the 02.157 record, rather than assume: that the atomic
cursor's contention really stays negligible at 2 MB granularity (too small and the `AtomicLong` becomes the
bottleneck, too large and the imbalance returns); that three cursors genuinely beats two and four on this
exact core count and cache rather than just sounding balanced; and that the native image's runtime-only
computation still satisfies the no-build-time-precompute rule while actually beating the JVM build on a job
this short. If those hold, the composition lands under 02.157 and the ladder ends here. The full module is in
the answer.
