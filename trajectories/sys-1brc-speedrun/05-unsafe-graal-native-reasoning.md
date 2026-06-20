02.157, down from 03.210 — the branchless-SWAR-over-Unsafe bet paid off, and the optimization log shows it
came from steady small cuts rather than one trick. So the algorithm is essentially right: parallel,
byte-keyed open addressing, scaled-integer aggregation, branchless register-width parsing over raw
addresses. There's no obvious *new* algorithm left to find. What's left is two kinds of overhead that the
previous rung still pays, and to get under two seconds I have to attack both at once: the fixed cost the
JVM imposes before and around the actual work, and the last branch-misprediction and load-stall slack
inside the hot loop.

Start with the fixed cost, because on a job that finishes in ~2 seconds it's a real fraction of the total.
The previous rung still runs on the JVM: it starts the VM, loads classes, and lets the JIT warm up by
interpreting and then recompiling the hot loop while the clock is already running. For a long job that
warmup amortizes to nothing, but here the whole run is two seconds — so the seconds of JVM startup and the
early interpreted iterations are measurable. The fix is ahead-of-time compilation: build the program with
GraalVM `native-image` into a standalone native binary. There's no VM to start, no class loading, no JIT
warmup — the hot loop is already optimized machine code from the first instruction. The challenge rules
forbid computing the answer at build time, so the native image must still read the file and do all the work
at runtime; I'm only removing the *startup and warmup*, not the computation. (The previous rung was already
a native binary on the board, so this isn't new in kind — but the finale leans on it fully and pairs it
with the rest.)

The other piece of fixed cost is the unmap, which the previous rung already addressed with the subprocess
trick — spawn a child worker, let it stream its output, and let the parent return before the OS finishes
reclaiming the ~12 GB mapping so the unmap latency falls outside the measured wall-clock. I keep that
exactly: `spawnWorker` re-launches the process with `--worker`, the worker does the work and prints, and
the unmap happens after the answer is out.

Now the hot loop, where the last real algorithmic refinement lives. The previous rung split the file into
one equal segment per core. The problem with an equal split is *imbalance*: the OS schedules eight threads
on eight cores, but page faults as the mmap warms, NUMA effects, and the luck of which thread hits more
distinct keys mean some threads finish their fixed slice early and sit idle while a straggler runs long.
The total time is the *slowest* thread, so idle cores at the end are wasted. The fix is **work stealing
with small segments**: instead of eight big fixed slices, carve the file into many small segments
(`SEGMENT_SIZE = 1<<21 = 2 MB`) and hand them out from a single shared `AtomicLong` cursor that every
thread `addAndGet`s. A thread that finishes a 2 MB segment immediately grabs the next one; no thread can be
left holding a long tail because the granularity is fine and the work flows to whichever core is free. This
is the move that keeps all eight cores busy right to the end — the credited insight that 2 MB
work-stealing segments beat an equal split.

Inside a segment there's a second, subtler scheduling idea: run **three cursors in the same thread**. I
split each 2 MB segment into three parts and advance a `Scanner` over each, interleaving their work in the
loop body — read `word1`, `word2`, `word3`; find three delimiters; do three hash lookups; parse three
numbers; record three results. The reason this helps is instruction-level parallelism and memory latency
hiding: each line's processing has a dependent chain (load → find delimiter → hash → probe → load entry →
update), and a single cursor stalls that core whenever a load misses cache. With three independent cursors
in flight, while cursor 1 is waiting on a memory load the CPU's out-of-order engine has cursors 2 and 3's
independent work to chew on, so the core stays fed. Three is the sweet spot — enough independent streams to
hide latency, few enough to keep their working sets in registers and L1.

The per-line work itself is the same family as before but tuned to shave the last branches. The delimiter
search is SWAR — `findDelimiter(word) = (input − 0x01..01) & ~input & 0x80..80` after XOR with `0x3B..3B` —
and I read the first **16 bytes** of every name up front as two words (`word`, `wordB`) and their two
delimiter masks, *unconditionally*. The credited insight here is that branching between the "<8 byte name"
and "8–16 byte name" cases costs more in misprediction than the wasted work of always doing both, so I do
both: compute `letterCount1` and `letterCount2` from the two masks, build the key from the two words with
precomputed `MASK1`/`MASK2` lookup tables (mask-based instead of bit-shifting — another credited idea,
because table lookups avoid variable shifts), and form the hash as `word ^ word2`. Only names longer than
16 bytes take a slow scalar-ish path. The number parse is the same branchless magic-constant SWAR
(`convertIntoNumber`, credited to the same author as the merykitty parse): find the `'.'` with
`~numberWord & 0x10101000`, recover the sign, align and multiply by `0x640a0001` to sum the digits into
bits 32–41, mask `& 0x3FF`, apply sign — temperature times ten, no branches.

The map is open addressing again, sized `1<<17`, but the entry is now a real `Result` object whose first
two fields are the name's two `long` words (`firstNameWord`, `secondNameWord`) plus the raw `nameAddress`.
On a probe the cheap check is comparing those two cached words directly (`existingResult.firstNameWord ==
word && existingResult.secondNameWord == word2`); only on the rare long-name collision do I walk the name
8 bytes at a time via `getLongAt` against the stored `nameAddress`, and probe `(tableIndex + 31) &
(len-1)` on collision. Aggregation keeps min/max as `short` and sum as `long` — the scaled-integer values
fit a `short` exactly — and `record` updates them branch-lean. Everything reads through `Scanner`'s
`Unsafe.getLong(pos)` on raw mmap addresses; the map's per-thread `Result[]` means no locks in the loop,
and a final `TreeMap` accumulation (`putIfAbsent` + `accumulate`) merges the per-thread results and sorts
for output. The name is only decoded to a `String` at the very end, in `calcName`, once per distinct
station.

So the finale is the same parallel byte-keyed scaled-integer SWAR algorithm, brought to its limit on three
axes simultaneously: an ahead-of-time GraalVM native binary that erases JVM startup and JIT warmup; a
work-stealing scheduler over 2 MB segments from a shared atomic cursor that keeps all eight cores busy to
the last byte; and a three-cursors-per-thread inner loop with unconditional 16-byte name reads and
table-mask key building that hides memory latency and strips the last data-dependent branches. Each of
those is one contributor's idea — the subprocess and Unsafe mmap, the work-stealing 2 MB segments, the
branch-free <8/8–16 unification, the mask-table lookups, the branchless number parse — composed into one
program.

This is the closing move, so let me set the bar and why I believe it clears it. The standing record at the
top of this ladder is the branchless-SWAR rung at 02.157 seconds. The bet is that the algorithm is already
right and what remains is overhead: the JVM's fixed startup/warmup tax, the idle-core tail of an equal file
split, and the per-line load stalls and mispredicted length branches. Removing the first with an AOT native
binary, the second with fine-grained work stealing, and the third with three interleaved cursors and
branchless 16-byte key handling should compound into the final sub-two-second result — making this the
fastest known way to aggregate a billion rows on this machine, and the rank-1 entry. The honest things I'd
want to validate against the 02.157 record: that work stealing's atomic-cursor contention stays negligible
at 2 MB granularity (too-small segments would make the `AtomicLong` the bottleneck; too-large would bring
back the imbalance), that three cursors is genuinely better than two or four on this exact core count and
cache, and that the native image's runtime-only computation still satisfies the no-build-time-precompute
rule while actually beating the JVM build on so short a job. If those hold, the composition lands under
02.157 and the ladder ends here. The full module is in the answer.
