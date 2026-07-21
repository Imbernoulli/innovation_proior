03.210, down from 05.979 — a 1.862× drop, right in the band I hoped for when I bet the byte-at-a-time scan
was the hot spot, so the diagnosis was sound: vectorizing the delimiter search and the key compare bought
almost exactly the "roughly halve" I predicted. Read the new number mechanically. 3.210 s across eight cores
is ~25.7 core-seconds for a billion rows, ~25.7 ns/row, down from ~47.8. So vectorizing the two
linear-in-length costs removed about `47.8 − 25.7 ≈ 22` ns/row, meaning the scan and byte-by-byte key
compare together had been ~46% of the per-row work — a clean confirmation the scan was where the time went.
The *remaining* ~25.7 ns/row is now everything else: the map probe and its possible cache-miss stall, the
number parse, the aggregate update, and whatever branch and codegen overhead the vector path itself carries.

That last item makes me doubt a *wider* vector is the right next lever. Two things bother me. First, the
payload is tiny relative to the tool: a name here averages ~7 bytes and rarely exceeds 16, so loading a
32-byte vector to find a `';'` almost always within the first eight bytes is a lot of lane machinery for a
short key, and the incubating Vector API's codegen is uneven — depending on the JDK it sometimes fails to
fold the vector path into tight machine code, leaving mask extraction and lane shuffles a register-level
approach would not pay. Second, every line still carries branches I have not killed: the long-name fallback
when the mask is zero, the tail near the segment end, and the `node == null` versus key-mismatch split inside
the probe. A billion lines means a billion trips through those, and mispredictions on data-dependent name
lengths dominate a tight loop. Size it: if even 10% of a billion lines mispredict one branch at ~15 cycles,
that is `1e8 × 15 / 2.5e9 ≈ 0.6` seconds of pure penalty — a fifth of the whole run spent on branches that
produce no answer, and not something a wider vector addresses.

So lay out the moves and force the choice. Wider is not available — Zen2's native width is 256 bits, so
`SPECIES_512` decomposes into two 256-bit ops for no gain. Keeping the Vector API and shaving branches leaves
the incubator-codegen variance and the wasted lane work on short payloads exactly where they are. Or I switch
philosophy entirely: do the data-parallelism *within a 64-bit register* — SWAR — for the whole hot loop, not
just the number, so it is branchless plain-`long` math the JIT compiles predictably with no incubator module,
and the register holds only 8 bytes so I am never doing 32-byte lane work for a 7-byte name. The previous
rung already used SWAR for the *number* parse, where it was clearly the elegant choice; the bet now is that
SWAR is the better tool for the *name* scan and the entire loop, precisely because the data is short enough
that one or two 8-byte words cover almost every name, and because branchless `long` arithmetic is what
removes the mispredictions I just priced at ~0.6 s. This is a genuinely different bet from the previous rung,
not a strict improvement on it — hand-rolled register-width SIMD the JIT turns into tight predictable code,
versus explicit wide-vector SIMD from the incubator — and I choose it because what is left to cut is branches
and codegen variance, not raw scan width. Counting the work on a 7-byte name makes the bet plausible: the
wide path issues a 256-bit load spanning 32 bytes (possibly straddling two cache lines), a lane-compare, a
mask-extraction to a general register, and a trailing-zero count; the SWAR path issues a single 8-byte load,
an XOR, a subtract, two ANDs, and a trailing-zero count — five or six plain integer ops on one register,
touching 8 bytes instead of 32, with no vector-register round trip. Not wider — narrower — but leaner and
more predictable on this payload.

The core SWAR identity for "is there a target byte in this 8-byte word" is the classic zero-byte hunt: XOR
the word with the target repeated eight times — `0x3B3B3B3B3B3B3B3B` for `';'` — so every matching byte
becomes `0x00`, then `(x − 0x0101010101010101) & ~x & 0x8080808080808080` lights the high bit of exactly the
zero bytes. A byte that became `0x00` borrows on `x − 1` and keeps its high bit through the AND; a
non-matching byte in `0x01..0x7F` does not borrow into the high bit, so the AND yields 0 — the identity finds
zero bytes and only zero bytes. `numberOfTrailingZeros` of that mask, shifted right by 3, gives the byte
index of the first delimiter, no vector and no loop. I read the name 8 bytes at a time and test with that
identity, and I deliberately **unroll to 16 bytes** — two `long`s — as the common case, because almost every
name fits in sixteen bytes: a 1–8-byte name is resolved by the first word, a 9–16-byte name by the second,
and only a longer name falls into the rare scanning loop. When a word has no delimiter I keep all eight
bytes; when it does, I build `mask = ~((highBitMask >>> 7) − 1)` and clear the bytes at and beyond the `';'`
so the partial word holds only the name's tail. That 16-byte shape is what makes the branch behaviour
predictable — the predictor learns "first word usually suffices," and the straight-line path retires without
a stall on the overwhelming majority of lines, extracting the key and leaving the cursor precisely on the
value byte with no separate delimiter-skip step.

I fold the hash *as I scan*: `hash ^= readBuffer1; hash ^= readBuffer2`, XOR-ing in the masked name words
themselves, then a final avalanche `hash ^= hash >> 32`. This is a different trade from the previous rung's
fixed first-4/last-4 FxHash — there the SIMD scan skipped the bytes, so I loaded them separately to hash;
here the SWAR scan already *has* both name words sitting in registers, so folding them in is nearly free and
needs no second read.

The deeper move needs `Unsafe`. The `MemorySegment`/buffer indirection still carries per-access bounds
checks; I drop them by taking the mmap's raw base address — `fileChannel.map(...).address()` — and walking it
with `UNSAFE.getLong(ptr)`, which fetches 8 bytes from an arbitrary address with no bounds check and no
object header. My in-file optimization log records the descent this produced on the development machine:
62000 ms initially, down to 6500 once mmap'd, 4200 with the custom map, 3900 with SWAR token checks, then the
single largest cut — 1900 ms — the moment raw `Unsafe` replaced the bounds-checked buffer, bottoming out
around 1200 ms. That log is the record of this philosophy paying off in steady increments rather than one
lucky trick, and the Unsafe step is the one that jumps.

For the map I keep open addressing but push the whole entry into a **flyweight `byte[]`** with a hand-laid
memory format read and written by `Unsafe` at fixed offsets: a `long` sum, an `int` min, an `int` max, an
`int` count, a length byte, then the name bytes. So a station's aggregate is one flat `byte[]` with no
boxing, and `updateEntry` is four `Unsafe` field read/writes. This finally spends the exact-integer-sum lever
I flagged two rungs ago: the temperature I feed in is the scaled integer straight from the parser, `min` and
`max` are `int`s holding scaled values in `[−999, 999]`, and `sum` is a `long`. A billion scaled values of up
to 999 sum to at most ~10^12, nowhere near a `long`'s ceiling, so the aggregate is now exact — the
floating-point drift from summing a billion tenths into a `double` is gone, traded for arithmetic that is
both faster and more correct. The entries stay cache-resident: ~40–45 bytes each across 413 stations is on
the order of 18 KB, sitting comfortably in L2 and much of it in L1. The key *compare* on a probe is itself
SWAR — compare the stored name to `readBuffer1`/`readBuffer2` 8 bytes at a time with `UNSAFE.getLong` on both
sides — so a 16-byte name is two `long` compares, no `Arrays.equals`. I also separate the table's *index*
from the fat entries, and this is a distinct cache win from the flyweight layout: if each slot held the whole
~40-byte entry inline, every probe step — even a miss — would drag 40 bytes through the cache hierarchy, so
by keeping slots slim the probe walks a dense contiguous array where several slots share a cache line and the
common "is this the right key" checks stay in L1, pulling the fat entry only on a hit worth updating.

For the merge I choose a `ConcurrentHashMap` where earlier rungs hand-merged: with a few hundred distinct
stations and eight threads each finishing with its own `Result[]`, the total merge is a few thousand
operations at the very end — far too little work to matter against a two-second parse, and too little
contention to justify a custom lock-free structure. So the hot loop is per-thread and lock-free and the
trivial tail uses the standard concurrent map, then sorts for output.

The number parse stays the branchless SWAR magic-constant trick, with one micro-optimization: the shift is
`min28 = dotPosition ^ 0b11100` instead of `28 − dotPosition`. On exactly the three values `dotPosition` can
take — 12, 20, 28 — the XOR computes the subtraction (`12^28 = 16 = 28−12`, `20^28 = 8`, `28^28 = 0`), and
XOR is one cheap ALU op with no dependency on a subtract's constant. The rest is as before: find the `'.'`
via `~numberBytes & 0x10101000`, recover the sign from `(invNumberBytes << 59) >> 63`, align, multiply by
`100*0x1000000 + 10*0x10000 + 1`, mask `& 0x3FF`, sign-apply. Now the *whole* line — scan, hash, key compare,
number — is uniform branchless `long` arithmetic over raw addresses.

One more idea the notes flag that I take: the subprocess trick. Unmapping ~12 GB at process exit is slow and
serialized — the kernel tears down the mapping's page-table entries — and on a job that finishes in a couple
of seconds that teardown, tens to over a hundred milliseconds, is a measurable few percent of the
wall-clock. If I spawn a child worker that does the actual work and streams its output, and let the parent
return as soon as the child has printed the answer, the OS reclaims the mapping *after* I have emitted the
last output byte, so the unmap latency falls outside the measured window. The threads are bare `Thread[]`,
one per processor, each running `processMemoryArea` and merging into the shared `ConcurrentHashMap`.

So this rung is the same parallel, byte-keyed, scaled-integer structure, but every per-line operation is now
branchless SWAR over raw `Unsafe` addresses instead of Vector-API lane ops: a 16-byte unrolled delimiter
scan, a hash folded from the scanned words, a SWAR key compare, the SWAR number parse, and flyweight `byte[]`
entries. Set the bar with the same budgeting: the standing time is 03.210, ~25.7 ns/row. Removing the ~0.6 s
of branch mispredictions (via the branchless 16-byte handling), the per-access bounds checks (via `Unsafe`),
and the incubator's uneven lane codegen (via plain `long` math) should each shave a slice; if they compound
to roughly a third off, `3.210 × (16/25.7) ≈ 2.0` seconds is the landing I aim at, plus whatever the
subprocess unmap trick removes from the tail. The falsifiable form: if branches and incubator variance really
were what remained, ~2 seconds; if I barely move, the true residual was memory-latency on the map probe, a
stall no branch-trimming touches, which becomes the next thing to chase. The costs are honest: SWAR
over-reads up to 16 bytes past a name, so I must guarantee slack past the last line, which the
newline-aligned boundaries and the mmap'd tail provide; and `Unsafe` is unforgiving — one wrong offset is a
segfault, not an exception. The full module is in the answer.
