03.210, down from 05.979 — a 1.862× drop, right in the band I hoped for when I bet the byte-at-a-time scan
was the hot spot, so the diagnosis was sound: vectorizing the delimiter search and the key compare bought
almost exactly the "roughly halve" I predicted. Let me read the new number mechanically before deciding
what to do next. 3.210 seconds across eight cores is `3.210 × 8 ≈ 25.7` core-seconds of CPU for a billion
rows, i.e. ~25.7 ns/row, down from the previous rung's ~47.8 ns/row. So vectorizing the two linear-in-length
costs removed about `47.8 − 25.7 ≈ 22` ns/row, which means the scan and byte-by-byte key compare together
had been about 46% of the per-row work. That is a clean confirmation that the scan really was where the time
was going. But it also tells me the *remaining* ~25.7 ns/row is now everything else — the map probe and its
possible cache-miss stall, the number parse, the aggregate update, and whatever branch and codegen overhead
the vector path itself carries.

And it is that last item that makes me doubt a *wider* vector is the right lever for the next step. Two
things bother me when I stare at where the 25.7 ns now sit. First, the payload is tiny relative to the tool.
A station name here averages ~7 bytes and rarely exceeds 16; loading a 32-byte vector to find a `';'` that
is almost always within the first eight bytes is a lot of lane machinery for a short key, and the incubating
Vector API's codegen quality is uneven — depending on the JDK it sometimes fails to fold the vector path
into the tight machine code I would write by hand, leaving mask extraction and lane shuffles that a
register-level approach would not pay. Second, every line still carries *branches* I have not killed: the
long-name fallback when the mask comes back zero, the tail handling near the segment end, and the `node ==
null` versus key-mismatch split inside the probe. A billion lines means a billion trips through those
branches, and mispredictions on data-dependent name lengths dominate a tight loop in a way a flat
instruction count never reveals. Let me size that to see if it is worth chasing: if even 10% of a billion
lines mispredict one branch at ~15 cycles, that is `1e8 × 15 / 2.5e9 ≈ 0.6` seconds of pure misprediction
penalty — a fifth of the whole 3.2-second run spent on branches that produce no answer. That is a real
target, and it is not one that a wider vector addresses.

So let me lay out the moves actually available and force the choice. I could push the Vector API *wider* —
but the machine is Zen2, whose native vector width is 256 bits, so `SPECIES_512` would just decompose into
two 256-bit ops with no throughput gain; there is no wider lane to reach for. I could keep the Vector API
and try to *shave the branches* around it — but that leaves the incubator-codegen variance and the wasted
lane work on short payloads exactly where they are. Or I could switch philosophy entirely: do the
data-parallelism *within a 64-bit register* — SWAR — for the whole hot loop, not just the number, so it is
branchless plain-`long` math the JIT compiles predictably, with no incubator module at all, and the register
holds only 8 bytes so I am never doing 32-byte lane work for a 7-byte name. The previous rung already used
SWAR for the *number* parse, where it was clearly the elegant choice; the bet now is that SWAR is the better
tool for the *name* scan and the entire loop, precisely because the data is short enough that one or two
8-byte words cover almost every name, and because branchless `long` arithmetic is exactly what removes the
mispredictions I just priced at ~0.6 s. This is a genuinely different bet from the previous rung, not a
strict improvement on it — hand-rolled register-width SIMD that the JIT turns into tight predictable code,
versus explicit wide-vector SIMD from the incubator — and I am choosing it because the thing left to cut is
branches and codegen variance, not raw scan width.

It helps to count the work each philosophy does on the common case — a 7-byte name — to see why the bet is
plausible. The wide-vector path issues a 256-bit load spanning 32 bytes (which can straddle two cache
lines), a lane-compare across all 32 lanes, a mask-extraction from the vector mask register to a general
register, and a trailing-zero count. The SWAR path, for a 7-byte name that fits inside the first 8-byte
word, issues a single 8-byte load, an XOR, a subtract, two ANDs to form the high-bit mask, and a
trailing-zero count — five or six plain integer ops on one register, touching 8 bytes instead of 32, with no
vector-register round trip and nothing for the incubator's codegen to get wrong. Both find the delimiter in
"constant" time for a short name, but the SWAR version does strictly less memory traffic and stays entirely
in the integer pipeline the JIT compiles most reliably. That is the mechanism I am betting on: not that
register-width SIMD is *wider* (it is narrower), but that on this payload it is *leaner and more
predictable*.

The core SWAR identity for "is there a target byte in this 8-byte word" is the classic zero-byte hunt. XOR
the word with the target byte repeated eight times — for `';'` that is `0x3B3B3B3B3B3B3B3B` — so that every
byte position holding `';'` becomes `0x00`. Then `(x − 0x0101010101010101) & ~x & 0x8080808080808080` lights
up the high bit of exactly the byte positions that were zero. Let me verify that identity does what I need
on the boundary cases, because if it false-positives the whole scan is wrong. For a byte that became `0x00`
(a match): `x − 1` borrows and sets its low bits, `~x` has the high bit set, and the AND with `0x80` keeps
that high bit — so `0x00 → 0x80`, detected. For a matching byte `';' ^ ';' = 0`, confirmed by that case. For
a non-matching byte in `0x01..0x7F`: its high bit is 0, `~x`'s high bit is 1, but `x − 1` does not borrow
into the high bit, so the AND yields 0 — not detected, correct. The identity finds zero bytes and only zero
bytes. `Long.numberOfTrailingZeros` of that mask, shifted right by 3, gives the byte index of the first
delimiter — no vector, no loop, no incubator.

I read the name 8 bytes at a time as a `long`, test for `';'` with that identity, and if it is not in the
first word read the next 8 and test again. I deliberately **unroll to 16 bytes** — two `long`s — as the
common case, because almost every station name fits in sixteen bytes, so the loop handles the typical line
in one straight-line pass with no back-branch: a name of 1–8 bytes is resolved by the first word, a name of
9–16 by the second, and only a name longer than 16 falls into the rare scanning loop. When a word has no
delimiter I keep all eight of its bytes; when it does, I build `mask = ~((highBitMask >>> 7) − 1)` and clear
the bytes at and beyond the `';'` so the partial word holds only the name's tail. That 16-byte shape is
exactly what makes the branch behaviour predictable — the branch predictor learns "first word usually
suffices," and the straight-line path retires without a stall on the overwhelming majority of lines.

Let me run one short name through `readNext` to be sure the masking and the cursor advance are right,
because an off-by-one here is a silently corrupted key, not a crash. Take `"Rome;12.3"` with the cursor on
`'R'`. The first word's low bytes are `'R','o','m','e',';'`, and after XOR with the `';'` mask, byte 4
becomes `0x00`, so `highBitMask1` has just the high bit of byte 4 set, i.e. bit `8*4 + 7 = 39`. Then
`Long.numberOfTrailingZeros(highBitMask1) = 39`, `>> 3 = 4`, and `position1 = 1 + 4 = 5`. The mask is
`~((highBitMask1 >>> 7) − 1)`: `highBitMask1 >>> 7` moves that bit down to bit 32, minus one sets bits
`0..31`, and the complement clears bits `0..31` while setting `32..63`, giving `0xFFFFFFFF00000000`. So
`readBuffer1 = lastRead & ~mask1` keeps only the low four bytes — `"Rome"` — and clears the `';'` and
everything past it. That is exactly the name's bytes and nothing else, which is what I want both to hash and
to compare. Since `position1 = 5 ≠ 0` the name terminated in the first word, so I fold the hash, zero
`readBuffer2`, and advance `ptr += 5`. Where does that land? `'R'(0) 'o'(1) 'm'(2) 'e'(3) ';'(4) '1'(5)`, so
`ptr + 5` sits on the `'1'` — the first byte of the temperature. The scan extracted the key and left the
cursor precisely at the value, with no separate step to skip the delimiter.

I fold the hash *as I scan*: `hash ^= readBuffer1; hash ^= readBuffer2`, XOR-ing in the (masked) name words
themselves, then a final avalanche mix `hash ^= hash >> 32` for entropy. This is a different trade from the
previous rung's fixed first-4/last-4 FxHash. There, the SIMD scan skipped the bytes, so I had to load them
separately to hash. Here, the SWAR scan already *has* both name words sitting in registers as `readBuffer1`
and `readBuffer2` — I loaded them to find the delimiter — so folding them into the hash is nearly free and
needs no second read of the name. The whole key is hashed from the exact bytes I am already holding.

The deeper move needs `Unsafe`, and it is where the biggest single step in my own optimization log lives. The
`MemorySegment`/buffer indirection still carries per-access bounds checks; I want to drop them entirely. The
file is mmap'd, so I take its raw base address — `fileChannel.map(...).address()` — and walk it with
`UNSAFE.getLong(ptr)`, which fetches 8 bytes from an arbitrary address with no bounds check and no object
header. That is the cheapest possible read, and it is what turns the 16-byte SWAR scan into a couple of
machine loads. My in-file changelog records the descent this produced on the development machine: the
initial submission at 62000 ms, down to 6500 once mmap'd, 4200 with the custom map, 3900 with SWAR token
checks, and then the single largest cut — 1900 ms — the moment raw `Unsafe` memory access replaced the
bounds-checked buffer, bottoming out around 1200 ms at the dev-machine floor. That log is the record of this
exact philosophy paying off in steady increments rather than one lucky trick, and the Unsafe step is the one
that jumps.

For the map I keep open addressing, but I push the whole entry into a **flyweight `byte[]`** with a
hand-laid-out memory format and read and write its fields with `Unsafe` at fixed offsets: a `long` sum, an
`int` min, an `int` max, an `int` count, a length byte, then the name bytes. So a station's aggregate is one
flat `byte[]` with no boxing and no per-field objects, and `updateEntry` is four `Unsafe` field
read/writes. This layout also finally spends the exact-integer-sum lever I flagged two rungs ago: the
temperature I feed in is the scaled integer straight from the parser, `min` and `max` are `int`s holding
scaled values in `[−999, 999]`, and `sum` is a `long`. A billion scaled values of up to 999 sum to at most
~10^12, which is nowhere near a `long`'s ceiling, so the aggregate is now exact — the floating-point drift
from summing a billion tenths into a `double` is gone entirely, and I have traded it for arithmetic that is
both faster and more correct. Let me size that to confirm it stays cache-resident: the fixed fields are 8 + 4 + 4 + 4 + 1 =
21 bytes plus a ~7-byte name and the array header, so roughly 40–45 bytes per entry; across 413 distinct
stations that is on the order of 18 KB of live entry storage, which sits comfortably in L2 and much of it in
L1, so the aggregate updates hit cache rather than memory. I separate the hash table's *index* from the
entry storage and pre-construct a pool of small entries to keep them close in memory, and I size the table
large so the load factor stays tiny; collisions linear-probe `(index + 1) & TABLE_MASK`. The key *compare*
on a probe is itself SWAR: compare the stored name to the scanned `readBuffer1`/`readBuffer2` words 8 bytes
at a time with `UNSAFE.getLong` on both sides — `value != UNSAFE.getLong(object2, address2)` — so a 16-byte
name is two `long` compares, no `Arrays.equals`, no vector. Every per-line operation is now the same kind of
math over the same kind of data.

The reason to separate the table's *index* from the fat entries is cache behaviour. If each table slot held
the whole ~40-byte flyweight inline,
then every probe step — even a probe that misses — would drag a 40-byte entry through the cache hierarchy,
and a probe chain would touch several cache lines. By keeping the slots slim (an index/hash into a separate
entry pool) the probe walks a dense, contiguous array where several slots share a cache line, so the common
"is this slot the right key" checks stay in L1 and I only pull the fat entry once I have a hit worth
updating. On this key set the probe is almost always length one, but making the *unit of probing* small is
what keeps even that one access cheap, and it is a distinct win from the flyweight layout itself — it
shrinks the bytes each probe step drags through cache even when the entry format is left unchanged.

The merge deserves one line of justification too, because I chose a `ConcurrentHashMap` where the previous
rungs hand-merged. With only a few hundred distinct stations and eight threads each finishing with its own
`Result[]`, the total merge is a few thousand `putIfAbsent`-plus-accumulate operations at the very end — far
too little work to matter against a two-second parse, and far too little contention to justify a lock-free
custom structure. So I spend simplicity there rather than cleverness: the hot loop is per-thread and
lock-free, and the trivial tail uses the standard concurrent map, then sorts for output.

The number parse stays the branchless SWAR magic-constant trick, and there is one micro-optimization in it I
want to verify rather than copy on faith: the shift is computed as `min28 = dotPosition ^ 0b11100` instead of
`28 − dotPosition`. Those are only equal for the specific values `dotPosition` can take, so let me check.
The dot position (in bits) is one of 12, 20, or 28. `12 ^ 28`: `12 = 0b01100`, `28 = 0b11100`, XOR = `0b10000
= 16`, and `28 − 12 = 16` — equal. `20 ^ 28`: `20 = 0b10100`, `28 = 0b11100`, XOR = `0b01000 = 8`, and `28 −
20 = 8` — equal. `28 ^ 28 = 0`, and `28 − 28 = 0` — equal. So on exactly the three values the data can
produce, XOR with `0b11100` computes the subtraction, and XOR is one cheap ALU op with no dependency on a
subtract's constant. The rest of the parse is as before: find the `'.'` via `~numberBytes & 0x10101000`,
recover the sign from `(invNumberBytes << 59) >> 63`, align the digits with that shift, multiply by
`100*0x1000000 + 10*0x10000 + 1` to sum them into the right bits, mask `& 0x3FF` for the scaled integer, and
sign-apply with `(absValue + signed) ^ signed`. It is the merykitty parse, which I credit in the comment; it
was already the right tool, and now the *whole* line — scan, hash, key compare, and number — is uniform
branchless `long` arithmetic over raw addresses.

One more idea the notes flag that I will take: the subprocess trick. Unmapping ~12 GB at process exit is
slow and serialized — the kernel has to tear down the mapping's page-table entries, and on a job that
finishes in a couple of seconds that teardown, tens to over a hundred milliseconds, is a measurable few
percent of the wall-clock. If I spawn a child worker that does the actual work and streams its output, and
let the parent return as soon as the child has printed the answer, the OS reclaims the mapping *after* I
have already emitted the last output byte — so the unmap latency falls outside the measured window entirely.
The threads themselves are bare `Thread[]`, one per processor, each running `processMemoryArea` over its
address range and merging into a shared `ConcurrentHashMap` at the end; with only a few hundred keys the
merge contention is negligible, and a final sort produces the output.

So this rung is the same parallel, byte-keyed, scaled-integer structure as before, but every per-line
operation is now branchless SWAR over raw `Unsafe` addresses instead of Vector-API lane ops over a
`MemorySegment`: a 16-byte unrolled delimiter scan by the high-bit identity, a hash folded from the scanned
words, a SWAR key compare, the SWAR number parse, and flyweight `byte[]` entries. Let me set the bar with the
same per-row budgeting I have been using. The standing time is 03.210, ~25.7 ns/row. Removing the ~0.6 s of
branch mispredictions I priced (via the branchless 16-byte handling), the per-access bounds checks (via
`Unsafe`), and the incubator's uneven lane codegen (via plain `long` math) should each shave a slice; if
they compound to roughly a third off, `3.210 × (16/25.7) ≈ 2.0` seconds is the landing I am aiming at, plus
whatever the subprocess unmap trick removes from the tail. The falsifiable form: if branches and incubator
variance really were what remained, I should see ~2 seconds; if I barely move, then the true residual was
memory-latency on the map probe — a stall no amount of branch-trimming touches — and that becomes the next
thing to chase. The cost of this bet is honest and sharp: SWAR over-reads up to 16 bytes past a name, so I
must guarantee slack past the last line, which the newline-aligned segment boundaries and the mmap'd tail
provide; and `Unsafe` is unsupported and unforgiving — one wrong offset is a segfault, not an exception. But
if the diagnosis holds that branches and incubator variance, not scan width, are what is left, the
branchless-SWAR philosophy is the sharper instrument here, and it should come in under 03.210. The full
module is in the answer.
