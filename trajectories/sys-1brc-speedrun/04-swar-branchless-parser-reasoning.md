03.210, down from 05.979 — vectorizing the delimiter search and key compare bought roughly the 1.8× I
hoped for, which confirms the scan really was the hot spot. But the Vector API got me there by reaching for
the incubating SIMD module, and when I stare at where the remaining time goes I start to doubt that a wider
vector is the right lever for the *next* step. Two things bother me. First, a station name here averages
around seven bytes and rarely exceeds sixteen; loading a 32-byte vector to find a `';'` that's almost
always in the first eight bytes is doing a lot of lane work for a short payload, and the incubator's
codegen quality is uneven — sometimes the vector path doesn't fold into the tight machine code I'd write by
hand. Second, every line still has *branches* I haven't killed: the long-name fallback, the tail handling,
the `node == null` versus key-mismatch split, the per-byte residue. A billion lines means a billion trips
through those branches, and branch mispredictions on data-dependent name lengths are exactly the kind of
cost that doesn't show up in a flat instruction count but dominates a tight loop.

So let me try a different philosophy for the same data-parallelism, one that needs no incubator module and
gives the JIT plain integer arithmetic it compiles predictably: do the SIMD *within a 64-bit register*.
The merykitty rung already used this — SWAR — for the *number* parse, where it was clearly the elegant
choice. The bet now is that SWAR is also the better tool for the *name* scan and the whole hot loop, not
just the number, precisely because it's branchless plain-`long` math the compiler turns into a handful of
predictable instructions, and because the data is short enough that one or two 8-byte words cover almost
every name.

The core SWAR identity for "is there a target byte in this 8-byte word" is the classic one: XOR the word
with a mask of the target byte repeated eight times (for `';'` that's `0x3B3B…3B`), then
`(x - 0x0101…01) & ~x & 0x8080…80` lights up the high bit of every byte position that was zero after the
XOR — i.e. every position that held the target. `Long.numberOfTrailingZeros` of that, shifted right by 3,
gives the byte index of the first delimiter. No vector, no loop, no incubator. So I read the name 8 bytes
at a time as a `long`, test for `';'` with that identity, and if it's not in the first word I read the next
8 and test again. I deliberately unroll to **16 bytes** — two `long`s — as the common case, because almost
every station name fits in sixteen bytes, so the loop handles the typical line in one straight-line pass
with no back-branch. When a word has no delimiter I keep its bytes whole; when it does, I mask off the
bytes at and beyond the `';'` so the partial word holds only the name's tail. This is the rewrite the
optimization notes mark as "rewrite loop for 16 b" — the 16-byte shape is what makes the branch behavior
predictable.

I fold the hash *as I scan*: `hash ^= readBuffer1; hash ^= readBuffer2`, XOR-ing in the (masked) name words
themselves, then a final avalanche mix `hash ^= hash >> 17` for entropy. The hash is therefore computed
from the exact bytes I'm already loading to compare — no second pass over the name, no separate
first-4/last-4 read — which is a touch cheaper than the FxHash-from-offsets approach and keeps everything
in the words I already have in registers.

Now the deeper move, the one that needs `Unsafe`: get rid of the `MemorySegment`/buffer indirection and
the per-access bounds checks entirely. The file is mmap'd; I take its raw base address
(`fileChannel.map(...).address()`) and walk it with `UNSAFE.getLong(ptr)`, which fetches 8 bytes from an
arbitrary address with no bounds check and no object header. That's the cheapest possible read, and it's
what turns the 16-byte SWAR scan into a couple of machine loads. The notes log this as "added unsafe memory
access: 1900 ms" — a big single step — and a string of follow-ons ("storing only the address",
"improved layout/predictability") that are all about keeping the hot data small and the access pattern
linear so the prefetcher and cache stay happy.

For the map I keep open addressing, but I push the whole entry into a **flyweight `byte[]`** with a
hand-laid-out memory format and read/write its fields with `Unsafe` at fixed offsets: a `long` sum, an
`int` min, an `int` max, an `int` count, a length byte, then the name bytes. So a station's aggregate is
one flat `byte[]` with no boxing and no per-field objects, and `updateEntry` is four `Unsafe` field
read/writes. I separate the hash table's *index* from the entry storage ("separate hash from entries",
"separate hash from entries: 1550 ms"), pre-construct a pool of small entries to keep them close in memory
("changed to flyweight byte[]"), and size the table at `1<<19` buckets so the load factor stays tiny.
Collisions linear-probe `(index+1) & TABLE_MASK`. The key *compare* on a probe is itself SWAR: compare the
stored name to the scanned `readBuffer1`/`readBuffer2` words 8 bytes at a time with `UNSAFE.getLong` on
both sides — `value != UNSAFE.getLong(entry, addr)` — so a 16-byte name is two long compares, no
`Arrays.equals`, no vector.

The number parse I keep as the same branchless SWAR magic-constant trick — load the value word, find the
`'.'` via `~numberBytes & 0x10101000`, recover the sign from `(invNumberBytes << 59) >> 63`, align the
digits and multiply by `100*0x1000000 + 10*0x10000 + 1` to sum them into the right bits, `& 0x3FF` for the
scaled integer, sign-apply with `(absValue + signed) ^ signed`. It's the merykitty parse, which I credit in
the comment; it was already the right tool, and now the *whole* line — scan, hash, key compare, and number
— is uniform branchless `long` arithmetic over raw addresses.

One more thing the notes flag that I'll take: the subprocess trick. Unmapping ~12 GB at process exit is
slow and serialized; if I spawn a child worker that does the actual work and let the parent return as soon
as the child has streamed its output, the OS reclaims the mapping after I've already printed the answer, so
the unmap latency falls off the measured wall-clock. The threads are bare `Thread[]`, one per processor,
each running `processMemoryArea` over its address range and merging into a shared `ConcurrentHashMap` at the
end (a few hundred keys, so contention is negligible), then sorted for output.

So this rung is the same parallel, byte-keyed, scaled-integer structure as before, but every per-line
operation is now branchless SWAR over raw `Unsafe` addresses instead of Vector-API lane ops over a
`MemorySegment`: 16-byte unrolled delimiter scan by the high-bit identity, hash folded from the scanned
words, SWAR key compare, SWAR number parse, flyweight `byte[]` entries. It's a genuinely different bet from
the previous rung — hand-rolled register-width SIMD that the JIT compiles to tight predictable code, versus
explicit wide-vector SIMD from the incubator — and the optimization log is the record of that bet paying
off in small steady increments (62000 ms → 6500 ms once mmap'd → 4200 with the custom map → 3900 with SWAR
token checks → 1900 with Unsafe → 1200 at the floor on the dev machine).

The bar is 03.210. The bet is that on short names the branch-free 16-byte SWAR loop over raw addresses
beats the wide-vector path — fewer branches, no incubator-codegen variance, less lane work for a 7-byte
payload — and that the flyweight `Unsafe` entries and the subprocess unmap trick shave the rest, landing
around two seconds. The hedge is honest: SWAR over-reads up to 16 bytes, so I must guarantee slack past the
last line (the segment boundaries and the mmap'd tail give it), and `Unsafe` is unsupported and unforgiving
— one wrong offset is a segfault, not an exception. But if the diagnosis is right that branches and
incubator variance, not raw scan width, are what's left to cut, then the branchless-SWAR philosophy is the
sharper instrument here, and it should come in under 03.210. The full module is in the answer.
