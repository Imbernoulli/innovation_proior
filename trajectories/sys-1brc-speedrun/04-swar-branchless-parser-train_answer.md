The Vector-API rung came in at **00:03.210**, roughly the 1.8× I hoped for, which confirms the scan really was the hot spot. But it got there by reaching for the incubating SIMD module, and when I stare at where the remaining time goes I doubt a *wider* vector is the right lever for the next step. Two things bother me. A station name here averages ~7 bytes and rarely exceeds 16, so loading a 32-byte vector to find a `';'` that's almost always in the first eight is a lot of lane work for a short payload — and the incubator's codegen quality is uneven, sometimes failing to fold the vector path into the tight machine code I'd write by hand. And every line still has *branches* I haven't killed: the long-name fallback, the tail handling, the `node == null` versus key-mismatch split. A billion lines means a billion trips through those branches, and mispredictions on data-dependent name lengths dominate a tight loop in a way a flat instruction count never shows.

So I try a different philosophy for the same data-parallelism, one needing no incubator module and giving the JIT plain integer arithmetic it compiles predictably: do the SIMD *within a 64-bit register*. The method is the **SWAR branchless parser over raw `Unsafe` addresses with flyweight-`byte[]` map entries**. The previous rung already used SWAR for the *number*; the bet now is that SWAR is the better tool for the *name* scan and the whole hot loop too, because it is branchless plain-`long` math the compiler turns into a handful of predictable instructions, and the data is short enough that one or two 8-byte words cover almost every name.

The core SWAR identity for "is there a target byte in this 8-byte word" is the classic one: XOR the word with the target byte repeated eight times (for `';'` that is `0x3B3B…3B`), then $(x - \mathtt{0x0101…01})\ \&\ \lnot x\ \&\ \mathtt{0x8080…80}$ lights up the high bit of every byte position that became zero after the XOR — i.e. every position that held the target. $\text{numberOfTrailingZeros}$ of that, shifted right by 3, gives the byte index of the first delimiter — no vector, no loop, no incubator. I read the name 8 bytes at a time as a `long`, test for `';'` with that identity, and if it's not in the first word read the next 8 and test again. I deliberately **unroll to 16 bytes** — two `long`s — as the common case, since almost every name fits in sixteen, so the loop handles the typical line in one straight-line pass with no back-branch. When a word has no delimiter I keep its bytes whole; when it does, I build `mask = ~((highBitMask >>> 7) - 1)` and clear the bytes at and beyond the `';'` so the partial word holds only the name's tail. The 16-byte shape is exactly what makes the branch behavior predictable.

I fold the hash *as I scan*: `hash ^= readBuffer1; hash ^= readBuffer2`, XOR-ing in the (masked) name words themselves, then a final avalanche mix `hash ^= hash >> 32` for entropy. The hash is therefore computed from the exact bytes already loaded to compare — no second pass, no separate first-4/last-4 read — which is a touch cheaper than FxHash-from-offsets and keeps everything in registers.

The deeper move needs `Unsafe`: drop the `MemorySegment`/buffer indirection and the per-access bounds checks entirely. The file is mmap'd; I take its raw base address (`fileChannel.map(...).address()`) and walk it with `UNSAFE.getLong(ptr)`, which fetches 8 bytes from an arbitrary address with no bounds check and no object header — the cheapest possible read, and what turns the 16-byte SWAR scan into a couple of machine loads. (This is the single biggest step in my own optimization log, dropping the dev-machine time to 1900 ms, on a descent of 62000 → mmap 6500 → custom map 4200 → SWAR token checks 3900 → Unsafe 1900 → ~1200 at the floor.)

For the map I keep open addressing but push the whole entry into a **flyweight `byte[]`** with a hand-laid-out memory format and read/write its fields with `Unsafe` at fixed offsets: a `long` sum, an `int` min, an `int` max, an `int` count, a length byte, then the name bytes. A station's aggregate is one flat `byte[]` with no boxing and no per-field objects, and `updateEntry` is four `Unsafe` field read/writes. I separate the hash table's *index* from the entry storage, pre-construct a pool of small entries to keep them close in memory, and size the table large so the load factor stays tiny; collisions linear-probe `(index+1) & TABLE_MASK`. The key *compare* on a probe is itself SWAR: compare the stored name to the scanned `readBuffer1`/`readBuffer2` words 8 bytes at a time with `UNSAFE.getLong` on both sides, so a 16-byte name is two `long` compares — no `Arrays.equals`, no vector.

The number parse stays the branchless SWAR magic-constant trick: load the value word, find the `'.'` via $\lnot\text{numberBytes}\ \&\ \mathtt{0x10101000}$, recover the sign from $(\text{invNumberBytes} \ll 59) \gg 63$, align the digits with a shift of $28 - \text{dotPosition}$, multiply by $100\cdot\mathtt{0x1000000} + 10\cdot\mathtt{0x10000} + 1$ to sum them into bits 32–41, mask `& 0x3FF` for the scaled integer, and sign-apply with $(\text{absValue} + \text{signed}) \oplus \text{signed}$. The whole line — scan, hash, key compare, and number — is now uniform branchless `long` arithmetic over raw addresses. One more idea I take: the **subprocess trick**. Unmapping ~12 GB at exit is slow and serialized, so I spawn a child worker that does the work and streams its output, and let the parent return before the OS finishes reclaiming the mapping — the unmap latency falls outside the measured wall-clock. Threads are bare `Thread[]`, one per processor, each running `processMemoryArea` over its address range and merging into a shared `ConcurrentHashMap` (a few hundred keys, negligible contention), then sorted for output.

This is a genuinely different bet from the previous rung — hand-rolled register-width SIMD the JIT compiles to tight predictable code, versus explicit wide-vector SIMD from the incubator. On short names the branch-free 16-byte SWAR loop over raw addresses should beat the wide-vector path — fewer branches, no incubator-codegen variance, far less lane work for a 7-byte payload — and the flyweight `Unsafe` entries plus the subprocess unmap shave the rest, landing around two seconds. The cost of the bet: SWAR over-reads up to 16 bytes, so I must guarantee slack past the last line (the segment boundaries and mmap'd tail give it), and `Unsafe` is unforgiving — one wrong offset is a segfault, not an exception.

```java
private static final long DELIMITER_MASK = 0x3B3B3B3B3B3B3B3BL;   // ';' x8

// scan the name 8 bytes at a time, unrolled to 16; mask off bytes at/after ';', fold the hash
private boolean readNext() {
    long lastRead = UNSAFE.getLong(ptr);
    entryLength += 16;
    long comparisonResult1 = (lastRead ^ DELIMITER_MASK);
    long highBitMask1 = (comparisonResult1 - 0x0101010101010101L) & (~comparisonResult1 & 0x8080808080808080L);
    boolean noContent1 = highBitMask1 == 0;
    long mask1 = noContent1 ? 0 : ~((highBitMask1 >>> 7) - 1);
    int position1 = noContent1 ? 0 : 1 + (Long.numberOfTrailingZeros(highBitMask1) >> 3);
    readBuffer1 = lastRead & ~mask1;
    hash ^= readBuffer1;
    if (position1 != 0) { hash ^= hash >> 32; readBuffer2 = 0; ptr += position1; return false; }

    lastRead = UNSAFE.getLong(ptr + 8);                          // second 8-byte word
    long comparisonResult2 = (lastRead ^ DELIMITER_MASK);
    long highBitMask2 = (comparisonResult2 - 0x0101010101010101L) & (~comparisonResult2 & 0x8080808080808080L);
    boolean noContent2 = highBitMask2 == 0;
    long mask2 = noContent2 ? 0 : ~((highBitMask2 >>> 7) - 1);
    int position2 = noContent2 ? 0 : 1 + (Long.numberOfTrailingZeros(highBitMask2) >> 3);
    readBuffer2 = lastRead & ~mask2;
    hash ^= readBuffer2;
    hash ^= hash >> 32;
    if (position2 != 0) { ptr += position2 + 8; return false; }
    ptr += 16; return true;                                      // name longer than 16 -> keep scanning
}

private static final long DOT_BITS = 0x10101000;
private static final long MAGIC_MULTIPLIER = (100 * 0x1000000 + 10 * 0x10000 + 1);

// branchless SWAR parse -> temperature*10
private int readTemperature() {
    final long numberBytes = UNSAFE.getLong(ptr);
    final long invNumberBytes = ~numberBytes;
    final int dotPosition = Long.numberOfTrailingZeros(invNumberBytes & DOT_BITS);
    final long signed = (invNumberBytes << 59) >> 63;
    final int min28 = (dotPosition ^ 0b11100);
    final long minusFilter = ~(signed & 0xFF);
    final long digits = ((numberBytes & minusFilter) << min28) & 0x0F000F0F00L;
    ptr += (dotPosition >> 3) + 3;
    final long absValue = ((digits * MAGIC_MULTIPLIER) >>> 32) & 0x3FF;
    return (int) ((absValue + signed) ^ signed);
}

// flyweight byte[] entry: long sum, int min, int max, int count, byte len, byte[] name — Unsafe fields
public static void updateEntry(final byte[] entry, final int temp) {
    int entryMin = UNSAFE.getInt(entry, ENTRY_MIN);
    int entryMax = UNSAFE.getInt(entry, ENTRY_MAX);
    long entrySum = UNSAFE.getLong(entry, ENTRY_SUM) + temp;
    int entryCount = UNSAFE.getInt(entry, ENTRY_COUNT) + 1;
    if (temp < entryMin)      UNSAFE.putInt(entry, ENTRY_MIN, temp);
    else if (temp > entryMax) UNSAFE.putInt(entry, ENTRY_MAX, temp);
    UNSAFE.putInt(entry, ENTRY_COUNT, entryCount);
    UNSAFE.putLong(entry, ENTRY_SUM, entrySum);
}

// SWAR key compare on a probe: two long compares for a 16-byte name
private static boolean compare(final long value1, final Object object2, final long address2) {
    return value1 != UNSAFE.getLong(object2, address2);
}
```
