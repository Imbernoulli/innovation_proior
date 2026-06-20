**Problem (from step 3).** The Vector-API rung (03.210 s) confirmed the scan was the hot spot, but it
leans on the incubating SIMD module (uneven codegen) and loads a 32-byte vector for names that are usually
~7 bytes, and the hot loop still has data-dependent branches (long-name fallback, tail, null-vs-mismatch)
that mispredict a billion times.

**Key idea.** Do the data-parallelism *within a 64-bit register* — SWAR — for the whole hot loop, not
just the number, so it's branchless plain-`long` math the JIT compiles predictably, with no incubator.
Read the name 8 bytes at a time as a `long` and find `';'` with the high-bit identity
`(x − 0x01..01) & ~x & 0x80..80` after XOR-ing in `0x3B..3B`; unroll to **16 bytes** (the common name
length) so the typical line is one straight-line pass. Fold the hash from the scanned words themselves.
Walk the mmap'd file by **raw `Unsafe` addresses** (`getLong(ptr)`, no bounds check). Store each station's
aggregate as a **flyweight `byte[]`** with `Unsafe` field accesses at fixed offsets, in an open-addressing
table; compare keys with SWAR `long` compares. Keep the branchless SWAR magic-constant number parse
(merykitty). Spawn a child worker so the slow ~12 GB unmap happens after output is already printed.

**Why it works.** On short names a 16-byte branch-free SWAR scan over raw addresses beats a wide vector:
fewer branches, no incubator-codegen variance, and far less lane work per 7-byte payload. The hash is
computed from words already in registers (no second pass), key compares are two `long` compares for a
16-byte name, entries are flat `byte[]`s with no boxing, and the subprocess trick removes the serial unmap
from the measured wall-clock. The author's in-file optimization log records the descent (62000 ms → mmap
6500 → custom map 4200 → SWAR token checks 3900 → Unsafe 1900 → ~1200 at the dev-machine floor).

**Change / code.** The 16-byte SWAR delimiter scan with in-line hashing, the SWAR number parse, and the
flyweight-`byte[]` update.

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

// branchless SWAR parse (idea: merykitty) -> temperature*10
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

**Target.** Beat 00:03.210 — into the ~2-second range — by replacing wide-vector lane ops with a
branchless 16-byte SWAR loop over raw `Unsafe` addresses, flyweight entries, and the subprocess unmap
trick.
