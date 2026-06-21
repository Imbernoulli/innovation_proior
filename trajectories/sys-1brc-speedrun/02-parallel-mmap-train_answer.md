The baseline came back at **04:49.679**, and the profile is exactly what the code predicted: one core busy, seven idle, and a torrent of short-lived allocation underneath. That hands me two large multipliers sitting untouched, and they compound, so I take both at once. The machine has eight cores and the baseline uses one — an up-to-8× ceiling left on the table. And every line pays for a decoded `String`, a `String[]` from `split`, and a boxed `Double`, billions of objects whose only purpose is to be discarded, with the parse doing far more work than the fixed data shape requires. The baseline's time is dominated by per-row overhead, not by the trivial min/max/sum arithmetic, so the move is to attack the per-row overhead and the single-threadedness together.

The method is **parallel memory-mapped segments with a custom byte-keyed open-addressing map** (the spullara approach). Threading sets the shape of everything else, so I start there. To use eight cores I give each one an independent slice of the file, let it build its own aggregate, and merge the slices at the end — which is sound precisely because $(\min,\max,\text{sum},\text{count})$ is associative, the same property the baseline's combiner already relied on. I split into `numberOfSegments = availableProcessors()` segments. The subtlety is that I cannot cut on raw byte offsets: a blind cut lands mid-line, so one segment sees a truncated `station;temp` while its neighbour sees the other half. The fix is to choose each boundary by starting from the nominal offset `i * segmentSize` and scanning forward to the next `'\n'`, so every segment begins exactly at a line start and ends at a line end; the previous segment's end *is* this segment's start, so there is no overlap and no gap. (Below a 1 MB threshold the split is pointless, so I use one segment.) A `parallel()` stream over the segments fans them across the cores.

The bigger structural change is the per-row overhead. The baseline reads *decoded lines*; I stop decoding entirely and work on raw bytes. `FileChannel.map(READ_ONLY, start, len)` maps a segment straight into memory as a `MappedByteBuffer`, so I read bytes out of the page cache with no `read()` syscalls, no per-line `String` allocation, and no UTF-8 decode of all 12 GB. The inner loop becomes a pointer marching through mapped memory: from the current position, walk byte by byte accumulating the station name into a scratch `byte[100]` until I hit `';'`, then parse the temperature, then advance past the newline.

Three pieces of that loop carry the win. First, the **number parse**. `Double.parseDouble` is a general float parser, but the data is always one of `D.D`, `DD.D`, `-D.D`, `-DD.D` — one optional minus, one or two integer digits, a dot, exactly one fractional digit. I never need a `double` during parsing; I read it as a *scaled integer*. I peek at the bytes directly: if the first byte is `'-'`, set `negative = -1` and step past it; then if the byte two positions ahead is `'.'`, the integer part is one digit and the value is `d0*10 + d1`; otherwise it is two digits and the value is `d0*100 + d1*10 + d2`, each digit being `byte - '0'`. The whole parse is a handful of integer adds and multiplies with no function call and no floating point, and I divide by ten only when I finally store `temp / 10.0`.

Second, and where the real allocation savings live, the **per-station lookup**. The baseline hashes a `String` key into a `HashMap` per row; I never want to materialize a `String` per row at all. The station name is a range of bytes in my scratch buffer, so I look it up *as bytes* in a small custom map specialized to this task: a flat open-addressing table. I accumulate the key's hash *as I scan it to the delimiter* — `hash = 31*hash + b`, the same polynomial `String.hashCode` uses, computed for free during the delimiter walk — and use `hash & (MAPSIZE - 1)` to pick a slot in a power-of-two array. `MAPSIZE = 1<<17 = 128K` slots sits comfortably above the 10,000-station ceiling, so the table stays sparse and probe chains short. On a hit I update the slot's `Result` (min/max/sum/count) in place. On a miss — a `null` slot, or a slot whose stored key bytes don't match — I linear-probe forward, `slot = (slot+1) & (MAPSIZE-1)`, until I find the match or an empty slot, then create the `Result` and copy the name bytes in *once*. Key comparison is `Arrays.equals` over the byte ranges, comparing bytes directly with no decode. This is the move that kills the billions of allocations: a `String` is created exactly once per *distinct* station (a few hundred), at merge time, instead of once per *row*.

Open addressing with linear probing is the right structure here rather than chaining: there are no per-node list objects, everything lives in two flat arrays (`Result[] slots` and `byte[][] keys`) so the table is cache-friendly, and with the table 100× larger than the key set the load factor is tiny and collisions rare. Each segment-thread gets its own `ByteArrayToResultMap`, so there is zero contention in the hot loop — no locks, no shared state. Only at the very end do I drain each thread's map into a list of `(byte[] key, Result)` entries, decode each key to a `String` exactly once, and `Collectors.toMap` them into a `TreeMap` with a merge function that combines stations colliding across threads. The `TreeMap` keeps the output sorted, same as the baseline. Every layer the baseline paid for per row — the decoded `String`, the `split` array, the boxed `Double`, the `String` hash, the single core — is gone or amortized, so a five-minute single-core job should fall into single-digit seconds.

```java
public static void main(String[] args) throws IOException, ExecutionException, InterruptedException {
    var filename = args.length == 0 ? FILE : args[0];
    var file = new File(filename);

    var resultsMap = getFileSegments(file).stream().map(segment -> {
        var resultMap = new ByteArrayToResultMap();
        long segmentEnd = segment.end();
        try (var fileChannel = (FileChannel) Files.newByteChannel(Path.of(filename), StandardOpenOption.READ)) {
            var bb = fileChannel.map(FileChannel.MapMode.READ_ONLY, segment.start(), segmentEnd - segment.start());
            var buffer = new byte[100];                 // up to 100 chars for a city name
            int startLine;
            int limit = bb.limit();
            while ((startLine = bb.position()) < limit) {
                int currentPosition = startLine;
                byte b;
                int offset = 0, hash = 0;
                while (currentPosition != segmentEnd && (b = bb.get(currentPosition++)) != ';') {
                    buffer[offset++] = b;
                    hash = 31 * hash + b;               // hash the key while scanning to ';'
                }
                int temp, negative = 1;
                if (bb.get(currentPosition) == '-') { negative = -1; currentPosition++; }
                if (bb.get(currentPosition + 1) == '.') {   // D.D  -> one integer digit
                    temp = negative * ((bb.get(currentPosition) - '0') * 10 + (bb.get(currentPosition + 2) - '0'));
                    currentPosition += 3;
                } else {                                    // DD.D -> two integer digits
                    temp = negative * ((bb.get(currentPosition) - '0') * 100 + ((bb.get(currentPosition + 1) - '0') * 10 + (bb.get(currentPosition + 3) - '0')));
                    currentPosition += 4;
                }
                if (bb.get(currentPosition) == '\r') currentPosition++;
                currentPosition++;
                resultMap.putOrMerge(buffer, 0, offset, temp / 10.0, hash);
                bb.position(currentPosition);
            }
            return resultMap;
        } catch (IOException e) { throw new RuntimeException(e); }
    }).parallel().flatMap(partition -> partition.getAll().stream())
            .collect(Collectors.toMap(e -> new String(e.key()), Entry::value, CalculateAverage_spullara::merge, TreeMap::new));

    System.out.println(resultsMap);
}

// newline-aligned segment boundaries, one segment per core
private static List<FileSegment> getFileSegments(File file) throws IOException {
    int numberOfSegments = Runtime.getRuntime().availableProcessors();
    long fileSize = file.length();
    long segmentSize = fileSize / numberOfSegments;
    List<FileSegment> segments = new ArrayList<>(numberOfSegments);
    if (segmentSize < 1_000_000) { segments.add(new FileSegment(0, fileSize)); return segments; }
    try (RandomAccessFile randomAccessFile = new RandomAccessFile(file, "r")) {
        for (int i = 0; i < numberOfSegments; i++) {
            long segStart = i * segmentSize;
            long segEnd = (i == numberOfSegments - 1) ? fileSize : segStart + segmentSize;
            segStart = findSegment(i, 0, randomAccessFile, segStart, segEnd);
            segEnd = findSegment(i, numberOfSegments - 1, randomAccessFile, segEnd, fileSize);
            segments.add(new FileSegment(segStart, segEnd));
        }
    }
    return segments;
}

// open-addressing, byte-keyed map; key compared as bytes, String made once per distinct station
class ByteArrayToResultMap {
    public static final int MAPSIZE = 1024 * 128;       // 128K slots >> 10K key ceiling
    Result[] slots = new Result[MAPSIZE];
    byte[][] keys = new byte[MAPSIZE][];

    public void putOrMerge(byte[] key, int offset, int size, double temp, int hash) {
        int slot = hash & (slots.length - 1);
        var slotValue = slots[slot];
        while (slotValue != null && (keys[slot].length != size || !Arrays.equals(keys[slot], 0, size, key, offset, size))) {
            slot = (slot + 1) & (slots.length - 1);     // linear probe
            slotValue = slots[slot];
        }
        Result value = slotValue;
        if (value == null) {
            slots[slot] = new Result(temp);
            byte[] bytes = new byte[size];
            System.arraycopy(key, offset, bytes, 0, size);
            keys[slot] = bytes;
        } else {
            value.min = Math.min(value.min, temp);
            value.max = Math.max(value.max, temp);
            value.sum += temp;
            value.count += 1;
        }
    }
}
```
