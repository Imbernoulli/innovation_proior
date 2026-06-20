**Problem (from baseline).** The streams baseline takes 04:49.679 because it runs on one core while seven
idle, and pays per-row overhead on every one of a billion lines: a decoded `String`, a `String[]` from
`split`, a boxed `Double` from `parseDouble`, and a `String`-keyed `HashMap` probe — billions of
throwaway objects, a full UTF-8 decode of ~12 GB, and a general-purpose float parser.

**Key idea.** Take both multipliers at once. (1) Split the file into one segment per core at
newline-aligned boundaries (seek to the nominal offset, scan forward to the next `'\n'`) and process the
segments with `parallel()`. (2) Inside each segment, stop decoding: `FileChannel.map` the segment and run
a raw byte-pointer loop — scan the name to `';'` while accumulating `hash = 31*hash + b`, parse the
temperature as a *scaled integer* (`D.D`/`DD.D`/`-D.D`/`-DD.D` → integer adds, no float), and update a
per-thread **open-addressing hash map keyed on the name's bytes** in place. Merge the per-thread maps into
a `TreeMap` at the end, decoding each distinct station to a `String` exactly once.

**Why it works.** Min/max/sum/count is associative, so per-segment aggregation parallelizes cleanly with a
final elementwise merge. Memory-mapping removes syscalls and the 12 GB decode; the scaled-integer parse
removes the float work the fixed `D.D` data never needed; the byte-keyed open-addressed table (128K slots
for ≤10K keys, so a tiny load factor and short linear-probe chains) removes the per-row `String`/box
allocation — a `String` is created once per *distinct station*, not once per *row*. Per-thread maps mean
no locks in the hot loop. Together these turn a five-minute single-core job into a single-digit-seconds
eight-core one.

**Change / code.** The mmap + parallel segment loop, the scaled-integer parse, and the byte-keyed
open-addressing map.

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

**Target.** Beat 04:49.679 by roughly two orders of magnitude — into single-digit seconds — by using all
eight cores and removing the per-row `String`/`split`/box/float overhead.
