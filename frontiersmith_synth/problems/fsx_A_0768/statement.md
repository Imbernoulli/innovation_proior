# Multi-Scale Motif Coder

## Story

You are given a symbol sequence `seq` of length `N`, drawn from a small alphabet.
Hidden inside the sequence are a handful of exact repeated motifs, planted at
**irregular, multi-scale spacing**: some motifs are short and their repeats sit close
together, other motifs are much longer and their repeats are scattered far apart, and
both families are interleaved with each other and with random background symbols.

Your job is to compress the sequence into a **dictionary** of reusable motifs plus a
**segmentation** of the sequence into literal runs and dictionary references, so as to
minimize the total encoded size.

## Input (public instance, one JSON object on stdin)

```json
{
  "name": "trace9101",
  "n": 220,
  "seq": [3, 17, 22, ...],
  "alphabet_size": 40,
  "bits_per_symbol": 6,
  "ptr_bits": 6,
  "dict_header_bits": 6,
  "max_dict_entries": 64,
  "max_motif_len": 40
}
```

- `seq`: a list of `n` integers, each in `[0, alphabet_size)`.
- `bits_per_symbol`, `ptr_bits`, `dict_header_bits`: the tunable costs used in the
  scoring formula below (read them from the instance -- don't assume fixed values).
- `max_dict_entries`, `max_motif_len`: hard caps on your output (see below).

## Output (one JSON object on stdout)

```json
{
  "dictionary": [[3, 17, 22, 9], [5, 5, 30, 30, 30, 8, 8]],
  "segments": [{"type": "lit", "len": 12}, {"type": "ref", "dict_idx": 0}, ...]
}
```

- `dictionary`: a list of at most `max_dict_entries` motifs, each a list of between 2
  and `max_motif_len` integers in `[0, alphabet_size)`.
- `segments`: a list that partitions `[0, n)` **in order, with no gaps or overlaps**.
  Each element is either `{"type": "lit", "len": L}` (the next `L` symbols of `seq` are
  encoded literally) or `{"type": "ref", "dict_idx": i}` (the next
  `len(dictionary[i])` symbols of `seq` must **exactly equal** `dictionary[i]`).

Any of the following makes the instance score `0.0`: the segments don't exactly cover
`[0, n)`; a `ref` segment's `dict_idx` is out of range or its content doesn't match
`seq` at that position; a dictionary entry has fewer than 2 or more than
`max_motif_len` symbols, or a symbol outside `[0, alphabet_size)`; more than
`max_dict_entries` declared entries; a crash, a timeout, or output that isn't the JSON
object above.

## Objective and scoring (deterministic)

Let `used` be the set of dictionary entries referenced by at least one `ref` segment.
The evaluator computes:

```
dict_cost = sum( dict_header_bits + bits_per_symbol * len(e)  for e in used )
seg_cost  = sum( bits_per_symbol * L   for each {"type": "lit", "len": L} )
          + sum( ptr_bits              for each {"type": "ref", ...} )
y_cand    = dict_cost + seg_cost
y_base    = bits_per_symbol * n          (the whole sequence as one literal run)
```

and normalizes (minimization):

```
r = clamp( 0.1 * y_base / max(y_cand, 1e-12), 0, 1 )
```

Reproducing the pure-literal baseline scores `~0.1`; doing worse scores below `0.1`;
genuine compression scores higher, capped at `1.0`. Your final score is the mean of `r`
over all instances.

## Notes on the cost model

A `ref` segment costs a **flat** `ptr_bits`, no matter how long the motif it points to
is -- so capturing a long repeat as one reference is very cheap. But every dictionary
entry you declare costs `dict_header_bits + bits_per_symbol * len(e)` **once**,
whether you reference it once or many times, so declaring motifs that end up barely
reused is a net loss. Some instances have their motif repeats clustered close
together; others scatter them across a wide, irregular span with unrelated content (and
a second, different-length motif family) in between -- a segmentation strategy that
only looks at a bounded local neighborhood around each position will miss those.

## Constraints

`n` up to a few hundred, `max_dict_entries <= 64`, `max_motif_len <= 40`. Your program
is run in an isolated subprocess and sees only the public instance above; it never
measures wall-clock time.
