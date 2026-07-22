# TIER: greedy
# The "obvious" first pass: classic fixed-WINDOW LZ77-style parsing. Scan left to right;
# at each position, search only the last WINDOW symbols already processed for the LONGEST
# matching substring, and if it clears MIN_MATCH, emit a reference (registering its content
# as a new dictionary entry, or reusing one already registered with the same content);
# otherwise extend the current literal run by one symbol. No global search, no notion of
# which length SCALES are structurally dominant, and -- critically -- a match can never
# reach further back than WINDOW symbols.
#
# This does fine when a motif's repeats happen to land close together, but it is BLIND to
# any repeat separated by more than WINDOW symbols: on the irregularly-spaced, multi-scale
# instances (long motifs recurring far apart, interleaved with a different-length short
# motif family) it re-encodes every out-of-reach repeat completely literally, leaving most
# of the achievable compression on the table. It also has no way to reason about whether a
# match is worth registering as a dictionary entry at all -- it takes the first match that
# clears MIN_MATCH regardless of expected reuse.
import sys, json

inst = json.load(sys.stdin)
seq = inst["seq"]
n = inst["n"]

WINDOW = 110
MIN_MATCH = 4
MAX_MATCH = 40

dictionary = []
content_to_idx = {}
segments = []
i = 0
lit_start = None


def flush_lit(end):
    global lit_start
    if lit_start is not None and lit_start < end:
        segments.append({"type": "lit", "len": end - lit_start})
    lit_start = None


while i < n:
    wstart = max(0, i - WINDOW)
    best_len = 0
    max_l = min(MAX_MATCH, n - i)
    for j in range(wstart, i):
        L = 0
        while L < max_l and j + L < i and seq[j + L] == seq[i + L]:
            L += 1
        if L >= MIN_MATCH and L > best_len:
            best_len = L
    if best_len >= MIN_MATCH:
        flush_lit(i)
        content = seq[i:i + best_len]
        key = tuple(content)
        if key in content_to_idx:
            idx = content_to_idx[key]
        else:
            idx = len(dictionary)
            dictionary.append(content)
            content_to_idx[key] = idx
        segments.append({"type": "ref", "dict_idx": idx})
        i += best_len
    else:
        if lit_start is None:
            lit_start = i
        i += 1
flush_lit(n)

print(json.dumps({"dictionary": dictionary, "segments": segments}))
