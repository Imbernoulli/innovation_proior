import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_input(path):
    toks = open(path).read().split()
    it = iter(toks)
    k = int(next(it))
    sigma = int(next(it))
    Lmax = int(next(it))
    automata = []
    for _ in range(k):
        n = int(next(it))
        table = []
        for _l in range(sigma):
            row = [int(next(it)) for _ in range(n)]
            table.append(row)
        automata.append((n, table))
    return k, sigma, Lmax, automata


def apply_word(n, table, word_letters):
    """word_letters: list of ints. Returns size of final distinct-state set."""
    S = set(range(n))
    for l in word_letters:
        if len(S) == 1:
            # once collapsed to a single state it stays collapsed under any further
            # letters (deterministic total DFA) -- short-circuit for speed
            break
        S = {table[l][s] for s in S}
    return len(S)


def contribution(n, m):
    if n <= 1:
        return 1.0
    return (n - m) / (n - 1)


def total_score(automata, word_letters):
    total = 0.0
    for n, table in automata:
        m = apply_word(n, table, word_letters)
        total += contribution(n, m)
    return total


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    k, sigma, Lmax, automata = parse_input(in_path)

    # ---- sanity byte cap on raw output before any parsing ----
    raw = open(out_path, "rb").read()
    if len(raw) > 100000:
        fail("output too large (%d bytes)" % len(raw))

    try:
        text = raw.decode("utf-8", errors="strict")
    except Exception:
        fail("output not valid utf-8")

    # ---- extract the word: must be a single token on a single content line ----
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    if len(lines) == 0:
        word = ""
    elif len(lines) == 1 and len(lines[0].split()) <= 1:
        word = lines[0]
    else:
        fail("output must be a single line containing at most one token")

    # ---- validate alphabet + length budget ----
    if len(word) > Lmax:
        fail("word length %d exceeds budget Lmax=%d" % (len(word), Lmax))
    valid_chars = set(str(d) for d in range(sigma))
    for ch in word:
        if ch not in valid_chars:
            fail("character %r not a valid letter (alphabet is 0..%d)" % (ch, sigma - 1))

    word_letters = [int(ch) for ch in word]

    F = total_score(automata, word_letters)

    # ---- internal baseline B: repeat a single fixed letter (index 0) for up to
    #      Lmax steps -- a trivial, structure-oblivious recipe (no search at all) ----
    base_letter = 0
    base_len = min(Lmax, 60)
    base_word = [base_letter] * base_len
    B = total_score(automata, base_word)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f k=%d Ratio: %.6f" % (F, B, k, ratio))


if __name__ == "__main__":
    main()
