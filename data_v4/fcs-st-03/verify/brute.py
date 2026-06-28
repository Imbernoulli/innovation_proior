import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    m = int(data[idx]); idx += 1
    pats = []
    for _ in range(m):
        p = data[idx]; idx += 1
        w = int(data[idx]); idx += 1
        pats.append((p, w))
    # text is the next token; may be absent if the generator emitted an empty text.
    if idx < len(data):
        text = data[idx]; idx += 1
    else:
        text = ""

    total = 0
    n = len(text)
    for p, w in pats:
        lp = len(p)
        if lp == 0 or lp > n:
            continue
        # count occurrences of p in text (overlapping), naive O(n*lp) scan
        c = 0
        start = 0
        while True:
            j = text.find(p, start)
            if j == -1:
                break
            c += 1
            start = j + 1
        total += c * w
    print(total)

main()
