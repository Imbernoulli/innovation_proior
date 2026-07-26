# TIER: trivial
# Do-nothing baseline: predict the constant mean of the training tension
# readings T_obs for every held-out design, ignoring p1,p2,d and the whole
# recurrence structure entirely. Reproduces the checker's own internal
# baseline -> ~0.1.
import sys, re


def spacify(expr):
    """Re-emit an arithmetic expression with every token whitespace-separated
    (purely cosmetic for ast.parse, which is whitespace-insensitive) so
    numeric literals are standalone tokens."""
    toks = re.findall(r"\*\*|\d+\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?"
                       r"|[A-Za-z_]\w*|[+\-*/(),]", expr)
    return " ".join(toks)


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.0")
        return
    n = int(data[0])
    vals = data[2:]
    tobs = []
    idx = 0
    for _ in range(n):
        d = int(vals[idx])
        row = vals[idx: idx + d + 2]
        tobs.append(float(row[-1]))
        idx += d + 2
    mean_t = sum(tobs) / len(tobs) if tobs else 0.0
    print(spacify("%.10g" % mean_t))


if __name__ == "__main__":
    main()
