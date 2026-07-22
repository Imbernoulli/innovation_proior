# TIER: greedy
import sys

def main():
    data = sys.stdin.read().split("\n")
    d = int(data[0].split()[0])
    coeffs = list(map(int, data[1].split()))

    lines = []

    def emit(line):
        lines.append(line)
        return len(lines)

    # Horner's rule: v = a_d; for i = d-1 .. 0: v = v*x + a_i
    # This is the textbook, minimal-multiplication-count evaluator (exactly d
    # multiplications, provably optimal in COUNT for a generic polynomial) --
    # but every multiplication depends on the previous one, so it is a single
    # sequential chain of depth d.
    v = emit(f"C {coeffs[d]}")
    for i in range(d - 1, -1, -1):
        v = emit(f"M {v} 0")
        c = emit(f"C {coeffs[i]}")
        v = emit(f"A {v} {c}")

    out = [str(len(lines))] + lines
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
