# TIER: trivial
import sys

def main():
    data = sys.stdin.read().split("\n")
    d = int(data[0].split()[0])
    coeffs = list(map(int, data[1].split()))

    lines = []  # instruction lines, produce wires 1,2,3,...

    def emit(line):
        lines.append(line)
        return len(lines)  # newly produced wire index

    running = None
    for i in range(0, d + 1):
        if i == 0:
            term_wire = emit(f"C {coeffs[i]}")
        else:
            # rebuild x^i from scratch: no sharing with any other term's power chain
            cur = 0  # wire 0 == x == x^1
            for step in range(2, i + 1):
                cur = emit(f"M {cur} 0")
            const_wire = emit(f"C {coeffs[i]}")
            term_wire = emit(f"M {cur} {const_wire}")
        if running is None:
            running = term_wire
        else:
            running = emit(f"A {running} {term_wire}")

    out = [str(len(lines))] + lines
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
