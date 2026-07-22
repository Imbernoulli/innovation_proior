# TIER: strong
import sys

def main():
    data = sys.stdin.read().split("\n")
    d = int(data[0].split()[0])
    coeffs = list(map(int, data[1].split()))
    m = d // 2

    lines = []

    def emit(line):
        lines.append(line)
        return len(lines)

    const1 = emit("C 1")

    # Memoized binary-exponentiation power ladder: power(n) returns the wire
    # holding x^n, built by square-and-multiply and SHARED across every call
    # site (the sub-expressions needed for different pairs overlap heavily).
    # This gives O(log n) multiplicative depth per fresh power instead of the
    # O(n) sequential chain a left-to-right accumulation would need.
    power_cache = {0: const1, 1: 0}  # power(0) = the constant 1; power(1) = x (wire 0)

    def power(n):
        if n in power_cache:
            return power_cache[n]
        if n % 2 == 0:
            h = power(n // 2)
            w = emit(f"M {h} {h}")
        else:
            p1 = power(n - 1)
            w = emit(f"M {p1} 0")
        power_cache[n] = w
        return w

    # Symmetry-balancing insight: because a_i == a_{d-i} (planted palindrome),
    # pair term i with term d-i instead of treating each monomial separately:
    #   a_i*x^i + a_i*x^{d-i} = a_i * x^i * (1 + x^{d-2i})
    # This roughly halves the number of distinct coefficient-scaling terms
    # (m pairs instead of d+1 monomials) while every power still comes from
    # the shared log-depth ladder above.
    running = None
    for i in range(0, m):
        pdi = power(d - 2 * i)
        t1 = emit(f"A {pdi} {const1}")          # 1 + x^(d-2i)
        if i == 0:
            t2 = t1                              # x^0 == 1, no multiplication needed
        else:
            pi = power(i)
            t2 = emit(f"M {pi} {t1}")            # x^i * (1 + x^(d-2i))
        ac = emit(f"C {coeffs[i]}")
        t3 = emit(f"M {t2} {ac}")                # a_i * (x^i + x^(d-i))
        running = t3 if running is None else emit(f"A {running} {t3}")

    # Middle (self-paired) term a_m * x^m.
    pm = power(m)
    amc = emit(f"C {coeffs[m]}")
    termm = emit(f"M {pm} {amc}")
    running = emit(f"A {running} {termm}") if running is not None else termm

    out = [str(len(lines))] + lines
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
