#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE congestion-controller instance to stdout.

Format (whitespace separated tokens, one record per line):
  T C Qmax n_comp
  base_rtt_ego init_cwnd_ego
  ALPHA BETA GAMMA
  n_comp lines, each:  TYPE p1 p2 p3 base_rtt demand
    TYPE=AIMD  : p1=init_win p2=inc_step p3=0            demand=-1 (uncapped: use FAIR)
    TYPE=CONST : p1=rate     p2=0        p3=0             demand=rate
    TYPE=ONOFF : p1=rate_on  p2=period_on p3=period_off   demand=avg rate

testId 1..10 is a hand-authored difficulty/trap ladder: 1-2 are single-flow
(no competitors) so raw throughput is all that matters; 3-10 plant multi-flow
regimes (delay-sensitive CONST flows, bursty ONOFF flows, competing AIMD
flows, tight buffers, and a high-fairness-weight extreme) where a program
that only reacts to its own loss starves everyone else.
"""
import sys

CASES = {
    1: dict(T=40, C=12, Qmax=20, base_rtt_ego=4, init=2, A=1.0, B=1.0, G=4.0, comps=[]),
    2: dict(T=50, C=16, Qmax=8, base_rtt_ego=3, init=2, A=1.0, B=4.0, G=3.0, comps=[]),
    3: dict(T=50, C=14, Qmax=8, base_rtt_ego=4, init=2, A=5.0, B=3.0, G=3.0, comps=[
        ('CONST', 6, 0, 0, 5, 6),
    ]),
    4: dict(T=60, C=18, Qmax=10, base_rtt_ego=3, init=2, A=6.0, B=3.0, G=3.0, comps=[
        ('CONST', 4, 0, 0, 5, 4),
        ('CONST', 4, 0, 0, 6, 4),
        ('ONOFF', 8, 4, 4, 4, 4),
    ]),
    5: dict(T=50, C=16, Qmax=9, base_rtt_ego=4, init=2, A=6.0, B=4.0, G=2.0, comps=[
        ('ONOFF', 12, 4, 4, 4, 6),
    ]),
    6: dict(T=70, C=20, Qmax=11, base_rtt_ego=4, init=2, A=8.0, B=4.0, G=3.0, comps=[
        ('CONST', 4, 0, 0, 5, 4),
        ('CONST', 4, 0, 0, 5, 4),
        ('CONST', 3, 0, 0, 6, 3),
        ('ONOFF', 6, 3, 3, 4, 3),
    ]),
    7: dict(T=60, C=18, Qmax=9, base_rtt_ego=4, init=2, A=5.0, B=3.0, G=3.0, comps=[
        ('CONST', 6, 0, 0, 4, 6),
        ('CONST', 5, 0, 0, 5, 5),
    ]),
    8: dict(T=80, C=24, Qmax=13, base_rtt_ego=4, init=3, A=7.0, B=4.0, G=3.0, comps=[
        ('CONST', 6, 0, 0, 5, 6),
        ('CONST', 5, 0, 0, 6, 5),
        ('ONOFF', 10, 4, 4, 4, 5),
    ]),
    9: dict(T=90, C=28, Qmax=15, base_rtt_ego=4, init=1, A=9.0, B=4.0, G=3.0, comps=[
        ('CONST', 4, 0, 0, 5, 4),
        ('CONST', 4, 0, 0, 5, 4),
        ('CONST', 3, 0, 0, 6, 3),
        ('CONST', 4, 0, 0, 4, 4),
        ('ONOFF', 8, 4, 4, 4, 4),
    ]),
    10: dict(T=50, C=16, Qmax=9, base_rtt_ego=4, init=2, A=9.0, B=2.0, G=2.0, comps=[
        ('CONST', 6, 0, 0, 5, 6),
    ]),
}


def main():
    tid = int(sys.argv[1])
    tid = ((tid - 1) % 10) + 1
    c = CASES[tid]
    out = []
    out.append(f"{c['T']} {c['C']} {c['Qmax']} {len(c['comps'])}")
    out.append(f"{c['base_rtt_ego']} {c['init']}")
    out.append(f"{c['A']:.2f} {c['B']:.2f} {c['G']:.2f}")
    for (typ, p1, p2, p3, brtt, demand) in c['comps']:
        out.append(f"{typ} {p1} {p2} {p3} {brtt} {demand}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
