# Selftest Covering (Python/Format-C)
Cover elements to minimize F = sum(c_i covered) + sum(p_i uncovered). Baseline covers nothing (B=sum p).
Score = min(1000,100*B/F)/1000. Input: "n" then n lines "p c". Output: "k" then k covered indices (1-based).
