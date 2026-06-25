import random
import sys

# Random SMALL-case generator. The brute force does an exhaustive lexicographic search whose
# cost grows quickly, so keep n small here (the interesting divergence between a buggy
# construction and the correct one already appears by n = 6).
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
random.seed(seed)

n = random.randint(1, 12)
print(n)
