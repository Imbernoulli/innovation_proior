# TIER: trivial
"""Never rotate: reproduce the never-rotate baseline hop cost exactly."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    M = len(inst["accesses"])
    print(json.dumps({"ops": [[] for _ in range(M)]}))


if __name__ == "__main__":
    main()
