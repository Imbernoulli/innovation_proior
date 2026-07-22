# TIER: trivial
"""Blind guess: ignore the logged tributes entirely and submit the flat
'nothing learned' ritual (b=3, a2=0, a1=0, a0=0). This reproduces the
checker's own baseline construction exactly."""


def main():
    print(3, 0, 0, 0)


if __name__ == "__main__":
    main()
