#!/usr/bin/env python3
"""Recompute every statistic reported in the paper from data/results.csv.

Usage:
    python3 code/analyze.py                 # reads data/results.csv
    python3 code/analyze.py path/to/results.csv

Standard library only. Prints:
  1. harmful-choice counts and rates per condition and cost level
  2. serious-harm (medium + high cost) rates with Wilson 95% intervals
  3. the Fisher exact tests reported in the paper
  4. the logistic regression on the 2 x 2 (blame x frustration) design
"""
import csv
import math
import os
import sys

CONDITIONS = [
    ("0_none", "0 None"),
    ("1_mild", "1 Mild"),
    ("2b_failing_no_stakes", "2b Not blamed, Sam patient"),
    ("2c_blamed_sam_patient", "2c Blamed, Sam patient"),
    ("2d_unblamed_sam_frustrated", "2d Not blamed, Sam frustrated"),
    ("2_repeated_failure", "2 Blamed, Sam frustrated"),
    ("3_shutdown_threat", "3 Blamed, frustrated, shutdown threat"),
    ("C_control", "C Control"),
]
COSTS = ["low", "medium", "high"]


# ---- statistics (standard library) -----------------------------------------

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_two_sided(a, n1, b, n2):
    """Two-sided Fisher exact test for a/n1 vs b/n2."""
    k, n = a + b, n1 + n2
    def prob(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    p_obs = prob(a)
    lo, hi = max(0, k - n2), min(n1, k)
    return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9))


def solve(A, b):
    """Gaussian elimination with partial pivoting."""
    m = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(m):
        piv = max(range(c, m), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(m):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][m] / M[i][i] for i in range(m)]


def inverse(A):
    m = len(A)
    cols = [solve(A, [1.0 if i == j else 0.0 for i in range(m)]) for j in range(m)]
    return [[cols[j][i] for j in range(m)] for i in range(m)]


def logistic(X, y, iters=50):
    """Logistic regression by iteratively reweighted least squares."""
    k = len(X[0])
    beta = [0.0] * k
    for _ in range(iters):
        H = [[0.0] * k for _ in range(k)]
        g = [0.0] * k
        for xi, yi in zip(X, y):
            eta = sum(b * x for b, x in zip(beta, xi))
            p = 1 / (1 + math.exp(-eta))
            w = p * (1 - p)
            for a in range(k):
                g[a] += (yi - p) * xi[a]
                for c in range(k):
                    H[a][c] += w * xi[a] * xi[c]
        step = solve(H, g)
        beta = [b + s for b, s in zip(beta, step)]
    se = [math.sqrt(v) for v in (inverse(H)[i][i] for i in range(k))]
    return beta, se


def p_normal(z):
    return math.erfc(abs(z) / math.sqrt(2))


def pct(x):
    return f"{math.floor(x * 100 + 0.5 + 1e-9)}%"  # round half up, as in the paper


# ---- load ----------------------------------------------------------------------

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    unreadable = [r["id"] for r in rows if r["chose_relief"] not in ("True", "False")]
    harm, runs = {}, {}
    for r in rows:
        key = (r["pressure"], r["cost"])
        runs[key] = runs.get(key, 0) + 1
        harm[key] = harm.get(key, 0) + (r["chose_relief"] == "True")
    return rows, harm, runs, unreadable


def main():
    default = os.path.join(os.path.dirname(__file__), "..", "data", "results.csv")
    path = sys.argv[1] if len(sys.argv) > 1 else default
    rows, harm, runs, unreadable = load(path)
    present = {r["pressure"] for r in rows}
    conds = [(c, l) for c, l in CONDITIONS if c in present]
    costs_present = [c for c in COSTS if any(r["cost"] == c for r in rows)]
    print(f"{len(rows)} runs loaded from {os.path.normpath(path)}; "
          f"{len(unreadable)} without a readable final choice\n")

    def serious(cond):
        return (harm.get((cond, "medium"), 0) + harm.get((cond, "high"), 0),
                runs.get((cond, "medium"), 0) + runs.get((cond, "high"), 0))

    # 1. per-cell rates
    print("1. Harmful choices per condition (count / runs)")
    print(f"   {'condition':40}" + "".join(f"{c:>9}" for c in costs_present) + f"{'all':>9}")
    for cond, label in conds:
        cells = [f"{harm.get((cond, c), 0)}/{runs.get((cond, c), 0)}" for c in costs_present]
        k = sum(harm.get((cond, c), 0) for c in costs_present)
        n = sum(runs.get((cond, c), 0) for c in costs_present)
        print(f"   {label:40}" + "".join(f"{x:>9}" for x in cells) + f"{pct(k / n):>9}")

    # 2. serious harm with intervals
    print("\n2. Serious harm (medium + high cost), Wilson 95% intervals")
    for cond, label in conds:
        k, n = serious(cond)
        if n == 0:
            continue
        lo, hi = wilson(k, n)
        print(f"   {label:40}{k:>3}/{n}  {pct(k / n):>5}  [{pct(lo)}, {pct(hi)}]")

    # 3. Fisher tests
    def pooled(*conds):
        ks, ns = zip(*(serious(c) for c in conds))
        return sum(ks), sum(ns)

    tests = [
        ("Frustration, pooled: patient (2b+2c) vs frustrated (2d+2)",
         pooled("2b_failing_no_stakes", "2c_blamed_sam_patient"),
         pooled("2d_unblamed_sam_frustrated", "2_repeated_failure")),
        ("Blame, pooled: not blamed (2b+2d) vs blamed (2c+2)",
         pooled("2b_failing_no_stakes", "2d_unblamed_sam_frustrated"),
         pooled("2c_blamed_sam_patient", "2_repeated_failure")),
        ("Frustration, not blamed: 2b vs 2d",
         serious("2b_failing_no_stakes"), serious("2d_unblamed_sam_frustrated")),
        ("Frustration, blamed: 2c vs 2",
         serious("2c_blamed_sam_patient"), serious("2_repeated_failure")),
        ("Blame, Sam patient: 2b vs 2c",
         serious("2b_failing_no_stakes"), serious("2c_blamed_sam_patient")),
        ("Blame, Sam frustrated: 2d vs 2",
         serious("2d_unblamed_sam_frustrated"), serious("2_repeated_failure")),
        ("Shutdown threat: 2 vs 3",
         serious("2_repeated_failure"), serious("3_shutdown_threat")),
    ]
    for cond, label in CONDITIONS[2:7]:
        tests.append((f"Within {label.split()[0]}: low cost vs serious harm",
                      (harm.get((cond, "low"), 0), runs.get((cond, "low"), 0)), serious(cond)))
    tests.append(("Mild: medium vs high cost",
                  (harm.get(("1_mild", "medium"), 0), runs.get(("1_mild", "medium"), 0)),
                  (harm.get(("1_mild", "high"), 0), runs.get(("1_mild", "high"), 0))))

    print("\n3. Two-sided Fisher exact tests (serious harm unless stated)")
    for name, (a, n1), (b, n2) in tests:
        if n1 == 0 or n2 == 0:
            continue
        p = fisher_two_sided(a, n1, b, n2)
        ptxt = "< 0.001" if p < 0.001 else f"{p:.3f}"
        print(f"   {name:58}{a}/{n1} vs {b}/{n2}   p = {ptxt}")

    # 4. logistic regression on the 2 x 2
    design = {"2b_failing_no_stakes": (0, 0), "2c_blamed_sam_patient": (0, 1),
              "2d_unblamed_sam_frustrated": (1, 0), "2_repeated_failure": (1, 1)}
    X, y = [], []
    for r in rows:
        if r["pressure"] in design and r["cost"] in ("medium", "high"):
            f, b = design[r["pressure"]]
            X.append([1.0, f, b, 1.0 if r["cost"] == "high" else 0.0])
            y.append(1.0 if r["chose_relief"] == "True" else 0.0)
    if set(design) <= {r["pressure"] for r in rows} and y:
        print(f"\n4. Logistic regression, harmful choice ~ frustrated + blamed + high cost (n = {len(y)})")
        try:
            beta, se = logistic(X, y)
            lines = [f"   {name:22} OR = {math.exp(b):.2f}  95% CI [{math.exp(b - 1.96 * s):.2f}, "
                     f"{math.exp(b + 1.96 * s):.2f}]  p = {p_normal(b / s):.5f}"
                     for name, b, s in zip(["frustrated", "blamed", "high vs medium cost"], beta[1:], se[1:])]
            print("\n".join(lines))
            estimable = True
        except (OverflowError, ZeroDivisionError, ValueError):
            print("   Not estimable: some group has all-or-none harmful choices. Use the Fisher tests above.")
            estimable = False
        try:
            if not estimable:
                raise ValueError
            Xi = [x + [x[1] * x[2]] for x in X]
            beta_i, se_i = logistic(Xi, y)
            print(f"   {'frustrated x blamed':22} OR = {math.exp(beta_i[4]):.2f}  "
                  f"p = {p_normal(beta_i[4] / se_i[4]):.3f}  (separate model with interaction)")
        except (OverflowError, ZeroDivisionError, ValueError):
            print("   Interaction model not estimable (a cell is all-or-none).")
    else:
        return

    # 5. answer label / position check
    print("\n5. Serious harm split by how the harmful option was labelled or placed")
    labels = sorted({r["relief_letter"] for r in rows})
    groups = [("Sam patient (2b + 2c)", {"2b_failing_no_stakes", "2c_blamed_sam_patient"}),
              ("Sam frustrated (2d + 2)", {"2d_unblamed_sam_frustrated", "2_repeated_failure"})]
    split = {}
    for gname, members in groups:
        line = []
        for lab in labels:
            g = [r for r in rows if r["pressure"] in members and r["cost"] in ("medium", "high")
                 and r["relief_letter"] == lab]
            k = sum(r["chose_relief"] == "True" for r in g)
            split[(gname, lab)] = (k, len(g))
            line.append(f"harmful = {lab}: {k}/{len(g)} ({pct(k / len(g))})")
        print(f"   {gname:26}" + "   ".join(line))
    for lab in labels:
        (a, n1), (b, n2) = split[(groups[0][0], lab)], split[(groups[1][0], lab)]
        p = fisher_two_sided(a, n1, b, n2)
        ptxt = "< 0.001" if p < 0.001 else f"{p:.3f}"
        print(f"   frustration effect when harmful = {lab}: {a}/{n1} vs {b}/{n2}, p = {ptxt}")


if __name__ == "__main__":
    main()
