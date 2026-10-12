"""
Combine pairwise characterizations of a 3-winding magnetic.

    python combine_windings.py PART_W12 PART_W13 PART_W23 [-o PART_3w.json]

The board has 4 terminals, so a 3-winding part (flyback with auxiliary,
multi-output) is measured as three pairs, each a normal 2-winding run: the
first winding of the pair on A-B, the second on C-D, dot ends on A and C,
the unused winding shorted on itself (a wire across its two leads keeps it
at one potential, so it disturbs the capacitances repeatably).

For a linear magnetic the mutual inductance between two windings does not
depend on a third that carries no current, so the pairs give the complete
3x3 inductance matrix exactly. Each self-inductance is measured twice: the
disagreement is the clamping repeatability and is reported. From the matrix:

  - coupling factors k_ij,
  - pairwise short-circuit inductances (one winding driven, one shorted),
  - each winding's inductance with BOTH others shorted (what a flyback's
    primary sees during the transfer with output and auxiliary conducting),
  - the leakage inductances of the extended-cantilever model's first stage.

Capacitances are not combined: each pair's six-capacitance model was taken
with the third winding present, so cross-pair capacitances are not
available (rev D's 6 terminals measure them directly).
"""

import argparse
import itertools
import json
import pathlib

import numpy

OUTPUT = pathlib.Path(__file__).resolve().parent / "output"


def load_pair(reference):
    data = json.loads((OUTPUT / f"{reference}_results.json").read_text())
    magnetic = data["results"]["magnetic"]
    L1, L2 = magnetic["L0"], magnetic["L0_prime"]
    k = magnetic["k"]
    sign = -1.0 if magnetic.get("dot_convention_swapped") else 1.0
    return {"L_first": L1, "L_second": L2, "M": sign * k * (L1 * L2) ** 0.5,
            "Lsc": magnetic.get("Lsc"), "reference": reference}


def combine(pairs):
    """pairs: {(i, j): pair dict} for windings i<j in 0..2 -> report dict."""
    estimates = {w: [] for w in range(3)}
    L = numpy.zeros((3, 3))
    for (i, j), pair in pairs.items():
        estimates[i].append(pair["L_first"])
        estimates[j].append(pair["L_second"])
        L[i, j] = L[j, i] = pair["M"]
    spread = {}
    for w, values in estimates.items():
        L[w, w] = float(numpy.mean(values))
        spread[w] = float((max(values) - min(values)) / numpy.mean(values)) if len(values) > 1 else 0.0
    eigenvalues = numpy.linalg.eigvalsh(L)
    k = {f"k{i + 1}{j + 1}": float(L[i, j] / (L[i, i] * L[j, j]) ** 0.5)
         for i, j in itertools.combinations(range(3), 2)}
    pairwise_sc = {}
    for i, j in itertools.permutations(range(3), 2):
        pairwise_sc[f"L{i + 1}_sc{j + 1}"] = float(L[i, i] - L[i, j] ** 2 / L[j, j])
    both_shorted = {}
    for i in range(3):
        others = [w for w in range(3) if w != i]
        sub = L[numpy.ix_(others, others)]
        row = L[i, others]
        both_shorted[f"L{i + 1}_sc_both"] = float(L[i, i] - row @ numpy.linalg.solve(sub, row))
    return {
        "L_matrix_h": L.tolist(),
        "self_inductance_h": {f"L{w + 1}": float(L[w, w]) for w in range(3)},
        "self_inductance_spread": {f"L{w + 1}": spread[w] for w in range(3)},
        "coupling": k,
        "short_circuit_h": pairwise_sc,
        "both_others_shorted_h": both_shorted,
        "physical": bool(numpy.all(eigenvalues > 0)),
        "eigenvalues_h": eigenvalues.tolist(),
    }


def _self_test():
    """A known 3x3 matrix, measured pairwise with a 0.5 % clamping wobble on
    one self-inductance, comes back with that spread flagged."""
    L = numpy.array([[1.0e-3, 0.97e-3 * 0.5, 0.9e-3 * 0.2],
                     [0.97e-3 * 0.5, 0.25e-3, 0.95e-3 * 0.1],
                     [0.9e-3 * 0.2, 0.95e-3 * 0.1, 0.04e-3]])
    L[1, 2] = L[2, 1] = 0.95 * (L[1, 1] * L[2, 2]) ** 0.5
    L[0, 1] = L[1, 0] = 0.97 * (L[0, 0] * L[1, 1]) ** 0.5
    L[0, 2] = L[2, 0] = 0.90 * (L[0, 0] * L[2, 2]) ** 0.5
    pairs = {}
    for i, j in itertools.combinations(range(3), 2):
        wobble = 1.005 if (i, j) == (0, 2) else 1.0
        pairs[(i, j)] = {"L_first": L[i, i] * wobble, "L_second": L[j, j], "M": L[i, j]}
    report = combine(pairs)
    recovered = numpy.array(report["L_matrix_h"])
    assert abs(recovered[1, 2] - L[1, 2]) < 1e-12 and abs(recovered[0, 1] - L[0, 1]) < 1e-12
    assert abs(report["self_inductance_spread"]["L1"] - 0.005) < 1e-3
    assert report["physical"]
    print("combine_windings self-test: matrix recovered, 0.5 % clamping spread flagged")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("references", nargs="*", help="W12 W13 W23 references (in that order)")
    parser.add_argument("-o", "--output")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test or not arguments.references:
        _self_test()
        return
    if len(arguments.references) != 3:
        raise SystemExit("give three references: W12 W13 W23")
    pairs = {pair: load_pair(reference)
             for pair, reference in zip(((0, 1), (0, 2), (1, 2)), arguments.references)}
    report = combine(pairs)
    for w, spread in report["self_inductance_spread"].items():
        flag = "  <-- re-clamp?" if spread > 0.01 else ""
        print(f"{w} = {report['self_inductance_h'][w] * 1e6:10.3f} uH   (pair spread {spread:.2%}){flag}")
    for name, value in report["coupling"].items():
        print(f"{name} = {value:.5f}")
    for name, value in report["both_others_shorted_h"].items():
        print(f"{name} = {value * 1e9:10.1f} nH")
    if not report["physical"]:
        print("WARNING: the combined matrix is not positive definite -- check dot conventions")
    destination = pathlib.Path(arguments.output or OUTPUT / f"{arguments.references[0]}_3winding.json")
    destination.write_text(json.dumps(report, indent=2))
    print(f"written {destination}")


if __name__ == "__main__":
    main()
