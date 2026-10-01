"""
Unit tests for TransformerModel -- runs with no instruments connected.

    python scripts/test_model.py

Every test builds synthetic data from a KNOWN circuit, runs the extraction,
and checks the known values come back.  That is the only way to be confident
in extraction maths whose inputs normally come from hardware.
"""

import math
import sys

import numpy
import pandas

import TransformerModel as tm


PASSED = []
FAILED = []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")


def close(a, b, tolerance=1e-9):
    """Relative comparison. Must not fall back to an absolute floor of 1.0 --
    every inductance and capacitance here is far below 1, so that would make
    the comparison vacuously true."""
    return abs(a - b) <= tolerance * max(abs(b), 1e-30)


# --------------------------------------------------------------------------


def test_cogitore_equations():
    """The two formulas I previously and wrongly flagged as non-standard."""
    print("\n[COG94] eq (3), (4), (5)")
    L0, Lsc = 1.0e-3, 1.9e-4
    k = tm.coupling_coefficient(L0, Lsc)
    check("k = sqrt(1 - Lsc/L0)", close(float(k), math.sqrt(1 - 0.19)))
    check("Lp = L0(1+k)/2  [eq 5]", close(tm.magnetizing_inductance(L0, k), L0 * (1 + k) / 2))
    check("ls = Lsc/k      [eq 4]", close(tm.leakage_inductance(Lsc, k), Lsc / k))
    # These must NOT be the T-model values -- guard against a well-meaning "fix"
    check("Lp is not the T-model k*L0",
          not close(tm.magnetizing_inductance(L0, k), k * L0, 1e-3),
          f"{tm.magnetizing_inductance(L0,k):.6e} vs T-model {k*L0:.6e}")
    try:
        tm.coupling_coefficient(1e-3, 2e-3)
        check("Lsc > L0 raises", False)
    except ValueError:
        check("Lsc > L0 raises", True)


def test_turns_ratio_convention():
    """eta = sqrt(L2/L1) per [BLA94] Table 1, used consistently everywhere."""
    print("\nTurns-ratio convention")
    L1, L2 = 1.0e-3, 16.0e-3          # 1:4 transformer
    eta = tm.turns_ratio(L1, L2)
    check("eta = sqrt(L2/L1) = N2/N1", close(float(eta), 4.0))
    k = 0.98
    check("effective N2/N1 = eta/k", close(float(tm.effective_turns_ratio(L1, L2, k)), 4.0 / 0.98))
    # The old inductance method used the reciprocal; make the difference visible
    check("reciprocal differs by eta^2 = 16x", close(eta ** 2 / (1 / eta) ** 2, 256.0, 1e-9))


def test_mutual_and_dot_convention():
    print("\nMutual inductance and dot convention")
    L1, L2, k = 1.0e-3, 4.0e-3, 0.95
    M = k * math.sqrt(L1 * L2)
    Lcum = L1 + L2 + 2 * M
    Ldif = L1 + L2 - 2 * M
    check("M = (Lcum - Ldif)/4", close(tm.mutual_inductance(Lcum, Ldif), M))

    eta = tm.turns_ratio(L1, L2)
    Lm = tm.magnetizing_from_mutual(M, eta)
    check("Lm = M/eta = k*L1  (primary-referred)", close(Lm, k * L1),
          f"got {Lm:.6e}, k*L1 = {k*L1:.6e}")
    # The old code's M/n with n = sqrt(L1/L2) gave the secondary-referred value
    n_old = math.sqrt(L1 / L2)
    check("old M/n gave k*L2 (secondary-referred)", close(M / n_old, k * L2))

    _, _, swapped = tm.check_dot_convention(Ldif, Lcum)
    check("reversed secondary detected", swapped)
    _, _, not_swapped = tm.check_dot_convention(Lcum, Ldif)
    check("correct wiring not flagged", not not_swapped)


def test_two_gap_resistance():
    """R = Rw + K*L^2, exact for two gaps, over-determined for more."""
    print("\nAC winding resistance, multi-gap")
    Rw_true, K_true = 0.42, 1.3e5
    L = numpy.array([1.0e-3, 7.0e-4, 5.0e-4, 3.0e-4, 1.5e-4])
    R = Rw_true + K_true * L ** 2

    two = tm.separate_winding_resistance(L[:2], R[:2])
    check("two gaps recover Rw exactly", close(two["Rw"], Rw_true, 1e-6), f"{two['Rw']:.6f}")

    many = tm.separate_winding_resistance(L, R)
    check("five gaps recover Rw exactly", close(many["Rw"], Rw_true, 1e-6))
    check("clean data -> tiny residual", many["residual_max"] < 1e-9,
          f"residual {many['residual_max']:.2e}")

    # Model violation must show up in the residual
    Rw_drift = Rw_true * numpy.array([1.0, 1.08, 1.16, 1.24, 1.32])
    bad = tm.separate_winding_resistance(L, Rw_drift + K_true * L ** 2)
    check("gap-dependent Rw raises the residual", bad["residual_max"] > 0.01,
          f"residual {bad['residual_max']*100:.2f}%")

    # Conditioning warning
    tight = tm.separate_winding_resistance([1.0e-3, 9.9e-4], [Rw_true + K_true * 1e-6,
                                                             Rw_true + K_true * 9.801e-7])
    check("near-equal gaps flagged by high noise gain", tight["noise_gain"] > 20,
          f"gain {tight['noise_gain']:.1f}")


def test_deembed_guard():
    print("\nParallel-capacitance de-embedding guard")
    f = numpy.array([1e4, 1e5, 1e6, 1e7, 1e8])
    R, valid = tm.deembed_parallel_capacitance(
        numpy.full_like(f, 5.0), numpy.full_like(f, 1e-3), 100e-12, f)
    check("out-of-range points flagged, not silent NaN", (~valid).any() and valid.any(),
          f"{valid.sum()}/{len(f)} usable")
    check("valid points are finite", numpy.isfinite(R[valid]).all())


def _synthetic_impedance(L, C, Rp, frequencies):
    """|Z| and phase of Rp || L || C -- a parallel resonance."""
    w = 2 * math.pi * frequencies
    Y = 1.0 / Rp + 1.0 / (1j * w * L) + 1j * w * C
    Z = 1.0 / Y
    return pandas.DataFrame({
        "measurement_index": 0,
        "frequency": frequencies,
        "magnitude": numpy.abs(Z),
        "phase": numpy.degrees(numpy.angle(Z)),
    })


def test_resonance_detection():
    print("\nResonance detection")
    L, C, Rp = 1.0e-3, 100e-12, 50e3
    f_expected = 1.0 / (2 * math.pi * math.sqrt(L * C))
    frequencies = numpy.logspace(4, 7.5, 801)
    data = _synthetic_impedance(L, C, Rp, frequencies)

    found = tm.detect_resonances(data)
    maxima = [r for r in found if r["type"] == "local maximum"]
    check("parallel resonance found", len(maxima) == 1, f"{len(maxima)} maxima")
    if maxima:
        error = abs(maxima[0]["frequency"] - f_expected) / f_expected
        check("resonance frequency within 1%", error < 0.01, f"{error*100:.3f}% off")
        check("confirmed by phase sign change", maxima[0]["phase_confirmed"] is True)

    # The whole point of a RELATIVE prominence: the old absolute prominence of
    # 2 ohms passes at one impedance level and fails at others. Test both ends.
    # Note Q = Rp*sqrt(C/L), so Rp must keep Q > 1 for a resonance to exist at
    # all -- a 5 ohm peak on 1 mH / 100 pF is simply an overdamped circuit,
    # not a detection failure.
    high_z = _synthetic_impedance(L, C, 5e6, frequencies)
    check("works at 5 MOhm peak (Q = 1580)",
          len([r for r in tm.detect_resonances(high_z) if r["type"] == "local maximum"]) == 1)

    # Low-impedance DUT: 1 uH / 10 nF gives sqrt(L/C) = 10 ohm, so Rp = 50 ohm
    # is a genuine resonance with Q = 5 and a 50 ohm peak.
    L_low, C_low, Rp_low = 1.0e-6, 10.0e-9, 50.0
    low_frequencies = numpy.logspace(4, 8, 801)
    low_z = _synthetic_impedance(L_low, C_low, Rp_low, low_frequencies)
    low_maxima = [r for r in tm.detect_resonances(low_z) if r["type"] == "local maximum"]
    check("works at 50 Ohm peak (Q = 5)", len(low_maxima) == 1, f"{len(low_maxima)} maxima")
    if low_maxima:
        expected = 1.0 / (2 * math.pi * math.sqrt(L_low * C_low))
        check("low-impedance resonance frequency within 1%",
              abs(low_maxima[0]["frequency"] - expected) / expected < 0.01)


def test_capacitance_round_trip():
    """Build resonances from KNOWN Cij, extract, and check they come back."""
    print("\nSix-capacitance round trip [BLA94] Table 1")
    truth = {"C11": 82e-12, "C12": -14e-12, "C13": 23e-12,
             "C22": 47e-12, "C23": -9e-12, "C33": 65e-12}
    eta = 2.0
    L0, Lsc = 2.0e-3, 1.1e-4

    measured = {}
    for link, expressions in tm.WINDING_LINKS.items():
        sums = {}
        for key, expression in expressions.items():
            sums[key] = float(expression(truth, eta))
        measured[link] = sums

    result = tm.solve_six_capacitances(measured, truth["C33"], eta)
    check("solver converged", result["converged"])
    check("used all 15 equations", result["equations_used"] == 15, str(result["equations_used"]))
    worst = 0.0
    for name in tm.CAPACITANCE_UNKNOWNS:
        error = abs(result["coefficients"][name] - truth[name]) / abs(truth[name])
        worst = max(worst, error)
    check("all five unknowns within 0.1%", worst < 1e-3, f"worst {worst*100:.4f}%")

    # Under-determination must raise, not return nonsense
    try:
        tm.solve_six_capacitances({"B-D": measured["B-D"]}, truth["C33"], eta)
        check("too few equations raises", False)
    except ValueError:
        check("too few equations raises", True)

    # Negative capacitances must survive -- [BLA94] allows them
    check("negative C12 recovered with sign",
          result["coefficients"]["C12"] < 0, f"{result['coefficients']['C12']:.3e}")


def test_resonance_to_capacitance():
    print("\nResonance -> capacitance sums")
    L0, Lsc = 1e-3, 1e-4
    C1_C2, C1_C3, C2_C3 = 120e-12, 300e-12, 250e-12
    f = lambda L, C: 1.0 / (2 * math.pi * math.sqrt(L * C))
    resonances = {
        "open": [
            {"frequency": f(L0, C1_C2), "type": "local maximum"},
            {"frequency": f(Lsc, C2_C3), "type": "local minimum"},
        ],
        "short": [{"frequency": f(Lsc, C1_C3), "type": "local maximum"}],
    }
    sums = tm.capacitance_sums_from_resonances(resonances, L0, Lsc)
    check("C1+C2 from open maximum with L0", close(sums["C1_C2"], C1_C2, 1e-9))
    check("C1+C3 from short maximum with Lsc", close(sums["C1_C3"], C1_C3, 1e-9))
    check("C2+C3 from open minimum with Lsc", close(sums["C2_C3"], C2_C3, 1e-9))

    missing = tm.capacitance_sums_from_resonances({"open": [], "short": []}, L0, Lsc)
    check("missing resonances give None, not a crash",
          all(v is None for v in missing.values()))


def test_reciprocity():
    """Z0*Zsc' = Z0'*Zsc -- the relay self-test."""
    print("\nReciprocity self-test [BLA94] II-C")
    frequencies = numpy.logspace(3, 6, 60)

    def frame(magnitude):
        return pandas.DataFrame({"measurement_index": 0, "frequency": frequencies,
                                 "magnitude": magnitude, "phase": numpy.zeros_like(frequencies)})

    Z0 = frame(numpy.full_like(frequencies, 1000.0))
    Z0p = frame(numpy.full_like(frequencies, 4000.0))
    Zsc = frame(numpy.full_like(frequencies, 100.0))
    Zscp = frame(numpy.full_like(frequencies, 400.0))     # 1000*400 == 4000*100
    ok, worst, _ = tm.check_reciprocity(Z0, Z0p, Zsc, Zscp)
    check("consistent set passes", ok and worst < 1e-12, f"worst {worst:.2e}")

    bad = frame(numpy.full_like(frequencies, 400.0 * 1.5))  # as if a relay misfired
    ok2, worst2, _ = tm.check_reciprocity(Z0, Z0p, Zsc, bad)
    check("a wrong relay state is caught", not ok2, f"worst {worst2*100:.1f}%")


def test_linearity():
    print("\nLinearity check [COG94] III")
    frequencies = numpy.logspace(4, 6, 40)

    def frame(values):
        return pandas.DataFrame({"measurement_index": 0, "frequency": frequencies,
                                 "inductance": values})

    base = numpy.full_like(frequencies, 1e-3)
    ok, worst, _ = tm.check_linearity(frame(base), frame(base * 1.001))
    check("linear DUT passes", ok, f"worst {worst*100:.2f}%")
    ok2, worst2, _ = tm.check_linearity(frame(base), frame(base * 1.15))
    check("saturating DUT fails", not ok2, f"worst {worst2*100:.1f}%")


def test_full_summary():
    print("\nEnd-to-end magnetic summary")
    L1, L2, k = 1.2e-3, 4.8e-3, 0.97
    M = k * math.sqrt(L1 * L2)
    summary = tm.magnetic_summary(
        L0=L1, Lsc=L1 * (1 - k ** 2), L0_prime=L2,
        L_cum=L1 + L2 + 2 * M, L_dif=L1 + L2 - 2 * M)
    check("k recovered", close(summary["k"], k, 1e-6), f"{summary['k']:.6f}")
    check("eta recovered", close(summary["eta"], 2.0, 1e-6))
    check("k from mutual agrees with k from Lsc", summary["k_consistency"] < 1e-6,
          f"{summary['k_consistency']:.2e}")
    check("Lm from mutual = k*L1", close(summary["Lm_from_mutual"], k * L1, 1e-6))
    check("Lp uses eq (5)", close(summary["Lp_magnetizing"], L1 * (1 + k) / 2, 1e-9))


def test_plateau():
    print("\nLow-frequency resistance plateau")
    frequencies = numpy.logspace(2, 5, 200)
    r1 = 0.35
    resistance = r1 + 1e-12 * frequencies ** 2
    data = pandas.DataFrame({"measurement_index": 0, "frequency": frequencies,
                             "resistance": resistance})
    value, used_to, flatness = tm.winding_resistance_from_plateau(data)
    check("r1 recovered from plateau", close(value, r1, 5e-3), f"{value:.4f} vs {r1}")
    check("flatness reported", flatness < 0.05, f"{flatness:.4f}")

    steep = pandas.DataFrame({"measurement_index": 0, "frequency": frequencies,
                              "resistance": r1 * (1 + (frequencies / 1e3) ** 2)})
    _, _, bad_flatness = tm.winding_resistance_from_plateau(steep)
    check("no plateau -> poor flatness flagged", bad_flatness > 0.1, f"{bad_flatness:.3f}")


if __name__ == "__main__":
    print("=" * 68)
    print("TransformerModel test suite -- no instruments required")
    print("=" * 68)
    for test in (test_cogitore_equations, test_turns_ratio_convention,
                 test_mutual_and_dot_convention, test_two_gap_resistance,
                 test_deembed_guard, test_resonance_detection,
                 test_capacitance_round_trip, test_resonance_to_capacitance,
                 test_reciprocity, test_linearity, test_full_summary, test_plateau):
        test()
    print("\n" + "=" * 68)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
    print("=" * 68)
    sys.exit(1 if FAILED else 0)
