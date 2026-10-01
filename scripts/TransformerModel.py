"""
Lumped-element two-winding transformer model and extraction maths.

Everything here is a pure function of measured data. No instrument, no relay
board, no plotting -- so it can be unit-tested against synthetic data with
nothing connected (see test_model.py).

References, all implemented literally:

  [COG94] Cogitore, Keradec & Barbaroux, "The two-winding transformer: an
          experimental method to obtain a wide frequency range equivalent
          circuit", IEEE Trans. Instrum. Meas. 43(2) 364-371, 1994.
          doi:10.1109/19.293449
  [BLA94] Blache, Keradec & Cogitore, "Stray capacitances of two winding
          transformers: equivalent circuit, measurements, calculation and
          lowering", IEEE IAS Annual Meeting 1994, 1211-1217.
          doi:10.1109/IAS.1994.377552
  [SCH98] Schellmanns, Berrouche & Keradec, "Multiwinding transformers: a
          successive refinement method to characterize a general equivalent
          circuit", IEEE Trans. Instrum. Meas. 47(5) 1316-1321, 1998.
          doi:10.1109/19.746603

CONVENTIONS -- read before touching anything.

  eta  is the coupling ratio of [COG94] Fig. 1/2, defined so the magnetic
       quadrupole is SYMMETRIC.  From [BLA94] Table 1, where eta**2 multiplies
       C22 when referring the secondary capacitance to the primary,

           eta = sqrt(L0_secondary / L0_primary)

       i.e. eta ~ N2/N1.  This is the ONLY definition used in this file.  The
       old code defined it both ways in different methods; that is what
       `turns_ratio()` exists to prevent.

  Lp   is the PARALLEL inductance of [COG94] Fig. 2, which draws it as two
       elements of 2*Lp either side of the coupler.  It is NOT the magnetizing
       inductance of a conventional T-model: for a T you would get k*L0, here
       you get L0*(1+k)/2.  Both are right in their own circuit.  Do not
       "fix" one into the other.
"""

import math

import numpy
import pandas
from scipy.optimize import least_squares
from scipy.signal import find_peaks

# --------------------------------------------------------------------------
# Magnetic coupling -- [COG94] section II-B
# --------------------------------------------------------------------------


def coupling_coefficient(L0, Lsc):
    """[COG94] eq (3): k = sqrt(1 - Lsc/L0).

    L0  -- primary open-circuit inductance
    Lsc -- primary short-circuit inductance (secondary shorted)
    """
    ratio = Lsc / L0
    if numpy.any(numpy.asarray(ratio) >= 1.0):
        raise ValueError(
            f"Lsc/L0 = {ratio} >= 1: the short-circuit inductance cannot exceed "
            "the open-circuit one. Check that config 2 really shorts the secondary."
        )
    return numpy.sqrt(1.0 - ratio)


def turns_ratio(L0_primary, L0_secondary):
    """Coupling ratio eta of [COG94], = sqrt(L0_secondary / L0_primary) ~ N2/N1.

    [COG94] warns this "does not coincide systematically with the turn number
    ratio"; it converges to it for k >= 0.98.  Use `effective_turns_ratio` when
    coupling is loose and the mutual inductance is available.
    """
    return numpy.sqrt(L0_secondary / L0_primary)


def effective_turns_ratio(L0_primary, L0_secondary, k):
    """Turns ratio corrected for imperfect coupling: N2/N1 = eta / k.

    From M = k*sqrt(L1*L2): the ideal-transformer ratio implied by the mutual
    inductance is k*sqrt(L1/L2) referred one way, so the physical turns ratio
    differs from `turns_ratio` by 1/k.  At k = 0.95 that is a 5% correction on
    every referred quantity.
    """
    return turns_ratio(L0_primary, L0_secondary) / k


def magnetizing_inductance(L0, k):
    """[COG94] eq (5): Lp = L0 * (1 + k) / 2.

    Parallel inductance of the SYMMETRIC equivalent circuit (Fig. 2 draws it as
    2*Lp either side of the coupler).  Not the T-model magnetizing inductance.
    """
    return L0 * (1.0 + k) / 2.0


def leakage_inductance(Lsc, k):
    """[COG94] eq (4): ls = Lsc / k.  Series leakage of the symmetric circuit."""
    return Lsc / k


def mutual_inductance(L_cumulative, L_differential):
    """M = (Lcum - Ldif) / 4, from series-aiding and series-opposing measurements.

    Not from the papers -- [COG94] uses only L0, Lsc and L0' -- but a valid and
    useful independent cross-check on k and eta.
    """
    return (L_cumulative - L_differential) / 4.0


def check_dot_convention(L_cumulative, L_differential):
    """Return (L_cum, L_dif, was_swapped) with the aiding measurement first.

    A DUT clamped with the secondary reversed makes Lcum < Ldif and every
    derived quantity silently follows the sign.  [SCH98] notes the sign of eta
    depends on terminal labelling, so an instrument that accepts arbitrary
    magnetics must detect this rather than assume it.
    """
    if numpy.mean(L_cumulative) < numpy.mean(L_differential):
        return L_differential, L_cumulative, True
    return L_cumulative, L_differential, False


def magnetizing_from_mutual(M, eta):
    """Primary-referred magnetizing inductance from the mutual inductance.

    Lm = M / eta with eta = N2/N1, since M = k*sqrt(L1*L2) and
    M/eta = M*sqrt(L1/L2) = k*L1.

    The old code computed M/n with n = sqrt(L1/L2), i.e. the reciprocal, which
    returned the SECONDARY-referred value and then subtracted it from a
    primary-referred one.
    """
    return M / eta


# --------------------------------------------------------------------------
# Losses -- [COG94] section II-C
# --------------------------------------------------------------------------


def winding_resistance_from_plateau(data, max_frequency=None):
    """r1 from the low-frequency plateau of |Z| -- [COG94] II-C.

    "The low-frequency plateau of Z0 equals r1, and that of Z0' equals eta^2*r2."

    Takes the median of the flattest low-frequency decade to reject noise.
    Returns (resistance, frequency_used, plateau_flatness) where flatness is the
    relative spread over the averaged window -- above ~0.1 the sweep never
    reached the plateau and the number should not be trusted.
    """
    frame = data.sort_values("frequency")
    if max_frequency is not None:
        frame = frame[frame["frequency"] <= max_frequency]
    if len(frame) < 3:
        raise ValueError(
            "Not enough low-frequency points to find a resistive plateau. "
            "Start the sweep lower (100 Hz - 1 kHz)."
        )
    window = max(3, len(frame) // 5)
    head = frame.head(window)
    values = head["resistance"].to_numpy()
    resistance = float(numpy.median(values))
    flatness = float((values.max() - values.min()) / abs(resistance)) if resistance else float("inf")
    return resistance, float(head["frequency"].iloc[-1]), flatness


def safe_reference_frequency(resonances, preferred=10_000.0, margin=20.0,
                             available=None):
    """Pick a frequency low enough that capacitance has not yet lifted L.

    [COG94] reads the inductances off "the first ascending part" of the Bode
    plot. Measuring a decade below the first resonance leaves a 1% error, since
    the apparent inductance is inflated by 1/(1 - (f/f_res)**2); a factor of 20
    brings that to 0.25%.  Returns the highest usable frequency at or below
    `preferred`.
    """
    maxima = [r["frequency"] for r in resonances if r["type"] == "local maximum"]
    ceiling = min(maxima) / margin if maxima else preferred
    chosen = min(preferred, ceiling)
    if available is not None:
        low, high = float(numpy.min(available)), float(numpy.max(available))
        chosen = float(numpy.clip(chosen, low, high))
    return chosen


def inductance_lift(reference_frequency, resonance_frequency):
    """Relative over-read of L at `reference_frequency`: 1/(1-(f/fr)^2) - 1."""
    if not resonance_frequency:
        return 0.0
    ratio = reference_frequency / resonance_frequency
    if ratio >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - ratio ** 2) - 1.0


def core_loss_resistance(z_data, resonance_frequency):
    """Rp = |Z0| at the first parallel resonance -- [COG94] II-C.

    "The value of Rp ... is masked by parallel inductances on the low-frequency
    side or by parallel capacitors on the high-frequency side.  Luckily, these
    two effects compensate each other at resonance frequency f2, so that the
    modulus of Z0 measured at this frequency identifies with Rp."
    """
    averaged = average_cycles(z_data)
    index = (averaged["frequency"] - resonance_frequency).abs().idxmin()
    return float(averaged.loc[index, "magnitude"]), float(averaged.loc[index, "frequency"])


# --------------------------------------------------------------------------
# AC winding resistance -- multi-gap separation of core and winding loss
# --------------------------------------------------------------------------


def separate_winding_resistance(inductances, resistances):
    """Least-squares split of R_measured = Rw + K*L^2 over two or more gaps.

    Physical basis.  With a complex permeability mu = mu' - j*mu'' and a gap lg,
    the reluctance model gives

        a = lg + le*mu'/|mu|^2      b = le*mu''/|mu|^2      P = mu0*Ae*N^2
        L      = P*a/(a^2+b^2)
        R_core = w*P*b/(a^2+b^2) = (w*b/P) * L^2 * (1 + b^2/a^2)

    Only lg varies with the gap and it appears solely inside `a`, so
    K = w*b/P is gap-independent and R = Rw + K*L^2.  The neglected
    (1 + b^2/a^2) term shrinks as the gap grows, so gapping IMPROVES the
    approximation.

    Two gaps give the exact closed form
        Rw = R1 - L1^2 (R1 - R2) / (L1^2 - L2^2)
    which is what the old code implemented, correctly.  Three or more gaps
    over-determine the system, and the residual then TESTS the assumption that
    Rw is gap-independent -- it is not, if fringing flux from the gap drives
    proximity loss in nearby conductors.

    Returns dict with Rw, K, residual_max (relative), and conditioning gain.
    """
    L = numpy.asarray(inductances, dtype=float)
    R = numpy.asarray(resistances, dtype=float)
    if L.shape != R.shape or L.size < 2:
        raise ValueError("Need at least two (inductance, resistance) pairs.")

    design = numpy.vstack([numpy.ones_like(L), L ** 2]).T
    (Rw, K), *_ = numpy.linalg.lstsq(design, R, rcond=None)
    predicted = design @ numpy.array([Rw, K])
    residual = numpy.abs(R - predicted) / numpy.abs(R)

    order = numpy.argsort(L)[::-1]
    L1, L2 = L[order[0]], L[order[-1]]
    gain = abs(L1 ** 2 / (L1 ** 2 - L2 ** 2)) if L1 != L2 else float("inf")

    return {
        "Rw": float(Rw),
        "K": float(K),
        "residual_max": float(residual.max()),
        "residual_rms": float(numpy.sqrt((residual ** 2).mean())),
        "noise_gain": float(gain),
        "inductance_ratio": float(L2 / L1) if L1 else float("nan"),
        "n_gaps": int(L.size),
    }


def deembed_parallel_capacitance(resistance, inductance, Cp, frequency):
    """Remove the shunt self-capacitance from a measured series resistance.

    Inversion of the parallel-C network.  The two square roots are
    sqrt(1+u)*sqrt(1-u) with u = 2*Rm*Cp*w*(L*Cp*w^2 - 1), real only for
    |u| <= 1.  That holds at and below resonance; above it the expression is
    complex and the old code produced NaN silently.  Here it is checked.
    """
    w = 2.0 * math.pi * numpy.asarray(frequency, dtype=float)
    Rm = numpy.asarray(resistance, dtype=float)
    L = numpy.asarray(inductance, dtype=float)

    u = 2.0 * Rm * Cp * w * (L * Cp * w ** 2 - 1.0)
    valid = numpy.abs(u) <= 1.0
    result = numpy.full_like(Rm, numpy.nan)
    if numpy.any(valid):
        denominator = 2.0 * Cp ** 2 * w[valid] ** 2 * Rm[valid]
        result[valid] = (1.0 - numpy.sqrt(1.0 - u[valid] ** 2)) / denominator
    return result, valid


# --------------------------------------------------------------------------
# Resonances -- the input to every capacitance value
# --------------------------------------------------------------------------


def average_cycles(data):
    """Average repeated measurement cycles onto a single frequency grid."""
    columns = [c for c in data.columns if c != "measurement_index"]
    return data[columns].groupby("frequency", as_index=False).mean().sort_values("frequency").reset_index(drop=True)


def detect_resonances(data, prominence_decades=0.08, use_phase=True):
    """Find series and parallel resonances in an impedance sweep.

    The old implementation called find_peaks on the LINEAR magnitude with
    prominence=2 -- meaning 2 ohms absolute.  Near a 100 kOhm parallel
    resonance ordinary ripple clears that; on a low-impedance DUT real
    resonances fall below it.  Detection here is on log10|Z| with a RELATIVE
    prominence in decades, so it behaves identically at any impedance level.

    Both papers plot phase alongside modulus because phase is the more reliable
    indicator; when `use_phase` and a phase column exists, every candidate is
    confirmed against a phase zero-crossing in the right direction.

    Returns a list of dicts sorted by frequency, each with keys
    frequency / impedance_magnitude / type / prominence / phase_confirmed.
    """
    averaged = average_cycles(data)
    if "magnitude" not in averaged.columns:
        raise ValueError("detect_resonances needs a 'magnitude' column (Z/phase sweep).")

    frequency = averaged["frequency"].to_numpy()
    magnitude = averaged["magnitude"].to_numpy()
    if numpy.any(magnitude <= 0):
        raise ValueError("Non-positive impedance magnitude in sweep.")
    log_magnitude = numpy.log10(magnitude)

    phase = averaged["phase"].to_numpy() if "phase" in averaged.columns else None

    log_frequency = numpy.log10(frequency)

    found = []
    for sign, kind in ((+1.0, "local maximum"), (-1.0, "local minimum")):
        indexes, properties = find_peaks(sign * log_magnitude, prominence=prominence_decades)
        for index, prominence in zip(indexes, properties["prominences"]):
            peak_frequency, peak_magnitude = _interpolate_peak(
                log_frequency, log_magnitude, index)
            found.append(
                {
                    "frequency": peak_frequency,
                    "impedance_magnitude": peak_magnitude,
                    "type": kind,
                    "prominence_decades": float(prominence),
                    "phase_confirmed": _confirm_with_phase(phase, index, kind),
                    "grid_frequency": float(frequency[index]),
                }
            )

    return sorted(found, key=lambda r: r["frequency"])


def _interpolate_peak(log_frequency, log_magnitude, index):
    """Parabolic refinement of a peak sitting between sweep points.

    A log sweep of 401 points over 10 kHz - 40 MHz steps 2.1% per point, so
    taking the nearest grid point costs up to ~1% in frequency -- and every
    capacitance goes as 1/f**2, so that is ~2% on the answer before any
    measurement noise. Fitting a parabola through the three points around the
    peak removes most of it.
    """
    if index <= 0 or index >= len(log_magnitude) - 1:
        return float(10.0 ** log_frequency[index]), float(10.0 ** log_magnitude[index])

    y0, y1, y2 = log_magnitude[index - 1], log_magnitude[index], log_magnitude[index + 1]
    denominator = y0 - 2.0 * y1 + y2
    if denominator == 0.0:
        return float(10.0 ** log_frequency[index]), float(10.0 ** log_magnitude[index])

    # Vertex offset in units of the (uniform in log f) grid step.
    offset = 0.5 * (y0 - y2) / denominator
    if abs(offset) > 1.0:
        return float(10.0 ** log_frequency[index]), float(10.0 ** log_magnitude[index])

    step = log_frequency[index + 1] - log_frequency[index]
    peak_log_frequency = log_frequency[index] + offset * step
    peak_log_magnitude = y1 - 0.25 * (y0 - y2) * offset
    return float(10.0 ** peak_log_frequency), float(10.0 ** peak_log_magnitude)


def _confirm_with_phase(phase, index, kind):
    """A parallel resonance takes phase +->-; a series resonance -->+."""
    if phase is None:
        return None
    low = max(0, index - 3)
    high = min(len(phase) - 1, index + 3)
    if high - low < 2:
        return None
    before, after = phase[low], phase[high]
    if kind == "local maximum":
        return bool(before > 0.0 > after)
    return bool(before < 0.0 < after)


def check_reciprocity(Z0, Z0_prime, Zsc, Zsc_prime, tolerance=0.05):
    """[BLA94] II-C: Z0 * Zsc' = Z0' * Zsc, for any linear two-port.

    "These four quantities are linked ... but their redundancy is useful when
    double checking."

    This is an end-to-end verification that the relay matrix actually produced
    the four topologies the script asked for -- checkable at every frequency
    point with no reference standard.  Given that the dominant failure mode of
    a switched fixture is a relay silently in the wrong state returning a
    plausible sweep, this is the most valuable self-test available.

    Returns (passed, max_relative_error, per_frequency_frame).
    """
    frames = [average_cycles(d)[["frequency", "magnitude"]] for d in (Z0, Z0_prime, Zsc, Zsc_prime)]
    merged = frames[0].rename(columns={"magnitude": "Z0"})
    for frame, name in zip(frames[1:], ("Z0p", "Zsc", "Zscp")):
        merged = merged.merge(frame.rename(columns={"magnitude": name}), on="frequency", how="inner")
    if merged.empty:
        raise ValueError("The four sweeps share no common frequency points.")

    left = merged["Z0"] * merged["Zscp"]
    right = merged["Z0p"] * merged["Zsc"]
    merged["relative_error"] = (left - right).abs() / left.abs()
    worst = float(merged["relative_error"].max())
    return worst <= tolerance, worst, merged


def check_linearity(low_drive, high_drive, column="inductance", tolerance=0.02):
    """[COG94] III: acquire a curve twice at two drive levels to prove linearity.

    "a curve can be acquired twice, with two different supply voltages, to
    check whether the behavior of the transformer is linear."

    The entire six-capacitance model assumes linear operation.  On an automated
    instrument this costs one extra sweep and turns an unstated assumption into
    a reported precondition.
    """
    a = average_cycles(low_drive)[["frequency", column]]
    b = average_cycles(high_drive)[["frequency", column]]
    merged = a.merge(b, on="frequency", suffixes=("_low", "_high"))
    if merged.empty:
        raise ValueError("Linearity sweeps share no common frequency points.")
    deviation = (merged[f"{column}_high"] - merged[f"{column}_low"]).abs() / merged[f"{column}_low"].abs()
    worst = float(deviation.max())
    return worst <= tolerance, worst, merged


# --------------------------------------------------------------------------
# Electrostatic model -- the six capacitances of [BLA94] Table 1
# --------------------------------------------------------------------------

# Each entry maps a primary/secondary winding link to the three capacitance
# sums measurable with that link, as symbolic combinations of the matrix
# coefficients (C11, C12, C13, C22, C23, C33) and eta.  Transcribed directly
# from [BLA94] Table 1 / [COG94] eq (9).
#
#   "C1_C3" is C1+C3, from the SHORT-circuit resonance
#   "C2_C3" is C2+C3, from the second OPEN-circuit resonance
#   "C1_C2" is C1+C2, from the first OPEN-circuit resonance

WINDING_LINKS = {
    # B connected to D  ->  V3 = 0
    "B-D": {
        "C1_C3": lambda c, n: c["C11"],
        "C2_C3": lambda c, n: n ** 2 * c["C22"],
        "C1_C2": lambda c, n: c["C11"] + n ** 2 * c["C22"] + 2 * n * c["C12"],
    },
    # A connected to C  ->  V3 = V1 - V2
    "A-C": {
        "C1_C3": lambda c, n: c["C11"] + c["C33"] + 2 * c["C13"],
        "C2_C3": lambda c, n: n ** 2 * (c["C22"] + c["C33"] - 2 * c["C23"]),
        "C1_C2": lambda c, n: (
            c["C11"] + c["C33"] + 2 * c["C13"]
            + n ** 2 * (c["C22"] + c["C33"] - 2 * c["C23"])
            + 2 * n * (c["C12"] - c["C33"] - c["C13"] + c["C23"])
        ),
    },
    # B connected to C
    "B-C": {
        "C1_C3": lambda c, n: c["C11"],
        "C2_C3": lambda c, n: n ** 2 * (c["C22"] + c["C33"] - 2 * c["C23"]),
        "C1_C2": lambda c, n: (
            c["C11"]
            + n ** 2 * (c["C22"] + c["C33"] - 2 * c["C23"])
            + 2 * n * (c["C12"] - c["C13"])
        ),
    },
    # A connected to D  ->  V3 = V1
    "A-D": {
        "C1_C3": lambda c, n: c["C11"] + c["C33"] + 2 * c["C13"],
        "C2_C3": lambda c, n: n ** 2 * c["C22"],
        "C1_C2": lambda c, n: (
            c["C11"] + c["C33"] + 2 * c["C13"]
            + n ** 2 * c["C22"]
            + 2 * n * (c["C12"] + c["C23"])
        ),
    },
    # No connection: V3 follows from capacitive division, I3 = 0
    "floating": {
        "C1_C3": lambda c, n: c["C11"] - c["C13"] ** 2 / c["C33"],
        "C2_C3": lambda c, n: n ** 2 * (c["C22"] - c["C23"] ** 2 / c["C33"]),
        "C1_C2": lambda c, n: (
            c["C11"] - c["C13"] ** 2 / c["C33"]
            + n ** 2 * (c["C22"] - c["C23"] ** 2 / c["C33"])
            + 2 * n * (c["C12"] - c["C13"] * c["C23"] / c["C33"])
        ),
    },
}

CAPACITANCE_UNKNOWNS = ("C11", "C12", "C13", "C22", "C23")
PICOFARAD = 1e-12


def capacitance_sums_from_resonances(resonances, L0, Lsc):
    """Turn one link's resonances into the three measurable capacitance sums.

    [COG94] section II-D / [BLA94] eq (8): with ls and Lp known, three
    independent capacitance sums follow from the resonance frequencies.

      C1+C2  <- first parallel (maximum) resonance of the OPEN sweep,   with L0
      C1+C3  <- first parallel (maximum) resonance of the SHORT sweep,  with Lsc
      C2+C3  <- second (minimum) resonance of the OPEN sweep,           with Lsc

    Missing resonances yield None rather than an exception: [SCH98] exists
    precisely because some key frequencies fall outside the instrument range,
    and the solver below simply uses fewer equations.
    """
    def first_frequency(items, kind):
        for item in items:
            if item["type"] == kind:
                return item["frequency"]
        return None

    def capacitance(frequency, inductance):
        if frequency is None:
            return None
        return 1.0 / (inductance * (2.0 * math.pi * frequency) ** 2)

    open_resonances = resonances["open"]
    short_resonances = resonances["short"]

    return {
        "C1_C2": capacitance(first_frequency(open_resonances, "local maximum"), L0),
        "C1_C3": capacitance(first_frequency(short_resonances, "local maximum"), Lsc),
        "C2_C3": capacitance(first_frequency(open_resonances, "local minimum"), Lsc),
    }


def solve_six_capacitances(measured_sums, C33, eta, initial_guess_pf=100.0):
    """Solve [BLA94] Table 1 for the six capacitance matrix coefficients.

    `measured_sums` maps a link name from WINDING_LINKS to a dict of
    {"C1_C2": value or None, "C1_C3": ..., "C2_C3": ...} in farads.
    `C33` is measured directly with both windings shorted -- [BLA94] II-D:
    "this coefficient ... matches the capacitance measured between the two
    windings when they are short-circuited."

    Solved in PICOFARADS.  The old code seeded fsolve at 1 farad for quantities
    of order 1e-10 and set the Cauchy loss knee nine orders of magnitude above
    the residuals, so the robust loss never engaged and the tolerances were met
    almost immediately at whatever the start point was.

    [BLA94] notes some capacitances are legitimately NEGATIVE -- "only
    capacitances which can be measured directly have to be positive" -- so no
    positivity bound is applied to the off-diagonal terms.
    """
    equations = []
    for link, sums in measured_sums.items():
        if link not in WINDING_LINKS:
            raise KeyError(f"Unknown winding link {link!r}; expected one of {sorted(WINDING_LINKS)}")
        for key, expression in WINDING_LINKS[link].items():
            measured = sums.get(key)
            if measured is None:
                continue
            equations.append((link, key, expression, measured / PICOFARAD))

    if len(equations) < len(CAPACITANCE_UNKNOWNS):
        raise ValueError(
            f"Only {len(equations)} usable resonance equations for "
            f"{len(CAPACITANCE_UNKNOWNS)} unknowns. Widen the sweep or add links."
        )

    C33_pf = C33 / PICOFARAD

    def residuals(x):
        values = dict(zip(CAPACITANCE_UNKNOWNS, x))
        values["C33"] = C33_pf
        return [expression(values, eta) - measured for _, _, expression, measured in equations]

    start = numpy.full(len(CAPACITANCE_UNKNOWNS), initial_guess_pf)
    solution = least_squares(residuals, start, loss="cauchy", f_scale=initial_guess_pf * 0.1,
                             xtol=1e-12, ftol=1e-12, gtol=1e-12)

    coefficients = {name: float(value) * PICOFARAD for name, value in zip(CAPACITANCE_UNKNOWNS, solution.x)}
    coefficients["C33"] = C33
    final = residuals(solution.x)
    scale = max(abs(measured) for _, _, _, measured in equations)
    return {
        "coefficients": coefficients,
        "equations_used": len(equations),
        "residual_max_pf": float(numpy.max(numpy.abs(final))),
        "residual_relative": float(numpy.max(numpy.abs(final)) / scale) if scale else float("nan"),
        "converged": bool(solution.success),
    }


def gamma_capacitances(coefficients, eta):
    """[COG94] eq (8): the placed capacitors y4, y5, y6 of the Fig. 1 circuit.

    Only three of the six survived text extraction from the scanned paper --
    equation (8) is typeset across a two-column break and y1..y3 are lost.
    They are NOT guessed here.  If you need the full placed-capacitor set for
    a SPICE netlist, read them off Fig. 1 / eq (8) of the printed paper and
    add them below.

    This costs nothing physically: the six matrix coefficients Cij returned by
    `solve_six_capacitances` already determine the entire electrostatic
    behaviour.  [BLA94] is explicit that the y placement "may even be rather
    free, provided energy WE stocked by the six capacitors together can be
    identified with WE" -- the y are one convenient layout, not the answer.

    Only directly measurable capacitances must be positive; the rest may be
    negative because couplings can reduce the stored energy.
    """
    c = coefficients
    n = eta
    return {
        "y4": c["C33"] + c["C13"] + n * c["C23"],
        "y5": -n * c["C23"],
        "y6": -c["C13"],
    }


def practical_capacitances(coefficients, eta):
    """The three numbers a magnetics designer actually asks for, from Cij.

    C33 is the direct inter-winding capacitance (measured, not fitted).
    The self terms are the energy-equivalent capacitance seen across each
    winding with the other winding floating and unexcited.
    """
    c = coefficients
    return {
        "C_primary_self": c["C11"],
        "C_secondary_self": c["C22"],
        "C_interwinding": c["C33"],
        "C_primary_secondary_coupling": c["C12"],
        "C_secondary_referred_to_primary": eta ** 2 * c["C22"],
    }


# --------------------------------------------------------------------------
# Convenience: full magnetic summary from the three canonical measurements
# --------------------------------------------------------------------------


def magnetic_summary(L0, Lsc, L0_prime, L_cum=None, L_dif=None):
    """Everything [COG94] II-B yields, plus optional mutual-inductance checks."""
    k = coupling_coefficient(L0, Lsc)
    eta = turns_ratio(L0, L0_prime)
    summary = {
        "L0": L0,
        "Lsc": Lsc,
        "L0_prime": L0_prime,
        "k": float(k),
        "eta": float(eta),
        "turns_ratio_N2_N1": float(effective_turns_ratio(L0, L0_prime, k)),
        "Lp_magnetizing": float(magnetizing_inductance(L0, k)),
        "ls_leakage": float(leakage_inductance(Lsc, k)),
    }
    if L_cum is not None and L_dif is not None:
        L_cum, L_dif, swapped = check_dot_convention(L_cum, L_dif)
        M = mutual_inductance(L_cum, L_dif)
        summary.update(
            {
                "L_cumulative": L_cum,
                "L_differential": L_dif,
                "dot_convention_swapped": swapped,
                "M": float(M),
                "Lm_from_mutual": float(magnetizing_from_mutual(M, eta)),
                "k_from_mutual": float(M / math.sqrt(L0 * L0_prime)),
            }
        )
        summary["k_consistency"] = abs(summary["k_from_mutual"] - summary["k"]) / summary["k"]
    return summary
