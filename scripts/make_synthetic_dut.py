"""
Write cached CSVs for a transformer with KNOWN parameters, so the whole
acquisition -> extraction -> report pipeline can be exercised offline.

    python make_synthetic_dut.py
    python MagneticCharacterizer.py SYNTHETIC --cache

Everything the characterizer reports should come back equal to the truth
printed here.  This is the end-to-end counterpart to test_model.py, which
tests the maths in isolation.

Sweeps are synthesized from the three-capacitor circuit of [BLA94] Fig. 4:

    Z_open(w) = (r + jwL0) * (1 - w^2/ws^2) / (1 - w^2/wp^2)  ||  Rp
        wp = 1/sqrt(L0 * (C1+C2))     parallel resonance, a maximum
        ws = 1/sqrt(Lsc * (C2+C3))    series resonance, a minimum

    Z_short(w) = (r + jwLsc)  ||  1/(jw(C1+C3))
        parallel resonance at 1/sqrt(Lsc * (C1+C3))

which is exactly the structure `capacitance_sums_from_resonances` inverts.
"""

import math
import pathlib

import numpy
import pandas

import RelayBoardController as rbc
import TransformerModel as tm

# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------

REFERENCE = "SYNTHETIC"

L0 = 2.40e-3          # primary open-circuit inductance
K = 0.982             # coupling
ETA = 2.0             # N2/N1, so L0' = eta^2 * L0
R1 = 0.185            # primary winding resistance, ohm
R2 = 0.640            # secondary winding resistance, ohm
RP = 46_000.0         # core loss resistance, ohm

TRUE_C = {            # [BLA94] capacitance matrix coefficients, farad
    "C11": 78.0e-12,
    "C12": -12.0e-12,
    "C13": 21.0e-12,
    "C22": 44.0e-12,
    "C23": -8.0e-12,
    "C33": 61.0e-12,
}

LSC = L0 * (1.0 - K ** 2)
L0_PRIME = ETA ** 2 * L0

POINTS = 401
RL_BAND = (100.0, 1_000_000.0)
Z_BAND = (10_000.0, 40_000_000.0)


def _frame(frequencies, **columns):
    data = {"measurement_index": 0, "frequency": frequencies}
    data.update(columns)
    single = pandas.DataFrame(data)
    second = single.copy()
    second["measurement_index"] = 1
    return pandas.concat([single, second], ignore_index=True)


def _log_sweep(band):
    return numpy.logspace(math.log10(band[0]), math.log10(band[1]), POINTS)


def open_impedance(w, L, r, C1_C2, C2_C3, Lsc, Rp):
    wp = 1.0 / math.sqrt(L * C1_C2)
    ws = 1.0 / math.sqrt(Lsc * C2_C3)
    ideal = (r + 1j * w * L) * (1.0 - (w / ws) ** 2) / (1.0 - (w / wp) ** 2)
    return 1.0 / (1.0 / ideal + 1.0 / Rp)


def short_impedance(w, Lsc, r, C1_C3):
    return 1.0 / (1.0 / (r + 1j * w * Lsc) + 1j * w * C1_C3)


def link_sums(link):
    """The three capacitance sums for one winding link, from [BLA94] Table 1."""
    expressions = tm.WINDING_LINKS[link]
    return {key: float(expression(TRUE_C, ETA)) for key, expression in expressions.items()}


def write_rl(path, frequencies, impedance):
    w = 2.0 * math.pi * frequencies
    _frame(frequencies, resistance=impedance.real,
           inductance=impedance.imag / w).to_csv(path, index=False)


def write_z(path, frequencies, impedance):
    _frame(frequencies, magnitude=numpy.abs(impedance),
           phase=numpy.degrees(numpy.angle(impedance))).to_csv(path, index=False)


def main():
    output = pathlib.Path(__file__).parent.resolve() / "output"
    output.mkdir(parents=True, exist_ok=True)

    def target(number, kind):
        name = rbc.CONFIGS[number]["name"]
        return output / f"{REFERENCE}_cfg{number:02d}_{name}_{kind}.csv"

    floating = link_sums("floating")
    rl_frequencies = _log_sweep(RL_BAND)
    z_frequencies = _log_sweep(Z_BAND)
    w_rl = 2.0 * math.pi * rl_frequencies
    w_z = 2.0 * math.pi * z_frequencies

    # --- config 1: Z0, primary with secondary open -------------------------
    write_rl(target(1, "RL"), rl_frequencies,
             open_impedance(w_rl, L0, R1, floating["C1_C2"], floating["C2_C3"], LSC, RP))
    write_z(target(1, "Z"), z_frequencies,
            open_impedance(w_z, L0, R1, floating["C1_C2"], floating["C2_C3"], LSC, RP))

    # --- config 2: Zsc, secondary shorted ----------------------------------
    r_sc = R1 + R2 / ETA ** 2
    write_rl(target(2, "RL"), rl_frequencies, short_impedance(w_rl, LSC, r_sc, floating["C1_C3"]))
    write_z(target(2, "Z"), z_frequencies, short_impedance(w_z, LSC, r_sc, floating["C1_C3"]))

    # --- config 3: Z0', secondary with primary open ------------------------
    # Referred through eta^2: impedance scales, capacitance scales inversely.
    write_rl(target(3, "RL"), rl_frequencies,
             open_impedance(w_rl, L0_PRIME, R2, floating["C1_C2"] / ETA ** 2,
                            floating["C2_C3"] / ETA ** 2, LSC * ETA ** 2, RP * ETA ** 2))
    write_z(target(3, "Z"), z_frequencies,
            open_impedance(w_z, L0_PRIME, R2, floating["C1_C2"] / ETA ** 2,
                           floating["C2_C3"] / ETA ** 2, LSC * ETA ** 2, RP * ETA ** 2))

    # --- config 4: Zsc', primary shorted -----------------------------------
    # Chosen so Z0 * Zsc' == Z0' * Zsc exactly: the reciprocity self-test must
    # pass on consistent data, which is only meaningful if it is not built in
    # by construction of a single formula.
    Z0 = open_impedance(w_z, L0, R1, floating["C1_C2"], floating["C2_C3"], LSC, RP)
    Z0p = open_impedance(w_z, L0_PRIME, R2, floating["C1_C2"] / ETA ** 2,
                         floating["C2_C3"] / ETA ** 2, LSC * ETA ** 2, RP * ETA ** 2)
    Zsc = short_impedance(w_z, LSC, r_sc, floating["C1_C3"])
    write_z(target(4, "Z"), z_frequencies, Z0p * Zsc / Z0)

    # --- configs 5, 6: series aiding and opposing --------------------------
    M = K * math.sqrt(L0 * L0_PRIME)
    for number, sign in ((5, +1.0), (6, -1.0)):
        L_series = L0 + L0_PRIME + 2.0 * sign * M
        r_series = R1 + R2
        write_rl(target(number, "RL"), rl_frequencies,
                 r_series + 1j * w_rl * L_series)

    # --- config 7: C33 measured directly -----------------------------------
    _frame(z_frequencies,
           capacitance=numpy.full_like(z_frequencies, TRUE_C["C33"])).to_csv(
        target(7, "Cs"), index=False)

    # --- configs 8..13: winding links --------------------------------------
    for link, numbers in rbc.LINK_CONFIGS.items():
        if link == "floating":
            continue                      # already written as configs 1 and 2
        sums = link_sums(link)
        write_z(target(numbers["open"], "Z"), z_frequencies,
                open_impedance(w_z, L0, R1, sums["C1_C2"], sums["C2_C3"], LSC, RP))
        write_z(target(numbers["short"], "Z"), z_frequencies,
                short_impedance(w_z, LSC, r_sc, sums["C1_C3"]))

    # --- linearity pair: identical, i.e. perfectly linear -------------------
    linear = open_impedance(w_rl, L0, R1, floating["C1_C2"], floating["C2_C3"], LSC, RP)
    for suffix in ("RL_drive_low", "RL_drive_high"):
        write_rl(target(1, suffix), rl_frequencies, linear)

    # --- three gaps for the AC resistance recipe ---------------------------
    # R = Rw + K*L^2 with Rw genuinely gap-independent, so the fit residual
    # must come back near zero.
    Rw_true, K_core = 0.240, 9.0e4
    gap_frequencies = numpy.logspace(3, math.log10(200_000.0), 201)
    for label, inductance in (("gap1", 2.40e-3), ("gap2", 1.20e-3), ("gap3", 6.0e-4)):
        skin = Rw_true * (1.0 + 0.30 * numpy.sqrt(gap_frequencies / 1e4))
        resistance = skin + K_core * inductance ** 2
        _frame(gap_frequencies, resistance=resistance,
               inductance=numpy.full_like(gap_frequencies, inductance)).to_csv(
            output / f"{REFERENCE}_acr_{label}_RL.csv", index=False)

    print("=" * 70)
    print(f"Synthetic DUT '{REFERENCE}' written to {output}")
    print("=" * 70)
    print("GROUND TRUTH")
    print(f"  k                     {K:.5f}")
    print(f"  eta (N2/N1)           {ETA:.4f}")
    print(f"  L0                    {L0*1e6:.3f} uH")
    print(f"  Lsc                   {LSC*1e6:.3f} uH")
    print(f"  Lp = L0(1+k)/2        {tm.magnetizing_inductance(L0, K)*1e6:.3f} uH")
    print(f"  ls = Lsc/k            {tm.leakage_inductance(LSC, K)*1e6:.3f} uH")
    print(f"  r1                    {R1*1e3:.2f} mohm")
    print(f"  r2                    {R2*1e3:.2f} mohm")
    print(f"  Rp                    {RP:.0f} ohm")
    for name, value in TRUE_C.items():
        print(f"  {name}                   {value*1e12:.2f} pF")
    print(f"  Rw (10 kHz, 3 gaps)   {Rw_true*(1+0.30)*1e3:.2f} mohm")
    print("=" * 70)
    print("\nNow run:  python MagneticCharacterizer.py SYNTHETIC --cache")


if __name__ == "__main__":
    main()
