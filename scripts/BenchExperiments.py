"""
Bench experiments behind the rev B findings, as reusable functions.

Each one is what was run by hand while bringing the board up; bench.py
exposes them on the command line. All need the instruments except
`consistency`, which works on cached data only.
"""

import json
import pathlib

import numpy
import pandas

import CapacitanceFit as cfit
import RelayBoardController as rbc
import TransformerModel as tm

HERE = pathlib.Path(__file__).parent.resolve()
OUTPUT = HERE / "output"


def _z_median(characterizer, relay_word, band=(1e6, 2e7)):
    """Series L and R of a board-only path in the current calibration."""
    characterizer.relay_board._apply_word(relay_word)
    data = characterizer.bode_100.take_Z_phase_measurement(
        start_frequency=1e5, stop_frequency=3e7, number_of_measurement_cycles=2,
        number_of_measurement_points=201, source_power_dbm=13)
    averaged = tm.average_cycles(data)
    f = averaged.frequency.to_numpy()
    z = averaged.magnitude.to_numpy() * numpy.exp(1j * numpy.radians(averaged.phase.to_numpy()))
    keep = (f > band[0]) & (f < band[1])
    return float(numpy.median((z.imag / (2 * numpy.pi * f))[keep])), float(numpy.median(z.real[keep]))


def column_map(characterizer):
    """Every column and LINK loop relative to column A (DUT isolated).

    With the calibration of path A-B the OSL SHORT runs through A's column, so
    HI -> t column -> LO measures column t minus column A, and
    A column -> LINK -> t column -> LO measures the LINK loop to t.
    """
    isolate = sum(1 << b for b in rbc.ISOLATE_BITS.values())
    bit = rbc.MATRIX_BITS
    word = lambda pairs: isolate | sum(1 << bit[p] for p in pairs)
    characterizer._set_config(1)
    base = _z_median(characterizer, word([("A", "HI"), ("A", "LO"), ("B", "LO")]))
    result = {"reference": "column A (calibration of path A-B)", "column": {}, "link_from_A": {}}
    for terminal in "BCD":
        L, R = _z_median(characterizer, word([("A", "HI"), (terminal, "HI"), (terminal, "LO")]))
        result["column"][terminal] = {"L_nh": (L - base[0]) * 1e9, "R_mohm": (R - base[1]) * 1e3}
        L, R = _z_median(characterizer, word([("A", "HI"), ("A", "LINK"), (terminal, "LINK"), (terminal, "LO")]))
        result["link_from_A"][terminal] = {"L_nh": (L - base[0]) * 1e9, "R_mohm": (R - base[1]) * 1e3}
    characterizer.relay_board.reset()
    for kind in ("column", "link_from_A"):
        print(f"  {kind:12s} " + "  ".join(f"{t}: {v['L_nh']:+6.1f} nH {v['R_mohm']:+6.1f} mohm"
                                         for t, v in result[kind].items()))
    return result


def link_calibration_test(characterizer, sweep=None):
    """Lcum with its own OSL vs with a LINK-free OSL of path A-D.

    Proved (2026-10-11) that the LINK rail in the calibration state is not
    what makes LINK-linked states read low (<0.06 pF difference).
    """
    sweep = sweep or dict(start_frequency=1e4, stop_frequency=5e7, number_of_measurement_cycles=2,
                          number_of_measurement_points=401, source_power_dbm=13)
    rbc.CONFIGS.setdefault(20, rbc._config(20, "AD_nolink", "A", "D", (), "Lcum without LINK, for a LINK-free OSL"))

    def admittance(data):
        averaged = tm.average_cycles(data)
        z = averaged.magnitude.to_numpy() * numpy.exp(1j * numpy.radians(averaged.phase.to_numpy()))
        return averaged.frequency.to_numpy(), 1 / z

    characterizer._set_config(5)
    f, own = admittance(characterizer.bode_100.take_Z_phase_measurement(**sweep))
    characterizer._set_config(20)
    characterizer.relay_board.set_config(5)
    _, other = admittance(characterizer.bode_100.take_Z_phase_measurement(**sweep))
    characterizer.relay_board.reset()
    difference = (own - other).imag / (2 * numpy.pi * f) * 1e12
    keep = (f > 1e6) & (f < 2e7)
    value = float(numpy.median(numpy.abs(difference[keep])))
    print(f"  Lcum own-OSL vs LINK-free-OSL: median |dC| {value:.3f} pF (1-20 MHz)")
    return {"median_abs_dC_pf": value}


def drive_sweep(characterizer, levels=(-30, -20, -10, 0, 5, 10, 13), config_number=1):
    """Amplitude permeability: L and R at 10 kHz / 100 kHz vs drive, plus
    the 100 ohm LOAD standard at each level (calibration drive independence)."""
    characterizer._set_config(config_number)
    rows = []
    for dbm in levels:
        averaged = tm.average_cycles(characterizer.bode_100.take_Rs_Ls_measurement(
            1e3, 1e6, number_of_measurement_cycles=2, number_of_measurement_points=61, source_power_dbm=dbm))
        row = {"dbm": dbm}
        for f in (1e4, 1e5):
            i = int(numpy.argmin(numpy.abs(averaged.frequency - f)))
            row[f"L_{int(f/1e3)}k_uH"] = float(averaged.inductance[i] * 1e6)
            row[f"R_{int(f/1e3)}k_ohm"] = float(averaged.resistance[i])
        rows.append(row)
    characterizer.relay_board.set_calibration_mode("LOAD")
    for row in rows:
        data = characterizer.bode_100.take_Z_phase_measurement(1e3, 1e6, 2, 21, row["dbm"])
        mean = data.groupby("frequency").magnitude.mean()
        row["load_error_pct"] = float(numpy.median(numpy.abs(mean / rbc.CALIBRATION_LOAD_OHM - 1)) * 100)
    characterizer.relay_board.reset()
    frame = pandas.DataFrame(rows)
    print(frame.to_string(index=False))
    return frame


def consistency(reference):
    """Model-independent checks on cached data (no instruments).

    * open-link differences against the ground-free [BLA94] relations for the
      reference's eta (two numbers, C33 and u, explain four differences);
    * the Lcum identity 4*Y(Lcum) = Y(BC open) for a ground-free DUT -- its
      residual is the guarded-measurement fixture term;
    * reciprocity raw and with board paths subtracted.
    """
    summary = cfit.differential_summary(str(OUTPUT), reference)
    od = {k: v["value"] for k, v in summary["open_differences"].items()}
    C33 = summary["electrostatic"]["C33_pf"]
    f5, Y5 = cfit.load_admittance(str(OUTPUT), reference, 5)
    _, Y12 = cfit.load_admittance(str(OUTPUT), reference, 12)
    keep = (f5 >= 1e6) & (f5 <= 8e6)
    identity = float(numpy.median(cfit.effective_capacitance(f5, 4 * Y5.mean(0) - Y12.mean(0))[keep]) * 1e12)
    result = {
        "C33_pf": C33,
        "open_differences_pf": od,
        "AC_minus_BD_pf (0 for any lumped DUT)": od["AC"],
        "BC_AD_mean_minus_C33_pf (clamp terms)": (od["BC"] + od["AD"]) / 2 - C33,
        "Lcum_identity_pf (fixture, 0 if ground-free)": identity,
        "u_open_pf": summary["u_open"]["value"],
        "u_short_pf": summary["u_short"]["value"],
        "link_net_to_ground_pf": summary["link_net_to_ground_pf"],
    }
    for key, value in result.items():
        if not isinstance(value, dict):
            print(f"  {key:46s} {value:+8.3f}")
    return result


def manual_jumper_check(path_prefix, eta):
    """The ground-free [BLA94] relations on an old manual-jumper dataset
    (C:/Users/.../Lab/output/<ref>_advanced_capacitance_characterization_data_)."""
    from scipy.optimize import least_squares

    def Y(name):
        averaged = tm.average_cycles(pandas.read_csv(f"{path_prefix}{name}.csv"))
        z = averaged.magnitude * numpy.exp(1j * numpy.radians(averaged.phase))
        return averaged.frequency.to_numpy(), 1 / z.to_numpy()

    f, ybd = Y("group_1_B_with_D_open")
    keep = (f > 1e6) & (f < 8e6)
    w = 2 * numpy.pi * f
    m = {k: float(numpy.median(((Y(n)[1] - ybd).imag / w)[keep]) * 1e12)
         for k, n in (("AC", "group_2_A_with_C_open"), ("BC", "group_3_B_with_C_open"),
                      ("AD", "group_4_A_with_D_open"), ("floating", "group_5_all_floating_open"))}
    c33 = tm.average_cycles(pandas.read_csv(f"{path_prefix}group_6_A_with_B_and_C_with_D.csv"))
    c33 = float(numpy.median(c33[(c33.frequency > 1e5) & (c33.frequency < 5e6)].capacitance) * 1e12)

    def predict(p):
        C, u = p
        return {"AD": C + 2 * u, "BC": eta * (eta * C - 2 * u), "AC": (1 - eta) ** 2 * C + 2 * (1 - eta) * u,
                "floating": -u * u / C}

    s = least_squares(lambda p: [predict(p)[k] - m[k] for k in m] + [p[0] - c33], [c33, 0])
    worst = max(abs(m[k] - predict(s.x)[k]) for k in m)
    print(f"  C33 direct {c33:.2f} pF, fit C33 {s.x[0]:.2f}, u {s.x[1]:.2f}; worst residual {worst:.2f} pF")
    return {"C33_direct_pf": c33, "C33_fit_pf": float(s.x[0]), "u_pf": float(s.x[1]), "worst_residual_pf": worst}
