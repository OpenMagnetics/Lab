"""
ngspice check of the rev C calibration architecture: does the SHORT standard
follow the DUT path, and what does that buy?

    python simulate_calibration.py

A lumped model of one measurement path (2-wire mode) is simulated in ngspice
(KiCad's ngspice.dll, in-process via scripts/SpiceModel.NgSpice) the way the
on-board B-WIC bridge sees it -- V at the HI feed over the current into the LO
shunt -- in four states: OPEN, SHORT, LOAD and DUT. The analyzer's three-term
OSL correction is then applied with ideal standard values, exactly as the
Bode 100 does, and the corrected result is compared with the true DUT plus the
two uncalibrated arms (the best a calibration at the iso-relay plane can do).

Two SHORT standards are compared on the same fixture:
  rev B   iso open, the HI terminal's own LO relay closed: the short runs
          through ONE terminal bus and skips the LO terminal's bus, matrix
          contact and the LO rail stretch to it
  rev C  iso energized, SHORT relays of the HI and LO terminals closed: the
          short runs through BOTH buses, both iso contacts (NO throw) and a
          Kelvin star -- the DUT path minus arms and DUT, plus two short-relay
          contacts and the star copper

Element values: 0.8 nH/mm and 0.04 pF/mm for 0.5 mm tracks with no plane
underneath; G6K closed contact 50 mohm + 5 nH, open contact 0.25 pF; lengths
from the rev C layout (parasitic_report.py) where routed, else estimates.
"""

import os
import sys

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
from SpiceModel import NgSpice  # noqa: E402

L_MM, C_MM = 0.8e-9, 0.04e-12
R_CONTACT, L_CONTACT, C_OPEN = 0.05, 5e-9, 0.25e-12
R_LOAD = 100.0

# rev C geometry (mm), terminal pair A (HI) - B (LO), a same-side pair
GEOMETRY = {
    "rail_hi": 52.5,      # HI feed -> column A NO pad (parasitic_report)
    "rail_lo": 105.1,     # column B NO pad -> LO feed (parasitic_report)
    "rail_hi_load": 96.5, # HI feed -> column E
    "rail_lo_load": 83.1, # column E -> LO feed
    "bus": 40.0,          # iso COM -> matrix COM, per terminal (estimate)
    "bus_hi_lo": 12.0,    # HI relay COM -> LO relay COM along one bus (rev B SHORT)
    "load_bus": 8.0,      # load-column relay COM -> R5, per side
    "arm": 9.7,           # clamp -> iso NC (force arm, parasitic_report)
    "short_feed": 10.0,   # iso NO throw -> SHORT relay COM
    "star": 8.0,          # between the two SHORT relays (same side)
}


def segment(name, a, b, length_mm, r=0.0):
    """Series R-L of a track, half its capacitance to ground at each end."""
    c = length_mm * C_MM / 2
    return [f"l{name} {a} {name}_m {length_mm * L_MM:.4e}",
            f"r{name} {name}_m {b} {max(r, 1e-4):.4e}",
            f"c{name}a {a} 0 {c:.4e}", f"c{name}b {b} 0 {c:.4e}"]


def contact(name, a, b, closed=True):
    if closed:
        return [f"l{name} {a} {name}_m {L_CONTACT:.4e}", f"r{name} {name}_m {b} {R_CONTACT}"]
    return [f"c{name} {a} {b} {C_OPEN:.4e}"]


def circuit(state, dut, short_style="relays"):
    """Netlist of the A-B path in one state. ``dut`` is a SPICE element body
    between nodes da and db (the clamps)."""
    g = GEOMETRY
    lines = ["* fixture path A-B", "vsrc src 0 dc 0 ac 1", "rs src hi 50"]
    # HI feed -> column A, LO: column B -> feed -> shunt
    lines += segment("rh", "hi", "colA", g["rail_hi"])
    lines += segment("rl", "colB", "lo", g["rail_lo"])
    lines += ["rsh lo 0 2.35"]
    # matrix contacts and buses (always closed in this configuration)
    lines += contact("kah", "colA", "busA_m")
    lines += contact("kbl", "colB", "busB_m")
    lines += segment("ba", "busA_m", "comA", g["bus"])
    lines += segment("bb", "busB_m", "comB", g["bus"])
    # the load column hangs on the rails through open contacts unless LOAD
    load = state == "LOAD"
    lines += segment("rhe", "hi", "colE_h", g["rail_hi_load"])
    lines += segment("rle", "colE_l", "lo", g["rail_lo_load"])
    lines += contact("keh", "colE_h", "ce", closed=load)
    lines += contact("kel", "colE_l", "cf", closed=load)
    lines += segment("lbe", "ce", "r5a", g["load_bus"])
    lines += segment("lbf", "cf", "r5b", g["load_bus"])
    lines += [f"r5 r5a r5b {R_LOAD}"]
    # isolation relays: NC to the clamp arm, NO to the SHORT feed
    measuring = state == "DUT"
    for t in ("A", "B"):
        lines += contact(f"i{t}nc", f"com{t}", f"arm{t}", closed=measuring)
        lines += contact(f"i{t}no", f"com{t}", f"iso{t}", closed=not measuring)
        lines += segment(f"ar{t}", f"arm{t}", f"d{t.lower()}", g["arm"])
        lines += segment(f"sf{t}", f"iso{t}", f"sh{t}", g["short_feed"])
    if measuring:
        lines += ["xdut da db dut", ".subckt dut 1 2", *dut.splitlines(), ".ends"]
    shorting = state == "SHORT"
    if short_style == "relays":
        lines += contact("sa", "shA", "star", closed=shorting)
        lines += contact("sb", "shB", "star_b", closed=shorting)
        lines += segment("st", "star", "star_b", g["star"])
    else:
        # rev B: SHORT relays absent (stubs end open); the short is the HI
        # terminal's own LO relay closing its bus onto the LO rail at column A
        lines += [f"csa shA 0 1e-15", f"csb shB 0 1e-15"]
        lines += segment("bhl", "busA_m", "busA_lo", g["bus_hi_lo"])
        lines += contact("kal", "busA_lo", "colA_lo", closed=shorting)
        # column A's tap on the LO rail sits (rail_lo + 44 mm) from the feed
        lines += segment("rla", "colA_lo", "lo", g["rail_lo"] + 44.0)
    lines += [".ac dec 30 1e3 3e7", ".end"]
    return lines


def measured(state, dut="", short_style="relays"):
    spice = NgSpice.get()
    spice.run(circuit(state, dut, short_style))
    f = spice.vector("frequency").real
    v_hi = spice.vector("hi")
    i_lo = spice.vector("lo") / 2.35
    return f, v_hi / i_lo


def corrected(zm, zo, zs, zl, r_load=R_LOAD):
    """Three-term OSL with ideal standards (open = inf, short = 0, load)."""
    c = (zs - zl) / (r_load * (zl - zo))
    return (zm - zs) / (c * (zo - zm))


DUTS = {
    "1 uH + 20 mohm (leakage)": ("l1 1 m 1e-6\nr1 m 2 0.02", lambda w: 0.02 + 1j * w * 1e-6),
    "1 ohm (Rac)": ("r1 1 2 1", lambda w: 1.0 + 0 * w),
    "10 pF (Cps)": ("c1 1 2 10e-12", lambda w: 1 / (1j * w * 10e-12)),
}


def arm_impedance(w):
    """Two uncalibrated force arms in series (the iso-plane floor)."""
    return 2 * (1e-4 + 1j * w * GEOMETRY["arm"] * L_MM)


def main():
    print(__doc__.split("\n\n")[0].strip())
    for style in ("B", "relays"):
        f, zo = measured("OPEN", short_style=style)
        _, zs = measured("SHORT", short_style=style)
        _, zl = measured("LOAD", short_style=style)
        print(f"\nSHORT standard of rev {style}:")
        for name, (body, truth) in DUTS.items():
            _, zm = measured("DUT", body, style)
            z = corrected(zm, zo, zs, zl)
            w = 2 * numpy.pi * f
            reference = truth(w) + (arm_impedance(w) if "pF" not in name else 0)
            residual = z - reference
            cells = []
            for target in (1e4, 1e5, 1e6, 1e7):
                i = int(numpy.argmin(abs(f - target)))
                if "pF" in name:
                    c_true = 10e-12
                    c_meas = -1 / (w[i] * z[i].imag)
                    cells.append(f"{target:7.0e} dC {1e15 * (c_meas - c_true):+7.1f} fF")
                else:
                    cells.append(f"{target:7.0e} dR {1e3 * residual[i].real:+6.1f} mohm "
                                 f"dL {1e9 * residual[i].imag / w[i]:+6.1f} nH")
            print(f"  {name:26s} " + "  ".join(cells))
    print("\nResidual = corrected - (DUT + arms): what the OSL leaves in the result. The arms "
          "are the iso-plane floor; 2-wire fixture compensation or Kelvin removes the rest.")


if __name__ == "__main__":
    main()
