"""
ngspice validation of the rev B calibration architecture.

    python simulate_calibration.py

Builds a full parasitic model of the fixture from the ACTUAL board geometry
(trace lengths from generate_pcb.py placement constants, relay parasitics from
the Omron G6K-RF datasheet), then:

  1. Simulates the analyzer's view of the four cal states -- OPEN, SHORT,
     LOAD, and a known DUT -- across 1 kHz .. 40 MHz.
  2. Applies the standard OSL correction (the same maths the Bode 100 runs):

         Zdut = Zstd_load * (Zm-Zo)(Zs-Zl) / [(Zm-Zl)(Zo-Zs)] ... solved via
         the bilinear three-standard formulation.

  3. Compares the corrected result against the true DUT, and against what an
     UNCALIBRATED measurement or a rails-referenced load (the rejected design)
     would have produced.

This answers the design question directly: do the fixture parasitics cancel
through calibration, and how much does the "load measured like the DUT"
architecture matter?

Fixture parasitic model (per element, from geometry):
  trace L ~ 0.8 nH/mm (0.5 mm trace, no plane underneath)
  trace C ~ 0.04 pF/mm (no plane under the signal region)
  relay closed contact: 50 mohm + 5 nH (through-package)
  relay open contact:   0.25 pF (20 dB isolation at 1 GHz in 50R)
  bridge feed: 8 nH + 1 pF (rails into the on-board B-WIC bridge -- rev B
  integrates the bridge, so the old edge-tab + external-adapter interface
  is gone; the BNC/coax side is behind the bridge and inside the Bode's
  own calibration plane)
"""

import math
import os
import re
import subprocess
import sys

NGSPICE = r"C:\Users\alfon\Downloads\ngspice-45.2_64\Spice64\bin\ngspice.exe"
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------- geometry (mm)
BUS_LENGTH = 44.0          # iso COM (y 59.6) .. LINK COM (y 97.6), + stubs
ARM_LENGTH = 11.4          # clamp pad .. iso NC pad
RAIL_LENGTH = 60.0         # average rail run from descent to a column tap
DESCENT_LENGTH = 30.0      # rail row .. bridge band (y 75.8/87.8 -> 111.5)

L_PER_MM = 0.8e-9
C_PER_MM = 0.04e-12
R_CONTACT = 0.05
L_CONTACT = 5e-9
C_OPEN_CONTACT = 0.25e-12
L_TAB = 8e-9               # bridge feed (was 20 nH tab + adapter in rev A)
C_TAB = 1e-12
R_CAL = 100.0

# The known DUT: 2.4 mH || 60 pF || 46k -- same scale as the SYNTHETIC test.
DUT_L, DUT_C, DUT_R = 2.4e-3, 60e-12, 46e3

FREQUENCIES = [1e3, 10e3, 100e3, 1e6, 5e6, 10e6, 20e6, 40e6]


def fixture_deck(termination, label):
    """One-port fixture as seen from the analyzer tabs; `termination` is the
    SPICE snippet across the terminal plane nodes tp / tn."""
    bus = BUS_LENGTH
    return f"""* rev B fixture -- {label}
* analyzer port
Vprobe port 0 dc 0 ac 1
Rsense port hi 1u
* tabs + descents (constant, both cal and measure)
Ltabh hi n1 {L_TAB + DESCENT_LENGTH * L_PER_MM}
Ctabh n1 0 {C_TAB + DESCENT_LENGTH * C_PER_MM}
Lrailh n1 n2 {RAIL_LENGTH * L_PER_MM}
Crailh n2 0 {RAIL_LENGTH * C_PER_MM}
* HI-side matrix contact + column bus
Rkh n2 n3 {R_CONTACT}
Lkh n3 n4 {L_CONTACT}
Lbush n4 tp {bus * L_PER_MM}
Cbush tp 0 {bus * C_PER_MM}
* open contacts of the three unused columns hanging on the HI rail
Copen1 n2 x1 {C_OPEN_CONTACT}
Rx1 x1 0 1G
Copen2 n2 x2 {C_OPEN_CONTACT}
Rx2 x2 0 1G
Copen3 n2 x3 {C_OPEN_CONTACT}
Rx3 x3 0 1G
* LO side, mirrored
Lbusl tn n5 {bus * L_PER_MM}
Cbusl tn 0 {bus * C_PER_MM}
Lkl n5 n6 {L_CONTACT}
Rkl n6 n7 {R_CONTACT}
Lraill n7 n8 {RAIL_LENGTH * L_PER_MM}
Craill n8 0 {RAIL_LENGTH * C_PER_MM}
Ltabl n8 0 {L_TAB + DESCENT_LENGTH * L_PER_MM}
Ctabl n8 0 {C_TAB + DESCENT_LENGTH * C_PER_MM}
{termination}
.control
ac dec 20 1k 40meg
wrdata {label}.txt v(hi) i(Vprobe)
.endc
.end
"""


TERMINATIONS = {
    # OPEN: iso energized -- terminal plane ends at an open contact.
    "open": f"Copen tp tpo {C_OPEN_CONTACT}\nRox tpo 0 1G\n"
            f"Copen2b tn tno {C_OPEN_CONTACT}\nRox2 tno 0 1G",
    # SHORT: the first HI terminal's LO relay also closed -- the bridge runs
    # back down the SAME column bus through a second contact.
    "short": f"Rsc tp s1 {R_CONTACT}\nLsc s1 tn {L_CONTACT}",
    # LOAD: the 100R column -- contact + full bus on each side (by layout,
    # identical bus length to a terminal column).
    "load": f"Rld tp l1 {R_CAL}\nCld tp tn 0.3p\nLld l1 tn 1n",
    # DUT: iso contact + clamp arm on each side, then the device.
    "dut": (f"Riso1 tp d1 {R_CONTACT}\nLiso1 d1 d2 {L_CONTACT}\n"
            f"Larm1 d2 d3 {ARM_LENGTH * L_PER_MM}\n"
            f"Ldut d3 d4 {DUT_L}\nCdut d3 d4 {DUT_C}\nRdut d3 d4 {DUT_R}\n"
            f"Larm2 d4 d5 {ARM_LENGTH * L_PER_MM}\n"
            f"Liso2 d5 d6 {L_CONTACT}\nRiso2 d6 tn {R_CONTACT}"),
    # The REJECTED design's load: straight across the rails, no column bus.
    "load_at_rails": None,   # handled specially below
}


def run_deck(label, termination):
    deck_path = os.path.join(HERE, f"sim_{label}.cir")
    with open(deck_path, "w") as handle:
        handle.write(fixture_deck(termination, f"sim_{label}"))
    result = subprocess.run([NGSPICE, "-b", deck_path], capture_output=True,
                            text=True, cwd=HERE, timeout=120)
    data_path = os.path.join(HERE, f"sim_{label}.txt")
    if not os.path.exists(data_path):
        print(result.stdout[-1500:])
        print(result.stderr[-1500:])
        raise RuntimeError(f"ngspice produced no output for {label}")
    return read_impedance(data_path)


def read_impedance(path):
    """wrdata AC layout: freq re(v) im(v) freq re(i) im(i)."""
    table = {}
    for line in open(path):
        fields = [float(x) for x in line.split()]
        if len(fields) < 6:
            continue
        frequency = fields[0]
        v = complex(fields[1], fields[2])
        i = complex(fields[4], fields[5])
        if abs(i) > 0:
            table[round(frequency, 3)] = v / i
    return table


def osl_correct(Zm, Zo, Zs, Zl, Z_load_definition):
    """Standard three-term one-port correction (identical to the analyzer's):
    maps measured Zm through the error two-port defined by the three
    standards, assuming they are ideal open / short / Z_load_definition."""
    # Bilinear transform: Z = (a*Zm + b) / (c*Zm + 1); solve a,b,c from the
    # three (measured, true) pairs: (Zo, inf), (Zs, 0), (Zl, Zload).
    # true = Zload * (Zm-Zs)(Zo-Zl) / [ (Zo-Zm)(Zl-Zs) ]  -- classic form.
    return Z_load_definition * (Zm - Zs) * (Zo - Zl) / ((Zo - Zm) * (Zl - Zs))


def true_dut(frequency):
    w = 2 * math.pi * frequency
    y = 1 / DUT_R + 1 / complex(0, w * DUT_L) + complex(0, w * DUT_C)
    return 1 / y


def main():
    if not os.path.exists(NGSPICE):
        raise SystemExit(f"ngspice not found at {NGSPICE}")

    print("Simulating fixture states (ngspice)...")
    measured = {}
    for label in ("open", "short", "load", "dut"):
        measured[label] = run_deck(label, TERMINATIONS[label])
        print(f"  {label}: {len(measured[label])} frequency points")

    # The rejected architecture: load applied straight across the rails
    # (no matrix contact, no column bus). Same OPEN and SHORT.
    rails_load = (f"Rld n2 lr {R_CAL}\nLld lr n7 1n")
    deck = fixture_deck("Copen tp tpo 0.25p\nRox tpo 0 1G", "sim_unused")
    # splice the rails load into an OPEN-terminated fixture:
    deck = deck.replace("Copen tp tpo 0.25p\nRox tpo 0 1G",
                        "Copen tp tpo 0.25p\nRox tpo 0 1G\n" + rails_load)
    path = os.path.join(HERE, "sim_load_rails.cir")
    open(path, "w").write(deck.replace("sim_unused", "sim_load_rails"))
    subprocess.run([NGSPICE, "-b", path], capture_output=True, text=True,
                   cwd=HERE, timeout=120)
    measured["load_rails"] = read_impedance(os.path.join(HERE, "sim_load_rails.txt"))

    print(f"\n{'freq':>10} | {'true DUT':>12} | {'raw error':>9} | "
          f"{'OSL error':>9} | {'rails-load error':>16}")
    print("-" * 70)
    worst_osl, worst_rails = 0.0, 0.0
    for frequency in FREQUENCIES:
        key = min(measured["dut"], key=lambda f: abs(f - frequency))
        Zm = measured["dut"][key]
        Zo = measured["open"][key]
        Zs = measured["short"][key]
        Zl = measured["load"][key]
        Zlr = measured["load_rails"][key]
        Zt = true_dut(key)

        corrected = osl_correct(Zm, Zo, Zs, Zl, R_CAL)
        corrected_rails = osl_correct(Zm, Zo, Zs, Zlr, R_CAL)

        raw_error = abs(Zm - Zt) / abs(Zt)
        osl_error = abs(corrected - Zt) / abs(Zt)
        rails_error = abs(corrected_rails - Zt) / abs(Zt)
        worst_osl = max(worst_osl, osl_error)
        worst_rails = max(worst_rails, rails_error)
        print(f"{key:10.3g} | {abs(Zt):12.4g} | {raw_error:8.2%} | "
              f"{osl_error:8.2%} | {rails_error:15.2%}")

    print("-" * 70)
    print(f"worst-case corrected error, rev B architecture : {worst_osl:.2%}")
    print(f"worst-case corrected error, load-at-rails      : {worst_rails:.2%}")
    print("\nResidual sources in the rev B number: the DUT arms and iso contacts")
    print("sit beyond the cal plane (~8 nH + 0.4 pF + 100 mohm per side).")

    for label in ("open", "short", "load", "dut", "load_rails"):
        for suffix in (".cir", ".txt"):
            path = os.path.join(HERE, f"sim_{label}{suffix}")
            if os.path.exists(path):
                os.remove(path)
    return worst_osl


if __name__ == "__main__":
    worst = main()
    sys.exit(0 if worst < 0.05 else 1)
