"""
AC check of the rev C Kelvin front-end (schematic values), in ngspice.

    python simulate_frontend.py

Circuit as drawn in generate_schematic.py build_frontend(): AC-coupled unity
buffers (U7/U8), 0.1 % four-resistor difference stage (U9), buffered VMID
(U10), 49.9 R + 47 uF into the Bode 100's CH1 (50 R or 1 Mohm). OPA356 as a macro
model: 200 MHz GBW, single pole, 100 dB DC gain.

Reports differential gain/phase 10 Hz-50 MHz and the common-mode rejection
with a worst-case 0.1 % mismatch of the difference-stage resistors.
"""

import os
import sys

import numpy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
from SpiceModel import NgSpice  # noqa: E402

OPAMP = """
.subckt opamp inp inn out vcc vee
* 100 dB, single pole at 2 kHz -> 200 MHz GBW; 10 R output
egain n1 0 inp inn 1e5
rp n1 n2 1k
cp n2 0 79.6n
ebuf n3 0 n2 0 1
ro n3 out 10
.ends
"""


def netlist(mismatch, drive, load=50.0):
    """``drive``: 'diff' (SNS_HI = +1/2, SNS_LO = -1/2) or 'cm' (both +1)."""
    r_ref = 1000.0 * (1 + mismatch)
    r_fb = 1000.0 * (1 - mismatch)
    hi, lo = ("0.5", "-0.5") if drive == "diff" else ("1", "1")
    return f"""* rev C Kelvin front-end
{OPAMP}
vsup vcc 0 dc 5
vhi shi 0 dc 0 ac {hi}
vlo slo 0 dc 0 ac {lo}
* VMID: 18k/10k, 10 uF, buffered
r30 vcc vdiv 18k
r31 vdiv 0 10k
c23 vdiv 0 10u
xu10 vdiv vmid vmid vcc 0 opamp
* HI input
c20 shi finp 1u
r20 finp vmid 100k
r21 finp fp 100
xu7 fp fbp fbp vcc 0 opamp
* LO input
c21 slo finn 1u
r22 finn vmid 100k
r23 finn fn 100
xu8 fn fbn fbn vcc 0 opamp
* difference stage
r26 fbp fdp 1k
r27 fdp vmid {r_ref}
r24 fbn fdm 1k
r25 fdm famp {r_fb}
xu9 fdp fdm famp vcc 0 opamp
* output into the Bode 50 R input
r28 famp fr 49.9
c22 fr fout 47u
r29 fout 0 100k
rbode fout 0 {load}
.ac dec 40 10 50e6
.end
""".splitlines()


def response(mismatch, drive, load=50.0):
    spice = NgSpice.get()
    spice.run(netlist(mismatch, drive, load))
    return spice.vector("frequency").real, spice.vector("fout")


def main():
    frequency, diff = response(0.0, "diff")
    _, cm = response(0.001, "cm")
    _, diff_mm = response(0.001, "diff")
    print("Kelvin front-end, V(Bode CH1) / (V(SNS_HI) - V(SNS_LO)), into 50 R")
    print(f"{'f':>10} {'|G| dB':>8} {'phase':>8} {'CMRR dB (0.1% mm)':>18}")
    for target in (10, 100, 1e3, 1e4, 1e5, 1e6, 3e6, 1e7, 2e7, 5e7):
        i = int(numpy.argmin(abs(frequency - target)))
        gain = diff[i]
        cmrr = 20 * numpy.log10(abs(diff_mm[i]) / abs(cm[i]))
        print(f"{frequency[i]:10.3g} {20 * numpy.log10(abs(gain)):8.2f} "
              f"{numpy.degrees(numpy.angle(gain)):8.2f} {cmrr:18.1f}")
    for load in (50.0, 1e6):
        _, gain = response(0.0, "diff", load)
        reference = abs(gain[numpy.argmin(abs(frequency - 1e3))])
        flat = frequency[abs(abs(gain) / reference - 1) < 0.01]
        print(f"CH1 at {'50 R' if load == 50 else '1 Mohm'}: +/-1 % flat {flat.min():.3g} Hz"
              f" .. {flat.max():.3g} Hz, gain {20 * numpy.log10(reference):.2f} dB")
    print("(gain/phase sit inside the Kelvin OSL; only their stability matters)")


if __name__ == "__main__":
    main()
