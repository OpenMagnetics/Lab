"""
SPICE model of a characterized transformer, and ngspice simulation of every
relay configuration so the model can be laid over the measured sweeps.

    python SpiceModel.py REFERENCE            writes output/REFERENCE.lib and
                                              output/REFERENCE_spice_vs_measured.csv

The subcircuit is the [COG94] two-winding model in SPICE-native form:
coupled inductors L1 = L0 and L2 = L0' with coupling k (equivalent to the
magnetizing + leakage T model), series winding resistances r1 and r2, the
core-loss resistance Rp across the primary, and the six [BLA94] capacitances
as six branch capacitors between the terminals A, B (primary, dot on A) and
C, D (secondary, dot on C). Values come from output/REFERENCE_results.json.

Simulation testbenches reproduce what the bench measures: HI driven at 1 V,
the current sensed in the LO rail only (the bridge is a guarded, three-
terminal measurement), LINK terminals tied together and floating, floating
terminals left open. Optionally the measured fixture terms (LINK net and
isolated clamps to ground) are added, which is what the raw data contain.

ngspice runs in-process through its shared library; KiCad ships one
(ngspice.dll), which is found automatically on Windows. Set NGSPICE_LIBRARY
to use another.
"""

import ctypes
import glob
import json
import os
import pathlib
import sys

import numpy

import RelayBoardController as rbc

HERE = pathlib.Path(__file__).parent.resolve()
OUTPUT = HERE / "output"


# --------------------------------------------------------------------------
# ngspice shared library
# --------------------------------------------------------------------------

def _find_library():
    explicit = os.environ.get("NGSPICE_LIBRARY")
    if explicit:
        return explicit
    candidates = sorted(glob.glob(r"C:\Program Files\KiCad\*\bin\ngspice.dll"), reverse=True)
    candidates += ["libngspice.so", "libngspice.so.0", "libngspice.dylib"]
    for candidate in candidates:
        if os.path.isabs(candidate) and not os.path.exists(candidate):
            continue
        return candidate
    raise RuntimeError("ngspice shared library not found; install KiCad or set NGSPICE_LIBRARY")


class _VectorInfo(ctypes.Structure):
    _fields_ = [("v_name", ctypes.c_char_p), ("v_type", ctypes.c_int), ("v_flags", ctypes.c_short),
                ("v_realdata", ctypes.POINTER(ctypes.c_double)),
                ("v_compdata", ctypes.POINTER(ctypes.c_double)), ("v_length", ctypes.c_int)]


class NgSpice:
    """Minimal in-process ngspice: load a netlist, run, read complex vectors."""

    _instance = None

    def __init__(self):
        path = _find_library()
        if os.path.isabs(path) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(os.path.dirname(path))
        self.library = ctypes.CDLL(path)
        self.output = []
        send_char = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
        send_stat = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)
        controlled_exit = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool,
                                           ctypes.c_int, ctypes.c_void_p)
        # Keep references: ctypes callbacks must outlive the library's use of them.
        self._callbacks = (send_char(lambda s, i, p: self.output.append(s.decode(errors="ignore")) or 0),
                           send_stat(lambda s, i, p: 0),
                           controlled_exit(lambda *a: 0))
        self.library.ngSpice_Init(*self._callbacks, None, None, None, None)
        self.library.ngGet_Vec_Info.restype = ctypes.POINTER(_VectorInfo)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def run(self, lines):
        self.output.clear()
        self.library.ngSpice_Command(b"destroy all")
        self.library.ngSpice_Command(b"remcirc")
        array = (ctypes.c_char_p * (len(lines) + 1))(*[line.encode() for line in lines], None)
        self.library.ngSpice_Circ(array)
        self.library.ngSpice_Command(b"run")

    def vector(self, name):
        info = self.library.ngGet_Vec_Info(name.encode())
        if not info:
            raise KeyError(f"ngspice vector {name!r} not found; output: {''.join(self.output[-5:])}")
        info = info.contents
        n = info.v_length
        if info.v_compdata:
            data = numpy.ctypeslib.as_array(info.v_compdata, shape=(2 * n,)).reshape(n, 2)
            return data[:, 0] + 1j * data[:, 1]
        return numpy.ctypeslib.as_array(info.v_realdata, shape=(n,)).copy()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def model_parameters(results):
    """Element values (SI) from a characterization results dict."""
    magnetic = results["magnetic"]
    losses = results.get("losses", {})
    branches = results.get("capacitance_differential", {}).get("branches_pf", {})
    return {
        "L1": magnetic["L0"],
        "L2": magnetic["L0_prime"],
        "k": min(magnetic["k"], 0.999999),
        "r1": losses.get("r1", 0.1),
        "r2": losses.get("r2_secondary", 0.1),
        "Rp": losses.get("Rp") or 1e9,
        "C": {name: value * 1e-12 for name, value in branches.items()},
    }


def subcircuit(reference, parameters, name=None):
    """.subckt text. Pins: A B (primary, dot A) C D (secondary, dot C)."""
    name = name or f"OM_{reference}".replace("-", "_")
    p = parameters
    lines = [
        f"* {reference}: OpenMagnetics bench extraction (small-signal)",
        "* [COG94] coupled-inductor model + [BLA94] six capacitances as branch capacitors",
        f".SUBCKT {name} A B C D",
        f"R1 A N1 {p['r1']:.6g}",
        f"L1 N1 B {p['L1']:.6g}",
        f"R2 C N2 {p['r2']:.6g}",
        f"L2 N2 D {p['L2']:.6g}",
        f"K12 L1 L2 {p['k']:.8f}",
        f"RP A B {p['Rp']:.6g}",
    ]
    for branch, value in p["C"].items():
        if abs(value) < 1e-16:
            continue
        note = "  ; negative branch: an energy coefficient, valid in AC analysis" if value < 0 else ""
        lines.append(f"C{branch} {branch[0]} {branch[1]} {value:.6g}{note}")
    lines.append(f".ENDS {name}")
    return name, lines


def testbench(model_name, model_lines, config, fixture=None, start=1e4, stop=5e7, points_per_decade=100):
    """AC testbench of one relay configuration: Z = V(HI) / I(LO rail).

    With `fixture`, the measured board terms are included as on the bench:
    LINK net and isolated clamps to ground (unmeasured return) and, where it
    is known (fixture_paths.json), the series board path of the state.
    """
    series = (fixture or {}).get("series", {}).get(config["number"])
    lines = ["* relay configuration " + config["name"], *model_lines]
    if series:
        lines += ["VSRC src 0 AC 1", f"RPATH src srcl {series[1]:.6g}", f"LPATH srcl hi {series[0]:.6g}"]
    else:
        lines.append("VSRC hi 0 AC 1")
    lines.append("VLO lo 0 DC 0")
    node = {}
    for terminal in "ABCD":
        if terminal in config["HI"]:
            node[terminal] = "hi"
        elif terminal in config["LO"]:
            node[terminal] = "lo"
        elif terminal in config["LINK"]:
            node[terminal] = "link"
        else:
            node[terminal] = f"float_{terminal}"
    lines.append(f"XDUT {node['A']} {node['B']} {node['C']} {node['D']} {model_name}")
    # DC paths for floating nodes (no effect at AC).
    for name in set(node.values()):
        if name not in ("hi", "lo"):
            lines.append(f"RDC_{name} {name} 0 1e12")
    if fixture:
        # What the raw data contain: strays to board ground are not measured
        # (they return outside VLO), exactly as on the bench.
        if config["LINK"]:
            lines.append(f"CLINKG link 0 {fixture.get('link_net_to_ground_pf', 0) * 1e-12:.6g}")
        for terminal in rbc.floating_terminals(config):
            lines.append(f"CFG_{terminal} float_{terminal} 0 {fixture.get('clamp_to_ground_pf', 0) * 1e-12:.6g}")
            lines.append(f"CFH_{terminal} float_{terminal} hi {fixture.get('clamp_to_hi_pf', 0) * 1e-12:.6g}")
    lines += [f".ac dec {points_per_decade} {start:g} {stop:g}", ".end"]
    return lines


def simulate(results, config_number, fixture=None, start=1e4, stop=5e7, model=None):
    """(frequency, Z) of a model in one relay configuration. `model` is
    (name, lines) from subcircuit/broadband_subcircuit; default small-signal."""
    name, model = model or subcircuit("model", model_parameters(results), name="DUTMODEL")
    spice = NgSpice.get()
    spice.run(testbench(name, model, rbc.CONFIGS[config_number], fixture, start, stop))
    frequency = numpy.real(spice.vector("frequency"))
    current = spice.vector("i(vlo)")
    # ngspice reports the current flowing into VLO's + node (lo -> 0 through
    # the source): the DUT current I_LO.
    return frequency, 1.0 / current


def fixture_terms(results):
    d = results.get("capacitance_differential", {})
    terms = {"link_net_to_ground_pf": d.get("link_net_to_ground_pf", 0.0),
             "clamp_to_ground_pf": d.get("clamp_to_ground_pf", {}).get("value", 0.0),
             "clamp_to_hi_pf": d.get("clamp_to_HI_pf", {}).get("value", 0.0),
             "series": {}}
    paths = results.get("fixture_paths")
    if paths:
        for number, (driven, shorted) in {2: ("AB", "CD"), 4: ("CD", "AB")}.items():
            terms["series"][number] = (
                paths[driven]["other_column"]["L_h"] + paths[shorted]["link_loop"]["L_h"],
                paths[driven]["other_column"]["R_ohm"] + paths[shorted]["link_loop"]["R_ohm"])
    return terms


def load_results(reference):
    return json.loads((OUTPUT / f"{reference}_results.json").read_text())["results"]


def compare(reference, results, model, fixture=True):
    """Measured vs simulated for every cached configuration -> DataFrame."""
    import CapacitanceFit as cfit
    import pandas
    rows = []
    terms = fixture_terms(results) if fixture else None
    for number in sorted(rbc.CONFIGS):
        try:
            f_meas, Y = cfit.load_admittance(str(OUTPUT), reference, number)
        except FileNotFoundError:
            continue
        z_meas = 1.0 / Y.mean(axis=0)
        f_sim, z_sim = simulate(results, number, terms, model=model)
        z_sim = numpy.interp(f_meas, f_sim, z_sim.real) + 1j * numpy.interp(f_meas, f_sim, z_sim.imag)
        for f, zm, zs in zip(f_meas, z_meas, z_sim):
            rows.append((number, f, abs(zm), numpy.degrees(numpy.angle(zm)), abs(zs), numpy.degrees(numpy.angle(zs))))
    return pandas.DataFrame(rows, columns=["config", "frequency", "measured_mag", "measured_phase",
                                           "spice_mag", "spice_phase"])


def main():
    reference = sys.argv[1] if len(sys.argv) > 1 else "RM_concentric_20261011"
    results = load_results(reference)
    small_name, small = subcircuit(reference, model_parameters(results))
    broadband = broadband_parameters(reference, results)
    wide_name, wide = broadband_subcircuit(reference, broadband)
    library = OUTPUT / f"{reference}.lib"
    library.write_text(chr(10).join(small) + chr(10) * 2 + chr(10).join(wide) + chr(10))
    print(f"Wrote {library}: {small_name} (small-signal, constant elements) and {wide_name} "
          f"(broadband: leakage fit {broadband['leakage']['median_error']*100:.1f} %, "
          f"core+dielectric fit {broadband['shunt']['median_error']*100:.1f} %)")
    frame = compare(reference, results, (wide_name, wide))
    for number, group in frame.groupby("config"):
        band = group.frequency <= 1e7
        ratio = (group.spice_mag[band] / group.measured_mag[band] - 1).abs()
        print(f"  config {number:2d} {rbc.CONFIGS[number]['name']:18s} median |Z| error to 10 MHz: {ratio.median()*100:6.2f} %")
    destination = OUTPUT / f"{reference}_spice_vs_measured.csv"
    frame.to_csv(destination, index=False)
    print(f"Wrote {destination}")




# --------------------------------------------------------------------------
# Broadband model: passive networks fitted to the measured sweeps
# --------------------------------------------------------------------------
#
# Constant L, k, R and C reproduce neither the ferrite's mu(f), nor the
# leakage's proximity roll-off (1.0 -> 0.68 uH from 10 kHz to 1 MHz on the RM
# part), nor the self-capacitance, which on both measured parts behaves as a
# LOSSY, RELAXING dielectric: a passive network of RL branches plus a constant
# C misses the open-circuit sweep by 14-24 %, while RL plus Debye RC branches
# fits it to ~2 % from 10 kHz to 30 MHz. So:
#   * magnetizing node M-B: RL branches (core) + RC branches (self-capacitance,
#     dispersive) + a conductance, fitted by NNLS to the open-circuit
#     admittance of config 1 directly (no subtraction near resonance);
#   * A-M: leakage as a Foster ladder (R0, L_inf, R_i || L_i), fitted to the
#     shorted state with its board path removed;
#   * ideal transformer (eta) to C-D, and the four interwinding branch
#     capacitors AC, AD, BC, BD at the terminals (the self branches AB, CD are
#     inside the fitted network).
# Every element is a positive R, L or C: the subcircuit is passive and causal
# and runs in any SPICE. It is valid at the drive level of the sweeps it was
# fitted to (CAPACITANCE_SWEEP: 13 dBm).

from scipy.optimize import nnls


def _nnls(columns, target):
    """Relative-error NNLS with column normalisation; returns (x, rel. error)."""
    A = numpy.stack(columns, axis=1)
    scale = 1.0 / numpy.abs(target)
    Aw = A * scale[:, None]
    M = numpy.vstack([Aw.real, Aw.imag])
    b = numpy.concatenate([(target * scale).real, (target * scale).imag])
    norms = numpy.linalg.norm(M, axis=0)
    norms[norms == 0] = 1.0
    x, _ = nnls(M / norms, b, maxiter=200000)
    x = x / norms
    return x, numpy.abs(A @ x / target - 1)


def _leakage_fit(frequency, impedance):
    w = 2 * numpy.pi * frequency
    tau = numpy.logspace(numpy.log10(1 / (2 * numpy.pi * 1e8)), numpy.log10(1 / (2 * numpy.pi * 1e3)), 24)
    columns = [1j * w * t / (1 + 1j * w * t) for t in tau] + [numpy.ones_like(w, dtype=complex), 1j * w * 1e-6]
    x, error = _nnls(columns, impedance)
    sections = [(float(r), float(r * t)) for r, t in zip(x[:len(tau)], tau) if r > 0]
    return {"sections": sections, "R0": float(x[-2]), "L_inf": float(x[-1]) * 1e-6,
            "median_error": float(numpy.median(error))}


def _shunt_fit(frequency, admittance):
    """Y = sum 1/(R_i + jw L_i) + sum 1/(R_j + 1/(jw C_j)) + G (all >= 0)."""
    w = 2 * numpy.pi * frequency
    tau_l = numpy.logspace(numpy.log10(1 / (2 * numpy.pi * 3e8)), numpy.log10(1 / (2 * numpy.pi * 10)), 40)
    tau_c = numpy.logspace(numpy.log10(1 / (2 * numpy.pi * 1e9)), numpy.log10(1 / (2 * numpy.pi * 1e4)), 30)
    columns = ([1 / (1 + 1j * w * t) for t in tau_l]                 # coefficient 1/R_i, L_i = R_i tau
               + [1j * w * t / (1 + 1j * w * t) for t in tau_c]      # coefficient 1/R_j, C_j = tau / R_j
               + [numpy.ones_like(w, dtype=complex)])
    x, error = _nnls(columns, admittance)
    rl = [(1 / g, t / g) for g, t in zip(x[:len(tau_l)], tau_l) if g > 0]
    rc = [(1 / g, t * g) for g, t in zip(x[len(tau_l):len(tau_l) + len(tau_c)], tau_c) if g > 0]
    return {"rl": [(float(r), float(l)) for r, l in rl], "rc": [(float(r), float(c)) for r, c in rc],
            "G": float(x[-1]), "median_error": float(numpy.median(error))}


def broadband_parameters(reference, results, kind="Zhd", band=(1e4, 3e7)):
    """Fit the shunt (core + dielectric) and leakage networks to the sweeps."""
    import CapacitanceFit as cfit
    d = results["capacitance_differential"]
    C_short = d.get("C11", {}).get("value", 0.0) * 1e-12

    f, Y0 = cfit.load_admittance(str(OUTPUT), reference, 1, kind)
    _, Ysc = cfit.load_admittance(str(OUTPUT), reference, 2, kind)
    y0, ysc = Y0.mean(axis=0), Ysc.mean(axis=0)
    w = 2 * numpy.pi * f

    z_sc = 1.0 / (ysc - 1j * w * C_short)
    path = results.get("fixture_paths")
    if path:
        L_path = path["AB"]["other_column"]["L_h"] + path["CD"]["link_loop"]["L_h"]
        R_path = path["AB"]["other_column"]["R_ohm"] + path["CD"]["link_loop"]["R_ohm"]
        z_sc = z_sc - (R_path + 1j * w * L_path)
    leak_band = (f >= band[0]) & (f <= 2e7)
    leakage = _leakage_fit(f[leak_band], z_sc[leak_band])

    open_band = (f >= band[0]) & (f <= band[1])
    shunt = _shunt_fit(f[open_band], y0[open_band])
    branches = d.get("branches_pf", {})
    interwinding = {k: branches[k] * 1e-12 for k in ("AC", "AD", "BC", "BD") if k in branches}
    return {"leakage": leakage, "shunt": shunt, "eta": results["magnetic"]["eta"],
            "C": interwinding, "drive_dbm": 13, "band": band}


def broadband_subcircuit(reference, parameters, name=None):
    """Passive broadband .subckt (pins A B C D) from broadband_parameters."""
    name = name or f"OM_{reference}_BB".replace("-", "_")
    p = parameters
    lines = [f"* {reference}: broadband model, fitted {p['band'][0]:g}-{p['band'][1]:g} Hz at "
             f"{p['drive_dbm']} dBm drive (passive: R, L, C >= 0)",
             f".SUBCKT {name} A B C D",
             f"RLK0 A LK0 {max(p['leakage']['R0'], 1e-6):.6g}",
             f"LLKI LK0 LK1 {max(p['leakage']['L_inf'], 1e-12):.6g}"]
    node = "LK1"
    for i, (r, l) in enumerate(p["leakage"]["sections"]):
        nxt = f"LK{i + 2}"
        lines += [f"RLS{i} {node} {nxt} {r:.6g}", f"LLS{i} {node} {nxt} {l:.6g}"]
        node = nxt
    lines.append(f"RLM {node} M 1e-6")
    for i, (r, l) in enumerate(p["shunt"]["rl"]):
        lines += [f"RSL{i} M SL{i} {r:.6g}", f"LSL{i} SL{i} B {l:.6g}"]
    for i, (r, c) in enumerate(p["shunt"]["rc"]):
        lines += [f"RSC{i} M SC{i} {r:.6g}", f"CSC{i} SC{i} B {c:.6g}"]
    if p["shunt"]["G"] > 0:
        lines.append(f"RSG M B {1 / p['shunt']['G']:.6g}")
    lines += ["* ideal transformer, ratio eta = N2/N1 (dots on A and C)",
              f"EXF C1 D M B {p['eta']:.8f}", "VXF C1 C DC 0", f"FXF M B VXF {p['eta']:.8f}",
              "* interwinding branch capacitors"]
    for branch, value in p["C"].items():
        if abs(value) >= 1e-16:
            lines.append(f"C{branch} {branch[0]} {branch[1]} {value:.6g}")
    lines.append(f".ENDS {name}")
    return name, lines


if __name__ == "__main__":
    main()
