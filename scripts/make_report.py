"""
PDF characterization report from cached sweeps and results -- no instruments.

    python make_report.py DUT_REFERENCE [--before OLD_REFERENCE] [--out file.pdf]

Reads output/{reference}_*.csv, output/{reference}_results.json and the
calibration sidecars, and writes output/{reference}_report.pdf. --before adds
a page comparing the capacitance link differences with an earlier run (used
here to show the rev B fixture fixes).
"""

import argparse
import datetime
import glob
import json
import os
import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy
import pandas

import CapacitanceFit as cfit
import RelayBoardController as rbc
import TransformerModel as tm

HERE = pathlib.Path(__file__).parent.resolve()
OUTPUT = HERE / "output"
CALIBRATIONS = HERE / "calibrations"

# Reference categorical palette, light mode, fixed order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
PAGE = (11.69, 8.27)   # A4 landscape

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "legend.frameon": False, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
    "axes.titlecolor": INK, "figure.facecolor": "white",
})


# ------------------------------------------------------------------ helpers

def sweep_path(reference, number, kind):
    return OUTPUT / f"{reference}_cfg{number:02d}_{rbc.CONFIGS[number]['name']}_{kind}.csv"


def load(reference, number, kind):
    path = sweep_path(reference, number, kind)
    return tm.average_cycles(pandas.read_csv(path)) if path.exists() else None


def new_page(pdf_state, title, subtitle=None):
    figure = plt.figure(figsize=PAGE)
    pdf_state["page"] += 1
    figure.text(0.04, 0.955, title, fontsize=15, fontweight="bold", color=INK)
    if subtitle:
        figure.text(0.04, 0.925, subtitle, fontsize=9, color=INK2)
    figure.text(0.96, 0.02, f"{pdf_state['reference']}  ·  page {pdf_state['page']}",
                fontsize=7, color=MUTED, ha="right")
    return figure


def text_block(figure, x, y, lines, size=8.5, width=None, color=INK, line_height=None):
    """Write wrapped lines downward from (x, y) in figure coordinates."""
    line_height = line_height or size * 0.0021
    for line in lines:
        bold = line.startswith("**")
        content = line.strip("*")
        wrapped = textwrap.wrap(content, width) if width and content else [content]
        for piece in wrapped or [""]:
            figure.text(x, y, piece, fontsize=size, color=color,
                        fontweight="bold" if bold else "normal", va="top")
            y -= line_height
    return y


def table(figure, rect, header, rows, col_widths=None, size=8):
    axis = figure.add_axes(rect)
    axis.axis("off")
    t = axis.table(cellText=rows, colLabels=header, loc="upper left", cellLoc="left",
                   colLoc="left", colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(size)
    t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor("#f3f2ee")
        else:
            cell.set_text_props(color=INK)
    return axis


def log_frequency_axis(axis):
    axis.set_xscale("log")
    axis.set_xlabel("Frequency (Hz)")


# ------------------------------------------------------------------ pages

def page_summary(pdf, state, results, differential, warnings, calibration_meta):
    description = state.get("description")
    figure = new_page(state, f"Magnetic characterization — {state['reference']}",
                      (f"DUT: {description}  ·  " if description else "")
                      + f"Bode 100 + relay board rev B · automatic per-path OSL · "
                      f"report {datetime.datetime.now():%Y-%m-%d %H:%M}")
    magnetic = results.get("magnetic", {})
    losses = results.get("losses", {})
    estat = differential.get("electrostatic", {})

    rows = []
    if magnetic:
        rows += [
            ["Coupling coefficient k", f"{magnetic['k']:.5f}", "[COG94] from L0, Lsc at 10 kHz", "good"],
            ["Turns ratio N2/N1", f"{magnetic['turns_ratio_N2_N1']:.4f}", "eta = sqrt(L0'/L0)", "good"],
            ["Open-circuit inductance L0", f"{magnetic['L0']*1e6:.2f} µH", "0 dBm, 10 kHz", "drive-dependent"],
            ["Magnetizing inductance Lp", f"{magnetic['Lp_magnetizing']*1e6:.2f} µH", "L0 (1+k)/2", "drive-dependent"],
            ["Leakage inductance ls", f"{magnetic['ls_leakage']*1e6:.3f} µH", "Lsc / k", "includes ~2 arms"],
            ["Mutual M (Lcum, Ldif)", f"{magnetic.get('M', float('nan'))*1e6:.2f} µH",
             f"k from M: {magnetic.get('k_from_mutual', float('nan')):.4f}", "cross-check"],
        ]
    if losses:
        rows += [
            ["Primary resistance r1", f"{losses['r1']*1e3:.1f} mΩ", "low-f plateau", "incl. ~100 mΩ fixt."],
            ["Secondary resistance r2", f"{losses['r2_secondary']*1e3:.1f} mΩ", "low-f plateau", "incl. ~100 mΩ fixt."],
        ]
        if losses.get("Rp"):
            rows.append(["Core loss resistance Rp", f"{losses['Rp']:.0f} Ω", "at first resonance", "good"])
    if differential:
        rows += [
            ["Interwinding C33", f"{differential['C33']['forward_pf']:.2f} pF",
             f"direct; reversed {differential['C33']['reverse_pf']:.2f}", "good (±0.05)"],
            ["C13 [BLA94]", f"{differential['C13']['value']:+.2f} pF",
             "leakage-cancelling short pair 9/11", f"±{differential['C13']['spread']:.1f} sys."],
            ["C23 [BLA94]", f"{differential['C23']['value']:+.2f} pF",
             "leakage-cancelling short pair 14/15", f"±{differential['C23']['spread']:.1f} sys."],
            ["C13 + C23 (open / shorts)", f"{differential['u_open']['value']:+.2f} / {differential['u_short']['value']:+.2f} pF",
             "two independent families",
             f"{abs(differential['u_open']['value'] - differential['u_short']['value']):.1f} pF apart"],
            ["Self-C, C11+η²C22+2ηC12", f"≥ {differential['self_capacitance_lower_bound_pf']:.1f} pF",
             "open-circuit, max over 1–10 MHz", "frequency-dependent"],
            ["C11 [BLA94]", f"{differential['C11']['value']:+.2f} pF", "shorted state 9, smooth-leakage fit",
             f"±{max(differential['C11']['spread'], 1.0):.1f} sys."],
            ["C22 [BLA94]", f"{differential['C22']['value']:+.2f} pF", "shorted state 14, smooth-leakage fit",
             f"±{max(differential['C22']['spread'], 1.0):.1f} sys."],
            ["C12 [BLA94]", f"{differential['C12']['value']:+.2f} pF", f"(S − C11 − C22)/2, S = {differential['S_open_pf']:.1f}",
             f"±{max(differential['C12']['spread'], 1.0):.1f} sys."],
            ["Branch caps AB/CD/AC/AD/BC/BD", " / ".join(f"{v:.1f}" for v in differential["branches_pf"].values()),
             "equivalent circuit (pF)", "SPICE-ready"],
            ["Fixture: clamp→HI / clamp→GND", f"{differential['clamp_to_HI_pf']['value']:.2f} / "
             f"{differential['clamp_to_ground_pf']['value']:.2f} pF", "open-link differences (guarded)", "fixture"],
            ["Fixture: LINK net → GND", f"{differential['link_net_to_ground_pf']:.1f} pF",
             "identity 4·Y(Lcum) = Y(BC)", "fixture"],
        ]
    table(figure, [0.04, 0.30, 0.62, 0.60], ["Parameter", "Value", "How", "Confidence"], rows,
          col_widths=[0.33, 0.17, 0.33, 0.17], size=8)

    checks = []
    if "reciprocity_error" in results:
        checks.append(f"Reciprocity Z0·Zsc' = Z0'·Zsc: worst {results['reciprocity_error']*100:.1f} % "
                      f"({'pass' if results.get('reciprocity_passed') else 'FAIL'} at 5 %)")
    if "linearity_deviation" in results:
        checks.append(f"Linearity −10 vs +10 dBm: L moves {results['linearity_deviation']*100:.1f} % "
                      "(MnZn small-signal permeability, Rayleigh region)")
    if differential:
        checks.append(f"Fixture check AC−BD (must be 0): {differential['open_differences']['AC']['value']:+.2f} pF")
        checks.append(f"Open-family cycle noise: {differential['open_cycle_noise_pf']:.3f} pF")
    y = text_block(figure, 0.69, 0.89, ["**Checks"] + checks, width=58)
    y = text_block(figure, 0.69, y - 0.02, ["**Warnings from the run"] + [f"• {w}" for w in warnings[:8]], width=58,
                   size=7.5)
    if calibration_meta:
        any_meta = next(iter(calibration_meta.values()))
        text_block(figure, 0.69, y - 0.02, ["**Instruments",
                                            any_meta.get("bode_100", ""), any_meta.get("relay_board", ""),
                                            f"OSL drive {any_meta.get('source_power_dbm')} dBm, "
                                            f"LOAD {any_meta.get('load_ohm')} Ω, {len(calibration_meta)} paths"],
                   width=58, size=7.5)
    text_block(figure, 0.04, 0.27, [
        "**Reading this report",
        "All values are small-signal (0 dBm for inductance, 13 dBm for the capacitance sweeps). Capacitances follow the "
        "[BLA94] Table 1 convention used in TransformerModel.WINDING_LINKS (V1 = VA−VB, V2 = VC−VD, V3 = VD−VB). "
        "'sys.' is the spread over analysis bands and over the one quantity the short-pair method cannot pin "
        "(their shared absolute level); it is a systematic, not a statistical, uncertainty.",
    ], width=150, size=8)
    pdf.savefig(figure)
    plt.close(figure)


def page_findings(pdf, state):
    figure = new_page(state, "What was found and fixed on the bench",
                      "Branch claude/exciting-allen-152izs (rev B board, automatic OSL, USB SCPI)")
    left = [
        "**Software / instrument",
        "• The OMICRON SCPI server only opens the Bode over USB on the first instrument command (~2.5 s); writes "
        "arriving in that window are silently dropped. The driver now forces it with a query on connect.",
        "• In IAD mode the Bode refuses to sweep with no active correction ('Calibration must be active' — the data "
        "query just hangs). The pre-OSL relay check now runs through the fresh OSL on the first path of a session.",
        "• A fresh SCPI session starts in reflection (S11/P1R) mode; the OSL was being taken there, leaving IAD "
        "uncorrected. The impedance setup is now configured before the OSL.",
        "",
        "**Relay board (driver + firmware source)",
        "• Config 8 (link_BD_open) put D alone on the LINK rail: B and D were never joined, so the 'B-D linked' "
        "sweep was really a floating-secondary sweep plus LINK-rail capacitance. Fixed to HI=A, LO=B+D. "
        "verify_board.py now checks link INTENT (it would have caught this).",
        "• Floating terminals stayed tied to their crossbar column: open contacts + bus copper loaded the DUT "
        "node with uncalibrated pF (the OSL isolates every terminal, so never saw it). Floating terminals are now "
        "isolated in MEAS. Calibrations stay valid: the OSL state is unchanged.",
        "• The driver now drives every relay itself and reads the word back, so behaviour no longer depends on "
        "the flashed firmware's table (board reports 1.1.1; repo source was 1.0.0, now 1.2.0, not yet flashed).",
        "• New electrostatic-only configs 16–19 (C33 reversed; LO-rail coupling diagnostics).",
        "",
        "**Measurement physics (2026-10-11)",
        "• The bridge senses current only in the LO rail, so the bench measures a GUARDED transfer admittance: a stray "
        "to ground at relative potential v adds Cg·v(v−1). This explained the series-aiding state looking ~15 pF short "
        "on every DUT (LINK net → GND ≈ 14 pF at v = ½) and showed that 'capacitance to ground' is not measurable here.",
        "• Board paths outside the OSL (LO terminal's column, LINK loop) are asymmetric between A-B and C-D by tens of "
        "nH. They are now measured automatically with the DUT isolated (calibrations/fixture_paths.json) and removed "
        "from the LINK-shorted states: reciprocity 7.5 % → ~2 %, Lsc −5 %.",
    ]
    right = [
        "**Effect on the capacitance data (same DUT)",
        "• AC−BD, which must be 0 for any lumped DUT: −3.31 pF → +0.14 pF.",
        "• (BC+AD)/2 − BD vs direct C33: 33.6 vs 18.8 pF (+14.8) → 23.0 vs 18.75 pF (+4.2). The remaining 4.2 pF "
        "matches ~1 pF from each isolated clamp (arm + open iso contact) to HI and to ground.",
        "",
        "**Why the original six-capacitance result was wrong",
        "• It read each sum off one resonance as 1/(L·ω²) with L from a 0 dBm, 10 kHz sweep. The resonance sweeps "
        "ran at 13 dBm (L 14 % higher) and the first resonance sits where the MnZn permeability is already dispersive "
        "and lossy. The short-circuit resonances are beyond 40 MHz, so only 1 of 3 equations per link existed.",
        "• The fixture problems above added ~15 pF of phantom C33 to two of the five links.",
        "",
        "**Cross-checks",
        "• The ground-free [BLA94] model fits the old manual-jumper data of 750341867 to ~1 pF: the method and "
        "the ported equations are sound; the discrepancy was this fixture.",
        "• C33 forward = reverse to 0.001 pF; repeat run within 0.02 pF; five electrostatic states consistent.",
    ]
    text_block(figure, 0.04, 0.88, left, width=82, size=8.2)
    text_block(figure, 0.52, 0.88, right, width=82, size=8.2)
    pdf.savefig(figure)
    plt.close(figure)


def page_calibration(pdf, state, calibration_meta):
    figure = new_page(state, "Calibration — automatic OSL per signal path",
                      "Raw standards checked before OSL (or through it on the first path); LOAD re-verified, "
                      "SHORT re-actuated after")
    rows = []
    for path, meta in sorted(calibration_meta.items()):
        raw = meta.get("standards_raw_ohm", {})
        rows.append([path, meta.get("acquired", ""),
                     f"{raw.get('OPEN', float('nan')):.3g}", f"{raw.get('SHORT', float('nan'))*1e3:.2f}",
                     f"{raw.get('LOAD', float('nan')):.3f}",
                     f"{meta.get('short_residual_ohm', float('nan'))*1e3:.1f}",
                     meta.get("standards_checked", "before_osl")])
    table(figure, [0.04, 0.08, 0.92, 0.82],
          ["Signal path", "Acquired", "OPEN (Ω)", "SHORT (mΩ)", "LOAD (Ω)", "SHORT re-act. (mΩ)", "Check"],
          rows, col_widths=[0.13, 0.17, 0.11, 0.11, 0.11, 0.15, 0.17], size=7.5)
    pdf.savefig(figure)
    plt.close(figure)


def page_inductance(pdf, state, reference):
    figure = new_page(state, "Inductance and coupling (RL sweeps, 0 dBm, 100 Hz – 1 MHz)")
    axes = figure.subplots(1, 3, gridspec_kw=dict(left=0.06, right=0.98, top=0.86, bottom=0.12, wspace=0.28))
    sets = [(1, "Z0 (primary, sec. open)"), (3, "Z0' (secondary, pri. open)"), (5, "Lcum / 4 (series aiding)")]
    for color, (number, label) in zip(SERIES, sets):
        d = load(reference, number, "RL")
        if d is not None:
            scale = 0.25 if number == 5 else 1.0
            axes[0].plot(d.frequency, d.inductance * 1e6 * scale, color=color, label=label)
    axes[0].set_title("Open-circuit inductance")
    axes[0].set_ylabel("Inductance (µH)")
    sets = [(2, "Zsc (primary, sec. shorted)"), (6, "Ldif (series opposing)")]
    for color, (number, label) in zip(SERIES, sets):
        d = load(reference, number, "RL")
        if d is not None:
            axes[1].plot(d.frequency, d.inductance * 1e9, color=color, label=label)
    axes[1].set_title("Leakage-type inductance")
    axes[1].set_ylabel("Inductance (nH)")
    d0, dsc = load(reference, 1, "RL"), load(reference, 2, "RL")
    if d0 is not None and dsc is not None:
        merged = d0.merge(dsc, on="frequency", suffixes=("_o", "_s"))
        k = numpy.sqrt(numpy.clip(1 - merged.inductance_s / merged.inductance_o, 0, 1))
        axes[2].plot(merged.frequency, k, color=SERIES[0])
        axes[2].set_title("Coupling k(f) = √(1 − Lsc/L0)")
        axes[2].set_ylabel("k")
    for axis in axes:
        log_frequency_axis(axis)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="best")
    # Below ~1 kHz the shorted states are resistive and Im(Z)/w is noise.
    axes[1].set_xlim(1e4, 1e6)
    axes[2].set_xlim(1e3, 1e6)
    shorted = load(reference, 2, "RL")
    if shorted is not None:
        visible = shorted[shorted.frequency >= 1e4].inductance * 1e9
        axes[1].set_ylim(0, 1.3 * float(visible.max()))
    if d0 is not None and dsc is not None:
        axes[2].set_ylim(0.99, 1.0005)
    pdf.savefig(figure)
    plt.close(figure)


def page_linearity_resistance(pdf, state, reference):
    figure = new_page(state, "Drive linearity and winding resistance")
    axes = figure.subplots(1, 2, gridspec_kw=dict(left=0.06, right=0.98, top=0.86, bottom=0.12, wspace=0.22))
    for color, (kind, label) in zip(SERIES, (("RL_drive_low", "−10 dBm"), ("RL_drive_high", "+10 dBm"))):
        d = load(reference, 1, kind)
        if d is not None:
            axes[0].plot(d.frequency, d.inductance * 1e6, color=color, label=label)
    axes[0].set_title("Z0 inductance at two drive levels (Rayleigh-region MnZn)")
    axes[0].set_ylabel("Inductance (µH)")
    for color, (number, label) in zip(SERIES, ((2, "Zsc (r1 + r2' + fixture)"), (4, "Zsc' — secondary side"),
                                              (6, "Ldif"))):
        d = load(reference, number, "RL")
        if d is not None:
            axes[1].plot(d.frequency, d.resistance * 1e3, color=color, label=label)
    axes[1].set_title("Series resistance of the shorted states")
    axes[1].set_ylabel("Resistance (mΩ)")
    axes[1].set_yscale("log")
    for axis in axes:
        log_frequency_axis(axis)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="best")
    pdf.savefig(figure)
    plt.close(figure)


def _subtract_path(data, path):
    """Remove a series board path (L, R) from a Z sweep frame."""
    if not path:
        return data
    data = data.copy()
    omega = 2 * numpy.pi * data["frequency"].to_numpy()
    z = data["magnitude"].to_numpy() * numpy.exp(1j * numpy.radians(data["phase"].to_numpy()))
    z = z - (path[1] + 1j * omega * path[0])
    data["magnitude"] = numpy.abs(z)
    data["phase"] = numpy.degrees(numpy.angle(z))
    return data


def _board_path(paths, config_number):
    """(L, R) the board adds to LINK-shorted configs 2 and 4 (see
    MagneticCharacterizer.characterize_fixture_paths)."""
    pairs = {2: ("AB", "CD"), 4: ("CD", "AB")}
    if not paths or config_number not in pairs:
        return None
    driven, shorted = pairs[config_number]
    return (paths[driven]["other_column"]["L_h"] + paths[shorted]["link_loop"]["L_h"],
            paths[driven]["other_column"]["R_ohm"] + paths[shorted]["link_loop"]["R_ohm"])


def page_reciprocity(pdf, state, reference):
    figure = new_page(state, "Switching self-test — reciprocity [BLA94] II-C",
                      "Z0·Zsc' = Z0'·Zsc holds for any linear two-port; a relay in the wrong state breaks it")
    axes = figure.subplots(1, 2, gridspec_kw=dict(left=0.06, right=0.98, top=0.86, bottom=0.12, wspace=0.22))
    sweeps = {n: pandas.read_csv(sweep_path(reference, n, "Zhd")) for n in (1, 2, 3, 4)
              if sweep_path(reference, n, "Zhd").exists()}
    paths = state.get("fixture_paths")
    worst_corrected = None
    if len(sweeps) == 4:
        _, worst, merged = tm.check_reciprocity(sweeps[1], sweeps[3], sweeps[2], sweeps[4])
        axes[0].plot(merged.frequency, merged.relative_error * 100, color=SERIES[0], label="raw")
        if paths:
            _, _, fixed = tm.check_reciprocity(sweeps[1], sweeps[3], _subtract_path(sweeps[2], _board_path(paths, 2)),
                                               _subtract_path(sweeps[4], _board_path(paths, 4)))
            axes[0].plot(fixed.frequency, fixed.relative_error * 100, color=SERIES[1],
                         label="board paths subtracted")
            worst_corrected = float(fixed[fixed.frequency <= 10e6].relative_error.max())
            axes[0].legend(loc="upper left")
        axes[0].axhline(5, color=SERIES[7], linewidth=1, linestyle="--")
        axes[0].text(merged.frequency.min(), 5.2, "5 % limit", color=INK2, fontsize=7.5)
        axes[0].set_title(f"Relative error (raw worst {worst*100:.1f} %"
                          + (f", corrected ≤10 MHz {worst_corrected*100:.1f} %)" if worst_corrected else ")"))
        axes[0].set_ylabel("|Z0·Zsc' − Z0'·Zsc| / |Z0·Zsc'|  (%)")
        for color, (n, label) in zip(SERIES, ((2, "Zsc (from primary)"), (4, "Zsc' (from secondary)"))):
            d = tm.average_cycles(sweeps[n])
            axes[1].plot(d.frequency, d.magnitude, color=color, label=label)
        axes[1].set_yscale("log")
        axes[1].set_title("The two short-circuit impedances")
        axes[1].set_ylabel("|Z| (Ω)")
        axes[1].legend()
    for axis in axes:
        log_frequency_axis(axis)
    text_block(figure, 0.06, 0.06, [
        "Raw, the identity is off by 6–8 %: the LINK loop that shorts the far winding and the LO terminal's column sit "
        "outside the OSL and differ between the two directions (A-B vs C-D). Measured board-only with the DUT isolated "
        "(fixture_paths.json) and subtracted, it closes to ~2 % — switching is correct and Lsc is corrected by the same "
        "path. A relay in the wrong state would give errors of orders of magnitude."], width=175)
    pdf.savefig(figure)
    plt.close(figure)


def pages_all_sweeps(pdf, state, reference):
    numbers = [n for n in sorted(rbc.CONFIGS) if sweep_path(reference, n, "Zhd").exists()]
    per_page = 6
    for start in range(0, len(numbers), per_page):
        chunk = numbers[start:start + per_page]
        figure = new_page(state, "Impedance sweeps, all configurations (10 kHz – 50 MHz, 801 pts × 4 cycles)",
                          "|Z| above, phase below; band = min..max over the four cycles")
        grid = figure.add_gridspec(4, 3, left=0.06, right=0.98, top=0.88, bottom=0.09, hspace=0.6, wspace=0.25,
                                   height_ratios=[1.6, 1, 1.6, 1])
        for i, number in enumerate(chunk):
            row, column = (i // 3) * 2, i % 3
            config = rbc.CONFIGS[number]
            raw = pandas.read_csv(sweep_path(reference, number, "Zhd"))
            g = raw.groupby("frequency")
            mag_ax = figure.add_subplot(grid[row, column])
            ph_ax = figure.add_subplot(grid[row + 1, column], sharex=mag_ax)
            f = g.magnitude.mean().index.to_numpy()
            mag_ax.fill_between(f, g.magnitude.min(), g.magnitude.max(), color=SERIES[0], alpha=0.25, linewidth=0)
            mag_ax.plot(f, g.magnitude.mean(), color=SERIES[0], linewidth=1.2)
            ph_ax.plot(f, g.phase.mean(), color=SERIES[1], linewidth=1.2)
            mag_ax.set_xscale("log")
            mag_ax.set_yscale("log")
            topology = (f"HI={'+'.join(config['HI'])}  LO={'+'.join(config['LO']) or '—'}"
                        + (f"  LINK={'+'.join(config['LINK'])}" if config["LINK"] else ""))
            mag_ax.set_title(f"{number}  {config['name']}", loc="left")
            mag_ax.text(0.0, 1.02, "", transform=mag_ax.transAxes)
            mag_ax.annotate(topology, xy=(1, 1.04), xycoords="axes fraction", ha="right", fontsize=7, color=INK2)
            mag_ax.set_ylabel("|Z| (Ω)")
            ph_ax.set_ylabel("Phase (°)")
            ph_ax.set_ylim(-95, 95)
            ph_ax.set_yticks([-90, 0, 90])
            plt.setp(mag_ax.get_xticklabels(), visible=False)
            ph_ax.set_xlabel("Frequency (Hz)")
        pdf.savefig(figure)
        plt.close(figure)


def page_electrostatic(pdf, state, reference, differential):
    figure = new_page(state, "Electrostatic-only states — C33 and LO-rail coupling",
                      "Each winding shorted on itself (no magnetics): configs 7, 16–19")
    axis = figure.add_axes([0.06, 0.14, 0.55, 0.72])
    labels = {7: "7  A+B → HI, C+D → LO", 16: "16  reversed", 17: "17  all → HI, LO empty",
              18: "18  A+B → HI, C+D on LINK, LO empty", 19: "19  C+D → HI, A+B on LINK, LO empty"}
    for color, number in zip(SERIES, (7, 16, 17, 18, 19)):
        f, Y = cfit.load_admittance(str(OUTPUT), reference, number)
        c = cfit.effective_capacitance(f, Y).mean(axis=0) / cfit.PICO
        axis.plot(f, c, color=color, label=labels[number])
    log_frequency_axis(axis)
    axis.set_ylim(-1, 30)
    axis.set_ylabel("Im(Y)/ω  (pF)")
    axis.set_title("Apparent capacitance")
    axis.legend(loc="upper left")
    estat = differential["electrostatic"]
    rows = [[str(k), f"{v[0]:.3f}"] for k, v in estat["measured_pf"].items()]
    table(figure, [0.66, 0.56, 0.30, 0.30], ["Config", "C (pF), 0.1–5 MHz"], rows, col_widths=[0.4, 0.6])
    text_block(figure, 0.66, 0.52, [
        "**The bridge measures a GUARDED transfer admittance",
        "Current is sensed only in the LO rail (RC1 shunt); strays to",
        "board ground return unmeasured. A stray Cg at a node at relative",
        "potential v adds Cg·v(v−1): nothing at 0 or 1 V, negative at ½.",
        "",
        f"C33 = {estat['C33_pf']:.2f} pF (7 vs 16 differ by {estat['C33_asymmetry_pf']:+.3f}):",
        "every node at 0 or 1 V, so no stray to ground enters.",
        "",
        "17–19 have an EMPTY LO rail: they read only the coupling into",
        f"the LO rail ({estat['LO_rail_coupling_pf']:.2f} pF from the DUT, "
        f"+{estat['LINK_to_LO_coupling_pf']:.2f} pF via LINK),",
        "not capacitance to ground — that is invisible to this bridge.",
        "Above ~10 MHz C33 rises as if ~100 nH were in series: the",
        "uncalibrated arms/columns plus the winding path.",
    ], size=8)
    pdf.savefig(figure)
    plt.close(figure)


def page_open_family(pdf, state, reference, differential, before):
    figure = new_page(state, "Open-circuit link differences — core-free",
                      "C(link) − C(B-D linked) = Im(Y_link − Y_BD)/ω; the magnetizing term is common and cancels "
                      "at every frequency")
    axis = figure.add_axes([0.06, 0.14, 0.50, 0.72])
    spectra = differential["open_spectra"]
    f = spectra["frequency"]
    for color, key in zip(SERIES, ("BC", "AD", "AC", "floating")):
        axis.plot(f, spectra[key]["spectrum"], color=color, label=f"{key} − BD")
    axis.axvspan(1e6, 8e6, color=GRID, alpha=0.6, linewidth=0)
    axis.text(1.1e6, 46, "analysis band", color=INK2, fontsize=7.5)
    log_frequency_axis(axis)
    axis.set_ylim(-5, 50)
    axis.set_xlim(2e5, 5e7)
    axis.set_ylabel("Capacitance difference (pF)")
    axis.legend(loc="upper left")
    od = differential["open_differences"]
    rows = [[k + " − BD", f"{v['value']:+.3f}", f"{v['spread']:.3f}"] for k, v in od.items()]
    table(figure, [0.60, 0.62, 0.36, 0.24], ["Difference", "pF", "band spread"], rows, col_widths=[0.4, 0.3, 0.3])
    lines = [
        "**What they give",
        f"u = C13 + C23 = (AD − BC)/4 = {differential['u_open']['value']:+.2f} pF",
        "AC − BD must be 0 for any lumped DUT (same node voltages):",
        f"measured {od['AC']['value']:+.2f} pF — the fixture check.",
        f"(BC + AD)/2 − BD = {(od['BC']['value'] + od['AD']['value'])/2:.2f} pF vs C33 "
        f"{differential['C33']['forward_pf']:.2f}: ~1 pF per isolated clamp.",
        "Below ~0.5 MHz the core (≈1 µF-equivalent) drifts 0.07 % between",
        "sweeps and dominates; above ~12 MHz series path inductance enters.",
    ]
    if before:
        lines += ["", "**Before the fixture fixes (same DUT)"] + [
            f"{k} − BD: {v:+.2f} pF" for k, v in before.items()]
    text_block(figure, 0.60, 0.58, lines, size=8)
    pdf.savefig(figure)
    plt.close(figure)


def page_short_pairs(pdf, state, reference, differential):
    figure = new_page(state, "C13 and C23 — shorted pairs that share one leakage",
                      "1/(Y_b − jωC_b) − (ΔR + jωΔL) = 1/(Y_a − jωC_a) at every frequency: leakage cancels, ΔC and "
                      "the fixture path difference remain")
    axes = figure.subplots(1, 2, gridspec_kw=dict(left=0.06, right=0.62, top=0.86, bottom=0.14, wspace=0.3))
    for axis, (pair, (a, b, meaning)) in zip(axes, cfit.SHORT_PAIRS.items()):
        f, Ya = cfit.load_admittance(str(OUTPUT), reference, a)
        _, Yb = cfit.load_admittance(str(OUTPUT), reference, b)
        for color, (Y, n) in zip(SERIES, ((Ya, a), (Yb, b))):
            z = 1.0 / Y.mean(axis=0)
            axis.plot(f, z.imag / (2 * numpy.pi * f) * 1e9, color=color,
                      label=f"{n} {rbc.CONFIGS[n]['name']}")
        log_frequency_axis(axis)
        axis.set_xlim(2e5, 5e7)
        axis.set_ylabel("Apparent inductance Im(Z)/ω (nH)")
        axis.set_title(f"{pair}: ΔC = {meaning}")
        axis.legend(loc="upper left")
    rows = [
        ["C13", f"{differential['C13']['value']:+.2f}", f"{differential['C13']['spread']:.2f}",
         f"{differential['C13']['min']:+.2f} … {differential['C13']['max']:+.2f}"],
        ["C23", f"{differential['C23']['value']:+.2f}", f"{differential['C23']['spread']:.2f}",
         f"{differential['C23']['min']:+.2f} … {differential['C23']['max']:+.2f}"],
        ["C13+C23 (shorts)", f"{differential['u_short']['value']:+.2f}", f"{differential['u_short']['spread']:.2f}", ""],
        ["C13+C23 (open)", f"{differential['u_open']['value']:+.2f}", "—", ""],
    ]
    table(figure, [0.645, 0.62, 0.34, 0.24], ["", "pF", "± sys.", "range (pF)"], rows,
          col_widths=[0.34, 0.16, 0.16, 0.34])
    dl = differential["fixture_dL_nh"]
    rms = differential["short_pair_rms"]
    text_block(figure, 0.66, 0.58, [
        "**Fit quality",
        f"primary pair rms {rms['primary']['value']*100:.3f} % (cycle noise level)",
        f"secondary pair rms {rms['secondary']['value']*100:.2f} %",
        "",
        "**Fixture series-inductance difference found",
        f"primary pair (9 → 11): {dl['primary']['value']:+.1f} nH",
        f"secondary pair (14 → 15): {dl['secondary']['value']:+.1f} nH",
        "The far winding is shorted through a different rail and",
        "columns in each state — the rev B2 SHORT asymmetry, now",
        "quantified; it would otherwise be read as ~tens of pF.",
        "",
        "Spread is over 4 analysis bands × 3 values of the shared",
        "level the pair cannot pin.",
    ], size=8)
    pdf.savefig(figure)
    plt.close(figure)


def page_self_capacitance(pdf, state, reference, differential):
    figure = new_page(state, "Self-capacitance across the windings — a spectrum, not a constant",
                      "Im(Y)/ω of the open-circuit states; the core contributes −L'/(ω²|L|²) ≤ 0, so each curve is a "
                      "lower bound on C11 + η²C22 + 2ηC12")
    axis = figure.add_axes([0.06, 0.14, 0.55, 0.72])
    for color, (number, label) in zip(SERIES, ((8, "8  B-D linked"), (1, "1  Z0 floating"), (3, "3  Z0' (secondary)"),
                                              (10, "10  A-C linked"))):
        f, Y = cfit.load_admittance(str(OUTPUT), reference, number)
        axis.plot(f, cfit.effective_capacitance(f, Y).mean(axis=0) / cfit.PICO, color=color, label=label)
    log_frequency_axis(axis)
    axis.set_xlim(1e6, 5e7)
    axis.set_ylim(-5, 20)
    axis.set_ylabel("Im(Y)/ω  (pF)")
    axis.legend(loc="lower right")
    at = differential.get("self_capacitance_at_pf", {})
    # Describe the curve from the data rather than assume one part's shape.
    f, Y = cfit.load_admittance(str(OUTPUT), reference, 8)
    c = cfit.effective_capacitance(f, Y).mean(axis=0) / cfit.PICO
    band = (f >= 2e6) & (f <= 10e6)
    peak_f = float(f[band][numpy.argmax(c[band])])
    falls = peak_f < 8e6 and c[band][-1] < c[band].max() - 0.5
    if falls:
        shape_lines = [
            f"The curve peaks near {peak_f/1e6:.0f} MHz and then falls. A core",
            "term alone (-L'/(w^2|L|^2) <= 0, shrinking with f) can only make it",
            "rise towards the true value, so the self-capacitance itself",
            "decreases with frequency: consistent with windings directly on",
            "MnZn ferrite, whose permittivity relaxes in the MHz range.",
        ]
    else:
        shape_lines = [
            "The curve rises monotonically towards a plateau, as expected when",
            "the core term (<= 0) fades with frequency: the self-capacitance",
            "behaves as a near-constant here. Above ~10 MHz series path",
            "inductance (fixture + winding) lifts it further, so the plateau",
            "estimate is taken below 10 MHz.",
        ]
    text_block(figure, 0.65, 0.84, [
        "**Reading",
        f"Maximum over 1–10 MHz (lower bound): {differential['self_capacitance_lower_bound_pf']:.2f} pF.",
        "Values: " + ", ".join(f"{k} {v:.1f}" for k, v in at.items()) + " pF.",
        "",
    ] + shape_lines + [
        "",
        "**Consequence",
        "C12 is taken as (S − C11 − C22)/2 with S = this curve's maximum",
        "over 1–10 MHz. Where the curve falls after its peak (windings on",
        "MnZn), S and therefore C12 are values near the first resonance,",
        "not constants; use this curve directly for the open-circuit SRF.",
    ], size=8)
    pdf.savefig(figure)
    plt.close(figure)


def page_limits(pdf, state):
    figure = new_page(state, "Method limits, validation and next steps")
    left = [
        "**Global nodal fit (CapacitanceFit.fit)",
        "All configurations simulated as one network: six DUT branch capacitors (exact re-parameterisation of "
        "[BLA94]), four to ground, fixture arms/columns, isolated-clamp and LINK capacitance, the core FREE at every "
        "frequency, the leakage as a smooth proximity/skin model. Variable-projection Jacobian; 13–35 s per fit.",
        "Synthetic validation (test_capacitance_fit.py: relaxing lossy ferrite, dispersive leakage, 0.03 % noise): "
        "C11, C13, C22, C23, C33 recovered to 0.02 pF, C12 to ~2 pF, every fixture term recovered.",
        "On real data it reaches 0.2–1 % per state once the bench is modelled as a guarded measurement with a "
        "series path per configuration, but C11/C22/C12 then move by ±10 pF with where that path sits (it scales "
        "the apparent C by ~1−2Ls/L): the reported C11/C22 come instead from the shorted states 9/14 with a "
        "smooth-leakage fit, cross-checked against the pair differences to ~1 pF (4 pF on the noisier 14/15).",
        "",
        "**Identifiability, stated plainly",
        "Any capacitance whose voltage pattern lives only across the winding ports (C11, C22, C12) is "
        "indistinguishable from a jωC inside the magnetic two-port. Self-capacitances are only visible by "
        "resonating against an inductance one is willing to model. The classic method assumes a constant L — "
        "exactly what a ferrite breaks.",
    ]
    right = [
        "**Next steps",
        "• Flash firmware 1.2.0 (config 8 fix, floating isolation) — the Python driver already compensates.",
        "• Remaining uncorrected: the clamp arms (~13 nH each, common to both directions) and the isolated-clamp "
        "coupling (~2 pF to GND, ~0.1 pF to HI). A shorting bar / empty-clamp step would remove both.",
        "• Rev B2: keep the LINK rail and columns away from the ground plane (~14 pF to GND today) and equalise the "
        "A-B and C-D column/LINK path lengths (±38 nH today); make the SHORT follow the DUT path.",
        "• Populate G6K-2F-RF-S (lower open-contact capacitance) as planned.",
        "• For C11/C22/C12 on ferrite parts: measure the core's complex permeability (single-turn or toroid "
        "sample) and fit with it fixed, or accept the self-capacitance spectrum as the deliverable.",
        "• Inductance values are small-signal and drive-dependent (12 % over 20 dB at 10 kHz).",
        "",
        "**References",
        "[BLA94] Blache, Kéradec, Cogitore, 'Stray capacitances of two winding transformers: equivalent circuit, "
        "measurements, calculation and lowering', IEEE IAS 1994.",
        "[COG94] Cogitore, Kéradec, Barbaroux, 'The two-winding ferrite core transformer: an experimental method to "
        "obtain a wide frequency range equivalent circuit', IEEE TIM 1994.",
        "Ferrite permittivity and parasitic capacitance: see e.g. 'The Parasitic Capacitance of Magnetic Components "
        "with Ferrite Cores Due to Time-Varying EM Field' (2018).",
    ]
    text_block(figure, 0.04, 0.88, left, width=82, size=8.2)
    text_block(figure, 0.52, 0.88, right, width=82, size=8.2)
    pdf.savefig(figure)
    plt.close(figure)


# ------------------------------------------------------------------ main

def before_differences(reference):
    """Open link differences of an earlier run (Z sweeps, old fixture)."""
    try:
        values = {}
        f_ref, Y_ref = None, None
        path = OUTPUT / f"{reference}_cfg08_link_BD_open_Z.csv"
        d = tm.average_cycles(pandas.read_csv(path))
        f = d.frequency.to_numpy()
        Y_ref = 1.0 / (d.magnitude * numpy.exp(1j * numpy.radians(d.phase))).to_numpy()
        keep = (f > 1e6) & (f < 8e6)
        for key, number in (("BC", 12), ("AD", 13), ("AC", 10), ("floating", 1)):
            d = tm.average_cycles(pandas.read_csv(sweep_path(reference, number, "Z")))
            Y = 1.0 / (d.magnitude * numpy.exp(1j * numpy.radians(d.phase))).to_numpy()
            values[key] = float(numpy.median(((Y - Y_ref).imag / (2 * numpy.pi * f))[keep]) * 1e12)
        return values
    except (FileNotFoundError, KeyError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("reference")
    parser.add_argument("--before", default=None)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    reference = arguments.reference

    results_path = OUTPUT / f"{reference}_results.json"
    payload = json.loads(results_path.read_text()) if results_path.exists() else {}
    results = payload.get("results", {})
    warnings = payload.get("warnings", [])
    differential = results.get("capacitance_differential")
    if not differential or "open_spectra" not in differential:
        differential = cfit.differential_summary(str(OUTPUT), reference)
        f = differential["open_spectra"]["frequency"]
        _, Y = cfit.load_admittance(str(OUTPUT), reference, cfit.OPEN_REFERENCE)
        c = cfit.effective_capacitance(f, Y).mean(axis=0) / cfit.PICO
        window = (f >= 1e6) & (f <= 10e6)
        differential["self_capacitance_lower_bound_pf"] = float(c[window].max())
        differential["self_capacitance_at_pf"] = {
            f"{x/1e6:g} MHz": float(c[numpy.argmin(numpy.abs(f - x))]) for x in (3e6, 5e6, 10e6, 20e6, 30e6)}

    calibration_meta = {}
    for path in sorted(glob.glob(str(CALIBRATIONS / "relay_board_*.json"))):
        try:
            meta = json.loads(pathlib.Path(path).read_text())
            calibration_meta[meta.get("signal_path", os.path.basename(path))] = meta
        except ValueError:
            pass

    destination = pathlib.Path(arguments.out) if arguments.out else OUTPUT / f"{reference}_report.pdf"
    state = {"page": 0, "reference": reference,
             "fixture_paths": results.get("fixture_paths")}
    dut_path = OUTPUT / f"{reference}_dut.json"
    if dut_path.exists():
        state["description"] = json.loads(dut_path.read_text()).get("description")
    before = before_differences(arguments.before) if arguments.before else None
    with PdfPages(destination) as pdf:
        page_summary(pdf, state, results, differential, warnings, calibration_meta)
        page_findings(pdf, state)
        page_calibration(pdf, state, calibration_meta)
        page_inductance(pdf, state, reference)
        page_linearity_resistance(pdf, state, reference)
        page_reciprocity(pdf, state, reference)
        page_electrostatic(pdf, state, reference, differential)
        page_open_family(pdf, state, reference, differential, before)
        page_short_pairs(pdf, state, reference, differential)
        page_self_capacitance(pdf, state, reference, differential)
        pages_all_sweeps(pdf, state, reference)
        page_limits(pdf, state)
        info = pdf.infodict()
        info["Title"] = f"Magnetic characterization — {reference}"
        info["Subject"] = "Bode 100 + OpenMagnetics relay board rev B"
    print(f"Wrote {destination} ({state['page']} pages)")


if __name__ == "__main__":
    main()
