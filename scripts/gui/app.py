"""
OpenMagnetics Lab -- bench GUI.

    python -m streamlit run scripts/gui/app.py      (or: python scripts/bench.py gui)

Browse every characterized DUT (results, sweeps, models, capacitance
analysis, fixture and calibration state), overlay ngspice simulations of the
extracted models on the measured data, compare DUTs, and drive the bench:
relay configurations with readback, single sweeps, full characterizations,
drive sweeps, board-path measurements and reports.
"""

import json
import pathlib
import subprocess
import sys
import time

import numpy
import pandas
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

import CapacitanceFit as cfit          # noqa: E402
import RelayBoardController as rbc     # noqa: E402
import TransformerModel as tm          # noqa: E402

OUTPUT = SCRIPTS / "output"
CALIBRATIONS = SCRIPTS / "calibrations"

# Categorical palette, fixed order (dataviz reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
QUANTITIES = {
    "|Z| (ohm)": ("mag", True),
    "phase (deg)": ("phase", False),
    "Ls = Im(Z)/w (H)": ("ls", True),
    "Rs = Re(Z) (ohm)": ("rs", True),
    "C = Im(Y)/w (pF)": ("ceff", False),
    "G = Re(Y) (S)": ("g", True),
}

st.set_page_config(page_title="OpenMagnetics Lab", page_icon="🧲", layout="wide")


# --------------------------------------------------------------- data access

@st.cache_data(ttl=10)
def references():
    found = sorted(p.name[:-len("_results.json")] for p in OUTPUT.glob("*_results.json"))
    return [r for r in found if not r.startswith(("BENCH", "PATHTEST", "LINKTEST", "DRIVETEST"))]


@st.cache_data(ttl=10)
def load_results(reference):
    path = OUTPUT / f"{reference}_results.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data(ttl=10)
def dut_description(reference):
    path = OUTPUT / f"{reference}_dut.json"
    return json.loads(path.read_text()).get("description", "") if path.exists() else ""


@st.cache_data(ttl=10)
def sweep_files(reference):
    rows = []
    for path in OUTPUT.glob(f"{reference}_cfg*.csv"):
        stem = path.stem[len(reference) + 1:]          # cfgNN_name_kind
        number = int(stem[3:5])
        name = rbc.CONFIGS.get(number, {}).get("name", "?")
        kind = stem[6 + len(name) + 1:] if stem[6:].startswith(name) else stem[6:]
        rows.append({"config": number, "name": name, "kind": kind, "path": str(path)})
    return pandas.DataFrame(rows).sort_values(["config", "kind"]) if rows else pandas.DataFrame()


@st.cache_data(ttl=60)
def load_sweep(path):
    """Complex Z per cycle -> dict of derived quantities (mean and spread)."""
    data = pandas.read_csv(path).sort_values(["measurement_index", "frequency"])
    cycles = data.measurement_index.nunique()
    f = data.frequency.to_numpy().reshape(cycles, -1)[0]
    if "magnitude" in data:
        z = (data.magnitude.to_numpy() * numpy.exp(1j * numpy.radians(data.phase.to_numpy()))).reshape(cycles, -1)
    elif "inductance" in data:
        w = 2 * numpy.pi * data.frequency.to_numpy()
        z = (data.resistance.to_numpy() + 1j * w * data.inductance.to_numpy()).reshape(cycles, -1)
    else:
        return {"f": f, "cs": data.capacitance.to_numpy().reshape(cycles, -1).mean(0)}
    z = cfit.reject_outliers(z) if cycles >= 3 else z
    w = 2 * numpy.pi * f
    zm = z.mean(axis=0)
    y = 1 / zm
    out = {"f": f, "z": zm, "cycles": cycles,
           "mag": numpy.abs(zm), "phase": numpy.degrees(numpy.angle(zm)),
           "ls": zm.imag / w, "rs": zm.real, "ceff": y.imag / w * 1e12, "g": y.real}
    if cycles > 1:
        spread = numpy.abs(z - zm).std(axis=0) / numpy.sqrt(cycles)
        out["rel_spread"] = spread / numpy.abs(zm)
    return out


def figure(title, x_title="Frequency (Hz)", y_title="", log_x=True, log_y=False, height=420):
    fig = go.Figure()
    fig.update_layout(title=title, height=height, margin=dict(l=60, r=20, t=50, b=50),
                      legend=dict(orientation="h", y=-0.2), template="plotly_white",
                      xaxis_title=x_title, yaxis_title=y_title, hovermode="x unified")
    fig.update_xaxes(type="log" if log_x else "linear", showgrid=True)
    fig.update_yaxes(type="log" if log_y else "linear", showgrid=True)
    return fig


def add_trace(fig, x, y, name, index, dash=None, band=None):
    color = SERIES[index % len(SERIES)]
    if band is not None:
        fig.add_trace(go.Scatter(x=numpy.concatenate([x, x[::-1]]), y=numpy.concatenate([y + band, (y - band)[::-1]]),
                                 fill="toself", fillcolor=color, opacity=0.15, line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=2, dash=dash)))


def metric_table(rows):
    st.dataframe(pandas.DataFrame(rows, columns=["Parameter", "Value", "Unit", "± (sys.)", "Method"]),
                 hide_index=True, use_container_width=True)


# --------------------------------------------------------------------- pages

def page_summary(reference, payload):
    results = payload.get("results", {})
    m, losses = results.get("magnetic", {}), results.get("losses", {})
    d = results.get("capacitance_differential", {})
    st.subheader("Magnetics")
    rows = []
    if m:
        rows += [
            ["Coupling k", f"{m['k']:.5f}", "", "", f"[COG94] L0 at {m.get('reference_frequency', 1e4)/1e3:.0f} kHz, "
                                                     f"Lsc at {m.get('leakage_reference_frequency', 1e4)/1e3:.0f} kHz"],
            ["Turns ratio N2/N1", f"{m['turns_ratio_N2_N1']:.4f}", "", "", "sqrt(L0'/L0), k-corrected"],
            ["Open-circuit L0", f"{m['L0']*1e6:.2f}", "µH", "", "small-signal drive"],
            ["Magnetizing Lp", f"{m['Lp_magnetizing']*1e6:.2f}", "µH", "", "L0 (1+k)/2"],
            ["Leakage ls", f"{m['ls_leakage']*1e6:.3f}", "µH", "", "Lsc/k, board path removed"],
        ]
    if losses:
        rows += [["r1", f"{losses['r1']*1e3:.1f}", "mΩ", "~100 mΩ fixture", "low-f plateau"],
                 ["r2", f"{losses['r2_secondary']*1e3:.1f}", "mΩ", "~100 mΩ fixture", "low-f plateau"]]
        if losses.get("Rp"):
            rows.append(["Core loss Rp", f"{losses['Rp']:.0f}", "Ω", "", "at first resonance"])
    metric_table(rows)
    drive = results.get("drive_linearity")
    if drive:
        st.caption("Amplitude permeability, L at 10 kHz: " + ", ".join(
            f"{dbm:+d} dBm {L*1e6:.1f} µH" for dbm, L in zip(drive["levels_dbm"], drive["L_10kHz"]))
                   + f" — reported at {drive['small_signal_dbm']:+d} dBm.")
    if d:
        st.subheader("Capacitances [BLA94] (V1=VA−VB, V2=VC−VD, V3=VD−VB)")
        rows = [[name, f"{d[name]['value']:+.2f}", "pF", f"{d[name].get('spread', 0):.2f}", method]
                for name, method in (("C11", "shorted state 9"), ("C22", "shorted state 14"),
                                     ("C12", "(S − C11 − C22)/2"), ("C13", "short pair 9/11"),
                                     ("C23", "short pair 14/15")) if name in d]
        rows.append(["C33", f"{d['C33']['forward_pf']:.2f}", "pF", "0.05", "direct, configs 7 and 16"])
        metric_table(rows)
        if "branches_pf" in d:
            st.markdown("**SPICE branch capacitors (pF):** " + ", ".join(
                f"C_{k} = {v:.2f}" for k, v in d["branches_pf"].items()))
        st.markdown(f"**Fixture terms:** clamp→HI {d.get('clamp_to_HI_pf', {}).get('value', 0):.2f} pF, "
                    f"clamp→GND {d.get('clamp_to_ground_pf', {}).get('value', 0):.2f} pF, "
                    f"LINK net→GND {d.get('link_net_to_ground_pf', 0):.1f} pF")
    if payload.get("warnings"):
        st.subheader("Warnings")
        for w in payload["warnings"]:
            st.warning(w)
    st.subheader("Downloads")
    columns = st.columns(4)
    for column, (label, path) in zip(columns, (("results JSON", OUTPUT / f"{reference}_results.json"),
                                               ("SPICE .lib", OUTPUT / f"{reference}.lib"),
                                               ("PDF report", OUTPUT / f"{reference}_report.pdf"),
                                               ("model vs data CSV", OUTPUT / f"{reference}_spice_vs_measured.csv"))):
        if path.exists():
            column.download_button(label, path.read_bytes(), file_name=path.name)
        else:
            column.caption(f"{label}: not generated yet")


def page_sweeps(reference, files):
    if files.empty:
        st.info("No sweeps cached for this DUT.")
        return
    left, right = st.columns([1, 3])
    with left:
        kinds = sorted(files.kind.unique())
        kind = st.selectbox("Sweep kind", kinds, index=kinds.index("Zhd") if "Zhd" in kinds else 0)
        available = files[files.kind == kind]
        labels = [f"{r.config:2d} {r.name}" for r in available.itertuples()]
        chosen = st.multiselect("Configurations", labels, default=labels[:3])
        quantity = st.selectbox("Quantity", list(QUANTITIES))
        show_band = st.checkbox("Cycle spread band (±1σ of the mean)", value=True)
    key, log_y = QUANTITIES[quantity]
    fig = figure(f"{reference} — {quantity}", y_title=quantity, log_y=log_y)
    for index, label in enumerate(chosen):
        row = available.iloc[labels.index(label)]
        s = load_sweep(row.path)
        if key not in s:
            continue
        y = s[key]
        band = None
        if show_band and "rel_spread" in s and key in ("mag", "ls", "rs"):
            band = numpy.abs(y) * s["rel_spread"]
        add_trace(fig, s["f"], y, label, index, band=band)
    with right:
        st.plotly_chart(fig, use_container_width=True)
        for label in chosen:
            row = available.iloc[labels.index(label)]
            config = rbc.CONFIGS[row.config]
            st.caption(f"{label}: HI={'+'.join(config['HI']) or '—'}  LO={'+'.join(config['LO']) or '—'}"
                       + (f"  LINK={'+'.join(config['LINK'])}" if config["LINK"] else "")
                       + f" — {config['purpose']}")


def page_model(reference, results):
    import SpiceModel as sm
    if "capacitance_differential" not in results or "magnetic" not in results:
        st.info("This DUT predates the capacitance analysis; re-run it (Bench tab, or "
                "`python bench.py characterize REF --offline`) to build its models.")
        return
    try:
        sm.NgSpice.get()
    except Exception as error:
        st.error(f"ngspice not available: {error}")
        return
    left, right = st.columns([1, 3])
    with left:
        kind = st.radio("Model", ["broadband (fitted networks)", "small-signal (constant L, k, R, C)"])
        with_fixture = st.checkbox("Include measured fixture terms", value=True)
        configs = [n for n in range(1, 17)]
        chosen = st.multiselect("Configurations", [f"{n:2d} {rbc.CONFIGS[n]['name']}" for n in configs],
                                default=[" 1 Z0", " 2 Zsc", " 7 C33"])
    try:
        if kind.startswith("broadband"):
            parameters = sm.broadband_parameters(reference, results)
            model = sm.broadband_subcircuit(reference, parameters, name="DUT")
            st.caption(f"Broadband fit: leakage {parameters['leakage']['median_error']*100:.1f} %, "
                       f"core + dielectric {parameters['shunt']['median_error']*100:.1f} % "
                       f"({len(parameters['shunt']['rl'])} RL, {len(parameters['shunt']['rc'])} RC branches)")
        else:
            model = sm.subcircuit(reference, sm.model_parameters(results), name="DUT")
    except Exception as error:
        st.error(f"Could not build the model: {error}")
        return
    fixture = sm.fixture_terms(results) if with_fixture else None
    mag = figure("|Z|: measured (solid) vs ngspice (dashed)", y_title="|Z| (Ω)", log_y=True)
    phase = figure("Phase", y_title="deg")
    errors = []
    for index, label in enumerate(chosen):
        number = int(label.split()[0])
        try:
            f, Y = cfit.load_admittance(str(OUTPUT), reference, number)
        except FileNotFoundError:
            continue
        zm = 1 / Y.mean(axis=0)
        fs, zs = sm.simulate(results, number, fixture, model=model)
        add_trace(mag, f, numpy.abs(zm), f"{label} measured", index)
        add_trace(mag, fs, numpy.abs(zs), f"{label} SPICE", index, dash="dash")
        add_trace(phase, f, numpy.degrees(numpy.angle(zm)), f"{label} measured", index)
        add_trace(phase, fs, numpy.degrees(numpy.angle(zs)), f"{label} SPICE", index, dash="dash")
        zi = numpy.interp(f, fs, zs.real) + 1j * numpy.interp(f, fs, zs.imag)
        band = f <= 1e7
        errors.append({"config": label, "median |Z| error ≤10 MHz (%)": float(numpy.median(numpy.abs(zi[band] / zm[band] - 1)) * 100)})
    with right:
        st.plotly_chart(mag, use_container_width=True)
        st.plotly_chart(phase, use_container_width=True)
        if errors:
            st.dataframe(pandas.DataFrame(errors), hide_index=True)
    with st.expander("SPICE subcircuit"):
        st.code("\n".join(model[1]), language="text")
        st.download_button("Download .subckt", "\n".join(model[1]), file_name=f"{reference}_{model[0]}.lib")


def page_inductance(reference, files, results):
    if files.empty:
        return
    c1, c2 = st.columns(2)
    fig = figure("Open-circuit inductance (RL sweeps)", y_title="L (µH)")
    for index, kind in enumerate(sorted(k for k in files.kind.unique() if k.startswith("RL"))):
        row = files[(files.config == 1) & (files.kind == kind)]
        if not row.empty:
            s = load_sweep(row.iloc[0].path)
            add_trace(fig, s["f"], s["ls"] * 1e6, f"Z0 {kind}", index)
    c1.plotly_chart(fig, use_container_width=True)
    fig = figure("Leakage inductance vs frequency (proximity effect)", y_title="Lsc (nH)")
    for index, (number, label) in enumerate(((2, "Zsc primary"), (4, "Zsc' secondary"), (9, "cfg 9"), (6, "Ldif"))):
        row = files[(files.config == number) & (files.kind == "Zhd")]
        if not row.empty:
            s = load_sweep(row.iloc[0].path)
            keep = s["f"] <= 2e7
            add_trace(fig, s["f"][keep], s["ls"][keep] * 1e9, label, index)
    c2.plotly_chart(fig, use_container_width=True)
    drive = results.get("drive_linearity")
    if drive:
        fig = figure("Amplitude permeability: L(10 kHz) vs drive", x_title="Source level (dBm)",
                     y_title="L (µH)", log_x=False)
        fig.add_trace(go.Scatter(x=drive["levels_dbm"], y=numpy.array(drive["L_10kHz"]) * 1e6,
                                 mode="lines+markers", line=dict(color=SERIES[0], width=2), marker=dict(size=9),
                                 name="L(10 kHz)"))
        st.plotly_chart(fig, use_container_width=True)


def page_capacitance(reference, results):
    d = results.get("capacitance_differential")
    if not d:
        st.info("No capacitance analysis for this DUT.")
        return
    try:
        spectra = cfit.open_differences(str(OUTPUT), reference)
    except FileNotFoundError:
        st.info("Dense sweeps missing.")
        return
    c1, c2 = st.columns(2)
    fig = figure("Open-link differences (core-free)", y_title="ΔC (pF)")
    for index, key in enumerate(("BC", "AD", "AC", "floating")):
        add_trace(fig, spectra["frequency"], spectra[key]["spectrum"], f"{key} − BD", index)
    fig.add_vrect(x0=1e6, x1=8e6, fillcolor="#e4e3df", opacity=0.4, line_width=0)
    fig.update_yaxes(range=[-5, 50])
    c1.plotly_chart(fig, use_container_width=True)
    fig = figure("Self-capacitance spectrum (open states, lower bound on S)", y_title="Im(Y)/ω (pF)")
    for index, number in enumerate((8, 1, 3, 10)):
        f, Y = cfit.load_admittance(str(OUTPUT), reference, number)
        add_trace(fig, f, cfit.effective_capacitance(f, Y).mean(0) * 1e12, f"{number} {rbc.CONFIGS[number]['name']}", index)
    fig.update_xaxes(range=[6, 7.7])
    fig.update_yaxes(range=[-5, 25])
    c2.plotly_chart(fig, use_container_width=True)
    fig = figure("Electrostatic-only states (no magnetics)", y_title="Im(Y)/ω (pF)")
    for index, number in enumerate((7, 16, 17, 18, 19)):
        try:
            f, Y = cfit.load_admittance(str(OUTPUT), reference, number)
        except FileNotFoundError:
            continue
        add_trace(fig, f, cfit.effective_capacitance(f, Y).mean(0) * 1e12, f"{number} {rbc.CONFIGS[number]['name']}", index)
    fig.update_yaxes(range=[-1, 30])
    st.plotly_chart(fig, use_container_width=True)
    st.json({k: d[k] for k in ("C11", "C22", "C12", "C13", "C23", "u_open", "u_short", "fixture_dL_nh",
                               "short_pair_rms") if k in d}, expanded=False)


def page_fixture():
    st.subheader("Per-path calibrations")
    rows = []
    for path in sorted(CALIBRATIONS.glob("relay_board_*.json")):
        meta = json.loads(path.read_text())
        raw = meta.get("standards_raw_ohm", {})
        rows.append({"path": meta.get("signal_path"), "acquired": meta.get("acquired"),
                     "board": meta.get("relay_board", "").split(",")[-1],
                     "OPEN Ω": raw.get("OPEN"), "SHORT mΩ": (raw.get("SHORT") or 0) * 1e3,
                     "LOAD Ω": raw.get("LOAD"), "SHORT re-act mΩ": (meta.get("short_residual_ohm") or 0) * 1e3,
                     "check": meta.get("standards_checked", "before_osl")})
    st.dataframe(pandas.DataFrame(rows), hide_index=True, use_container_width=True)
    paths = CALIBRATIONS / "fixture_paths.json"
    if paths.exists():
        st.subheader("Board-only series paths (subtracted from LINK-shorted states)")
        st.json(json.loads(paths.read_text()))
    for name, title in (("bench_column_map.json", "Column map"), ("BENCH_bench_accuracy.json", "Accuracy study"),
                        ("BENCH_RM_bench_accuracy.json", "Accuracy study (RM)")):
        path = OUTPUT / name
        if path.exists():
            with st.expander(title):
                st.json(json.loads(path.read_text()))


def page_compare(all_refs):
    chosen = st.multiselect("DUTs", all_refs, default=all_refs[:2])
    number = st.selectbox("Configuration", sorted(rbc.CONFIGS), format_func=lambda n: f"{n} {rbc.CONFIGS[n]['name']}")
    quantity = st.selectbox("Quantity ", list(QUANTITIES))
    key, log_y = QUANTITIES[quantity]
    fig = figure(f"Config {number} across DUTs", y_title=quantity, log_y=log_y)
    curves = {}
    for index, ref in enumerate(chosen):
        files = sweep_files(ref)
        row = files[(files.config == number) & (files.kind == "Zhd")] if not files.empty else files
        if row.empty:
            continue
        s = load_sweep(row.iloc[0].path)
        add_trace(fig, s["f"], s[key], ref, index)
        curves[ref] = s
    st.plotly_chart(fig, use_container_width=True)
    if len(curves) == 2:
        (a, sa), (b, sb) = curves.items()
        zb = numpy.interp(sa["f"], sb["f"], sb["mag"])
        fig = figure(f"FRA-style deviation |Z({b})|/|Z({a})| − 1", y_title="%")
        add_trace(fig, sa["f"], (zb / sa["mag"] - 1) * 100, "deviation", 0)
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------ bench control

def bench_characterizer(reference="GUI", bode_ip="127.0.0.1"):
    import MagneticCharacterizer as M
    return M.MagneticCharacterizer(reference, bode_ip=bode_ip)


def relay_diagram(word):
    names = {v: f"{t}{r[0]}" for (t, r), v in rbc.MATRIX_BITS.items()}
    names.update({rbc.LOAD_HI_BIT: "LdH", rbc.LOAD_LO_BIT: "LdL"})
    names.update({v: f"iso{t}" for t, v in rbc.ISOLATE_BITS.items()})
    fig = go.Figure()
    for bit in range(rbc.RELAY_COUNT):
        on = (word >> bit) & 1
        fig.add_trace(go.Scatter(x=[bit % 6], y=[-(bit // 6)], mode="markers+text", text=[names.get(bit, bit)],
                                 textposition="bottom center", showlegend=False,
                                 marker=dict(size=34, symbol="square",
                                             color=SERIES[2] if on else "#e4e3df", line=dict(color="#52514e", width=1))))
    fig.update_layout(height=260, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def page_bench():
    st.caption("Live control. Needs the Bode 100 SCPI server (python scripts/bench.py server) and the relay board on USB.")
    bode_ip = st.text_input("Bode SCPI address", "127.0.0.1")
    tab_relays, tab_sweep, tab_runs = st.tabs(["Relays", "Single sweep", "Runs & experiments"])
    with tab_relays:
        number = st.selectbox("Configuration", sorted(rbc.CONFIGS),
                              format_func=lambda n: f"{n} {rbc.CONFIGS[n]['name']} — {rbc.CONFIGS[n]['purpose']}")
        mode = st.radio("Mode", ["MEAS", "OPEN", "SHORT", "LOAD"], horizontal=True)
        st.plotly_chart(relay_diagram(rbc.measurement_word(rbc.CONFIGS[number])
                                      | (rbc.calibration_overlay(rbc.CONFIGS[number], mode) if mode != "MEAS" else 0)),
                        use_container_width=True)
        if st.button("Apply to the board and read back"):
            board = rbc.RelayBoardController()
            try:
                board.set_config(number)
                board.set_calibration_mode(mode)
                st.success(f"{board.identity}: readback {board.get_relay_states()}")
            finally:
                board.close()
    with tab_sweep:
        c1, c2, c3 = st.columns(3)
        number = c1.selectbox("Config ", sorted(rbc.CONFIGS), format_func=lambda n: f"{n} {rbc.CONFIGS[n]['name']}")
        start = c1.number_input("Start (Hz)", 10.0, 5e7, 1e4, format="%g")
        stop = c2.number_input("Stop (Hz)", 10.0, 5e7, 5e7, format="%g")
        points = c2.number_input("Points", 11, 2001, 401)
        rbw = c3.selectbox("RBW", ["1000Hz", "300Hz", "100Hz", "30Hz", "10Hz"], index=1)
        dbm = c3.slider("Drive (dBm)", -30, 13, 13)
        cycles = c3.number_input("Cycles", 1, 20, 2)
        if st.button("Calibrate (if needed) and sweep"):
            c = bench_characterizer(bode_ip=bode_ip)
            try:
                c._set_config(number)
                c.bode_100.receiver_bandwidth = rbw
                with st.spinner("Sweeping..."):
                    data = c.bode_100.take_Z_phase_measurement(start, stop, int(cycles), int(points), dbm)
                c.relay_board.reset()
            finally:
                c.relay_board.close()
                c.bode_100.close_visa_session()
            path = OUTPUT / f"GUI_{time.strftime('%Y%m%d_%H%M%S')}_cfg{number:02d}.csv"
            data.to_csv(path, index=False)
            s = load_sweep(str(path))
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            fig.add_trace(go.Scatter(x=s["f"], y=s["mag"], name="|Z|", line=dict(color=SERIES[0], width=2)), 1, 1)
            fig.add_trace(go.Scatter(x=s["f"], y=s["phase"], name="phase", line=dict(color=SERIES[1], width=2)), 2, 1)
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log", row=1, col=1)
            fig.update_layout(template="plotly_white", height=520)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Saved {path.name}")
    with tab_runs:
        reference = st.text_input("DUT reference", "NEW_DUT")
        describe = st.text_input("Description (core, material, turns...)", "")
        experiment = st.selectbox("Run", ["characterize (full, ~20 min)", "report", "spice", "check",
                                          "fixture-paths", "column-map", "drive", "accuracy", "link-cal",
                                          "consistency", "firmware (build only)"])
        if st.button("Start"):
            command = [sys.executable, "-u", str(SCRIPTS / "bench.py"), "--bode-ip", bode_ip]
            name = experiment.split()[0]
            command.append(name)
            if name in ("characterize", "report", "spice", "consistency"):
                command.append(reference)
            if name == "characterize" and describe:
                command += ["--describe", describe]
            st.code(" ".join(command))
            log = st.empty()
            lines = []
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       cwd=str(SCRIPTS))
            for line in process.stdout:
                lines.append(line.rstrip())
                log.code("\n".join(lines[-40:]))
            process.wait()
            (st.success if process.returncode == 0 else st.error)(f"Finished with exit code {process.returncode}")
            st.cache_data.clear()


# ---------------------------------------------------------------------- main

def main():
    st.title("OpenMagnetics Lab")
    all_refs = references()
    with st.sidebar:
        st.header("DUT")
        if not all_refs:
            st.warning("No characterized DUTs in scripts/output yet.")
            reference = None
        else:
            import os
            preferred = os.environ.get("OM_GUI_REFERENCE")
            newest = max(all_refs, key=lambda r: (OUTPUT / f"{r}_results.json").stat().st_mtime)
            default = preferred if preferred in all_refs else newest
            reference = st.selectbox("Reference", all_refs, index=all_refs.index(default))
            st.caption(dut_description(reference) or "no description")
        st.divider()
        st.caption("Bode 100 + OpenMagnetics relay board rev B\n\nAll pages read cached data; "
                   "'Bench' talks to the instruments.")
    pages = ["Summary", "Sweeps", "Model vs data (SPICE)", "Inductance", "Capacitance", "Fixture & calibration",
             "Compare DUTs", "Bench"]
    tabs = st.tabs(pages)
    payload = load_results(reference) if reference else {}
    results = payload.get("results", {})
    files = sweep_files(reference) if reference else pandas.DataFrame()
    with tabs[0]:
        if reference:
            page_summary(reference, payload)
    with tabs[1]:
        if reference:
            page_sweeps(reference, files)
    with tabs[2]:
        if reference and results:
            page_model(reference, results)
    with tabs[3]:
        if reference:
            page_inductance(reference, files, results)
    with tabs[4]:
        if reference:
            page_capacitance(reference, results)
    with tabs[5]:
        page_fixture()
    with tabs[6]:
        page_compare(all_refs)
    with tabs[7]:
        page_bench()


main()
