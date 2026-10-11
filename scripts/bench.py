"""
One entry point for everything the bench does -- every measurement, check,
model and report of the rev B bring-up is reproducible from here.

    python bench.py server                 start the Bode 100 SCPI server (foreground)
    python bench.py check                  instruments up? relay word readback for every config
    python bench.py characterize REF [--describe "T26 3C90 10+10 turns"] [--offline]
    python bench.py report REF [--before OLDREF]
    python bench.py spice REF              .lib (small-signal + broadband) and model-vs-data CSV
    python bench.py fixture-paths          board-only series paths (cached per board identity)
    python bench.py column-map             every column / LINK loop relative to column A
    python bench.py drive                  amplitude permeability + LOAD vs drive level
    python bench.py accuracy               RBW vs noise/time study, calibration vs drive
    python bench.py link-cal               Lcum with and without LINK in the OSL
    python bench.py consistency REF        model-independent checks on cached data
    python bench.py manual-check PREFIX ETA   ground-free check on an old manual-jumper dataset
    python bench.py firmware [--flash]     build the STM32 firmware; --flash: SYST:DFU + dfu-util
    python bench.py gui                    launch the GUI (Streamlit)
    python bench.py tests                  run every offline test suite

Instrument commands start the SCPI server for their own duration when none
is answering on --bode-ip (default 127.0.0.1).
"""

import argparse
import contextlib
import glob
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent.resolve()
ROOT = HERE.parent
OUTPUT = HERE / "output"
TOOLS = pathlib.Path(os.environ.get("OM_TOOLS", pathlib.Path.home() / "tools"))
SCPI_RUNNER = r"C:\Program Files\OMICRON\BodeAnalyzerSuite\OmicronLab.VectorNetworkAnalysis.ScpiRunner.exe"


# --------------------------------------------------------------- SCPI server

def scpi_answering(host, port=5025, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def bode_serial():
    listing = subprocess.run([SCPI_RUNNER, "--list-devices"], capture_output=True, text=True, timeout=60).stdout
    # The device line looks like "Bode100R2-LN604P" (model-serial).
    for line in listing.splitlines():
        line = line.strip()
        if line.startswith("Bode") and "-" in line and "|" not in line:
            return line.split("-", 1)[1]
    raise RuntimeError(f"No Bode 100 found by ScpiRunner:\n{listing}")


@contextlib.contextmanager
def scpi_server(host="127.0.0.1"):
    """Yield with a SCPI server answering on host:5025, starting one if needed.

    ScpiRunner stops when its stdin closes ("Press enter to stop"), so it is
    run with a pipe held open for as long as we need it.
    """
    if scpi_answering(host):
        yield
        return
    process = subprocess.Popen([SCPI_RUNNER, "-i", host, "-p", "5025", "-s", bode_serial()],
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            if scpi_answering(host):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("SCPI server did not come up")
        yield
    finally:
        with contextlib.suppress(Exception):
            process.stdin.write(b"\n")
            process.stdin.flush()
            process.wait(timeout=10)
        if process.poll() is None:
            process.kill()


def characterizer(arguments, reference="BENCH", offline=False):
    import MagneticCharacterizer as M
    return M.MagneticCharacterizer(reference, relay_board_port=arguments.port, bode_ip=arguments.bode_ip,
                                   offline=offline)


# ------------------------------------------------------------------ commands

def cmd_server(arguments):
    print(f"Starting ScpiRunner on {arguments.bode_ip}:5025 (Ctrl+C to stop)")
    subprocess.run([SCPI_RUNNER, "-i", arguments.bode_ip, "-p", "5025", "-s", bode_serial()])


def cmd_check(arguments):
    import RelayBoardController as rbc
    with scpi_server(arguments.bode_ip):
        c = characterizer(arguments)
        print("Bode:", c.bode_100.visa_session.query("*IDN?").strip())
        print("Board:", c.relay_board.identity)
        bad = 0
        for number in sorted(rbc.CONFIGS):
            c.relay_board.set_config(number)        # raises if the readback differs
            for mode in ("OPEN", "SHORT", "LOAD", "MEAS"):
                c.relay_board.set_calibration_mode(mode)
        c.relay_board.reset()
        print(f"All {len(rbc.CONFIGS)} configurations x 4 modes verified by relay readback.")


def cmd_characterize(arguments):
    command = [sys.executable, str(HERE / "MagneticCharacterizer.py"), arguments.reference,
               "--bode-ip", arguments.bode_ip]
    if arguments.port:
        command += ["--port", arguments.port]
    if arguments.describe:
        command += ["--describe", arguments.describe]
    if arguments.offline:
        command.append("--offline")
    elif arguments.cache:
        command.append("--cache")
    if arguments.offline:
        subprocess.run(command, check=True)
    else:
        with scpi_server(arguments.bode_ip):
            subprocess.run(command, check=True)


def cmd_report(arguments):
    command = [sys.executable, str(HERE / "make_report.py"), arguments.reference]
    if arguments.before:
        command += ["--before", arguments.before]
    subprocess.run(command, check=True)


def cmd_spice(arguments):
    subprocess.run([sys.executable, str(HERE / "SpiceModel.py"), arguments.reference], check=True)


def _instrument(function_name, arguments, save_as):
    import BenchExperiments as bx
    with scpi_server(arguments.bode_ip):
        c = characterizer(arguments)
        result = getattr(bx, function_name)(c)
    if hasattr(result, "to_csv"):
        result.to_csv(OUTPUT / f"bench_{save_as}.csv", index=False)
    else:
        (OUTPUT / f"bench_{save_as}.json").write_text(json.dumps(result, indent=2))
    print(f"Saved output/bench_{save_as}.*")


def cmd_fixture_paths(arguments):
    with scpi_server(arguments.bode_ip):
        c = characterizer(arguments)
        c.characterize_fixture_paths(allow_use_cache=False)


def cmd_accuracy(arguments):
    with scpi_server(arguments.bode_ip):
        subprocess.run([sys.executable, str(HERE / "bench_accuracy.py"), "BENCH", "--bode-ip", arguments.bode_ip]
                       + (["--port", arguments.port] if arguments.port else []), check=True)


def cmd_consistency(arguments):
    import BenchExperiments as bx
    result = bx.consistency(arguments.reference)
    (OUTPUT / f"{arguments.reference}_consistency.json").write_text(json.dumps(result, indent=2))


def cmd_manual_check(arguments):
    import BenchExperiments as bx
    bx.manual_jumper_check(arguments.prefix, arguments.eta)


def cmd_firmware(arguments):
    arm = TOOLS / "arm" / "bin"
    make = shutil.which("make") or next(iter(glob.glob(str(pathlib.Path.home() /
        "AppData/Local/Microsoft/WinGet/Packages/ezwinports.make*/bin/make.exe"))), None)
    if not make:
        raise SystemExit("make not found (winget install ezwinports.make)")
    environment = dict(os.environ, PATH=str(arm) + os.pathsep + os.environ["PATH"])
    opencm3 = TOOLS / "libopencm3"
    if not (opencm3 / "lib" / "libopencm3_stm32f0.a").exists():
        raise SystemExit(f"libopencm3 not built in {opencm3}: git clone it and make TARGETS=stm32/f0")
    subprocess.run([make, "-C", str(ROOT / "firmware"), f"OPENCM3_DIR={opencm3.as_posix()}"],
                   env=environment, check=True)
    if not arguments.flash:
        return
    import RelayBoardController as rbc
    dfu = TOOLS / "dfu-util-0.11-binaries" / "win64" / "dfu-util.exe"
    board = rbc.RelayBoardController(com_port=arguments.port)
    print("Running:", board.identity, "-> SYST:DFU")
    board.enter_dfu()
    time.sleep(4)
    subprocess.run([str(dfu), "-d", "0483:df11", "-a", "0", "-s", "0x08000000:leave", "-D",
                    str(ROOT / "firmware" / "build" / "relay_board.bin")], check=True)
    time.sleep(4)
    print("Now running:", rbc.RelayBoardController(com_port=arguments.port).identity)


def cmd_gui(arguments):
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(HERE / "gui" / "app.py")], check=True)


def cmd_tests(arguments):
    failures = 0
    for script in ("test_model.py", "test_calibration.py", "test_capacitance_fit.py"):
        print(f"\n=== {script}")
        failures += subprocess.run([sys.executable, str(HERE / script)]).returncode != 0
    print(f"\n=== verify_board.py")
    failures += subprocess.run([sys.executable, str(ROOT / "hardware" / "relay-board-revB" / "verify_board.py")]).returncode != 0
    raise SystemExit(1 if failures else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--bode-ip", default="127.0.0.1")
    parser.add_argument("--port", default=None, help="relay board VISA resource (auto-detected)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("server").set_defaults(func=cmd_server)
    sub.add_parser("check").set_defaults(func=cmd_check)
    p = sub.add_parser("characterize")
    p.add_argument("reference")
    p.add_argument("--describe")
    p.add_argument("--cache", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_characterize)
    p = sub.add_parser("report")
    p.add_argument("reference")
    p.add_argument("--before")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("spice")
    p.add_argument("reference")
    p.set_defaults(func=cmd_spice)
    sub.add_parser("fixture-paths").set_defaults(func=cmd_fixture_paths)
    sub.add_parser("column-map").set_defaults(func=lambda a: _instrument("column_map", a, "column_map"))
    sub.add_parser("drive").set_defaults(func=lambda a: _instrument("drive_sweep", a, "drive"))
    sub.add_parser("link-cal").set_defaults(func=lambda a: _instrument("link_calibration_test", a, "link_cal"))
    sub.add_parser("accuracy").set_defaults(func=cmd_accuracy)
    p = sub.add_parser("consistency")
    p.add_argument("reference")
    p.set_defaults(func=cmd_consistency)
    p = sub.add_parser("manual-check")
    p.add_argument("prefix")
    p.add_argument("eta", type=float)
    p.set_defaults(func=cmd_manual_check)
    p = sub.add_parser("firmware")
    p.add_argument("--flash", action="store_true")
    p.set_defaults(func=cmd_firmware)
    sub.add_parser("gui").set_defaults(func=cmd_gui)
    sub.add_parser("tests").set_defaults(func=cmd_tests)
    arguments = parser.parse_args()
    arguments.func(arguments)


if __name__ == "__main__":
    main()
