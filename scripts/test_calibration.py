"""
Offline tests of the automatic per-path OSL flow in MagneticCharacterizer.

Fake Bode 100 and relay board: the fake analyzer reads back an impedance that
depends on which standard the fake board presents, so a stuck relay can be
simulated.  No instruments needed:

    python scripts/test_calibration.py
"""

import datetime
import json
import pathlib
import sys
import tempfile

import pandas

import MagneticCharacterizer as mc
import RelayBoardController as rbc

FAILURES = []


def check(name, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


class FakeBoard:
    identity = "OpenMagnetics,RelayBoard,revB,0001"

    def __init__(self, broken=None):
        self.broken = broken          # None | "LOAD" | "SHORT" | "OPEN"
        self.mode = "MEAS"
        self.history = []

    def set_calibration_mode(self, mode):
        self.mode = mode
        self.history.append(mode)

    def calibration_name(self, config_number):
        return f"relay_board_{rbc.signal_path(rbc.CONFIGS[config_number])}.mcalx"

    def presented_ohm(self):
        nominal = {"OPEN": 5e6, "SHORT": 0.2, "LOAD": 100.3, "MEAS": 50.0}[self.mode]
        if self.broken == self.mode == "LOAD":
            return 5e6                # K13/K14 stuck open: LOAD looks like OPEN
        if self.broken == self.mode == "SHORT":
            return 5e6                # LO crossbar relay never closed
        if self.broken == self.mode == "OPEN":
            return 30.0               # iso relay welded: DUT still connected
        return nominal


class FakeSession:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.writes = []

    def write(self, text):
        self.writes.append(text)

    def query(self, text):
        if text == "*IDN?":
            return "OMICRON Lab,Bode 100,FAKE,1.0"
        if text == ":SENS:CORR:FULL:ACT?":
            return "1" if self.analyzer.osl_succeeds else "0"
        return "1"


class FakeBode:
    source_power = 13

    def __init__(self, board, osl_succeeds=True):
        self.board = board
        self.osl_succeeds = osl_succeeds
        self.visa_session = FakeSession(self)
        self.current_calibration_group = None
        self.current_calibration_file = None
        self.loaded = []

    def is_calibrated(self):
        return self.osl_succeeds

    def take_Z_phase_measurement(self, **_):
        return pandas.DataFrame({"measurement_index": 0, "frequency": [900.0, 1000.0, 1100.0],
                                 "magnitude": self.board.presented_ohm(), "phase": 0.0})

    def _verify_load_standard(self, load, max_attempts):
        if self.board.mode != "LOAD":
            raise RuntimeError(f"verified the load in mode {self.board.mode}")

    def calibrate(self, calibration_file, calibration_group, **_):
        self.loaded.append(calibration_file)


def characterizer(directory, board, bode):
    instance = object.__new__(mc.MagneticCharacterizer)
    instance.calibrations_path = pathlib.Path(directory)
    instance.relay_board = board
    instance.bode_100 = bode
    instance.warnings = []
    return instance


def stored_files(bode):
    return [w for w in bode.visa_session.writes if w.startswith(":MMEM:STOR:CORR")]


def test_good_path():
    print("\nAutomatic OSL, healthy board")
    with tempfile.TemporaryDirectory() as directory:
        board = FakeBoard()
        bode = FakeBode(board)
        c = characterizer(directory, board, bode)
        c._ensure_calibrated(1)
        sidecar = pathlib.Path(directory) / "relay_board_A-B.json"
        check("calibration stored on the instrument", len(stored_files(bode)) == 1)
        check("provenance sidecar written", sidecar.exists())
        check("board left in MEAS", board.mode == "MEAS", board.mode)
        check("drive level set before OSL",
              any(w.startswith(":SOUR:POW") for w in bode.visa_session.writes))
        metadata = json.loads(sidecar.read_text())
        check("raw standards recorded", set(metadata["standards_raw_ohm"]) == {"OPEN", "SHORT", "LOAD"})

        # second call reuses it
        pathlib.Path(directory, "relay_board_A-B.mcalx").write_text("fake")
        c._ensure_calibrated(1)
        check("fresh calibration is reused", bode.loaded and len(stored_files(bode)) == 1)


def test_broken_relays():
    for broken in ("LOAD", "SHORT", "OPEN"):
        print(f"\nAutomatic OSL, {broken} standard broken")
        with tempfile.TemporaryDirectory() as directory:
            board = FakeBoard(broken=broken)
            bode = FakeBode(board)
            c = characterizer(directory, board, bode)
            try:
                c._ensure_calibrated(1)
                raised = None
            except RuntimeError as error:
                raised = str(error)
            check("refuses to calibrate", raised is not None and broken in raised, raised or "no error")
            check("nothing stored", not stored_files(bode) and not list(pathlib.Path(directory).iterdir()))
            check("board left in MEAS", board.mode == "MEAS", board.mode)


def test_osl_failure_restores_meas():
    print("\nAutomatic OSL, instrument reports no correction")
    with tempfile.TemporaryDirectory() as directory:
        board = FakeBoard()
        bode = FakeBode(board, osl_succeeds=False)
        c = characterizer(directory, board, bode)
        try:
            c._ensure_calibrated(1)
            raised = False
        except RuntimeError:
            raised = True
        check("raises", raised)
        check("board left in MEAS", board.mode == "MEAS", board.mode)


def test_staleness():
    print("\nCalibration reuse rules")
    with tempfile.TemporaryDirectory() as directory:
        board = FakeBoard()
        bode = FakeBode(board)
        c = characterizer(directory, board, bode)
        mcalx = pathlib.Path(directory, "relay_board_A-B.mcalx")
        sidecar = mcalx.with_suffix(".json")
        mcalx.write_text("fake")
        check("no sidecar -> re-acquire", not c._calibration_is_current(str(mcalx)))

        def write(age_hours, **override):
            metadata = dict(c._calibration_provenance(),
                            acquired=(datetime.datetime.now()
                                      - datetime.timedelta(hours=age_hours)).isoformat())
            metadata.update(override)
            sidecar.write_text(json.dumps(metadata))

        write(1)
        check("fresh, same setup -> reuse", c._calibration_is_current(str(mcalx)))
        write(c.MAX_CALIBRATION_AGE_HOURS + 1)
        check("too old -> re-acquire", not c._calibration_is_current(str(mcalx)))
        write(1, relay_board="OpenMagnetics,RelayBoard,revB,0002")
        check("different board -> re-acquire", not c._calibration_is_current(str(mcalx)))
        write(1, source_power_dbm=0)
        check("different drive level -> re-acquire", not c._calibration_is_current(str(mcalx)))
        sidecar.write_text("{not json")
        check("corrupt sidecar -> re-acquire", not c._calibration_is_current(str(mcalx)))


if __name__ == "__main__":
    for test in (test_good_path, test_broken_relays, test_osl_failure_restores_meas, test_staleness):
        test()
    print("\n" + "=" * 68)
    print(f"{'all passed' if not FAILURES else f'{len(FAILURES)} FAILED: ' + ', '.join(FAILURES)}")
    print("=" * 68)
    sys.exit(1 if FAILURES else 0)
