"""
SCPI driver for the OpenMagnetics relay switching board (rev B).

ARCHITECTURE
------------
Four DUT terminals A, B, C, D reach three internal rails through a full
crossbar: every terminal can be tied to HI, LO or LINK independently.

    HI    -- Bode 100 port, "hot" side
    LO    -- Bode 100 port, return side
    LINK  -- floating tie bus, for shorts and winding links that must NOT
             touch either measurement node

Twelve matrix relays (4 terminals x 3 rails) express every topology the
Cogitore/Blache method needs, and nothing is hard-wired into a fixed set of
"configurations" the way rev A was.  Two isolation relays are CHANGEOVERS:
de-energized they connect each terminal node to its clamp (the DUT); energized
they connect it to the calibration standards instead.  That is what makes OSL
automatic -- the DUT can stay clamped while an OPEN standard is presented --
and it keeps exactly one relay contact between the terminal node and either the
DUT or the standard, so the calibration path mirrors the measurement path.

Four standards relays switch a copper short or a 100 ohm 0.1% resistor across
each measurement pair.

Because calibration is automatic, every configuration can be calibrated in its
OWN relay state.  Rev A grouped 21 configurations onto 4 shared calibrations,
which meant relays inside a group changed the fixture between calibration and
measurement -- fine for inductance, not for picofarads.

The STM32 firmware lives in firmware/ in this repository.
"""

import pyvisa

# --------------------------------------------------------------------------
# Relay map -- must match firmware/src/relay_map.h
# --------------------------------------------------------------------------

TERMINALS = ("A", "B", "C", "D")
RAILS = ("HI", "LO", "LINK")

#: Shift-register bit for each matrix connection.
MATRIX_BITS = {
    ("A", "HI"): 0, ("A", "LO"): 1, ("A", "LINK"): 2,
    ("B", "HI"): 3, ("B", "LO"): 4, ("B", "LINK"): 5,
    ("C", "HI"): 6, ("C", "LO"): 7, ("C", "LINK"): 8,
    ("D", "HI"): 9, ("D", "LO"): 10, ("D", "LINK"): 11,
}
#: K13/K14: the LOAD-standard column -- one 100R 0.1% between two full-length
#: column buses, reached through the same contact structure as a terminal.
LOAD_HI_BIT = 12             # K13: CAL_E -> RAIL_HI
LOAD_LO_BIT = 13             # K14: CAL_F -> RAIL_LO
#: K15..K18: per-terminal isolation. COM = terminal node, NC = clamp,
#: NO = nothing (the open contact IS the open standard).
ISOLATE_BITS = {"A": 14, "B": 15, "C": 16, "D": 17}
RELAY_COUNT = 18

CALIBRATION_LOAD_OHM = 100.0


# --------------------------------------------------------------------------
# Measurement configurations
# --------------------------------------------------------------------------
#
# Each entry says which terminals go to which rail.  The signal path -- and so
# the calibration -- is fully determined by this dict, which is why the board
# no longer needs a hand-maintained CALIBRATION_GROUPS table that could drift
# out of step with the wiring.

def _config(number, name, hi, lo, link=(), purpose=""):
    return {
        "number": number,
        "name": name,
        "HI": tuple(hi),
        "LO": tuple(lo),
        "LINK": tuple(link),
        "purpose": purpose,
    }


CONFIGS = {c["number"]: c for c in [
    # --- the four canonical two-port impedances, [COG94] II-A -------------
    _config(1, "Z0", "A", "B", (),
            "primary, secondary open -- L0"),
    _config(2, "Zsc", "A", "B", ("C", "D"),
            "primary, secondary shorted -- Lsc"),
    _config(3, "Z0p", "C", "D", (),
            "secondary, primary open -- L0'"),
    _config(4, "Zscp", "C", "D", ("A", "B"),
            "secondary, primary shorted -- Lsc'. NEW in rev B: completes the "
            "set so Z0*Zsc' = Z0'*Zsc can verify the relays every run"),

    # --- mutual inductance cross-check ------------------------------------
    _config(5, "Lcum", "A", "D", ("B", "C"),
            "series aiding (B-C joined) -- M = (Lcum-Ldif)/4"),
    _config(6, "Ldif", "A", "C", ("B", "D"),
            "series opposing (B-D joined)"),

    # --- direct interwinding capacitance, [BLA94] II-D --------------------
    _config(7, "C33", ("A", "B"), ("C", "D"), (),
            "both windings shorted, measure between them -- C33 directly"),

    # --- winding links for the six-capacitance extraction -----------------
    # With the secondary SHORTED, linking B-D and B-C give the same topology,
    # as do A-C and A-D. [BLA94] Table 1 agrees: their C1+C3 columns are equal
    # in each pair. Rev A spent four configurations on what are two states.
    _config(8, "link_BD_open", "A", "B", ("D",),
            "B-D linked, secondary open"),
    _config(9, "link_BD_short", "A", ("B", "C", "D"), (),
            "B-D (or B-C) linked, secondary shorted"),
    _config(10, "link_AC_open", ("A", "C"), "B", (),
            "A-C linked, secondary open"),
    _config(11, "link_AC_short", ("A", "C", "D"), "B", (),
            "A-C (or A-D) linked, secondary shorted"),
    _config(12, "link_BC_open", "A", ("B", "C"), (),
            "B-C linked, secondary open"),
    _config(13, "link_AD_open", ("A", "D"), "B", (),
            "A-D linked, secondary open"),

    # --- secondary-side mirrors -------------------------------------------
    # [COG94] III: "Among these four measurements, the least accurate is
    # eliminated ... some extra measurements often allow useful checkings."
    # Measuring the same link from the other winding gives a redundant
    # equation whose accuracy is better whenever the secondary impedance sits
    # in a friendlier part of the analyzer's range.
    _config(14, "link_BD_short_sec", "C", ("A", "B", "D"), (),
            "B-D linked and primary shorted, measured from the secondary "
            "-- redundant with config 9"),
    _config(15, "link_AC_short_sec", ("A", "B", "C"), "D", (),
            "A-C linked and primary shorted, measured from the secondary "
            "-- redundant with config 11"),
]}

#: Which winding-link case in TransformerModel.WINDING_LINKS each pair of
#: configurations feeds.  "floating" reuses Z0/Zsc, which is correct: with no
#: link the measurement is identical and only the interpretation differs.
LINK_CONFIGS = {
    "B-D": {"open": 8, "short": 9},
    "A-C": {"open": 10, "short": 11},
    "B-C": {"open": 12, "short": 9},
    "A-D": {"open": 13, "short": 11},
    "floating": {"open": 1, "short": 2},
}

def signal_path(config):
    """Stable key identifying the fixture state, used to name calibrations."""
    hi = "".join(sorted(config["HI"]))
    lo = "".join(sorted(config["LO"]))
    link = "".join(sorted(config["LINK"]))
    return f"{hi}-{lo}" + (f"-L{link}" if link else "")


def calibration_overlay(config, mode):
    """Extra relay bits to energize on top of a configuration for one standard.

    OPEN   isolate all four terminals; the matrix stays in the measurement
           state, so the calibration sees the exact fixture it will correct.
    SHORT  isolate, then ALSO close the LO crossbar relay of the first HI
           terminal: that terminal's bus then bridges RAIL_HI to RAIL_LO --
           a short presented through the same contact plane as the DUT.
    LOAD   isolate, then close K13+K14: the 100R column bridges the rails
           through one contact and a full column bus per side, mimicking a
           two-terminal DUT (Keysight impedance handbook: measure the load
           "in the same way as the DUT will be measured").
    """
    mode = mode.upper()
    if mode == "MEAS":
        return 0
    overlay = sum(1 << bit for bit in ISOLATE_BITS.values())
    if mode == "SHORT":
        first_hi = config["HI"][0]
        overlay |= 1 << MATRIX_BITS[(first_hi, "LO")]
    elif mode == "LOAD":
        overlay |= (1 << LOAD_HI_BIT) | (1 << LOAD_LO_BIT)
    elif mode != "OPEN":
        raise ValueError(f"Unknown calibration mode {mode!r}")
    return overlay


class RelayBoardController:
    IDN_PREFIX = "OpenMagnetics,RelayBoard"

    def __init__(self, com_port=None, baud_rate=115200, timeout_ms=5000):
        self.baud_rate = baud_rate
        self.timeout_ms = timeout_ms
        self.resource_string = com_port or self._auto_detect()
        self.visa_session = self._connect(self.resource_string)
        self.identity = self.visa_session.query("*IDN?").strip()
        self.current_config = None
        self.current_path = None
        self.cal_mode = "MEAS"

    # ------------------------------------------------------------ transport

    def _auto_detect(self):
        manager = pyvisa.ResourceManager()
        for resource in manager.list_resources():
            if "ASRL" not in resource:
                continue
            session = None
            try:
                session = manager.open_resource(resource)
                session.timeout = 2000
                session.read_termination = "\n"
                session.write_termination = "\n"
                session.baud_rate = self.baud_rate
                identity = session.query("*IDN?")
                if self.IDN_PREFIX in identity:
                    print(f"Found relay board at {resource}: {identity}")
                    return resource
            except Exception:
                continue
            finally:
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        pass
        raise RuntimeError(
            "Could not auto-detect the relay board. Check the USB cable and that "
            "the board enumerates as a serial port."
        )

    def _connect(self, resource_string):
        print(f"Connecting to relay board at {resource_string}")
        session = pyvisa.ResourceManager().open_resource(resource_string)
        session.timeout = self.timeout_ms
        session.read_termination = "\n"
        session.write_termination = "\n"
        session.baud_rate = self.baud_rate
        identity = session.query("*IDN?")
        if self.IDN_PREFIX not in identity:
            raise RuntimeError(f"Device at {resource_string} is not a relay board: {identity}")
        print(f"Connected: {identity}")
        return session

    def _command(self, text):
        self.visa_session.write(text)
        self.visa_session.query("*OPC?")
        error = self.visa_session.query("SYST:ERR?")
        if not error.strip().startswith("0"):
            raise RuntimeError(f"Relay board rejected {text!r}: {error.strip()}")

    # ----------------------------------------------------------- switching

    def set_config(self, config_number):
        if config_number not in CONFIGS:
            raise ValueError(f"Unknown config {config_number}; valid: {sorted(CONFIGS)}")
        config = CONFIGS[config_number]
        self._command(f"CONF:MEAS {config_number}")
        self.current_config = config_number
        self.current_path = signal_path(config)
        self.cal_mode = "MEAS"
        print(f"Config {config_number} ({config['name']}): "
              f"HI={'+'.join(config['HI'])} LO={'+'.join(config['LO'])}"
              + (f" LINK={'+'.join(config['LINK'])}" if config["LINK"] else ""))
        return config

    def get_config(self):
        return int(self.visa_session.query("CONF:MEAS?").strip())

    def set_calibration_mode(self, mode):
        """MEAS | OPEN | SHORT | LOAD, applied on top of the current config.

        OPEN/SHORT/LOAD energize the isolation relays, disconnecting the DUT and
        substituting the on-board standard -- so the DUT can stay clamped.
        """
        mode = mode.upper()
        if mode not in ("MEAS", "OPEN", "SHORT", "LOAD"):
            raise ValueError(f"Bad calibration mode {mode!r}")
        if self.current_config is None:
            raise RuntimeError("Set a measurement configuration before a calibration mode.")
        self._command(f"CAL:MODE {mode}")
        self.cal_mode = mode
        return mode

    def calibration_name(self, config_number=None):
        """File-safe name of the calibration owned by a configuration."""
        config = CONFIGS[config_number if config_number is not None else self.current_config]
        return f"relay_board_{signal_path(config)}.mcalx"

    def needs_recalibration(self, config_number):
        return signal_path(CONFIGS[config_number]) != self.current_path

    # ------------------------------------------------------------ low level

    def set_relay(self, relay_number, state):
        if not 0 <= relay_number < RELAY_COUNT:
            raise ValueError(f"Relay must be 0..{RELAY_COUNT - 1}, got {relay_number}")
        self._command(f"RELAY {relay_number} {1 if state else 0}")

    def get_relay_states(self):
        response = self.visa_session.query("RELAY:ALL?").strip()
        return [int(bit) for bit in response.split(",")]

    def self_test(self):
        """Firmware-side check: shift register readback and coil supply rail."""
        return self.visa_session.query("*TST?").strip()

    def reset(self):
        self._command("*RST")
        self.current_config = None
        self.current_path = None
        self.cal_mode = "MEAS"
        print("Relay board reset: all relays de-energized, DUT connected")

    def close(self):
        try:
            self.reset()
        finally:
            self.visa_session.close()


if __name__ == "__main__":
    print("=" * 70)
    print("Relay board self-test")
    print("=" * 70)
    controller = RelayBoardController()
    print(f"Firmware self-test: {controller.self_test()}")

    paths = {}
    for number in sorted(CONFIGS):
        config = controller.set_config(number)
        path = signal_path(config)
        paths.setdefault(path, []).append(number)
        print(f"    relays: {controller.get_relay_states()}")
        print(f"    calibration: {controller.calibration_name(number)}")
        print(f"    {config['purpose']}")

    print("\nDistinct signal paths (one calibration each):")
    for path, numbers in sorted(paths.items()):
        print(f"  {path:16s} <- configs {numbers}")

    print("\nCalibration overlays per config (relay bits added to the state):")
    for number, config in sorted(CONFIGS.items()):
        overlays = {m: calibration_overlay(config, m) for m in ("OPEN", "SHORT", "LOAD")}
        print(f"  config {number:2d} {config['name']:18s} "
              f"OPEN={overlays['OPEN']:05x} SHORT={overlays['SHORT']:05x} "
              f"LOAD={overlays['LOAD']:05x}")

    controller.close()
