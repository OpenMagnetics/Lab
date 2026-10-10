"""
SCPI driver for the OMICRON Bode 100 vector network analyzer.

One-port impedance measurement via the impedance-adapter (IAD) method.
Returns long-form pandas DataFrames indexed by measurement_index + frequency.

Calibration handling is deliberately paranoid: the Bode 100 holds exactly one
active OSL correction, and a switched fixture has several signal paths.  The
previous version returned early whenever ANY calibration was active, so a fresh
session against an instrument still holding another path's correction would
silently keep it under the new path's name.  Here the loaded path is tracked
per-process and a mismatch always forces a reload.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy
import pandas
import pyvisa


def _scpi_bool(response):
    """':SENS:CORR:FULL:ACT?' may answer '1', '+1', '1\\n', 'ON'."""
    text = str(response).strip().upper()
    if text in ("ON", "TRUE"):
        return True
    if text in ("OFF", "FALSE"):
        return False
    try:
        return int(float(text)) != 0
    except ValueError:
        raise ValueError(f"Cannot interpret SCPI boolean response {response!r}")


class MagneticMeasurer:
    """Bode 100 driver for the rev B relay board.

    Bode 100 front-panel setup for the integrated bridge (the board clones
    the B-WIC's internal network, per Omicron's bridge application note
    2015-09-16):

    - OUTPUT -> board SOURCE, CH1 -> board CH1, CH2 -> board CH2 (short,
      equal-length 50 R BNC leads).
    - CH1 and CH2 input impedance: 1 Mohm (the bridge presents divider
      nodes, not 50 R sources; 50 R inputs would re-load the bridge).
    - Receiver attenuators: lowest range that does not overload -- the
      RC1 shunt is 2.35 R, so CH2 sees small signals for large |Z| DUTs.
    - Impedance method on the instrument stays IAD (:SENS:Z:METH IAD),
      exactly as with a physical B-WIC; the OSL run through the relay
      matrix absorbs the board's bridge tolerances.
    """

    #: Default source level in dBm. [COG94] III advises raising it at high
    #: frequency for signal-to-noise and lowering it at low frequency to avoid
    #: core saturation, so every measurement method takes an override.
    DEFAULT_SOURCE_POWER_DBM = 13

    def __init__(self, SCPI_server_IP="192.168.96.135", scpi_port=5025, timeout_ms=20000):
        self.SCPI_timeout = timeout_ms
        self.VISA_server = f"TCPIP::{SCPI_server_IP}::{scpi_port}::SOCKET"

        self.sweep_type = "LOG"
        self.receiver_bandwidth = "300Hz"
        self.source_power = self.DEFAULT_SOURCE_POWER_DBM

        # Which calibration path this PROCESS last loaded onto the instrument.
        # Never inferred from the instrument -- see class docstring.
        self.current_calibration_group = None
        self.current_calibration_file = None

        self.visa_session = self.new_visa_session()

    # ---------------------------------------------------------------- session

    def new_visa_session(self):
        print(f"Connecting to VISA resource {self.VISA_server}")
        session = pyvisa.ResourceManager().open_resource(self.VISA_server)
        session.timeout = self.SCPI_timeout
        session.read_termination = "\n"
        print(f"Connected: {session.query('*IDN?')}")
        # The OMICRON SCPI server answers *IDN? itself but only opens and
        # initializes the USB instrument on the first device command (~2-3 s).
        # Writes that arrive during that window are silently dropped, so force
        # the initialization with a query before any setting is sent.
        session.query(":SENS:FREQ:STOP?")
        return session

    def close_visa_session(self):
        self.visa_session.close()

    def is_calibrated(self):
        return _scpi_bool(self.visa_session.query(":SENS:CORR:FULL:ACT?"))

    # ------------------------------------------------------------ calibration

    def calibrate(self, calibration_file, calibration_group, force_calibration=False,
                  calibration_load=100.0, max_attempts=3):
        """Ensure the instrument holds the OSL correction for `calibration_group`.

        Loads the stored .mcalx when one exists; otherwise runs an interactive
        OSL against physical standards and stores the result.

        `calibration_group` is REQUIRED. The old signature let it default to
        None, which is how a stale correction could be relabelled.
        """
        if calibration_group is None:
            raise ValueError("calibration_group is required; refusing to load an unlabelled calibration.")

        already_loaded = (
            self.current_calibration_group == calibration_group
            and self.current_calibration_file == calibration_file
            and self.is_calibrated()
        )
        if already_loaded and not force_calibration:
            return True

        if os.path.exists(calibration_file) and not force_calibration:
            self._load_calibration(calibration_file, max_attempts)
        else:
            self._run_interactive_osl(calibration_file, calibration_load, max_attempts)

        if not self.is_calibrated():
            raise RuntimeError(f"Instrument reports no active calibration after loading {calibration_file}")

        self.current_calibration_group = calibration_group
        self.current_calibration_file = calibration_file
        print(f"Calibration active: group {calibration_group} <- {os.path.basename(calibration_file)}")
        return True

    def _load_calibration(self, calibration_file, max_attempts):
        print(f"Loading calibration {calibration_file}")
        last_error = None
        for attempt in range(1, max_attempts + 1):
            self.visa_session.write(f':MMEM:LOAD:CORR:FULL "{calibration_file}"')
            self.visa_session.write("*WAI")
            self.visa_session.query("*OPC?")
            try:
                self.take_Rs_Ls_measurement(start_frequency=10000, stop_frequency=11000,
                                            number_of_measurement_cycles=1,
                                            number_of_measurement_points=11)
                return
            except pyvisa.errors.VisaIOError as error:
                last_error = error
                print(f"  attempt {attempt}/{max_attempts} timed out, retrying")
        raise RuntimeError(f"Could not load calibration {calibration_file} after {max_attempts} attempts") from last_error

    def _run_interactive_osl(self, calibration_file, calibration_load, max_attempts):
        """Manual open/short/load. Blocks on input() -- see RelayBoardController
        for the automatic path using the board's own standards."""
        print(f"Running interactive OSL (load = {calibration_load} ohm)")
        self.visa_session.write(f":SOUR:POW {self.source_power}")
        self.visa_session.write(f":SENS:CORR:LOAD {calibration_load}")
        self.visa_session.write(":CALC:ZPAR:DEF Z")

        for prompt, command in (
            ("Leave the fixture OPEN", ":SENS:CORR:FULL:OPEN"),
            ("Fit the SHORT standard", ":SENS:CORR:FULL:SHOR"),
            (f"Fit the {calibration_load} ohm LOAD standard", ":SENS:CORR:FULL:LOAD"),
        ):
            input(f"  {prompt} and press Enter...")
            self.visa_session.write(command)
            self.visa_session.write("*WAI")
            self.visa_session.query("*OPC?")

        if not self.is_calibrated():
            raise RuntimeError("OSL sequence completed but the instrument reports no active calibration.")

        self._verify_load_standard(calibration_load, max_attempts)
        print(f"Storing calibration to {calibration_file}")
        self.visa_session.write(f':MMEM:STOR:CORR "{calibration_file}"')

    def _verify_load_standard(self, calibration_load, max_attempts):
        """Re-measure the load standard. The old code asserted on a variable
        that was never bound in this branch, so the check raised
        UnboundLocalError instead of its own message."""
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                data = self.take_Rs_Ls_measurement(start_frequency=100, stop_frequency=1000,
                                                   number_of_measurement_cycles=1,
                                                   number_of_measurement_points=11)
                measured = float(data["resistance"].iloc[0])
                error = abs(measured - calibration_load) / calibration_load
                if error < 1e-3:
                    print(f"  load standard verified: {measured:.4f} ohm ({error*100:.4f}% error)")
                    return
                raise RuntimeError(
                    f"Load standard reads {measured:.4f} ohm but calibration_load is "
                    f"{calibration_load} ohm ({error*100:.2f}% error). Wrong standard fitted?"
                )
            except pyvisa.errors.VisaIOError as error:
                last_error = error
                print(f"  verification attempt {attempt}/{max_attempts} timed out")
        raise RuntimeError("Could not verify the load standard") from last_error

    # ----------------------------------------------------------- measurement

    def _configure_sweep(self, start_frequency, stop_frequency, number_of_measurement_points,
                         source_power_dbm=None):
        power = self.source_power if source_power_dbm is None else source_power_dbm
        session = self.visa_session
        session.write(f":SENS:FREQ:STAR {start_frequency}")
        session.write(f":SENS:FREQ:STOP {stop_frequency}")
        session.write(":CALC:PAR:DEF Z")
        session.write(":SENS:Z:METH IAD")
        session.write(f":SENS:SWE:POIN {number_of_measurement_points}")
        session.write(f":SENS:SWE:TYPE {self.sweep_type}")
        session.write(f":SENS:BAND {self.receiver_bandwidth}")
        session.write(f":SOUR:POW {power}")
        session.write(":TRIG:SOUR BUS")
        session.write(":INIT:CONT ON")

    def _acquire(self, number_of_measurement_points):
        """One triggered sweep -> (frequencies, primary array, secondary array).

        OMICRON documents :CALC:DATA:SDAT? as returning <array 1>,<array 2> --
        all primary values then all secondary, NOT interleaved pairs. The split
        below relies on that, so the length is asserted rather than assumed.
        """
        self.visa_session.write(":TRIG:SING")
        self.visa_session.query("*OPC?")

        raw = [float(v) for v in self.visa_session.query(":CALC:DATA:SDAT?").split(",")]
        frequencies = [float(v) for v in self.visa_session.query(":SENS:FREQ:DATA?").split(",")]

        expected = 2 * number_of_measurement_points
        if len(raw) != expected:
            raise RuntimeError(
                f":CALC:DATA:SDAT? returned {len(raw)} values, expected {expected} "
                f"(two blocks of {number_of_measurement_points}). Data format may have changed."
            )
        if len(frequencies) != number_of_measurement_points:
            raise RuntimeError(
                f":SENS:FREQ:DATA? returned {len(frequencies)} points, expected "
                f"{number_of_measurement_points}."
            )

        return (numpy.array(frequencies),
                numpy.array(raw[:number_of_measurement_points]),
                numpy.array(raw[number_of_measurement_points:]))

    def _sweep(self, start_frequency, stop_frequency, number_of_measurement_cycles,
               number_of_measurement_points, source_power_dbm, calc_format, build_columns):
        """Run N cycles and assemble one DataFrame.

        Built from a list of frames concatenated once. The old code called
        pandas.concat per ROW, which is O(n^2) and dominated runtime after the
        instrument itself.
        """
        self._configure_sweep(start_frequency, stop_frequency, number_of_measurement_points,
                              source_power_dbm)
        self.visa_session.write(calc_format)

        frames = []
        for measurement_index in range(number_of_measurement_cycles):
            frequencies, primary, secondary = self._acquire(number_of_measurement_points)
            columns = {"measurement_index": measurement_index, "frequency": frequencies}
            columns.update(build_columns(frequencies, primary, secondary))
            frames.append(pandas.DataFrame(columns))
        return pandas.concat(frames, ignore_index=True)

    def take_Rs_Ls_measurement(self, start_frequency=10000, stop_frequency=1000000,
                               number_of_measurement_cycles=2, number_of_measurement_points=201,
                               source_power_dbm=None):
        """Series resistance and inductance. :CALC:FORM SCOM -> real, imaginary."""
        def columns(frequencies, real, imaginary):
            return {"resistance": real, "inductance": imaginary / (2.0 * math.pi * frequencies)}

        self.visa_session.write(":CALC:ZPAR:DEF Z")
        return self._sweep(start_frequency, stop_frequency, number_of_measurement_cycles,
                           number_of_measurement_points, source_power_dbm, ":CALC:FORM SCOM", columns)

    def take_Z_phase_measurement(self, start_frequency=100, stop_frequency=40000000,
                                 number_of_measurement_cycles=1, number_of_measurement_points=201,
                                 source_power_dbm=None):
        """Impedance magnitude (ohm) and phase (degrees). :CALC:FORM SLIN."""
        def columns(frequencies, magnitude, phase):
            return {"magnitude": magnitude, "phase": phase}

        self.visa_session.write(":CALC:ZPAR:DEF Z")
        return self._sweep(start_frequency, stop_frequency, number_of_measurement_cycles,
                           number_of_measurement_points, source_power_dbm, ":CALC:FORM SLIN", columns)

    def take_Cs_measurement(self, start_frequency=10000, stop_frequency=1000000,
                            number_of_measurement_cycles=2, number_of_measurement_points=201,
                            source_power_dbm=None):
        """SERIES capacitance -- :CALC:ZPAR:DEF Cs.

        Renamed from take_Cp_measurement, which wrote Cs, stored a column called
        'capacitance' and labelled its plots 'Parallel capacitance'. For a lossy
        interwinding path Cs and Cp differ by (1 + D^2), so the name now matches
        the SCPI parameter. Used for C33, which [BLA94] II-D defines as the
        capacitance between the two windings when both are short-circuited.
        """
        def columns(frequencies, capacitance, _unused):
            return {"capacitance": capacitance}

        self.visa_session.write(":CALC:ZPAR:DEF Cs")
        return self._sweep(start_frequency, stop_frequency, number_of_measurement_cycles,
                           number_of_measurement_points, source_power_dbm, ":CALC:FORM REAL", columns)

    # Backwards-compatible alias for existing scripts.
    take_Cp_measurement = take_Cs_measurement

    # -------------------------------------------------------------- plotting

    @staticmethod
    def _averaged(data, columns):
        wanted = [c for c in columns if c in data.columns]
        return data[wanted + ["frequency"]].groupby("frequency", as_index=False).mean()

    def plot_RL(self, data, plot_resistance=True, resistance_label="Resistance",
                plot_inductance=True, inductance_label="Inductance", title=None):
        if not plot_resistance and not plot_inductance:
            raise ValueError("plot_RL called with both traces disabled -- nothing to draw.")

        averaged = self._averaged(data, ["resistance", "inductance"])
        series = []
        if plot_resistance:
            series.append(("resistance", resistance_label, "tab:red"))
        if plot_inductance:
            series.append(("inductance", inductance_label, "tab:blue"))

        figure, axis = plt.subplots()
        column, label, colour = series[0]
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel(label, color=colour)
        axis.plot(averaged["frequency"], averaged[column], color=colour)
        axis.tick_params(axis="y", labelcolor=colour)
        axis.set_xscale("log")

        if len(series) > 1:
            column, label, colour = series[1]
            twin = axis.twinx()
            twin.set_ylabel(label, color=colour)
            twin.plot(averaged["frequency"], averaged[column], color=colour)
            twin.tick_params(axis="y", labelcolor=colour)

        if title:
            axis.set_title(title)
        figure.tight_layout()
        plt.show()

    def plot_Z(self, data, title=None):
        averaged = self._averaged(data, ["magnitude", "phase"])
        figure, axis = plt.subplots()
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Impedance magnitude (ohm)", color="tab:red")
        axis.plot(averaged["frequency"], averaged["magnitude"], color="tab:red")
        axis.tick_params(axis="y", labelcolor="tab:red")
        axis.set_xscale("log")
        axis.set_yscale("log")

        twin = axis.twinx()
        twin.set_ylabel("Phase (deg)", color="tab:blue")
        twin.plot(averaged["frequency"], averaged["phase"], color="tab:blue")
        twin.tick_params(axis="y", labelcolor="tab:blue")
        twin.set_ylim(-95, 95)

        if title:
            axis.set_title(title)
        figure.tight_layout()
        plt.show()

    def plot(self, data, column, label, title=None, log_y=False):
        averaged = self._averaged(data, [column])
        figure, axis = plt.subplots()
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel(label, color="tab:red")
        axis.plot(averaged["frequency"], averaged[column], color="tab:red")
        axis.tick_params(axis="y", labelcolor="tab:red")
        axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        if title:
            axis.set_title(title)
        figure.tight_layout()
        plt.show()
