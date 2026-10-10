"""
Automated small-signal characterization of a two-winding magnetic.

Drives the Bode 100 and the relay board to extract the full [COG94] Fig. 1
equivalent circuit: coupling, magnetizing and leakage inductance, winding and
core loss resistances, and the six capacitance coefficients of [BLA94].

    characterizer = MagneticCharacterizer("my_transformer")
    results = characterizer.characterize_all()

Every raw sweep is cached to output/{reference}_{config}_{kind}.csv, so any
analysis change can be re-run with allow_use_cache=True and no instruments
connected.  The extraction maths lives in TransformerModel and is unit-tested
independently (python test_model.py).
"""

import argparse
import datetime
import json
import math
import os
import pathlib

import numpy
import pandas

import Bode100Analyzer
import CapacitanceFit as cfit
import RelayBoardController as rbc
import TransformerModel as tm


class MagneticCharacterizer:

    #: Inductance band. Starts low enough to reach the resistive plateau that
    #: [COG94] II-C reads r1 and r2 from -- rev A started at 10 kHz, which for
    #: most magnetics is already inductive, so those resistances were
    #: unreachable.
    INDUCTANCE_BAND = (100.0, 1_000_000.0)

    #: Resonance band for the capacitance work.
    RESONANCE_BAND = (10_000.0, 40_000_000.0)

    #: Capacitance-grade sweep ("Zhd"): the analyzer's full 50 MHz, dense, and
    #: four cycles so every point carries its own repeatability. The
    #: differential estimators need the whole band and the cycle scatter.
    CAPACITANCE_SWEEP = dict(band=(10_000.0, 50_000_000.0), points=801, cycles=4,
                             drive_dbm=13)
    CAPACITANCE_CONFIGS = (7, 16, 17, 18, 19, 1, 8, 10, 12, 13, 3, 2, 9, 11, 4, 14, 15, 5, 6)

    #: [COG94] III: lower drive at low frequency to avoid core saturation,
    #: higher at high frequency for signal-to-noise. Rev A used a fixed 13 dBm
    #: everywhere, which is wrong at both ends of a 100 Hz - 40 MHz sweep.
    DRIVE_LOW_BAND_DBM = 0
    DRIVE_HIGH_BAND_DBM = 13

    def __init__(self, reference="temp", relay_board_port=None, bode_ip=None,
                 auto_calibrate=True):
        self.reference = reference
        self.auto_calibrate = auto_calibrate

        here = pathlib.Path(__file__).parent.resolve()
        self.calibrations_path = here / "calibrations"
        self.output_path = here / "output"
        self.calibrations_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.bode_100 = None
        self.relay_board = None
        self.offline = False
        self.results = {}
        self.warnings = []
        # Sweeps taken in this session, keyed by everything that defines them,
        # so two recipes asking for the same sweep share one acquisition.
        self._session_sweeps = {}

        try:
            kwargs = {"SCPI_server_IP": bode_ip} if bode_ip else {}
            self.bode_100 = Bode100Analyzer.MagneticMeasurer(**kwargs)
            self.relay_board = rbc.RelayBoardController(com_port=relay_board_port)
            self.relay_board.reset()
        except Exception as error:
            self.offline = True
            print(f"\n*** No instruments ({type(error).__name__}: {error})")
            print("*** Running OFFLINE -- cached CSVs only, allow_use_cache must be True.\n")

    # ------------------------------------------------------------ warnings

    def warn(self, message):
        self.warnings.append(message)
        print(f"  !! {message}")

    # --------------------------------------------------------- acquisition

    def _cache_path(self, config_number, kind):
        name = rbc.CONFIGS[config_number]["name"]
        return self.output_path / f"{self.reference}_cfg{config_number:02d}_{name}_{kind}.csv"

    def _set_config(self, config_number):
        """Switch, then make sure the calibration matches THIS relay state.

        Rev A grouped many configurations onto one calibration, so relays inside
        a group changed the fixture between calibrating and measuring. With
        on-board standards every configuration can own its calibration, so it
        does.
        """
        config = self.relay_board.set_config(config_number)
        if self.auto_calibrate:
            self._ensure_calibrated(config_number)
        return config

    #: A stored calibration older than this is re-acquired. OSL is automatic
    #: and the DUT stays clamped, so re-running it is cheap next to measuring
    #: against a fixture that drifted with temperature since yesterday.
    MAX_CALIBRATION_AGE_HOURS = 24.0

    #: Pre-OSL relay sanity check. At 1 kHz the three standards sit decades
    #: apart whatever correction (or none) the instrument holds, so a stuck
    #: K13/K14, a dead SHORT relay or a wrong R5 shows up before it can be
    #: baked into a .mcalx. Bounds are deliberately loose: this catches
    #: broken relays, not fixture error.
    STANDARD_CHECK_BAND = (900.0, 1100.0)
    STANDARD_CHECK_LOAD_TOLERANCE = 0.20
    STANDARD_CHECK_OPEN_MIN_OHM = 10_000.0
    STANDARD_CHECK_SHORT_MAX_OHM = 5.0

    def _calibration_metadata_path(self, calibration_file):
        return pathlib.Path(calibration_file).with_suffix(".json")

    def _calibration_provenance(self):
        """What a stored calibration is only valid for."""
        return {
            "relay_board": self.relay_board.identity,
            "bode_100": self.bode_100.visa_session.query("*IDN?").strip(),
            "source_power_dbm": self.bode_100.source_power,
            "load_ohm": rbc.CALIBRATION_LOAD_OHM,
        }

    def _calibration_is_current(self, calibration_file):
        """A .mcalx is reused only with a sidecar that matches this setup.

        Files without one (acquired before provenance was recorded, or copied
        in by hand) are re-acquired rather than trusted.
        """
        metadata_path = self._calibration_metadata_path(calibration_file)
        if not (os.path.exists(calibration_file) and metadata_path.exists()):
            return False
        try:
            metadata = json.loads(metadata_path.read_text())
            acquired = datetime.datetime.fromisoformat(metadata["acquired"])
        except (ValueError, KeyError) as error:
            print(f"  Unreadable calibration metadata {metadata_path.name} ({error}); re-acquiring")
            return False
        age_hours = (datetime.datetime.now() - acquired).total_seconds() / 3600.0
        if age_hours > self.MAX_CALIBRATION_AGE_HOURS:
            print(f"  Calibration {os.path.basename(calibration_file)} is {age_hours:.0f} h old; re-acquiring")
            return False
        for key, value in self._calibration_provenance().items():
            if metadata.get(key) != value:
                print(f"  Calibration {os.path.basename(calibration_file)} was taken with "
                      f"{key}={metadata.get(key)!r}, now {value!r}; re-acquiring")
                return False
        return True

    def _standard_magnitude(self, mode):
        self.relay_board.set_calibration_mode(mode)
        start, stop = self.STANDARD_CHECK_BAND
        data = self.bode_100.take_Z_phase_measurement(
            start_frequency=start, stop_frequency=stop, number_of_measurement_cycles=1,
            number_of_measurement_points=3, source_power_dbm=self.bode_100.source_power)
        return float(data["magnitude"].median())

    def _check_standards(self, path):
        """Each standard must land in its own decade before OSL is trusted."""
        readings = {mode: self._standard_magnitude(mode) for mode in ("OPEN", "SHORT", "LOAD")}
        load = rbc.CALIBRATION_LOAD_OHM
        problems = [f"{mode} reads {value!r} (degenerate standards make the OSL singular)"
                    for mode, value in readings.items() if not math.isfinite(value)]
        if abs(readings["LOAD"] - load) > self.STANDARD_CHECK_LOAD_TOLERANCE * load:
            problems.append(f"LOAD reads {readings['LOAD']:.3g} ohm, expected ~{load:g} "
                            "(K13/K14 stuck, or R5 wrong/missing)")
        if readings["OPEN"] < self.STANDARD_CHECK_OPEN_MIN_OHM:
            problems.append(f"OPEN reads {readings['OPEN']:.3g} ohm "
                            "(an isolation relay did not open the DUT, or a standard relay is stuck closed)")
        if readings["SHORT"] > self.STANDARD_CHECK_SHORT_MAX_OHM:
            problems.append(f"SHORT reads {readings['SHORT']:.3g} ohm "
                            "(the first HI terminal's LO crossbar relay did not close)")
        if problems:
            raise RuntimeError(f"Calibration standards failed the relay check on path {path}: "
                               + "; ".join(problems))
        print("    standards OK: " + ", ".join(f"{m} {v:.3g} ohm" for m, v in readings.items()))
        return readings

    def _ensure_calibrated(self, config_number):
        calibration_file = str(self.calibrations_path / self.relay_board.calibration_name(config_number))
        path = rbc.signal_path(rbc.CONFIGS[config_number])

        if self._calibration_is_current(calibration_file):
            self.bode_100.calibrate(calibration_file, calibration_group=path)
            return

        print(f"  Calibrating path {path} (automatic, DUT stays clamped)")
        session = self.bode_100.visa_session
        # The board must never be left with the DUT switched out, whatever
        # fails below -- otherwise the next "measurement" is of a standard.
        try:
            # In IAD mode the Bode 100 refuses to sweep without an active
            # correction ("Calibration must be active" -- the data query just
            # hangs), so the raw check needs one already on the instrument.
            # On the first path of a session there is none: check the
            # standards through the fresh OSL instead, before it is stored.
            check_before_osl = self.bode_100.is_calibrated()
            standards = self._check_standards(path) if check_before_osl else None

            # The correction belongs to the measurement method active when it
            # is acquired: a fresh server session starts in one-port reflection
            # (S11/P1R), and an OSL taken there leaves IAD with no correction.
            # Put the instrument in the IAD impedance setup first. The FULL
            # correction is full-range, so the band chosen here does not limit
            # later sweeps.
            start, stop = self.STANDARD_CHECK_BAND
            self.bode_100._configure_sweep(start, stop, 3, self.bode_100.source_power)
            # Calibrate at an explicit, recorded drive level rather than
            # whatever the last sweep left on the source.
            session.write(f":SOUR:POW {self.bode_100.source_power}")
            session.write(f":SENS:CORR:LOAD {rbc.CALIBRATION_LOAD_OHM}")
            session.write(":CALC:ZPAR:DEF Z")
            for mode, command in (("OPEN", ":SENS:CORR:FULL:OPEN"),
                                  ("SHORT", ":SENS:CORR:FULL:SHOR"),
                                  ("LOAD", ":SENS:CORR:FULL:LOAD")):
                self.relay_board.set_calibration_mode(mode)
                session.write(command)
                session.write("*WAI")
                session.query("*OPC?")
                print(f"    {mode} standard applied")

            if not self.bode_100.is_calibrated():
                raise RuntimeError(f"Automatic OSL failed for path {path}")
            if standards is None:
                standards = self._check_standards(path)

            # Re-actuate and re-read through the new correction. LOAD must
            # come back as the defined value (catches noise and unstable
            # contacts); the SHORT residual is one sample of the contact
            # repeatability that sits outside the OSL plane.
            self.relay_board.set_calibration_mode("OPEN")
            self.relay_board.set_calibration_mode("LOAD")
            self.bode_100._verify_load_standard(rbc.CALIBRATION_LOAD_OHM, max_attempts=3)
            self.relay_board.set_calibration_mode("OPEN")
            short_residual = self._standard_magnitude("SHORT")
            print(f"    SHORT after re-actuation: {short_residual*1e3:.1f} mohm")
        finally:
            try:
                self.relay_board.set_calibration_mode("MEAS")
            except Exception as error:
                print(f"  !! could not return the relay board to MEAS: {error}")

        session.write(f':MMEM:STOR:CORR "{calibration_file}"')
        metadata = dict(self._calibration_provenance(),
                        acquired=datetime.datetime.now().isoformat(timespec="seconds"),
                        signal_path=path,
                        standards_raw_ohm=standards,
                        standards_checked="before_osl" if check_before_osl else "through_fresh_osl",
                        short_residual_ohm=short_residual)
        self._calibration_metadata_path(calibration_file).write_text(json.dumps(metadata, indent=2))
        self.bode_100.current_calibration_group = path
        self.bode_100.current_calibration_file = calibration_file
        print(f"    stored {os.path.basename(calibration_file)}")

    def measure(self, config_number, kind="RL", allow_use_cache=False, band=None,
                cycles=2, points=201, drive_dbm=None):
        """Acquire (or load) one sweep. kind is 'RL', 'Z', 'Cs' or 'Zhd'
        (capacitance-grade Z: CAPACITANCE_SWEEP settings)."""
        if kind == "Zhd":
            sweep = self.CAPACITANCE_SWEEP
            band, points, cycles, drive_dbm = sweep["band"], sweep["points"], sweep["cycles"], sweep["drive_dbm"]
        path = self._cache_path(config_number, kind)
        if allow_use_cache and path.exists():
            return pandas.read_csv(path)
        if self.offline:
            raise RuntimeError(
                f"Offline and no cached sweep at {path.name}. Connect the "
                "instruments, or run a configuration whose CSVs already exist."
            )

        start, stop = band or (self.INDUCTANCE_BAND if kind == "RL" else self.RESONANCE_BAND)
        if drive_dbm is None:
            drive_dbm = self.DRIVE_LOW_BAND_DBM if start < 1000 else self.DRIVE_HIGH_BAND_DBM

        # Recipes overlap (inductance and losses both need configs 1 and 3).
        # Re-measuring used to overwrite the CSV another recipe had already
        # analysed, so a later --cache run saw a different sweep than the live
        # one and gave slightly different results (eta 1.0032 vs 1.0037).
        key = (config_number, kind, start, stop, cycles, points, drive_dbm)
        if key in self._session_sweeps:
            return self._session_sweeps[key]

        self._set_config(config_number)
        method = {"RL": self.bode_100.take_Rs_Ls_measurement,
                  "Z": self.bode_100.take_Z_phase_measurement,
                  "Zhd": self.bode_100.take_Z_phase_measurement,
                  "Cs": self.bode_100.take_Cs_measurement}[kind]
        data = method(start_frequency=start, stop_frequency=stop,
                      number_of_measurement_cycles=cycles,
                      number_of_measurement_points=points,
                      source_power_dbm=drive_dbm)
        data.to_csv(path, index=False)
        self._session_sweeps[key] = data
        return data

    def value_at(self, data, frequency, parameter="inductance", tolerance=0.05):
        """Value at a frequency, refusing to silently return a far-off point.

        Rev A seeded its search at a relative error of 100 and never checked
        afterwards, so asking for 100 Hz on a sweep starting at 10 kHz returned
        the 10 kHz point into a variable named for 100 Hz.
        """
        averaged = tm.average_cycles(data)
        errors = (averaged["frequency"] - frequency).abs() / frequency
        index = errors.idxmin()
        if errors[index] > tolerance:
            available = averaged["frequency"]
            raise ValueError(
                f"No point within {tolerance*100:.0f}% of {frequency:g} Hz. "
                f"Sweep covers {available.min():g}..{available.max():g} Hz; "
                f"nearest is {available[index]:g} Hz."
            )
        return float(averaged.loc[index, "frequency"]), float(averaged.loc[index, parameter])

    # ------------------------------------------------------- verification

    def verify_switching(self, allow_use_cache=False):
        """[BLA94] II-C: Z0*Zsc' = Z0'*Zsc for any linear two-port.

        An end-to-end check that the relay matrix produced the four topologies
        the script asked for -- no reference standard needed, checkable at every
        frequency. The dominant failure mode of a switched fixture is a relay
        silently in the wrong state returning a plausible sweep; this catches it.
        """
        print("\n-- Switching self-test (reciprocity) --")
        sweeps = {name: self.measure(number, "Z", allow_use_cache, band=self.RESONANCE_BAND)
                  for name, number in (("Z0", 1), ("Zsc", 2), ("Z0p", 3), ("Zscp", 4))}
        # Judged below RECIPROCITY_MAX_HZ: above ~10 MHz the uncalibrated path
        # inductance and the ferrite's dispersive capacitance dominate the shorts.
        _, _, merged = tm.check_reciprocity(sweeps["Z0"], sweeps["Z0p"], sweeps["Zsc"], sweeps["Zscp"])
        merged = merged[merged.frequency <= self.RECIPROCITY_MAX_HZ]
        worst = float(merged.relative_error.max())
        passed = worst <= 0.05
        self.results["reciprocity_error"] = worst
        self.results["reciprocity_passed"] = passed
        if passed:
            print(f"  PASS -- worst deviation {worst*100:.2f}%")
        elif worst <= self.RECIPROCITY_FIXTURE_LIMIT:
            # A relay in the wrong state swaps an open for a short: orders of
            # magnitude, not percent. A few percent, smooth in frequency, is
            # the link path that shorts the far winding: it sits outside the
            # OSL plane and differs between the two directions (rev B2 TODO
            # "make the SHORT follow the DUT path"; ~26-59 nH on rev B).
            self.warn(f"Reciprocity off by {worst*100:.1f}% (<= {self.RECIPROCITY_MAX_HZ/1e6:g} MHz): "
                      "switching is correct; this is the uncalibrated link-path impedance in the "
                      "shorted states, which biases Lsc/ls by about that much.")
        else:
            self.warn(f"RECIPROCITY FAILED ({worst*100:.1f}%): a relay is probably in the "
                      "wrong state, or the DUT is non-linear. Results below are suspect.")
        return passed

    #: Upper frequency for the reciprocity identity (lumped windings).
    RECIPROCITY_MAX_HZ = 10e6
    #: Below this, a reciprocity error is fixture link impedance, not switching.
    RECIPROCITY_FIXTURE_LIMIT = 0.20

    def verify_linearity(self, config_number=1, allow_use_cache=False):
        """[COG94] III: acquire twice at two drive levels to prove linearity."""
        print("\n-- Linearity check --")
        low = self._cache_path(config_number, "RL_drive_low")
        high = self._cache_path(config_number, "RL_drive_high")
        if allow_use_cache and low.exists() and high.exists():
            low_data, high_data = pandas.read_csv(low), pandas.read_csv(high)
        elif self.offline:
            print("  skipped (offline, no cached drive sweeps)")
            return None
        else:
            self._set_config(config_number)
            low_data = self.bode_100.take_Rs_Ls_measurement(
                *self.INDUCTANCE_BAND, number_of_measurement_cycles=1, source_power_dbm=-10)
            low_data.to_csv(low, index=False)
            high_data = self.bode_100.take_Rs_Ls_measurement(
                *self.INDUCTANCE_BAND, number_of_measurement_cycles=1, source_power_dbm=10)
            high_data.to_csv(high, index=False)

        passed, worst, _ = tm.check_linearity(low_data, high_data)
        self.results["linearity_deviation"] = worst
        if passed:
            print(f"  PASS -- inductance moves {worst*100:.2f}% over a 20 dB drive change")
        else:
            self.warn(f"NON-LINEAR ({worst*100:.1f}% inductance change over 20 dB). The "
                      "six-capacitance model assumes linearity; lower the drive.")
        return passed

    # -------------------------------------------------- magnetic parameters

    def characterize_inductance(self, allow_use_cache=False):
        """[COG94] II-B: k, eta, Lp and ls from three (here five) sweeps."""
        print("\n-- Inductance and coupling --")
        sweeps = {name: self.measure(number, "RL", allow_use_cache)
                  for name, number in (("Z0", 1), ("Zsc", 2), ("Z0p", 3),
                                       ("Lcum", 5), ("Ldif", 6))}

        # [COG94] reads inductance off "the first ascending part" of the Bode
        # plot. Above that, self-capacitance inflates L by 1/(1-(f/fr)^2), so
        # the reference frequency is chosen from the measured resonance rather
        # than fixed at 10 kHz.
        reference_frequency = 10_000.0
        first_resonance = None
        try:
            resonances = tm.detect_resonances(
                self.measure(1, "Z", allow_use_cache, band=self.RESONANCE_BAND))
            maxima = [r["frequency"] for r in resonances if r["type"] == "local maximum"]
            first_resonance = min(maxima) if maxima else None
            reference_frequency = tm.safe_reference_frequency(
                resonances, preferred=10_000.0,
                available=tm.average_cycles(sweeps["Z0"])["frequency"])
        except Exception as error:
            self.warn(f"Could not locate the first resonance ({error}); "
                      "using a fixed 10 kHz reference frequency.")

        lift = tm.inductance_lift(reference_frequency, first_resonance)
        print(f"  reference frequency {reference_frequency:.0f} Hz"
              + (f" (resonance {first_resonance/1e3:.1f} kHz, "
                 f"inductance over-read {lift*100:.3f}%)" if first_resonance else ""))
        if lift > 0.01:
            self.warn(f"At {reference_frequency:.0f} Hz self-capacitance inflates the "
                      f"inductances by {lift*100:.1f}%. Start the RL sweep lower.")

        values = {}
        for name, data in sweeps.items():
            _, values[name] = self.value_at(data, reference_frequency, "inductance")

        summary = tm.magnetic_summary(
            L0=values["Z0"], Lsc=values["Zsc"], L0_prime=values["Z0p"],
            L_cum=values["Lcum"], L_dif=values["Ldif"])
        summary["reference_frequency"] = reference_frequency

        if summary.get("dot_convention_swapped"):
            self.warn("Series-aiding and series-opposing came back swapped -- the "
                      "secondary is clamped reversed. Corrected automatically.")
        consistency = summary.get("k_consistency")
        if consistency is not None and consistency > 0.05:
            self.warn(f"k from Lsc and k from mutual inductance disagree by "
                      f"{consistency*100:.1f}%; coupling may be too loose for the "
                      "strong-coupling approximations of [COG94].")

        print(f"  k    = {summary['k']:.5f}")
        print(f"  eta  = {summary['eta']:.5f}   (N2/N1 = {summary['turns_ratio_N2_N1']:.5f})")
        print(f"  Lp   = {summary['Lp_magnetizing']*1e6:.3f} uH   [COG94] eq (5), L0(1+k)/2")
        print(f"  ls   = {summary['ls_leakage']*1e6:.3f} uH   [COG94] eq (4), Lsc/k")
        self.results["magnetic"] = summary
        return summary

    def characterize_resistances(self, allow_use_cache=False):
        """[COG94] II-C: r1, r2 from the LF plateaus, Rp at parallel resonance."""
        print("\n-- Losses --")
        eta = self.results.get("magnetic", {}).get("eta")
        if eta is None:
            eta = self.characterize_inductance(allow_use_cache)["eta"]

        Z0_rl = self.measure(1, "RL", allow_use_cache)
        Z0p_rl = self.measure(3, "RL", allow_use_cache)

        # [COG94] II-C: "The low-frequency plateau of Z0 equals r1, and that of
        # Z0' equals eta^2*r2."  So the plateau of Z0' IS the physical secondary
        # winding resistance -- what a DC ohmmeter on the secondary reads -- and
        # dividing by eta^2 gives r2 as it appears in the equivalent circuit,
        # referred to the primary.  Report both; they differ by eta^2 and
        # confusing them is a factor-of-eta^2 error.
        r1, f1, flat1 = tm.winding_resistance_from_plateau(Z0_rl, max_frequency=5000)
        r2_secondary, f2, flat2 = tm.winding_resistance_from_plateau(Z0p_rl, max_frequency=5000)
        r2_circuit = r2_secondary / eta ** 2

        for name, flatness in (("r1", flat1), ("r2", flat2)):
            if flatness > 0.1:
                self.warn(f"{name}: the sweep never flattened (spread {flatness*100:.0f}%); "
                          "start lower than 100 Hz or treat this as an upper bound.")

        Z0_z = self.measure(1, "Z", allow_use_cache, band=self.RESONANCE_BAND)
        resonances = tm.detect_resonances(Z0_z)
        maxima = [r for r in resonances if r["type"] == "local maximum"]
        Rp = None
        if maxima:
            Rp, at_frequency = tm.core_loss_resistance(Z0_z, maxima[0]["frequency"])
            print(f"  Rp   = {Rp:.1f} ohm      at {at_frequency/1e3:.1f} kHz  [COG94] II-C")
        else:
            self.warn("No parallel resonance found in Z0, so Rp (core loss) is unavailable. "
                      "Widen the sweep above the self-resonant frequency.")

        print(f"  r1   = {r1*1e3:.2f} mohm   primary winding (plateau to {f1:.0f} Hz)")
        print(f"  r2   = {r2_secondary*1e3:.2f} mohm   secondary winding, as measured")
        print(f"         {r2_circuit*1e3:.2f} mohm   the same, referred to the primary "
              f"(/{eta**2:.3f})")
        self.warn("r1/r2 include the fixture: one isolation-relay contact plus clamp "
                  "wiring sit outside the calibration plane (order 100 mohm).")

        losses = {"r1": r1, "r2_secondary": r2_secondary, "r2_primary_referred": r2_circuit,
                  "Rp": Rp, "r1_plateau_flatness": flat1, "r2_plateau_flatness": flat2}
        self.results["losses"] = losses
        return losses

    def characterize_capacitance(self, allow_use_cache=False):
        """Capacitances from differences that cancel the core or the leakage.

        The [BLA94] resonance route (characterize_capacitance_resonance) needs
        L at each resonance; on ferrite the first open-circuit resonance sits
        where mu is already dispersive and lossy, and the short-circuit
        resonances sit at or past the analyzer's 50 MHz, where the fixture's
        tens of nH matter as much as the leakage. What this bench measures
        robustly instead:

          * C33 directly, both directions, and each winding's capacitance to
            ground (configs 7, 16-19: windings shorted, no magnetics at all);
          * C13 + C23 from open-circuit link differences, where the core term
            is common and cancels at every frequency;
          * C13 and C23 separately from pairs of shorted states that share one
            leakage, which cancels at every frequency;
          * the self-capacitance seen across the windings as a SPECTRUM, with
            its maximum as a lower bound -- C11, C22 and C12 separately are not
            identifiable without a model of the ferrite (CapacitanceFit's
            global fit does that for a lumped DUT and says so for this one).

        See CapacitanceFit.differential_summary.
        """
        print("\n-- Capacitances (differential) --")
        for number in self.CAPACITANCE_CONFIGS:
            self.measure(number, "Zhd", allow_use_cache)
        summary = cfit.differential_summary(str(self.output_path), self.reference)
        estat = summary["electrostatic"]
        spectra = summary.pop("open_spectra")
        f = spectra["frequency"]
        _, Y_open = cfit.load_admittance(str(self.output_path), self.reference, cfit.OPEN_REFERENCE)
        c_open = cfit.effective_capacitance(f, Y_open).mean(axis=0) / cfit.PICO
        window = (f >= 1e6) & (f <= 15e6)
        summary["self_capacitance_lower_bound_pf"] = float(c_open[window].max())
        summary["self_capacitance_at_pf"] = {
            f"{x/1e6:g} MHz": float(c_open[numpy.argmin(numpy.abs(f - x))]) for x in (3e6, 5e6, 10e6, 20e6, 30e6)}

        print(f"  C33 (interwinding)       {summary['C33']['forward_pf']:.2f} pF "
              f"(reverse {summary['C33']['reverse_pf']:.2f})")
        print(f"  to ground                primary {estat['C_primary_ground_pf']:.2f}, "
              f"secondary {estat['C_secondary_ground_pf']:.2f} pF")
        print(f"  C13                      {summary['C13']['value']:.2f} +- {summary['C13']['spread']:.2f} pF")
        print(f"  C23                      {summary['C23']['value']:.2f} +- {summary['C23']['spread']:.2f} pF")
        print(f"  C13+C23  open / shorts   {summary['u_open']['value']:.2f} / {summary['u_short']['value']:.2f} pF")
        print(f"  self-C across windings   >= {summary['self_capacitance_lower_bound_pf']:.2f} pF "
              "(open-circuit, core term removed only as a bound)")

        fixture_check = summary["open_differences"]["AC"]["value"]
        if abs(fixture_check) > 0.5:
            self.warn(f"AC-BD open difference is {fixture_check:.2f} pF; it is zero for any lumped "
                      "DUT, so the fixture is loading floating terminals.")
        gap = abs(summary["u_open"]["value"] - summary["u_short"]["value"])
        if gap > 1.0:
            self.warn(f"C13+C23 disagrees by {gap:.1f} pF between the open and the shorted "
                      "families -- the six-capacitance lumped model only approximately holds "
                      "for this part; treat C13/C23 to about that level.")
        self.results["capacitance_differential"] = summary
        return summary

    def characterize_capacitance_resonance(self, allow_use_cache=False):
        """[BLA94] Table 1: the six capacitance coefficients from resonances."""
        print("\n-- Capacitances --")
        magnetic = self.results.get("magnetic") or self.characterize_inductance(allow_use_cache)
        L0, Lsc, eta = magnetic["L0"], magnetic["Lsc"], magnetic["eta"]

        # C33 measured directly: [BLA94] II-D, "matches the capacitance measured
        # between the two windings when they are short-circuited".
        C33_data = self.measure(7, "Cs", allow_use_cache, band=self.RESONANCE_BAND)
        _, C33 = self.value_at(C33_data, 100_000.0, "capacitance")
        print(f"  C33  = {C33*1e12:.2f} pF   (direct, both windings shorted)")

        measured_sums = {}
        for link, numbers in rbc.LINK_CONFIGS.items():
            resonances = {}
            for state, number in numbers.items():
                data = self.measure(number, "Z", allow_use_cache, band=self.RESONANCE_BAND)
                resonances[state] = tm.detect_resonances(data)
            sums = tm.capacitance_sums_from_resonances(resonances, L0, Lsc)
            found = sum(1 for v in sums.values() if v is not None)
            measured_sums[link] = sums
            print(f"  link {link:9s}: {found}/3 resonance equations")
            if found == 0:
                self.warn(f"link {link}: no resonances detected -- widen the sweep.")

        solution = tm.solve_six_capacitances(measured_sums, C33, eta)
        coefficients = solution["coefficients"]
        if not solution["converged"]:
            self.warn("The capacitance solver did not converge.")
        if solution["residual_relative"] > 0.1:
            self.warn(f"Capacitance fit residual {solution['residual_relative']*100:.1f}% -- "
                      "the model is not describing the data; check resonance identification.")

        print(f"  solved from {solution['equations_used']} equations, "
              f"residual {solution['residual_relative']*100:.2f}%")
        for name in ("C11", "C12", "C13", "C22", "C23", "C33"):
            print(f"    {name} = {coefficients[name]*1e12:9.3f} pF")

        self.results["capacitance"] = {
            "coefficients": coefficients,
            "practical": tm.practical_capacitances(coefficients, eta),
            "gamma_partial": tm.gamma_capacitances(coefficients, eta),
            "equations_used": solution["equations_used"],
            "residual_relative": solution["residual_relative"],
        }
        return self.results["capacitance"]

    # ------------------------------------------------- AC winding resistance

    def characterize_ac_resistance(self, gap_labels=None, allow_use_cache=False,
                                   reference_frequency=10_000.0):
        """Separate winding from core loss using two or more gaps.

        R_measured = Rw + K*L^2, with K gap-independent -- see
        TransformerModel.separate_winding_resistance for the derivation.

        This is the ONE recipe that cannot run unattended: the core has to be
        physically re-gapped between measurements. Never schedule it inside an
        overnight sweep.

        Use three or more gaps. Two give the exact closed form but no way to
        tell whether Rw really is gap-independent -- fringing flux near the gap
        drives proximity loss that changes with the gap. With three or more the
        fit residual reports that directly.
        """
        print("\n-- AC winding resistance (multi-gap, MANUAL re-gapping) --")
        gap_labels = gap_labels or ["gap1", "gap2", "gap3"]
        if len(gap_labels) < 2:
            raise ValueError("Need at least two gaps.")
        if len(gap_labels) == 2:
            self.warn("Only two gaps: the fit cannot test whether Rw is gap-independent. "
                      "Three or more is strongly preferred.")

        sweeps = []
        for label in gap_labels:
            path = self.output_path / f"{self.reference}_acr_{label}_RL.csv"
            if allow_use_cache and path.exists():
                sweeps.append(pandas.read_csv(path))
                continue
            if self.offline:
                raise RuntimeError(f"Offline and no cached sweep for gap {label!r}.")
            input(f"  Fit the core with gap '{label}', keep the winding in place, press Enter...")
            self._set_config(1)
            data = self.bode_100.take_Rs_Ls_measurement(
                start_frequency=1000.0, stop_frequency=200_000.0,
                number_of_measurement_cycles=2, source_power_dbm=self.DRIVE_HIGH_BAND_DBM)
            data.to_csv(path, index=False)
            sweeps.append(data)

        # Row-wise: solve Rw at EVERY frequency, not from one scalar inductance.
        # Rev A extracted a single L per gap and applied it across the whole
        # sweep, which defeats the point of a frequency-resolved Rw(f).
        averaged = [tm.average_cycles(s) for s in sweeps]
        merged = averaged[0][["frequency"]].copy()
        for index, frame in enumerate(averaged):
            merged = merged.merge(
                frame[["frequency", "inductance", "resistance"]].rename(
                    columns={"inductance": f"L{index}", "resistance": f"R{index}"}),
                on="frequency", how="inner")
        if merged.empty:
            raise ValueError("The gap sweeps share no common frequency points.")

        rows = []
        for _, row in merged.iterrows():
            inductances = [row[f"L{i}"] for i in range(len(sweeps))]
            resistances = [row[f"R{i}"] for i in range(len(sweeps))]
            try:
                fit = tm.separate_winding_resistance(inductances, resistances)
            except Exception:
                continue
            rows.append({"frequency": row["frequency"], "Rw": fit["Rw"], "K": fit["K"],
                         "residual_max": fit["residual_max"], "Rc": resistances[0] - fit["Rw"]})
        result = pandas.DataFrame(rows)
        if result.empty:
            raise RuntimeError("Could not fit Rw at any frequency.")

        _, Rw_ref = self.value_at(
            result.assign(measurement_index=0), reference_frequency, "Rw")
        worst_residual = float(result["residual_max"].max())
        ratio = float(min(merged[f"L{i}"].iloc[0] for i in range(len(sweeps)))
                      / max(merged[f"L{i}"].iloc[0] for i in range(len(sweeps))))

        print(f"  gaps: {len(gap_labels)}   inductance ratio {ratio:.2f}")
        print(f"  Rw({reference_frequency/1e3:.0f} kHz) = {Rw_ref*1e3:.2f} mohm")
        print(f"  worst fit residual: {worst_residual*100:.2f}%")
        if ratio > 0.5:
            self.warn(f"Gap inductance ratio {ratio:.2f} is too close to 1. Noise gain "
                      "grows as 1/(1-ratio^2); aim for 0.5 or lower.")
        if len(gap_labels) > 2 and worst_residual > 0.03:
            self.warn(f"Residual {worst_residual*100:.1f}% suggests Rw is NOT gap-independent "
                      "-- most likely fringing-flux proximity loss. Move the winding away "
                      "from the gap.")

        self.results["ac_resistance"] = {
            "Rw_at_reference": Rw_ref, "reference_frequency": reference_frequency,
            "n_gaps": len(gap_labels), "inductance_ratio": ratio,
            "worst_residual": worst_residual,
        }
        result.to_csv(self.output_path / f"{self.reference}_ac_resistance.csv", index=False)
        return result

    # ------------------------------------------------------------ top level

    def characterize_all(self, allow_use_cache=False, skip_ac_resistance=True):
        """Everything that can run unattended."""
        print("=" * 70)
        print(f"Characterizing {self.reference}")
        print("=" * 70)
        self.verify_switching(allow_use_cache)
        self.verify_linearity(allow_use_cache=allow_use_cache)
        self.characterize_inductance(allow_use_cache)
        self.characterize_resistances(allow_use_cache)
        self.characterize_capacitance(allow_use_cache)
        if not skip_ac_resistance:
            self.characterize_ac_resistance(allow_use_cache=allow_use_cache)
        self.report()
        return self.results

    def report(self):
        print("\n" + "=" * 70)
        print(f"RESULTS -- {self.reference}")
        print("=" * 70)
        magnetic = self.results.get("magnetic", {})
        losses = self.results.get("losses", {})
        capacitance = self.results.get("capacitance", {}).get("coefficients", {})
        differential = self.results.get("capacitance_differential", {})

        if magnetic:
            print(f"  Coupling k               {magnetic['k']:.5f}")
            print(f"  Turns ratio N2/N1        {magnetic['turns_ratio_N2_N1']:.4f}")
            print(f"  Magnetizing Lp           {magnetic['Lp_magnetizing']*1e6:.3f} uH")
            print(f"  Leakage ls               {magnetic['ls_leakage']*1e6:.3f} uH")
        if losses:
            print(f"  Primary resistance r1    {losses['r1']*1e3:.2f} mohm")
            print(f"  Secondary resistance r2  {losses['r2_secondary']*1e3:.2f} mohm "
                  f"({losses['r2_primary_referred']*1e3:.2f} referred to primary)")
            if losses.get("Rp"):
                print(f"  Core loss Rp             {losses['Rp']:.1f} ohm")
        if differential:
            print(f"  Interwinding C33         {differential['C33']['forward_pf']:.2f} pF")
            print(f"  C13 / C23                {differential['C13']['value']:.2f} / "
                  f"{differential['C23']['value']:.2f} pF")
            print(f"  Self-C across windings   >= {differential['self_capacitance_lower_bound_pf']:.2f} pF")
        if capacitance:
            print(f"  Interwinding C33         {capacitance['C33']*1e12:.2f} pF")
            print(f"  Primary self C11         {capacitance['C11']*1e12:.2f} pF")
            print(f"  Secondary self C22       {capacitance['C22']*1e12:.2f} pF")

        print(f"\n  Small-signal measurement at {self.DRIVE_LOW_BAND_DBM}"
              f"/{self.DRIVE_HIGH_BAND_DBM} dBm -- these are small-signal "
              "permeability values, not values at operating flux.")

        if self.warnings:
            print(f"\n  {len(self.warnings)} warning(s):")
            for message in self.warnings:
                print(f"    - {message}")
        else:
            print("\n  No warnings.")

        destination = self.output_path / f"{self.reference}_results.json"
        with open(destination, "w") as handle:
            json.dump({"reference": self.reference, "results": self.results,
                       "warnings": self.warnings}, handle, indent=2, default=float)
        print(f"\n  Written to {destination}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("reference", nargs="?", default="temp",
                        help="DUT reference; names the cache and output files")
    parser.add_argument("--cache", action="store_true",
                        help="reuse cached CSVs instead of measuring (works offline)")
    parser.add_argument("--port", default=None, help="relay board VISA resource, e.g. ASRL5::INSTR")
    parser.add_argument("--bode-ip", default=None, help="Bode 100 SCPI server address")
    parser.add_argument("--ac-resistance", action="store_true",
                        help="also run the multi-gap AC resistance recipe (needs manual re-gapping)")
    parser.add_argument("--gaps", nargs="*", default=None, help="gap labels for --ac-resistance")
    parser.add_argument("--no-auto-calibrate", action="store_true")
    parser.add_argument("--describe", default=None,
                        help="free-text DUT description (core, material, turns...), stored in "
                             "output/{reference}_dut.json and shown in the report")
    arguments = parser.parse_args()

    if arguments.describe:
        here = pathlib.Path(__file__).parent.resolve() / "output"
        here.mkdir(parents=True, exist_ok=True)
        (here / f"{arguments.reference}_dut.json").write_text(json.dumps(
            {"reference": arguments.reference, "description": arguments.describe,
             "recorded": datetime.datetime.now().isoformat(timespec="seconds")}, indent=2))

    characterizer = MagneticCharacterizer(
        reference=arguments.reference, relay_board_port=arguments.port,
        bode_ip=arguments.bode_ip, auto_calibrate=not arguments.no_auto_calibrate)
    characterizer.characterize_all(allow_use_cache=arguments.cache,
                                   skip_ac_resistance=not arguments.ac_resistance)
    if arguments.ac_resistance and arguments.gaps:
        characterizer.characterize_ac_resistance(gap_labels=arguments.gaps,
                                                 allow_use_cache=arguments.cache)


if __name__ == "__main__":
    main()
