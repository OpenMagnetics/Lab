"""
Bench study of Bode 100 settings vs noise and accuracy (reproducible).

    python bench_accuracy.py [reference] [--bode-ip 127.0.0.1] [--port ASRL7::INSTR]

For a few representative relay configurations (open Z0, a shorted state,
pure C33) it sweeps the same band at several receiver bandwidths with N
repeat cycles and reports the cycle-to-cycle spread of the complex
impedance (the noise the analysis sees) and the time per sweep. It also
checks the drive independence of the calibration on the on-board 100 ohm
LOAD standard, and whether the SCPI server accepts internal calibration and
user-range correction. Writes output/{reference}_bench_accuracy.json.
"""

import argparse
import json
import time

import numpy
import pandas

import MagneticCharacterizer as M

CONFIGS = {1: "open Z0", 9: "shorted (B-D link)", 7: "C33 (windings shorted)"}
BANDWIDTHS_HZ = (1000, 300, 100, 30)


def complex_cycles(data):
    data = data.sort_values(["measurement_index", "frequency"])
    cycles = data.measurement_index.nunique()
    z = (data.magnitude.to_numpy() * numpy.exp(1j * numpy.radians(data.phase.to_numpy()))).reshape(cycles, -1)
    return data.frequency.to_numpy()[: z.shape[1]], z


def spread(z):
    """Median relative cycle-to-cycle scatter of the complex mean, |std|/sqrt(N)/|mean|."""
    mean = z.mean(axis=0)
    sigma = numpy.abs(z - mean).std(axis=0) / numpy.sqrt(z.shape[0])
    return float(numpy.median(sigma / numpy.abs(mean)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("reference", nargs="?", default="BENCH")
    parser.add_argument("--bode-ip", default="127.0.0.1")
    parser.add_argument("--port", default=None)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--points", type=int, default=201)
    arguments = parser.parse_args()

    c = M.MagneticCharacterizer(arguments.reference, relay_board_port=arguments.port, bode_ip=arguments.bode_ip)
    bode = c.bode_100
    session = bode.visa_session
    report = {"bandwidth": {}, "load_vs_drive": {}, "scpi": {}}

    original_bandwidth = bode.receiver_bandwidth
    for number, label in CONFIGS.items():
        c._set_config(number)
        report["bandwidth"][label] = {}
        for bandwidth in BANDWIDTHS_HZ:
            bode.receiver_bandwidth = f"{bandwidth}Hz"
            start = time.time()
            data = bode.take_Z_phase_measurement(start_frequency=10e3, stop_frequency=50e6,
                                                 number_of_measurement_cycles=arguments.cycles,
                                                 number_of_measurement_points=arguments.points,
                                                 source_power_dbm=13)
            seconds = (time.time() - start) / arguments.cycles
            f, z = complex_cycles(data)
            bands = {"10k-1M": (f < 1e6), "1M-10M": (f >= 1e6) & (f < 1e7), "10M-50M": f >= 1e7}
            entry = {"seconds_per_sweep": seconds, "spread_total": spread(z)}
            entry.update({name: spread(z[:, keep]) for name, keep in bands.items()})
            report["bandwidth"][label][bandwidth] = entry
            print(f"  {label:24s} RBW {bandwidth:5d} Hz: {seconds:6.1f} s/sweep, relative noise "
                  + "  ".join(f"{k} {v*1e6:7.1f} ppm" for k, v in entry.items() if k.startswith(("1", "s")) is False or k in bands)
                  )
    bode.receiver_bandwidth = original_bandwidth

    # Calibration vs drive level on the 100 ohm 0.1 % LOAD standard.
    c._set_config(1)
    c.relay_board.set_calibration_mode("LOAD")
    for dbm in (-20, -10, 0, 10, 13):
        data = bode.take_Z_phase_measurement(start_frequency=1e3, stop_frequency=1e6, number_of_measurement_cycles=2,
                                             number_of_measurement_points=31, source_power_dbm=dbm)
        mean = data.groupby("frequency").magnitude.mean()
        report["load_vs_drive"][dbm] = {"max_error": float(numpy.max(numpy.abs(mean / 100.0 - 1))),
                                        "median_error": float(numpy.median(numpy.abs(mean / 100.0 - 1)))}
        print(f"  LOAD at {dbm:+3d} dBm: median error {report['load_vs_drive'][dbm]['median_error']*100:.3f} %, "
              f"max {report['load_vs_drive'][dbm]['max_error']*100:.3f} %")
    c.relay_board.set_calibration_mode("MEAS")

    # Which accuracy-related SCPI commands this server accepts.
    for command in (":SENS:CORR:USER:ACT?", ":SYST:ICAL:EXEC?", ":SENS:SWE:DWEL?", ":SENS:AVER:COUN?"):
        try:
            session.timeout = 5000
            report["scpi"][command] = session.query(command).strip()
        except Exception as error:
            report["scpi"][command] = f"no reply ({type(error).__name__})"
        report["scpi"][command + " err"] = session.query(":SYST:ERR?").strip()
        print(f"  {command}: {report['scpi'][command]}  [{report['scpi'][command + ' err']}]")
    session.timeout = bode.SCPI_timeout

    c.relay_board.reset()
    destination = c.output_path / f"{arguments.reference}_bench_accuracy.json"
    destination.write_text(json.dumps(report, indent=2))
    print(f"Written to {destination}")


if __name__ == "__main__":
    main()
