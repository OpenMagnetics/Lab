"""
Clamp-plane open/short fixture compensation (Keysight / Voltech practice).

The automatic OSL sits at the isolation-relay plane: everything between it
and the DUT leads -- the arms, the clamps, and whatever the on-board SHORT
path differs from the DUT path by -- is not in it. Commercial practice
removes that with a one-time compensation at the DUT plane:

  Keysight Impedance Measurement Handbook 4.3: open/short compensation,
      Zdut = (Zm - Zs) / (1 - (Zm - Zs) * Yo),   Yo = 1 / (Zo - Zs)
  "Short measurement is performed by connecting the test fixture terminals
  directly together"; "the compensation must be performed with the same
  conditions" as the DUT measurement.
  Voltech AT series: a dummy header whose windings are replaced by shorting
  links, compensation stored per test.

Here: OPEN = nothing clamped, SHORT = one shorting bar (or a dummy header)
across all four clamps, both force and sense positions. Both are measured
through the normal per-path OSL for every signal path the configuration
table uses and stored per board identity in calibrations/fixture_compensation.json.
The shorting bar's own inductance can be declared (Keysight 4.7.5, "define
the short bar's inductance") and is added back to Zs.
"""

import datetime
import json
import pathlib

import numpy

FILE_NAME = "fixture_compensation.json"
#: Sweep grid of the compensation measurements (log), shared by every path.
BAND = (1e3, 5e7)
POINTS = 201


def complex_z(data):
    """Complex Z from a Z/phase or an Rs/Ls sweep (long-form, averaged)."""
    f = data["frequency"].to_numpy()
    if "magnitude" in data:
        return f, data["magnitude"].to_numpy() * numpy.exp(1j * numpy.radians(data["phase"].to_numpy()))
    return f, data["resistance"].to_numpy() + 1j * 2 * numpy.pi * f * data["inductance"].to_numpy()


def _interp(f_target, f, z):
    """Complex log-frequency interpolation (outside the band: nearest value)."""
    x, xt = numpy.log10(f), numpy.log10(numpy.clip(f_target, f.min(), f.max()))
    return numpy.interp(xt, x, z.real) + 1j * numpy.interp(xt, x, z.imag)


def compensate(f, zm, entry):
    """Keysight open/short compensation of a measured Z at frequencies f."""
    ff = numpy.asarray(entry["frequency"])
    zo = _interp(f, ff, numpy.asarray(entry["open_re"]) + 1j * numpy.asarray(entry["open_im"]))
    zs = _interp(f, ff, numpy.asarray(entry["short_re"]) + 1j * numpy.asarray(entry["short_im"]))
    zs = zs + 1j * 2 * numpy.pi * f * entry.get("short_bar_h", 0.0)
    yo = 1.0 / (zo - zs)
    return (zm - zs) / (1.0 - (zm - zs) * yo)


def apply(data, entry):
    """Return a copy of a Z/phase or Rs/Ls sweep with the compensation applied."""
    if entry is None:
        return data
    f, z = complex_z(data)
    z = compensate(f, z, entry)
    data = data.copy()
    if "magnitude" in data:
        data["magnitude"] = numpy.abs(z)
        data["phase"] = numpy.degrees(numpy.angle(z))
    else:
        data["resistance"] = z.real
        data["inductance"] = z.imag / (2 * numpy.pi * f)
    return data


class Store:
    """calibrations/fixture_compensation.json: {board, acquired, paths: {key: entry}}."""

    def __init__(self, calibrations_path, board_identity):
        self.path = pathlib.Path(calibrations_path) / FILE_NAME
        self.board = board_identity
        self.data = {"relay_board": board_identity, "paths": {}}
        if self.path.exists():
            stored = json.loads(self.path.read_text())
            if board_identity is None or stored.get("relay_board") == board_identity:
                self.data = stored

    def entry(self, key):
        return self.data.get("paths", {}).get(key)

    def put(self, key, state, f, z, short_bar_h=0.0):
        entry = self.data.setdefault("paths", {}).setdefault(key, {"frequency": list(map(float, f))})
        if not numpy.allclose(entry["frequency"], f):
            entry.clear()
            entry["frequency"] = list(map(float, f))
        entry[f"{state}_re"] = list(map(float, z.real))
        entry[f"{state}_im"] = list(map(float, z.imag))
        if state == "short":
            entry["short_bar_h"] = float(short_bar_h)
        entry[f"{state}_acquired"] = datetime.datetime.now().isoformat(timespec="seconds")

    def save(self):
        self.data["relay_board"] = self.board
        self.path.write_text(json.dumps(self.data))
        return self.path


def _self_test():
    """A known DUT behind a known fixture comes back exactly."""
    f = numpy.logspace(3, 7.5, 50)
    w = 2 * numpy.pi * f
    z_series = 0.1 + 1j * w * 30e-9          # arms + SHORT residual
    y_shunt = 1j * w * 2e-12                 # clamp strays
    def fixture(z_dut):
        return z_series + 1.0 / (1.0 / z_dut + y_shunt) if numpy.all(z_dut != 0) else z_series
    z_open = z_series + 1.0 / y_shunt
    entry = {"frequency": f, "open_re": z_open.real, "open_im": z_open.imag,
             "short_re": numpy.full_like(f, 0.1), "short_im": w * 30e-9}
    for z_dut in (1e-6 * 1j * w + 0.02, 1.0 + 0 * w, 1 / (1j * w * 10e-12)):
        recovered = compensate(f, fixture(z_dut), entry)
        assert numpy.allclose(recovered, z_dut, rtol=1e-9), "compensation does not invert the fixture"
    print("FixtureCompensation self-test: 3 DUTs recovered exactly through a series+shunt fixture")


if __name__ == "__main__":
    _self_test()
