"""
Global circuit fit of the six [BLA94] capacitances from every relay configuration.

Why not resonance frequencies
-----------------------------
[BLA94]/[COG94] read each capacitance sum off one resonance as 1/(L w^2),
with L taken from a low-frequency sweep. On a ferrite part that fails three
ways: the first open-circuit resonance sits where mu' is already rolling off
and mu'' is large (L at resonance is not L at 10 kHz); L itself depends on the
drive level the resonance sweep used; and the short-circuit resonances
(C11, C22 against the leakage) usually sit at or beyond the top of a 40-50 MHz
analyzer. One resonance per sweep also throws away the other 800 points.

What this does instead
----------------------
Every configuration is simulated as a nodal network and all of them are
fitted together, at every frequency, against the measured complex impedance:

  * DUT electrostatics: six inter-terminal branch capacitors (AB, CD, AC, AD,
    BC, BD) -- an exact, invertible re-parameterisation of [BLA94]'s
    C11..C33 -- plus one capacitor from each terminal to ground. The Bode LO
    side is ground, so the ground terms exist on the bench even though the
    ground-free model has none; leaving them out biases every link sweep.
  * Magnetics: a reciprocal two-port (admittance matrix Y11, Y12, Y22)
    left FREE at each frequency. Nothing is assumed about mu(f), core loss,
    proximity effect or leakage dispersion -- those are exactly what broke the
    1/(L w^2) formula. The magnetic two-port is shared by all configurations
    at a given frequency, which is what makes the capacitances separable:
    the configurations put different voltage patterns on the same capacitors
    while the magnetics stay the same.
  * Fixture (rev B relay board): the uncalibrated series impedance between
    the OSL plane and the DUT (clamp arm + isolation contact, plus a column
    bus for every terminal other than the first HI one), the capacitance of
    an isolated (floating) clamp to HI and to ground, and the LINK rail's
    capacitance to HI and to ground. Weak priors from DESIGN_NOTES.md keep
    these physical; the data decide.

The fit is a separable (variable-projection) least squares: for given global
parameters the per-frequency two-port is solved by Gauss-Newton, batched over
all frequencies, and the outer solver only sees the global parameters.

Identifiability is reported, not assumed: the outer Jacobian's singular
values and the parameter covariance come back with the result, and
`fit_by_cycle` repeats the fit on each measurement cycle independently.
"""

import glob
import os

import numpy
import pandas
from scipy.optimize import least_squares

import RelayBoardController as rbc

PICO = 1e-12
NANO = 1e-9

#: Branch capacitors of the DUT, ground-free part. [BLA94] mapping below.
DUT_BRANCHES = ("AB", "CD", "AC", "AD", "BC", "BD")
GROUND_BRANCHES = ("AG", "BG", "CG", "DG")

#: Global parameter vector layout: (name, scale, prior mean, prior sigma).
#: prior sigma None = no prior. Capacitances in pF, inductances in nH,
#: resistances in ohm. Priors only on fixture terms (DESIGN_NOTES.md budget).
PARAMETERS = (
    [(f"C_{b}", PICO, None, None) for b in DUT_BRANCHES]
    + [(f"C_{b}", PICO, 0.3, 1.0) for b in GROUND_BRANCHES]
    + [
        ("L_arm", NANO, 13.0, 15.0),     # clamp arm + iso contact, per terminal
        ("R_arm", 1.0, 0.05, 0.1),
        ("L_col", NANO, 35.0, 30.0),     # column bus + crossbar contact
        ("R_col", 1.0, 0.03, 0.1),
        ("C_fH", PICO, 0.5, 2.0),       # isolated clamp to HI rail
        ("C_fG", PICO, 0.5, 2.0),       # isolated clamp to ground
        ("C_LH", PICO, 1.0, 3.0),       # LINK rail to HI rail
        ("C_LG", PICO, 1.0, 3.0),       # LINK rail to ground
        # Magnetics, global part. The core (open-circuit impedance) stays
        # free at every frequency; the leakage and the turns ratio do not.
        ("eta2", 1.0, None, None),       # Z22/Z11, real part (eta^2)
        ("eta2_i", 1.0, 0.0, 0.01),      # ... imaginary part
        ("Lsc_hf", NANO, None, None),    # short-circuit inductance, HF limit
        ("Lsc_d", NANO, None, None),     # ... extra at low frequency (proximity)
        ("fc_sc", 1e6, None, None),      # ... relaxation frequency, MHz
        ("beta_sc", 1.0, 1.5, 1.0),      # ... relaxation exponent
        ("Rsc_0", 1.0, None, None),      # short-circuit resistance
        ("Rsc_s", 1.0, None, None),      # ... skin term, ohm per sqrt(MHz)
        ("Rsc_p", 1.0, None, None),      # ... proximity term, ohm per MHz
    ]
)

#: Why the leakage is parametric and the core is not: any capacitance whose
#: voltage pattern lives only across the winding PORTS (C11, C22, C12 -- the
#: quadratic form in V1, V2) is indistinguishable from a j*w*C term inside a
#: fully free magnetic two-port. A self-capacitance is only visible by
#: resonating against an inductance one is prepared to assume something about.
#: The resonance method assumes a constant L -- exactly what a ferrite breaks.
#: Here only the leakage gets a shape (smooth, monotone: proximity effect
#: lowers it, skin effect raises R), which is nearly air-cored physics; the
#: core is left completely free. C11 then comes from the primary-driven
#: shorts, C22 from the secondary-driven shorts and C11+C22-2*C12 from the
#: series-opposing (Ldif) state, none of which involves the core.
NAMES = [p[0] for p in PARAMETERS]

#: Physical bounds (display units). Inter-terminal branch capacitors may be
#: negative -- [BLA94]: only directly measurable capacitances must be
#: positive -- everything to ground, every fixture term and the leakage
#: model may not.
BOUNDS = dict(
    {f"C_{b}": (-100.0, 500.0) for b in DUT_BRANCHES},
    **{f"C_{b}": (0.0, 50.0) for b in GROUND_BRANCHES},
    L_arm=(0.0, 200.0), R_arm=(0.0, 5.0), L_col=(0.0, 300.0), R_col=(0.0, 5.0),
    C_fH=(0.0, 30.0), C_fG=(0.0, 30.0), C_LH=(0.0, 30.0), C_LG=(0.0, 30.0),
    eta2=(1e-3, 1e3), eta2_i=(-1.0, 1.0),
    Lsc_hf=(0.0, 1e9), Lsc_d=(0.0, 1e9), fc_sc=(0.05, 500.0), beta_sc=(0.3, 4.0),
    Rsc_0=(0.0, 1e3), Rsc_s=(0.0, 1e3), Rsc_p=(0.0, 1e3),
)
INDEX = {name: i for i, name in enumerate(NAMES)}


def blache_from_branches(c):
    """[BLA94] coefficients from branch capacitances (same units).

    Convention of TransformerModel.WINDING_LINKS: V1 = VA-VB, V2 = VC-VD,
    V3 = VD-VB (its table has "A-D linked -> V3 = V1" and "A-C linked ->
    V3 = V1-V2"). The branch energy sum C_xy (Vx-Vy)^2 equals V^T C V for
        C11 = AB+AC+AD   C22 = CD+AC+BC   C33 = AC+AD+BC+BD
        C12 = -AC        C13 = -(AC+AD)   C23 = AC+BC
    and the map is invertible, so nothing is lost by fitting branches.
    """
    return {
        "C11": c["AB"] + c["AC"] + c["AD"],
        "C22": c["CD"] + c["AC"] + c["BC"],
        "C33": c["AC"] + c["AD"] + c["BC"] + c["BD"],
        "C12": -c["AC"],
        "C13": -(c["AC"] + c["AD"]),
        "C23": c["AC"] + c["BC"],
    }


#: Linear map branch -> Blache, for propagating covariance.
BLACHE_ORDER = ("C11", "C12", "C13", "C22", "C23", "C33")
_BLACHE_MATRIX = numpy.array([
    [1, 0, 1, 1, 0, 0],    # C11 = AB+AC+AD
    [0, 0, -1, 0, 0, 0],   # C12 = -AC
    [0, 0, -1, -1, 0, 0],  # C13 = -(AC+AD)
    [0, 1, 1, 0, 1, 0],    # C22 = CD+AC+BC
    [0, 0, 1, 0, 1, 0],    # C23 = AC+BC
    [0, 0, 1, 1, 1, 1],    # C33 = AC+AD+BC+BD
], dtype=float)


# --------------------------------------------------------------------------
# Network model
# --------------------------------------------------------------------------

class Topology:
    """Node layout of one relay configuration.

    Nodes: HI, A, B, C, D, LINK (when used). LO is the reference (ground).
    """

    def __init__(self, config):
        self.config = config
        self.name = config["name"]
        self.uses_link = bool(config["LINK"])
        # Fixed layout for every configuration so all of them batch into one
        # solve; an unused LINK node is simply left with nothing attached.
        self.nodes = ["HI", "A", "B", "C", "D", "LINK"]
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.floating = rbc.floating_terminals(config)
        self.first_hi = config["HI"][0] if config["HI"] else None
        # (terminal, rail node or None for ground, is_first_hi)
        self.arms = []
        for rail in ("HI", "LO", "LINK"):
            for terminal in config[rail]:
                node = {"HI": "HI", "LO": None, "LINK": "LINK"}[rail]
                self.arms.append((terminal, node, rail == "HI" and terminal == self.first_hi))


def _stamp(Y, i, j, y):
    """Admittance y between node indexes i and j (None = ground). Y is (F,N,N)."""
    if i is not None:
        Y[:, i, i] += y
    if j is not None:
        Y[:, j, j] += y
    if i is not None and j is not None:
        Y[:, i, j] -= y
        Y[:, j, i] -= y


NODE_COUNT = 6


def _magnetic_patterns():
    """Stamp patterns of Y11, Y12, Y22 on the fixed node layout."""
    a, b, c, d = 1, 2, 3, 4
    P = numpy.zeros((3, NODE_COUNT, NODE_COUNT))
    for (r, s_, v) in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
        P[0, r, s_] += v
    for (r, s_, v) in ((c, c, 1), (d, d, 1), (c, d, -1), (d, c, -1)):
        P[2, r, s_] += v
    for (r, s_, v) in ((a, c, 1), (c, a, 1), (b, d, 1), (d, b, 1),
                       (a, d, -1), (d, a, -1), (b, c, -1), (c, b, -1)):
        P[1, r, s_] += v
    return P


MAGNETIC_PATTERNS = _magnetic_patterns()


def static_admittance(topology, omega, p):
    """Everything but the magnetics: (F, N, N) nodal admittance."""
    F = len(omega)
    N = NODE_COUNT
    ix = topology.index
    Y = numpy.zeros((F, N, N), dtype=complex)
    jw = 1j * omega

    # DUT capacitances
    for branch in DUT_BRANCHES:
        _stamp(Y, ix[branch[0]], ix[branch[1]], jw * p[f"C_{branch}"])
    for terminal in "ABCD":
        _stamp(Y, ix[terminal], None, jw * p[f"C_{terminal}G"])

    # Fixture: series arm (+column) from each assigned terminal to its rail
    z_arm = p["R_arm"] + jw * p["L_arm"]
    z_col = p["R_col"] + jw * p["L_col"]
    for terminal, rail, first in topology.arms:
        z = z_arm if first else z_arm + z_col
        _stamp(Y, ix[terminal], ix[rail] if rail else None, 1.0 / z)

    # Isolated clamps of floating terminals
    for terminal in topology.floating:
        _stamp(Y, ix[terminal], ix["HI"], jw * p["C_fH"])
        _stamp(Y, ix[terminal], None, jw * p["C_fG"])

    # LINK rail
    if topology.uses_link:
        _stamp(Y, ix["LINK"], ix["HI"], jw * p["C_LH"])
        _stamp(Y, ix["LINK"], None, jw * p["C_LG"])

    # A femtosiemens to ground on every node keeps a fully floating trial
    # state (and the unused LINK node) solvable; twelve orders below anything
    # measured here.
    Y[:, numpy.arange(N), numpy.arange(N)] += 1e-15
    return Y


def impedance_from_static(static, magnetic):
    """Input impedance HI->ground. static (..., F, N, N); magnetic (F, 3)
    with the two-port of port 1 = A-B, port 2 = C-D, dots on A and C."""
    Y = static + numpy.einsum("fm,mij->fij", magnetic, MAGNETIC_PATTERNS)
    excitation = numpy.zeros(Y.shape[:-1], dtype=complex)
    excitation[..., 0] = 1.0
    return numpy.linalg.solve(Y, excitation[..., None])[..., 0, 0]


def config_impedance(topology, omega, p, magnetic):
    """Input impedance HI->ground of one configuration, vectorised over f."""
    return impedance_from_static(static_admittance(topology, omega, p), magnetic)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

class Dataset:
    """Measured complex Z for a set of configurations on a common grid."""

    def __init__(self, frequency, impedance, sigma, configs):
        self.frequency = frequency              # (F,)
        self.impedance = impedance              # dict number -> (F,) complex
        self.sigma = sigma                      # dict number -> (F,) relative
        self.configs = configs                  # list of config numbers
        self.topologies = {n: Topology(rbc.CONFIGS[n]) for n in configs}

    def subset(self, indexes):
        """The same configurations at a subset of frequency indexes."""
        pick = numpy.asarray(indexes)
        return Dataset(self.frequency[pick],
                       {n: v[pick] for n, v in self.impedance.items()},
                       {n: v[pick] for n, v in self.sigma.items()}, self.configs)

    @classmethod
    def from_csv(cls, output_path, reference, kind="Zhd", configs=None, band=(2e5, 5e7),
                 points=140, cycle=None, relative_floor=5e-4, open_floor_pf=0.05,
                 short_floor_ohm=2e-3, short_floor_nh=0.5):
        """Load cached sweeps. cycle=None averages cycles; an int keeps one."""
        configs = configs or sorted(rbc.CONFIGS)
        frames = {}
        for number in configs:
            name = rbc.CONFIGS[number]["name"]
            path = os.path.join(output_path, f"{reference}_cfg{number:02d}_{name}_{kind}.csv")
            if os.path.exists(path):
                frames[number] = pandas.read_csv(path)
        if not frames:
            raise FileNotFoundError(f"No {kind} sweeps for {reference} in {output_path}")
        grid = numpy.array(sorted(next(iter(frames.values())).frequency.unique()))
        keep = (grid >= band[0]) & (grid <= band[1])
        grid = grid[keep]
        choose = numpy.unique(numpy.round(numpy.linspace(0, len(grid) - 1, points)).astype(int))
        grid = grid[choose]

        impedance, sigma = {}, {}
        for number, data in frames.items():
            data = data.sort_values(["measurement_index", "frequency"])
            cycles = data.measurement_index.nunique()
            z = (data.magnitude.to_numpy() * numpy.exp(1j * numpy.radians(data.phase.to_numpy())))
            z = z.reshape(cycles, -1)                                   # (cycles, all points)
            all_frequencies = data.frequency.to_numpy()[: z.shape[1]]
            pick = numpy.searchsorted(all_frequencies, grid)
            values = z[:, pick].T                                       # (F, cycles)
            if cycle is None:
                mean = values.mean(axis=1)
            else:
                mean = values[:, cycle]
            spread = numpy.abs(values - values.mean(axis=1, keepdims=True)).std(axis=1) \
                / numpy.sqrt(values.shape[1]) / numpy.abs(mean)
            # Calibration repeatability, not just cycle noise: an OPEN that
            # re-actuates to within ~0.05 pF and a SHORT to within a few mohm
            # (the post-OSL re-actuation checks) bound how well a near-open or
            # near-short state can be known, whatever its cycle scatter.
            omega = 2 * numpy.pi * grid
            open_term = omega * open_floor_pf * PICO * numpy.abs(mean)
            short_term = numpy.abs(short_floor_ohm + 1j * omega * short_floor_nh * NANO) / numpy.abs(mean)
            impedance[number] = mean
            sigma[number] = numpy.sqrt(spread ** 2 + relative_floor ** 2 + open_term ** 2 + short_term ** 2)
        return cls(grid, impedance, sigma, sorted(frames))


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------

def _parameters_si(x):
    return {name: value * scale for (name, scale, _, _), value in zip(PARAMETERS, x)}


def leakage_impedance(p, omega):
    """Primary short-circuit impedance (smooth physical model)."""
    f = omega / (2 * numpy.pi)
    L = p["Lsc_hf"] + p["Lsc_d"] / (1.0 + (f / p["fc_sc"]) ** p["beta_sc"])
    R = p["Rsc_0"] + p["Rsc_s"] * numpy.sqrt(f / 1e6) + p["Rsc_p"] * f / 1e6
    return R + 1j * omega * L


def magnetic_from_physical(q, p, omega):
    """Two-port admittance matrix. q (F,1) complex: log Zo (primary open),
    free per frequency. Zsc and r = Z22/Z11 come from the global parameters.
    These coordinates keep each family of sweeps attached to its own
    unknown -- Y11, Y12, Y22 agree to 0.1 % on a tightly coupled part."""
    Zo = numpy.exp(q[:, 0])
    Zsc = leakage_impedance(p, omega)
    r = p["eta2"] + 1j * p["eta2_i"]
    # Written so the square root stays near +1: sqrt(r*Zo*(Zo-Zsc)) crosses
    # the branch cut whenever a lossy Zo puts Zo^2 near the negative real
    # axis, silently flipping the mutual inductance at those frequencies.
    Z12 = Zo * numpy.sqrt(r * (1.0 - Zsc / Zo))
    det = r * Zo * Zsc
    return numpy.stack([1.0 / Zsc, -Z12 / det, 1.0 / (r * Zsc)], axis=1)


def _initial_physical(dataset, p):
    """Seed log Zo by continuation in frequency.

    At the bottom of the band the capacitances are negligible and the open
    sweep IS the core. Above the first resonance it is not (it turns
    capacitive), so each frequency is seeded from the solved one below it,
    scaled as an inductor, and solved before moving on.
    """
    omega = 2 * numpy.pi * dataset.frequency
    raw = numpy.log(dataset.impedance[1])[:, None]
    q = raw.copy()
    compiled = _Compiled(dataset, omega, p)
    q[0] = _solve_magnetic(None, omega[:1], p, raw[:1], 60, compiled.subset([0]))[0]
    for k in range(1, len(omega)):
        seed = q[k - 1].copy() + numpy.log(omega[k] / omega[k - 1])
        q[k] = _solve_magnetic(None, omega[k:k + 1], p, seed[None, :], 40, compiled.subset([k]))[0]
    return q


class _Compiled:
    """Static admittances of every configuration for one parameter set."""

    def __init__(self, dataset, omega, p):
        self.static = numpy.stack([static_admittance(dataset.topologies[n], omega, p)
                                   for n in dataset.configs])                  # (K,F,N,N)
        self.measured = numpy.stack([dataset.impedance[n] for n in dataset.configs])  # (K,F)
        self.sigma = numpy.stack([dataset.sigma[n] for n in dataset.configs])

    def subset(self, index):
        other = object.__new__(_Compiled)
        other.static = self.static[:, index]
        other.measured = self.measured[:, index]
        other.sigma = self.sigma[:, index]
        return other


def _residual_matrix(dataset, omega, p, magnetic, compiled=None):
    """(F, 2*K) real residuals, relative error over sigma."""
    compiled = compiled or _Compiled(dataset, omega, p)
    Y = compiled.static + numpy.einsum("fm,mij->fij", magnetic, MAGNETIC_PATTERNS)[None]
    excitation = numpy.zeros(Y.shape[:-1], dtype=complex)
    excitation[..., 0] = 1.0
    model = numpy.linalg.solve(Y, excitation[..., None])[..., 0, 0]               # (K,F)
    error = (model - compiled.measured) / compiled.measured / compiled.sigma
    return numpy.concatenate([error.real, error.imag], axis=0).T


def _solve_magnetic(dataset, omega, p, q, iterations=25, compiled=None):
    """Per-frequency Levenberg-Marquardt on log Zo, batched over frequency."""
    q = q.copy()
    F, NQ = q.shape
    compiled = compiled or _Compiled(dataset, omega, p)

    def model(qq):
        return _residual_matrix(dataset, omega, p, magnetic_from_physical(qq, p, omega), compiled)

    damping = numpy.full(F, 1e-3)
    base = model(q)
    cost = (base ** 2).sum(axis=1)
    h = 1e-7
    for _ in range(iterations):
        J = numpy.empty((F, base.shape[1], 2 * NQ))
        for k in range(2 * NQ):
            delta = numpy.zeros((F, NQ), dtype=complex)
            delta[:, k // 2] = h if k % 2 == 0 else 1j * h
            J[:, :, k] = (model(q + delta) - base) / h
        JT = numpy.transpose(J, (0, 2, 1))
        A = JT @ J
        g = (JT @ base[..., None])[..., 0]
        diag = numpy.einsum("fii->fi", A)
        improved_any = False
        for _attempt in range(6):
            M = A + damping[:, None, None] * numpy.einsum("fi,ij->fij", diag + 1e-30, numpy.eye(2 * NQ))
            step = -numpy.linalg.solve(M, g[..., None])[..., 0]
            step = numpy.clip(step, -0.5, 0.5)
            trial = q + step[:, 0::2] + 1j * step[:, 1::2]
            with numpy.errstate(all="ignore"):
                trial_base = model(trial)
            trial_cost = (trial_base ** 2).sum(axis=1)
            trial_cost[~numpy.isfinite(trial_cost)] = numpy.inf
            better = trial_cost < cost
            q[better] = trial[better]
            base[better] = trial_base[better]
            cost[better] = trial_cost[better]
            damping[better] = numpy.maximum(damping[better] / 5, 1e-9)
            damping[~better] *= 10
            improved_any = improved_any or better.any()
            if better.all():
                break
        if not improved_any or numpy.max(numpy.abs(step)) < 1e-10:
            break
    return q


def initial_parameters(dataset=None):
    """Prior means, capacitances at 5 pF, magnetics read off the sweeps."""
    x = numpy.array([mean if mean is not None else 5.0 for _, _, mean, _ in PARAMETERS], dtype=float)
    defaults = {"eta2": 1.0, "Lsc_hf": 500.0, "Lsc_d": 50.0, "fc_sc": 3.0, "Rsc_0": 0.3, "Rsc_s": 0.05,
                "Rsc_p": 0.01}
    if dataset is not None:
        z, f = dataset.impedance, dataset.frequency
        if 3 in z and 1 in z:
            defaults["eta2"] = float(numpy.real(z[3][0] / z[1][0]))
        short = z.get(9, z.get(2))
        if short is not None:
            i = int(numpy.argmin(numpy.abs(f - 1e6)))
            L = float(numpy.imag(short[i]) / (2 * numpy.pi * f[i]))
            defaults.update(Lsc_hf=0.9 * L / NANO, Lsc_d=0.1 * L / NANO,
                            Rsc_0=max(0.01, float(numpy.real(short[0]))))
    for name, value in defaults.items():
        x[INDEX[name]] = value
    if dataset is not None and short is not None:
        x = _seed_leakage(dataset, short, x)
    return x


def _seed_leakage(dataset, short, x):
    """Fit the leakage form to a shorted sweep below 5 MHz, where the
    self-capacitances are still negligible against the leakage reactance, so
    the global fit starts on the right side of every local minimum."""
    f = dataset.frequency
    keep = f <= 5e6
    if keep.sum() < 8:
        return x
    omega = 2 * numpy.pi * f[keep]
    names = ("Lsc_hf", "Lsc_d", "fc_sc", "beta_sc", "Rsc_0", "Rsc_s", "Rsc_p")
    start = numpy.array([x[INDEX[n]] for n in names])
    lower = numpy.array([BOUNDS[n][0] for n in names])
    upper = numpy.array([BOUNDS[n][1] for n in names])
    target = short[keep]

    def residuals(v):
        values = dict(zip(names, v))
        p = {n: values[n] * PARAMETERS[INDEX[n]][1] for n in names}
        error = (leakage_impedance(p, omega) - target) / target
        return numpy.concatenate([error.real, error.imag])

    solution = least_squares(residuals, numpy.clip(start, lower + 1e-9, upper - 1e-9),
                             bounds=(lower, upper), x_scale="jac")
    for name, value in zip(names, solution.x):
        x[INDEX[name]] = value
    return x


def fit(dataset, x0=None, fixed=None, verbose=0):
    """Fit all global parameters. Returns a result dict.

    fixed: optional {name: value_in_display_units} held constant.
    """
    omega = 2 * numpy.pi * dataset.frequency
    x0 = initial_parameters(dataset) if x0 is None else numpy.array(x0, dtype=float)
    fixed = fixed or {}
    free = [i for i, name in enumerate(NAMES) if name not in fixed]
    for name, value in fixed.items():
        x0[INDEX[name]] = value

    state = {"q": _initial_physical(dataset, _parameters_si(x0)), "z": None}
    prior_rows = [i for i in free if PARAMETERS[i][3] is not None]

    def expand(z):
        x = x0.copy()
        x[free] = z
        return x

    def profiled(z):
        """Residuals with the per-frequency magnetics solved out."""
        x = expand(z)
        p = _parameters_si(x)
        compiled = _Compiled(dataset, omega, p)
        state["q"] = _solve_magnetic(dataset, omega, p, state["q"], 30, compiled)
        state["z"] = numpy.array(z, copy=True)
        data = _residual_matrix(dataset, omega, p, magnetic_from_physical(state["q"], p, omega), compiled)
        prior = [(x[i] - PARAMETERS[i][2]) / PARAMETERS[i][3] for i in prior_rows]
        return numpy.concatenate([data.ravel(), prior])

    def jacobian(z):
        """Variable-projection (Kaufman) Jacobian.

        Derivatives with respect to the global parameters at FIXED magnetics,
        then, frequency by frequency, the component the two-port could absorb
        is projected out. At the inner optimum this is the Jacobian of the
        profiled residual, and it needs no inner re-solve per probe -- the
        finite-difference alternative re-solves 18 times and, worse, from a
        warm start each probe moves, which makes the derivatives inconsistent.
        """
        if state["z"] is None or not numpy.array_equal(state["z"], z):
            profiled(z)
        x = expand(z)
        p = _parameters_si(x)
        magnetic = magnetic_from_physical(state["q"], p, omega)
        compiled = _Compiled(dataset, omega, p)
        base = _residual_matrix(dataset, omega, p, magnetic, compiled)       # (F, R)
        F, R = base.shape

        Jx = numpy.empty((F, R, len(free)))
        for column, i in enumerate(free):
            step = 1e-3 if PARAMETERS[i][1] != 1.0 else 1e-4
            shifted = x.copy()
            shifted[i] += step
            ps = _parameters_si(shifted)
            Jx[:, :, column] = (_residual_matrix(dataset, omega, ps,
                                                 magnetic_from_physical(state["q"], ps, omega),
                                                 _Compiled(dataset, omega, ps)) - base) / step

        NQ = state["q"].shape[1]
        Jq = numpy.empty((F, R, 2 * NQ))
        h = 1e-7
        for k in range(2 * NQ):
            delta = numpy.zeros((F, NQ), dtype=complex)
            delta[:, k // 2] = h if k % 2 == 0 else 1j * h
            Jq[:, :, k] = (_residual_matrix(dataset, omega, p,
                                            magnetic_from_physical(state["q"] + delta, p, omega),
                                            compiled) - base) / h
        JqT = numpy.transpose(Jq, (0, 2, 1))
        gram = JqT @ Jq + 1e-12 * numpy.eye(2 * NQ) * numpy.trace(JqT @ Jq, axis1=1, axis2=2)[:, None, None]
        Jx = Jx - Jq @ numpy.linalg.solve(gram, JqT @ Jx)

        rows = [Jx.reshape(F * R, len(free))]
        if prior_rows:
            P = numpy.zeros((len(prior_rows), len(free)))
            for r, i in enumerate(prior_rows):
                P[r, free.index(i)] = 1.0 / PARAMETERS[i][3]
            rows.append(P)
        return numpy.vstack(rows)

    lower = numpy.array([BOUNDS[NAMES[i]][0] for i in free])
    upper = numpy.array([BOUNDS[NAMES[i]][1] for i in free])
    start = numpy.clip(x0[free], lower + 1e-9 * (upper - lower), upper - 1e-9 * (upper - lower))
    solution = least_squares(profiled, start, jac=jacobian, bounds=(lower, upper), method="trf",
                             x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12,
                             max_nfev=300, verbose=verbose)
    x = expand(solution.x)
    p = _parameters_si(x)
    state["q"] = _solve_magnetic(dataset, omega, p, state["q"], 80)
    state["magnetic"] = magnetic_from_physical(state["q"], p, omega)

    # Covariance from the profiled Jacobian, scaled by the reduced chi-square.
    J = solution.jac
    dof = max(1, len(solution.fun) - len(free))
    chi2 = float((solution.fun ** 2).sum() / dof)
    try:
        covariance_free = numpy.linalg.pinv(J.T @ J) * chi2
    except numpy.linalg.LinAlgError:
        covariance_free = numpy.full((len(free), len(free)), numpy.nan)
    covariance = numpy.zeros((len(NAMES), len(NAMES)))
    for a, i in enumerate(free):
        for b, j in enumerate(free):
            covariance[i, j] = covariance_free[a, b]
    singular = numpy.linalg.svd(J, compute_uv=False)

    branches_pf = numpy.array([x[INDEX[f"C_{b}"]] for b in DUT_BRANCHES])
    branch_cov = covariance[:6, :6]
    blache = _BLACHE_MATRIX @ branches_pf
    blache_cov = _BLACHE_MATRIX @ branch_cov @ _BLACHE_MATRIX.T

    per_config = {}
    for number in dataset.configs:
        model = config_impedance(dataset.topologies[number], omega, p, state["magnetic"])
        error = (model - dataset.impedance[number]) / dataset.impedance[number]
        per_config[number] = {"model": model, "rms_relative": float(numpy.sqrt(numpy.mean(numpy.abs(error) ** 2)))}

    return {
        "x": x,
        "parameters": {name: float(x[i]) for i, name in enumerate(NAMES)},
        "sigma": {name: float(numpy.sqrt(max(covariance[i, i], 0))) for i, name in enumerate(NAMES)},
        "blache_pf": dict(zip(BLACHE_ORDER, blache.tolist())),
        "blache_sigma_pf": dict(zip(BLACHE_ORDER, numpy.sqrt(numpy.clip(numpy.diag(blache_cov), 0, None)).tolist())),
        "blache_covariance": blache_cov,
        "chi2_reduced": chi2,
        "singular_values": singular,
        "magnetic": state["magnetic"],
        "frequency": dataset.frequency,
        "per_config": per_config,
        "success": bool(solution.success),
        "message": solution.message,
        "free": [NAMES[i] for i in free],
    }


def magnetic_summary(result):
    """Physical reading of the fitted two-port, per frequency."""
    omega = 2 * numpy.pi * result["frequency"]
    Y11, Y12, Y22 = result["magnetic"].T
    det = Y11 * Y22 - Y12 ** 2
    Z11, Z22, Z12 = Y22 / det, Y11 / det, -Y12 / det
    return pandas.DataFrame({
        "frequency": result["frequency"],
        "L_open_primary": (Z11 / (1j * omega)).real,
        "R_open_primary": Z11.real,
        "L_open_secondary": (Z22 / (1j * omega)).real,
        "L_short_primary": ((1.0 / Y11) / (1j * omega)).real,
        "R_short_primary": (1.0 / Y11).real,
        "L_short_secondary": ((1.0 / Y22) / (1j * omega)).real,
        "k": (Z12 / numpy.sqrt(Z11 * Z22)).real,
        "eta": numpy.sqrt(Z22 / Z11).real,
    })


# --------------------------------------------------------------------------
# Differential estimators -- what this bench can pin down without a model of
# the core or of distributed winding behaviour
# --------------------------------------------------------------------------
#
# The global fit above is exact for a lumped DUT and is validated on synthetic
# data (test_capacitance_fit.py). A real winding on MnZn ferrite is not fully
# lumped: windings sit directly on a high-permittivity, dispersive dielectric
# (MnZn ferrite), so turn-to-core capacitance changes with frequency, and the
# fixture path adds tens of nH above ~10 MHz. The estimators below
# use only differences in which the core (open family) or the leakage (short
# pairs) cancels at every frequency, plus electrostatic-only states, so they
# hold regardless.

def load_admittance(output_path, reference, number, kind="Zhd"):
    """(frequency, Y per cycle) for one cached sweep."""
    name = rbc.CONFIGS[number]["name"]
    path = os.path.join(output_path, f"{reference}_cfg{number:02d}_{name}_{kind}.csv")
    data = pandas.read_csv(path).sort_values(["measurement_index", "frequency"])
    cycles = data.measurement_index.nunique()
    z = (data.magnitude.to_numpy() * numpy.exp(1j * numpy.radians(data.phase.to_numpy()))).reshape(cycles, -1)
    return data.frequency.to_numpy()[: z.shape[1]], 1.0 / z


def effective_capacitance(frequency, admittance):
    """Im(Y)/omega -- the capacitance a sweep presents, per cycle."""
    return admittance.imag / (2 * numpy.pi * frequency)


def electrostatic_states(output_path, reference, band=(1e5, 5e6), kind="Zhd"):
    """Configs 7, 16-19 (windings shorted on themselves, no magnetics).

    Solves interwinding capacitance and each winding's capacitance to ground
    (the Bode LO side) from five states; the over-determination is the check.
    """
    measured = {}
    for number in (7, 16, 17, 18, 19):
        f, Y = load_admittance(output_path, reference, number, kind)
        keep = (f >= band[0]) & (f <= band[1])
        c = effective_capacitance(f, Y)[:, keep].mean(axis=0) / PICO
        measured[number] = (float(numpy.median(c)), float(numpy.std(c)))

    def model(v):
        Ciw, gP, gS = v
        return {7: Ciw + gP, 16: Ciw + gS, 17: gP + gS,
                18: gP + Ciw * gS / (Ciw + gS), 19: gS + Ciw * gP / (Ciw + gP)}

    solution = least_squares(lambda v: [model(v)[k] - measured[k][0] for k in measured],
                             [15.0, 0.5, 0.5], bounds=([0, 0, 0], [500, 50, 50]))
    fitted = model(solution.x)
    return {
        "measured_pf": measured,
        "C_interwinding_pf": float(solution.x[0]),
        "C_primary_ground_pf": float(solution.x[1]),
        "C_secondary_ground_pf": float(solution.x[2]),
        "residual_pf": {k: measured[k][0] - fitted[k] for k in measured},
    }


#: Open-family states: primary driven, secondary open, so the core term is
#: common to all and cancels in differences at every frequency.
OPEN_FAMILY = {"AC": 10, "BC": 12, "AD": 13, "floating": 1}
OPEN_REFERENCE = 8


def open_differences(output_path, reference, band=(1e6, 8e6), kind="Zhd"):
    """C(link) - C(B-D) for each open-circuit link, core-free.

    [BLA94] Table 1 with eta ~ 1 predicts AC-BD = 0, BC-BD = C33 - 2u,
    AD-BD = C33 + 2u, floating-BD = -u^2/C33 (u = C13 + C23 in the
    table's sign convention), so these four numbers both measure u and test
    the fixture (AC-BD is zero for any lumped DUT: same node voltages).
    """
    f, Yref = load_admittance(output_path, reference, OPEN_REFERENCE, kind)
    keep = (f >= band[0]) & (f <= band[1])
    out = {"frequency": f}
    for label, number in OPEN_FAMILY.items():
        _, Y = load_admittance(output_path, reference, number, kind)
        d = effective_capacitance(f, Y - Yref) / PICO              # per cycle
        mean = d.mean(axis=0)
        out[label] = {"spectrum": mean, "value": float(numpy.median(mean[keep])),
                      "band_spread": float(numpy.std(mean[keep])),
                      "cycle_spread": float(numpy.mean(numpy.std(d[:, keep], axis=0)))}
    return out


#: Short pairs sharing one leakage: (reference, linked) and what the
#: difference is in [BLA94] Table 1 terms (C1_C3 / C2_C3 columns).
SHORT_PAIRS = {
    "primary": (9, 11, "C33 + 2*C13"),     # B-D vs A-C linked, secondary shorted
    "secondary": (14, 15, "C33 - 2*C23"),  # same, primary shorted, measured from C-D
}


def short_pair_difference(output_path, reference, pair, band=(1e6, 8e6), base_pf=15.0, kind="Zhd"):
    """Capacitance difference of two shorted states with the leakage cancelled.

    Both states put the same leakage impedance Z_L(f) between the driven
    terminals; they differ by a capacitance and by the fixture path that
    shorts the other winding (different rail, different column). So
        1/(Y_b - jw C_b) - (dR + jw dL) = 1/(Y_a - jw C_a)
    at every frequency. dC = C_b - C_a is well determined; the shared level
    C_a is not (it trades against Z_L), so it is scanned by the caller.
    """
    a, b, _ = SHORT_PAIRS[pair]
    f, Ya = load_admittance(output_path, reference, a, kind)
    _, Yb = load_admittance(output_path, reference, b, kind)
    keep = (f >= band[0]) & (f <= band[1])
    w = 2 * numpy.pi * f[keep]
    ya, yb = Ya.mean(axis=0)[keep], Yb.mean(axis=0)[keep]

    def residuals(v):
        dC, dL, dR = v
        za = 1.0 / (ya - 1j * w * base_pf * PICO)
        zb = 1.0 / (yb - 1j * w * (base_pf + dC) * PICO) - (dR + 1j * w * dL * NANO)
        e = (zb - za) / za
        return numpy.concatenate([e.real, e.imag])

    s = least_squares(residuals, [20.0, 0.0, 0.0])
    return {"dC_pf": float(s.x[0]), "dL_nh": float(s.x[1]), "dR_ohm": float(s.x[2]),
            "rms_relative": float(numpy.sqrt(numpy.mean(residuals(s.x) ** 2)))}


def differential_summary(output_path, reference, kind="Zhd"):
    """Everything the differential estimators give, with a systematic spread
    taken over analysis bands and the unidentifiable shared short level."""
    states = electrostatic_states(output_path, reference, kind=kind)
    C33 = states["measured_pf"][7][0]
    C33_rev = states["measured_pf"][16][0]

    bands = ((5e5, 6e6), (1e6, 8e6), (1e6, 12e6), (2e6, 12e6))
    opens = [open_differences(output_path, reference, band, kind) for band in bands]
    open_values = {k: numpy.array([o[k]["value"] for o in opens]) for k in OPEN_FAMILY}
    u_open = (open_values["AD"] - open_values["BC"]) / 4.0

    C13s, C23s, dLs, rms = [], [], {"primary": [], "secondary": []}, {"primary": [], "secondary": []}
    for band in bands:
        for base in (5.0, 15.0, 25.0):
            p = short_pair_difference(output_path, reference, "primary", band, base, kind)
            s = short_pair_difference(output_path, reference, "secondary", band, base, kind)
            # Table sign convention: A-C link adds C33 + 2*C13 on the primary
            # side and C33 - 2*C23 on the secondary side.
            C13s.append((p["dC_pf"] - C33) / 2.0)
            C23s.append((C33 - s["dC_pf"]) / 2.0)
            dLs["primary"].append(p["dL_nh"])
            dLs["secondary"].append(s["dL_nh"])
            rms["primary"].append(p["rms_relative"])
            rms["secondary"].append(s["rms_relative"])
    C13s, C23s = numpy.array(C13s), numpy.array(C23s)

    def stat(values):
        values = numpy.asarray(values)
        return {"value": float(numpy.mean(values)), "spread": float(numpy.std(values)),
                "min": float(numpy.min(values)), "max": float(numpy.max(values))}

    return {
        "electrostatic": states,
        "C33": {"forward_pf": C33, "reverse_pf": C33_rev},
        "open_differences": {k: stat(v) for k, v in open_values.items()},
        "open_cycle_noise_pf": float(numpy.mean([opens[1][k]["cycle_spread"] for k in OPEN_FAMILY])),
        "u_open": stat(u_open),
        "C13": stat(C13s),
        "C23": stat(C23s),
        "u_short": stat(C13s + C23s),
        "fixture_dL_nh": {k: stat(v) for k, v in dLs.items()},
        "short_pair_rms": {k: stat(v) for k, v in rms.items()},
        "open_spectra": opens[1],
    }
