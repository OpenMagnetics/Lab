"""
Synthetic check of CapacitanceFit: build all relay configurations from a
known DUT + fixture with a deliberately unfriendly ferrite (relaxing, lossy
mu(f), drive-free) and frequency-dependent leakage, add noise, fit, and
compare. Run: python test_capacitance_fit.py
"""

import sys

import numpy

import CapacitanceFit as cf

FAILURES = []


def check(name, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        FAILURES.append(name)


def synthetic_magnetic(frequency, L1=260e-6, eta=1.0008, k=0.9985):
    """Two-port of a ferrite transformer with relaxing complex permeability
    and a leakage inductance that falls with frequency (proximity)."""
    omega = 2 * numpy.pi * frequency
    mu = 1.0 / (1.0 + 1j * frequency / 2.5e6) ** 0.8          # relaxation + loss
    L1f = L1 * mu
    L2f = eta ** 2 * L1f
    leakage = 2 * (1 - k) * L1 * (1.0 + 0.08 / (1 + (frequency / 3e6) ** 2))
    Lm = L1f - leakage / 2
    Lself1 = Lm + leakage / 2
    Lself2 = eta ** 2 * (Lm + leakage / 2)
    M = eta * Lm
    r1, r2 = 0.17 * (1 + numpy.sqrt(frequency / 2e5)), 0.19 * (1 + numpy.sqrt(frequency / 2e5))
    Z11 = r1 + 1j * omega * Lself1
    Z22 = r2 + 1j * omega * Lself2
    Z12 = 1j * omega * M
    det = Z11 * Z22 - Z12 ** 2
    return numpy.stack([Z22 / det, -Z12 / det, Z11 / det], axis=1)


TRUTH = {
    "C_AB": 6.0, "C_CD": 7.5, "C_AC": 9.0, "C_AD": 0.8, "C_BC": 1.2, "C_BD": 7.6,
    "C_AG": 0.15, "C_BG": 0.05, "C_CG": 0.2, "C_DG": 0.1,
    "L_arm": 11.0, "R_arm": 0.06, "L_col": 28.0, "R_col": 0.04,
    "C_fH": 1.0, "C_fG": 0.9, "C_LH": 0.6, "C_LG": 0.5,
}


def make_dataset(seed=1, noise=3e-4, configs=None):
    configs = configs or list(range(1, 20))
    frequency = numpy.geomspace(2e5, 5e7, 120)
    omega = 2 * numpy.pi * frequency
    # Magnetic globals are not used to generate: the synthetic two-port below
    # is deliberately NOT of the fitted form, to test robustness to that.
    x = numpy.array([TRUTH.get(name, 1.0) for name in cf.NAMES])
    p = cf._parameters_si(x)
    magnetic = synthetic_magnetic(frequency)
    rng = numpy.random.default_rng(seed)
    impedance, sigma = {}, {}
    for number in configs:
        topology = cf.Topology(cf.rbc.CONFIGS[number])
        z = cf.config_impedance(topology, omega, p, magnetic)
        z = z * (1 + noise * (rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape)))
        impedance[number] = z
        sigma[number] = numpy.full(z.shape, max(noise, 1e-4))
    return cf.Dataset(frequency, impedance, sigma, configs), magnetic


def test_recovers_blache():
    print("\nGlobal fit recovers the six capacitances (19 configs, 0.03 % noise)")
    dataset, _ = make_dataset()
    result = cf.fit(dataset)
    truth_branches = {b: TRUTH[f"C_{b}"] for b in cf.DUT_BRANCHES}
    truth = cf.blache_from_branches(truth_branches)
    for name in cf.BLACHE_ORDER:
        got, sigma = result["blache_pf"][name], result["blache_sigma_pf"][name]
        # C12 rests on the single series-opposing (Ldif) state and on the
        # leakage model; the others are pinned by several configurations.
        limit = 2.5 if name == "C12" else 0.15
        check(f"{name}", abs(got - truth[name]) < limit,
              f"fit {got:7.3f}  truth {truth[name]:7.3f} pF  (limit {limit})")
    # The synthetic two-port is deliberately not of the fitted form, so chi2
    # is not 1; it must still be small against the 1e5+ of a wrong model.
    check("model describes the data", result["chi2_reduced"] < 50, f"{result['chi2_reduced']:.2f}")


def test_original_fifteen_only():
    print("\nOnly the original 15 configurations (no ground-separating states)")
    dataset, _ = make_dataset(configs=list(range(1, 16)))
    result = cf.fit(dataset)
    truth = cf.blache_from_branches({b: TRUTH[f"C_{b}"] for b in cf.DUT_BRANCHES})
    worst = max(abs(result["blache_pf"][n] - truth[n]) for n in cf.BLACHE_ORDER if n != "C12")
    print(f"    worst Blache error (excluding C12) {worst:.3f} pF; C12 off by "
          f"{result['blache_pf']['C12'] - truth['C12']:+.2f} pF")
    check("still within 0.5 pF", worst < 0.5)


if __name__ == "__main__":
    test_recovers_blache()
    test_original_fifteen_only()
    print("\n" + "=" * 68)
    print("all passed" if not FAILURES else f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    print("=" * 68)
    sys.exit(1 if FAILURES else 0)
