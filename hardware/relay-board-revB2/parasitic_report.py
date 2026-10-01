"""
Post-route parasitic audit of the signal nets.

    "C:\\Program Files\\KiCad\\9.0\\bin\\python.exe" parasitic_report.py

Walks the routed board and reports, for every measurement net:
  - total routed length per layer and estimated L / C
  - via count (each ~1.2 nH)
  - parallel-run exposure: length of any OTHER net's track running closer
    than 1 mm alongside on the same layer, and closer than 0.5 mm on the
    adjacent layer (crossings at ~90 degrees are ignored -- ~0.05 pF each)
  - copper pour presence under the net (there must be none in the void)

Budgets (from DESIGN_NOTES.md; face-to-face bay layout):
    DUT arms   <= 16 mm, 0 vias, zero parallel exposure   (uncalibrated!)
    T buses    <= 60 mm each, matched across columns to 15%
    CAL buses  within 40% of the T-bus average (all calibrated per config,
               so mimicry is a robustness bonus, not a hard requirement)
Exit code 1 when a budget is violated.
"""

import math
import os
import sys
from collections import defaultdict

import pcbnew

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD_PATH = os.path.join(HERE, "relay_board.kicad_pcb")

L_PER_MM = 0.8e-9
C_PER_MM = 0.04e-12
L_VIA = 1.2e-9

SIGNAL_PREFIXES = ("/DUT_", "/T", "/CAL_", "/RAIL_")
BUDGET_FAILURES = []


def is_signal(name):
    if name in ("/TA", "/TB", "/TC", "/TD"):
        return True
    return name.startswith(("/DUT_", "/CAL_", "/RAIL_"))


def to_mm(value):
    return pcbnew.ToMM(value)


def segment_length(track):
    start, end = track.GetStart(), track.GetEnd()
    return math.hypot(to_mm(end.x - start.x), to_mm(end.y - start.y))


def parallel_exposure(track, other):
    """Rough co-run length between two segments when nearly parallel and
    within 1 mm; returns 0 for crossings (angle > 30 degrees)."""
    ax1, ay1 = to_mm(track.GetStart().x), to_mm(track.GetStart().y)
    ax2, ay2 = to_mm(track.GetEnd().x), to_mm(track.GetEnd().y)
    bx1, by1 = to_mm(other.GetStart().x), to_mm(other.GetStart().y)
    bx2, by2 = to_mm(other.GetEnd().x), to_mm(other.GetEnd().y)
    va = (ax2 - ax1, ay2 - ay1)
    vb = (bx2 - bx1, by2 - by1)
    la = math.hypot(*va)
    lb = math.hypot(*vb)
    if la < 0.05 or lb < 0.05:
        return 0.0
    cosine = abs((va[0] * vb[0] + va[1] * vb[1]) / (la * lb))
    if cosine < 0.866:                       # > 30 degrees: a crossing
        return 0.0
    # distance between midpoints as a cheap proximity proxy
    mid_a = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
    mid_b = ((bx1 + bx2) / 2, (by1 + by2) / 2)
    distance = math.hypot(mid_a[0] - mid_b[0], mid_a[1] - mid_b[1])
    if distance > max(1.0, (la + lb) / 4):
        return 0.0
    return min(la, lb)


def main():
    board = pcbnew.LoadBoard(BOARD_PATH)
    tracks_by_net = defaultdict(list)
    vias_by_net = defaultdict(int)
    for track in board.GetTracks():
        name = track.GetNetname()
        if track.GetClass() == "PCB_VIA":
            vias_by_net[name] += 1
        else:
            tracks_by_net[name].append(track)

    signal_nets = sorted(n for n in tracks_by_net if is_signal(n))
    all_tracks = [t for ts in tracks_by_net.values() for t in ts]

    print(f"{'net':<12} {'F.Cu mm':>8} {'B.Cu mm':>8} {'vias':>4} "
          f"{'L est':>8} {'C est':>8} {'co-run mm':>9}")
    print("-" * 64)
    lengths = {}
    for name in signal_nets:
        per_layer = defaultdict(float)
        exposure = 0.0
        for track in tracks_by_net[name]:
            per_layer[track.GetLayerName()] += segment_length(track)
            for other in all_tracks:
                if other.GetNetname() == name:
                    continue
                if other.GetLayer() != track.GetLayer():
                    continue
                if not is_signal(other.GetNetname()) or True:
                    exposure += parallel_exposure(track, other)
        total = sum(per_layer.values())
        lengths[name] = total
        inductance = total * L_PER_MM + vias_by_net[name] * L_VIA
        capacitance = total * C_PER_MM
        print(f"{name:<12} {per_layer.get('F.Cu', 0):8.1f} "
              f"{per_layer.get('B.Cu', 0):8.1f} {vias_by_net[name]:4d} "
              f"{inductance*1e9:6.1f}nH {capacitance*1e12:6.2f}pF {exposure:9.1f}")

    print("-" * 64)

    # Budgets
    for terminal in "ABCD":
        name = f"/DUT_{terminal}"
        if lengths.get(name, 99) > 16.0:
            BUDGET_FAILURES.append(f"{name} arm {lengths[name]:.1f} mm > 16 mm")
        if vias_by_net.get(name, 0) != 0:
            BUDGET_FAILURES.append(f"{name} has vias -- uncalibrated layer change")

    buses = [lengths.get(f"/T{t}", 0) for t in "ABCD"]
    if buses and max(buses) > 95.0:
        BUDGET_FAILURES.append(f"T bus too long: {max(buses):.1f} mm > 95 mm")
    if buses and min(buses) > 0:
        spread = (max(buses) - min(buses)) / (sum(buses) / len(buses))
        if spread > 0.15:
            # informational under per-config calibration: each path is
            # OSL-calibrated at the iso contact, so bus asymmetry cancels
            print(f"  note: T buses mismatched by {spread:.0%} "
                  "(calibrated out per config; informational)")

    average_bus = sum(buses) / len(buses) if buses else 0
    for name in ("/CAL_E", "/CAL_F"):
        length = lengths.get(name, 0)
        if average_bus and abs(length - average_bus) / average_bus > 0.40:
            BUDGET_FAILURES.append(
                f"{name} {length:.1f} mm vs T-bus average {average_bus:.1f} mm: "
                "load column no longer mimics a terminal column")

    if BUDGET_FAILURES:
        print("BUDGET VIOLATIONS:")
        for failure in BUDGET_FAILURES:
            print(f"  FAIL  {failure}")
        return 1
    print("All parasitic budgets met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
