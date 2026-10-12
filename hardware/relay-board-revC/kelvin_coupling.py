"""
Force-to-sense magnetic coupling of the rev C fixture, and what each SHORT
standard leaves behind -- 2-wire and Kelvin.

    python kelvin_coupling.py

Why this exists: on this board a terminal's force and sense run side by side
through the same relays (pole 1 / pole 2). A Kelvin measurement removes the
voltage DROP along the force path, but the sense loop still links the force
loop's magnetic flux: V_sense = I * (Z_dut + j w M_fs), with M_fs the mutual
inductance between the force path and the sense loop. A SHORT standard only
calibrates M_fs out if its current flows where the DUT current flows.

Method: Manhattan polylines of the force path and the sense loop for each
state (DUT, rev B-style SHORT, rev C SHORT relays), taken from the rev C
placement (generate_pcb.py constants and pad positions); partial inductances
of straight segments (Ruehli self, Neumann/Grover mutual for parallel
segments, zero for perpendicular ones) summed with current directions.
The DUT's own leads (inside the bay) are excluded: that is the measurement.

Bus routes are freerouted on the real board, so they are idealized here as
straight Manhattan runs between the pads they join; the numbers are
estimates of the size of each effect, not a substitute for a field solver.
"""

import math

MU = 4e-7 * math.pi
W = 0.5e-3                     # track width (m), thin-strip approximation


def self_partial(length):
    if length <= 0:
        return 0.0
    return MU * length / (2 * math.pi) * (math.log(2 * length / W) + 0.5 + 0.2235 * W / length)


def _f(u, d):
    return u * math.asinh(u / d) - math.sqrt(u * u + d * d)


def mutual_parallel(a1, a2, b1, b2, d):
    """Mutual partial inductance of two parallel filaments on the same axis
    coordinate ranges [a1,a2] and [b1,b2], separated by d (Grover)."""
    d = max(d, W)
    return MU / (4 * math.pi) * (_f(b2 - a1, d) - _f(b1 - a1, d) - _f(b2 - a2, d) + _f(b1 - a2, d))


def segments(polyline):
    out = []
    for (x1, y1), (x2, y2) in zip(polyline, polyline[1:]):
        if abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9:
            continue
        out.append(((x1 * 1e-3, y1 * 1e-3), (x2 * 1e-3, y2 * 1e-3)))
    return out


def mutual(seg_a, seg_b):
    """Signed mutual partial inductance of two directed Manhattan segments."""
    (ax1, ay1), (ax2, ay2) = seg_a
    (bx1, by1), (bx2, by2) = seg_b
    a_horizontal = abs(ay1 - ay2) < 1e-12
    b_horizontal = abs(by1 - by2) < 1e-12
    if a_horizontal != b_horizontal:
        return 0.0
    if a_horizontal:
        sa = 1 if ax2 > ax1 else -1
        sb = 1 if bx2 > bx1 else -1
        d = abs(ay1 - by1)
        if d < 1e-9 and max(min(ax1, ax2), min(bx1, bx2)) < min(max(ax1, ax2), max(bx1, bx2)):
            d = W                # overlapping collinear: adjacent strips
        m = mutual_parallel(min(ax1, ax2), max(ax1, ax2), min(bx1, bx2), max(bx1, bx2), d)
    else:
        sa = 1 if ay2 > ay1 else -1
        sb = 1 if by2 > by1 else -1
        d = abs(ax1 - bx1)
        if d < 1e-9 and max(min(ay1, ay2), min(by1, by2)) < min(max(ay1, ay2), max(by1, by2)):
            d = W
        m = mutual_parallel(min(ay1, ay2), max(ay1, ay2), min(by1, by2), max(by1, by2), d)
    return sa * sb * m


def loop_inductance(path):
    segs = segments(path)
    total = 0.0
    for i, a in enumerate(segs):
        for j, b in enumerate(segs):
            if i == j:
                length = math.hypot(a[1][0] - a[0][0], a[1][1] - a[0][1])
                total += self_partial(length)
            elif j > i:
                total += 2 * mutual(a, b)
    return total


def coupling(force, sense):
    return sum(mutual(a, b) for a in segments(force) for b in segments(sense))


# ------------------------------------------------------- rev C geometry (mm)
COL = {"A": 61.0, "B": 83.0, "C": 127.0, "D": 149.0}
ROW_HI, ROW_LO = 76.0, 88.0
Y_RAIL_HI, Y_SNS_HI = 79.8, 81.3
Y_RAIL_LO, Y_SNS_LO = 91.8, 93.3
FEED_HI_X, FEED_LO_X = 56.0, 160.0
FE_X = 88.0                                  # front-end taps on the sense rails
ISO = {"A": (76.0, 49.1, "W"), "B": (76.0, 60.7, "W"), "C": (134.0, 49.1, "E"), "D": (134.0, 60.8, "E")}
CLAMP = {"A": (85.0, 48.0), "B": (85.0, 56.0), "C": (125.0, 48.0), "D": (125.0, 56.0)}
SHORT_RELAY_X = {"W": 64.0, "E": 146.0}


def iso_pads(t):
    cx, cy, side = ISO[t]
    if side == "W":                          # rot 180: pole 1 east (x+3.5), rows mirrored
        p1, p2 = cx + 3.5, cx - 3.5
        return {"COM1": (p1, cy - 1.6), "NC1": (p1, cy + 0.6), "NO1": (p1, cy - 3.8),
                "COM2": (p2, cy - 1.6), "NC2": (p2, cy + 0.6), "NO2": (p2, cy - 3.8)}
    p1, p2 = cx - 3.5, cx + 3.5
    return {"COM1": (p1, cy + 1.6), "NC1": (p1, cy - 0.6), "NO1": (p1, cy + 3.8),
            "COM2": (p2, cy + 1.6), "NC2": (p2, cy - 0.6), "NO2": (p2, cy + 3.8)}


def bus(t, kind, row):
    """Iso COM -> matrix COM, Manhattan: horizontal at the COM row, then down
    the bus line (force 2.5 mm west of the column, sense 2.5 mm east of pole 2)."""
    pads = iso_pads(t)
    com = pads["COM1" if kind == "F" else "COM2"]
    x_bus = COL[t] - 2.5 if kind == "F" else COL[t] + 9.5
    x_pad = COL[t] if kind == "F" else COL[t] + 7.0
    return [com, (x_bus, com[1]), (x_bus, row + 1.6), (x_pad, row + 1.6)]


def arm(t, kind):
    pads = iso_pads(t)
    cx, cy = CLAMP[t]
    if kind == "F":
        nc = pads["NC1"]
        return [(cx, cy), (nc[0] + (2.6 if cx > nc[0] else -2.6), cy),
                (nc[0] + (2.6 if cx > nc[0] else -2.6), nc[1]), nc]
    nc = pads["NC2"]
    sy = cy - 2.54 if cx < 100 else cy + 2.54
    lane = cx - 3.0 if cx < 100 else cx + 3.0
    under = pads["NC1"][1] + (1.6 if pads["NC1"][1] > pads["COM1"][1] else -1.6)
    via = nc[0] - 1.9 if cx < 100 else nc[0] + 1.9
    return [(cx, sy), (lane, sy), (lane, under), (via, under), (via, nc[1]), nc]


def rev(polyline):
    return list(reversed(polyline))


def dut_paths(hi, lo):
    """Force path (feed -> DUT HI clamp, DUT LO clamp -> feed) and sense loop
    (front-end + -> DUT HI sense pad, DUT LO sense pad -> front-end -)."""
    f_in = [(FEED_HI_X, 110.0), (FEED_HI_X, Y_RAIL_HI), (COL[hi], Y_RAIL_HI), (COL[hi], ROW_HI + 1.6)]
    f_in += rev(bus(hi, "F", ROW_HI)) + rev(arm(hi, "F"))
    f_out = arm(lo, "F") + bus(lo, "F", ROW_LO) + [(COL[lo], Y_RAIL_LO), (FEED_LO_X, Y_RAIL_LO),
                                                     (FEED_LO_X, 112.0)]
    s_in = [(FE_X, 100.0), (FE_X, Y_SNS_HI), (COL[hi] + 7.0, Y_SNS_HI), (COL[hi] + 7.0, ROW_HI + 1.6)]
    s_in += rev(bus(hi, "S", ROW_HI)) + rev(arm(hi, "S"))
    s_out = arm(lo, "S") + bus(lo, "S", ROW_LO) + [(COL[lo] + 7.0, Y_SNS_LO), (FE_X, Y_SNS_LO),
                                                     (FE_X, 108.0)]
    return (f_in, f_out), (s_in, s_out)


def short_paths_b2(hi, lo):
    """SHORT relays: buses up to the iso NO throws, through the SHORT relays
    to the star (force and sense meet there)."""
    def to_star(t, kind):
        pads = iso_pads(t)
        no = pads["NO1" if kind == "F" else "NO2"]
        side = ISO[t][2]
        sx = SHORT_RELAY_X[side] + (3.5 if side == "W" else -3.5) * (1 if kind == "F" else -1)
        star_y = 55.0
        return [no, (no[0], no[1] - 1.2 if side == "W" else no[1] + 1.2),
                (sx, no[1] - 1.2 if side == "W" else no[1] + 1.2), (sx, star_y),
                (SHORT_RELAY_X[side], star_y)]
    f_in = [(FEED_HI_X, 110.0), (FEED_HI_X, Y_RAIL_HI), (COL[hi], Y_RAIL_HI), (COL[hi], ROW_HI + 1.6)]
    f_in += rev(bus(hi, "F", ROW_HI)) + to_star(hi, "F")
    f_out = rev(to_star(lo, "F")) + bus(lo, "F", ROW_LO) + [(COL[lo], Y_RAIL_LO), (FEED_LO_X, Y_RAIL_LO),
                                                           (FEED_LO_X, 112.0)]
    if ISO[hi][2] != ISO[lo][2]:              # cross-bay: star copper joins the sides
        f_in = f_in + [(SHORT_RELAY_X["W"], 55.0), (105.0, 66.5), (SHORT_RELAY_X["E"], 66.5)]
    s_in = [(FE_X, 100.0), (FE_X, Y_SNS_HI), (COL[hi] + 7.0, Y_SNS_HI), (COL[hi] + 7.0, ROW_HI + 1.6)]
    s_in += rev(bus(hi, "S", ROW_HI)) + to_star(hi, "S")
    s_out = rev(to_star(lo, "S")) + bus(lo, "S", ROW_LO) + [(COL[lo] + 7.0, Y_SNS_LO), (FE_X, Y_SNS_LO),
                                                           (FE_X, 108.0)]
    return (f_in, f_out), (s_in, s_out)


def short_paths_rev_b(hi, lo):
    """HI terminal's own LO relay: force down the HI column to the LO rail at
    that column; sense rails meet on the HI terminal's sense bus."""
    f = [(FEED_HI_X, 110.0), (FEED_HI_X, Y_RAIL_HI), (COL[hi], Y_RAIL_HI), (COL[hi], ROW_HI + 1.6),
         (COL[hi] - 2.5, ROW_HI + 1.6), (COL[hi] - 2.5, ROW_LO + 1.6), (COL[hi], ROW_LO + 1.6),
         (COL[hi], Y_RAIL_LO), (FEED_LO_X, Y_RAIL_LO), (FEED_LO_X, 112.0)]
    s = [(FE_X, 100.0), (FE_X, Y_SNS_HI), (COL[hi] + 7.0, Y_SNS_HI), (COL[hi] + 7.0, ROW_HI + 1.6),
         (COL[hi] + 9.5, ROW_HI + 1.6), (COL[hi] + 9.5, ROW_LO + 1.6), (COL[hi] + 7.0, ROW_LO + 1.6),
         (COL[hi] + 7.0, Y_SNS_LO), (FE_X, Y_SNS_LO), (FE_X, 108.0)]
    return (f, []), (s, [])


def evaluate(paths):
    (f_in, f_out), (s_in, s_out) = paths
    force = f_in + f_out if f_out else f_in
    m = coupling(f_in, s_in) + coupling(f_in, s_out) + coupling(f_out, s_in) + coupling(f_out, s_out) \
        if f_out else coupling(f_in, s_in)
    l2 = loop_inductance(f_in) + (loop_inductance(f_out) + 2 * coupling(f_in, f_out) if f_out else 0)
    return m, l2


def main():
    print("Force/sense coupling and loop inductance (nH), rev C placement\n")
    print(f"{'pair':8s} {'state':16s} {'2-wire L':>9s} {'Kelvin M':>9s}")
    for hi, lo in (("A", "B"), ("A", "C"), ("C", "D")):
        dut = evaluate(dut_paths(hi, lo))
        rows = {"DUT path": dut, "SHORT rev B": evaluate(short_paths_rev_b(hi, lo)),
                "SHORT relays": evaluate(short_paths_b2(hi, lo))}
        for state, (m, l2) in rows.items():
            print(f"{hi}-{lo:6s} {state:16s} {l2 * 1e9:9.1f} {m * 1e9:9.1f}")
        for state in ("SHORT rev B", "SHORT relays"):
            m, l2 = rows[state]
            print(f"{'':8s} residual vs DUT, {state:12s}: 2-wire {1e9 * (dut[1] - l2):+7.1f} nH, "
                  f"Kelvin {1e9 * (dut[0] - m):+7.1f} nH")
        print()
    print("Residual = what a SHORT taken in that state leaves in the corrected result "
          "(includes the arms, which only a clamp-plane short removes).")


if __name__ == "__main__":
    main()
