"""
Post-route track optimization: pull every route taut.

    "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" optimize_tracks.py [--dry-run]

Runs after route_pcb.py. Freerouting and the grid maze router both leave
detours: staircase runs, corners hugging long-gone obstacles, a via pair where
one layer would do. For each net, the outer-layer tracks are split into chains
between anchors (pads, vias, junctions, T-joins, chain ends) and each chain is

  1. straightened by line-of-sight: from every vertex, jump to the farthest
     later vertex the straight segment can reach;
  2. corner-cut: each interior corner is replaced by a chord across it, as
     deep as the clearance allows, then straightened again (repeat);

with every candidate segment checked exactly (no grid) against all other-net
copper on its layer: tracks, vias, pad outlines, holes, the board edge
(copper-to-edge rule) and no-track rule areas. Afterwards vias whose net
copper is on one layer only are removed, dangling stubs trimmed, zones
refilled and DRC re-run; the result is kept only if DRC is no worse.

Not touched: the generated rails (equal HI/LO feed length is a measurement
requirement, see parasitic_report.py), GND, the fine-pitch USB field, and the
inner layers (In1/In2 carry locked routing only inside the measurement void).
Locked tracks stay locked.
"""

import math
import os
import shutil
import subprocess
import sys

import numpy
import pcbnew

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD = os.path.join(HERE, "relay_board.kicad_pcb")
KICAD_CLI = r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe"

CLEARANCE = 0.2 + 0.015               # JLC 0.2 mm + rounding margin (nm grid)
EDGE_CLEARANCE = 0.5 + 0.015          # JLC copper-to-edge
HOLE_CLEARANCE = 0.25 + 0.015         # copper to a hole of another net / NPTH
OUTER = (pcbnew.F_Cu, pcbnew.B_Cu)
SKIP_NETS = {"/RAIL_HI", "/RAIL_LO", "/SNS_HI", "/SNS_LO", "/GND", ""}
SKIP_PREFIXES = ("/USB_",)
CORNER_FRACTIONS = (0.95, 0.85, 0.7, 0.5, 0.3, 0.15)
PASSES = 6
EPS = 1e-4                            # mm: coincident points


def uid(item):
    """Stable identity of a board item (id() of a SWIG wrapper is recycled)."""
    return item.m_Uuid.AsString()


def mm(value):
    return pcbnew.ToMM(value)


def vec(x, y):
    return pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))


# ----------------------------------------------------------------- geometry

def point_seg_distance(P, A, B):
    """Distance from points P (N x 2) to segments A-B (N x 2 each)."""
    V = B - A
    length2 = numpy.einsum("ij,ij->i", V, V)
    with numpy.errstate(divide="ignore", invalid="ignore"):
        t = numpy.where(length2 > 1e-12, numpy.einsum("ij,ij->i", P - A, V) / length2, 0.0)
    t = numpy.clip(t, 0.0, 1.0)
    return numpy.hypot(*(A + t[:, None] * V - P).T)


def seg_seg_distance(p1, p2, Q1, Q2):
    """Exact distance from segment p1-p2 to each segment Q1[k]-Q2[k]: zero if
    they cross, else the smallest endpoint-to-segment distance."""
    n = len(Q1)
    P1 = numpy.broadcast_to(numpy.asarray(p1, float), (n, 2))
    P2 = numpy.broadcast_to(numpy.asarray(p2, float), (n, 2))
    distance = numpy.minimum.reduce([point_seg_distance(P1, Q1, Q2), point_seg_distance(P2, Q1, Q2),
                                     point_seg_distance(Q1, P1, P2), point_seg_distance(Q2, P1, P2)])

    def orient(a, b, c):
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    o1, o2 = orient(P1, P2, Q1), orient(P1, P2, Q2)
    o3, o4 = orient(Q1, Q2, P1), orient(Q1, Q2, P2)
    crossing = (o1 * o2 < 0) & (o3 * o4 < 0)
    distance[crossing] = 0.0
    return distance


class Obstacles:
    """Capsules (segment + radius + required gap) per layer, tagged by net."""

    def __init__(self):
        self.items = {layer: [] for layer in OUTER}

    def add(self, layer, a, b, radius, gap, net, ref=None, net2=None):
        """`net2`: a second net that may touch this copper (a net tie's
        bridge belongs to both nets it ties)."""
        self.items[layer].append((a[0], a[1], b[0], b[1], radius, gap, net, ref,
                                  net if net2 is None else net2))

    def freeze(self):
        self.arrays = {}
        self.second = {}
        for layer, rows in self.items.items():
            data = numpy.array([r[:6] for r in rows], float) if rows else numpy.zeros((0, 6))
            nets = numpy.array([r[6] for r in rows], int) if rows else numpy.zeros(0, int)
            refs = [r[7] for r in rows]
            self.arrays[layer] = (data, nets, refs)
            self.second[layer] = numpy.array([r[8] for r in rows], int) if rows else numpy.zeros(0, int)

    def clear(self, layer, a, b, half, net, ignore=()):
        data, nets, refs = self.arrays[layer]
        if not len(data):
            return True
        lo_x, hi_x = min(a[0], b[0]), max(a[0], b[0])
        lo_y, hi_y = min(a[1], b[1]), max(a[1], b[1])
        reach = half + data[:, 4] + data[:, 5]
        near = ((numpy.minimum(data[:, 0], data[:, 2]) - reach <= hi_x)
                & (numpy.maximum(data[:, 0], data[:, 2]) + reach >= lo_x)
                & (numpy.minimum(data[:, 1], data[:, 3]) - reach <= hi_y)
                & (numpy.maximum(data[:, 1], data[:, 3]) + reach >= lo_y)
                & (nets != net) & (self.second[layer] != net))
        if ignore:
            near &= numpy.array([r not in ignore for r in refs])
        if not near.any():
            return True
        sub = data[near]
        distance = seg_seg_distance(a, b, sub[:, 0:2], sub[:, 2:4])
        return bool(numpy.all(distance >= half + sub[:, 4] + sub[:, 5]))


def build_obstacles(board):
    obstacles = Obstacles()
    for footprint in board.GetFootprints():
        for pad in footprint.Pads():
            net = pad.GetNetCode() or -1000 - len(obstacles.items[OUTER[0]])
            for layer in OUTER:
                if not pad.IsOnLayer(layer):
                    continue
                poly = pad.GetEffectivePolygon(layer, pcbnew.ERROR_OUTSIDE)
                for k in range(poly.OutlineCount()):
                    outline = poly.Outline(k)
                    points = [(mm(outline.CPoint(i).x), mm(outline.CPoint(i).y))
                              for i in range(outline.PointCount())]
                    for p, q in zip(points, points[1:] + points[:1]):
                        obstacles.add(layer, p, q, 0.0, CLEARANCE, net)
            if pad.HasHole():
                position = pad.GetPosition()
                centre = (mm(position.x), mm(position.y))
                radius = mm(max(pad.GetDrillSize().x, pad.GetDrillSize().y)) / 2
                hole_net = -1 if pad.GetAttribute() == pcbnew.PAD_ATTRIB_NPTH else net
                for layer in OUTER:
                    obstacles.add(layer, centre, centre, radius, HOLE_CLEARANCE, hole_net)
        # footprint copper without a net: a net tie's bridge is the copper of
        # both nets it ties (transparent to them, an obstacle to the rest);
        # anything else (logos) is foreign to every net
        tie = sorted({q.GetNetCode() for q in footprint.Pads() if q.GetNetCode() > 0})
        is_tie = footprint.GetReference().startswith("NT") and len(tie) == 2
        for item in footprint.GraphicalItems():
            if item.GetLayer() in OUTER and item.GetClass() == "PCB_SHAPE":
                box = item.GetBoundingBox()
                corners = [(mm(box.GetLeft()), mm(box.GetTop())), (mm(box.GetRight()), mm(box.GetTop())),
                           (mm(box.GetRight()), mm(box.GetBottom())), (mm(box.GetLeft()), mm(box.GetBottom()))]
                for p, q in zip(corners, corners[1:] + corners[:1]):
                    if is_tie:
                        obstacles.add(item.GetLayer(), p, q, 0.0, CLEARANCE, tie[0], net2=tie[1])
                    else:
                        obstacles.add(item.GetLayer(), p, q, 0.0, CLEARANCE, -2)
    for item in board.GetTracks():
        net = item.GetNetCode()
        if item.GetClass() == "PCB_VIA":
            p = item.GetPosition()
            centre = (mm(p.x), mm(p.y))
            for layer in OUTER:
                obstacles.add(layer, centre, centre, mm(item.GetWidth(layer)) / 2, CLEARANCE, net, uid(item))
                obstacles.add(layer, centre, centre, mm(item.GetDrillValue()) / 2, HOLE_CLEARANCE, net, uid(item))
        elif item.GetClass() == "PCB_TRACK" and item.GetLayer() in OUTER:
            s, e = item.GetStart(), item.GetEnd()
            obstacles.add(item.GetLayer(), (mm(s.x), mm(s.y)), (mm(e.x), mm(e.y)),
                          mm(item.GetWidth()) / 2, CLEARANCE, net, uid(item))
        elif item.GetClass() == "PCB_ARC" and item.GetLayer() in OUTER:
            s, m_, e = item.GetStart(), item.GetMid(), item.GetEnd()
            for p, q in ((s, m_), (m_, e)):
                obstacles.add(item.GetLayer(), (mm(p.x), mm(p.y)), (mm(q.x), mm(q.y)),
                              mm(item.GetWidth()) / 2 + 0.05, CLEARANCE, net, uid(item))
    for drawing in board.GetDrawings():
        if drawing.GetLayer() != pcbnew.Edge_Cuts:
            continue
        if drawing.GetShape() == pcbnew.SHAPE_T_CIRCLE:
            c = drawing.GetCenter()
            centre = (mm(c.x), mm(c.y))
            for layer in OUTER:
                obstacles.add(layer, centre, centre, mm(drawing.GetRadius()), EDGE_CLEARANCE, -3)
            continue
        if drawing.GetShape() == pcbnew.SHAPE_T_RECT:
            corners = drawing.GetRectCorners()
            points = [(mm(c.x), mm(c.y)) for c in corners]
        else:
            points = [(mm(drawing.GetStart().x), mm(drawing.GetStart().y)),
                      (mm(drawing.GetEnd().x), mm(drawing.GetEnd().y))]
        pairs = zip(points, points[1:] + points[:1]) if len(points) > 2 else [tuple(points)]
        for p, q in pairs:
            for layer in OUTER:
                obstacles.add(layer, p, q, 0.0, EDGE_CLEARANCE, -3)
    for zone in board.Zones():
        if not (zone.GetIsRuleArea() and zone.GetDoNotAllowTracks()):
            continue
        outline = zone.Outline().Outline(0)
        points = [(mm(outline.CPoint(i).x), mm(outline.CPoint(i).y)) for i in range(outline.PointCount())]
        for layer in OUTER:
            if zone.IsOnLayer(layer):
                for p, q in zip(points, points[1:] + points[:1]):
                    obstacles.add(layer, p, q, 0.0, 0.05, -4)
    obstacles.freeze()
    return obstacles


def in_area(board, name, point):
    for zone in board.Zones():
        if zone.GetIsRuleArea() and zone.GetZoneName() == name:
            if zone.Outline().Contains(vec(*point)):
                return True
    return False


# ------------------------------------------------------------------ chains

def key(point):
    return (round(point[0] / EPS), round(point[1] / EPS))


def on_segment(p, a, b, tolerance):
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    if length2 < 1e-12:
        return False
    t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / length2
    if t <= 1e-6 or t >= 1 - 1e-6:
        return False
    return math.hypot(ax + t * vx - p[0], ay + t * vy - p[1]) <= tolerance


def chains_for(board, net, layer, tracks, anchors, covers):
    """Split one net's tracks on one layer into chains of equal width between
    anchor points. Returns [(points, width, locked, [tracks])]. `covers` are
    (polygon-tester) callables for the net's pads/vias on this layer: a vertex
    inside one is a connection and never moves."""
    cache = {}

    def covered(k):
        if k not in cache:
            point = vec(k[0] * EPS, k[1] * EPS)
            cache[k] = any(test(point) for test in covers)
        return cache[k]

    ends = {}
    for track in tracks:
        for point in (track.a, track.b):
            ends.setdefault(key(point), []).append(track)
    # T-joins: an endpoint landing inside another segment freezes that segment
    frozen = set()
    for track in tracks:
        for other in tracks:
            if other is track:
                continue
            for point in (other.a, other.b):
                if on_segment(point, track.a, track.b, track.width / 2):
                    frozen.add(id(track))
                    anchors.add(key(point))
    used = set()
    chains = []

    def is_anchor(k):
        group = ends[k]
        if k in anchors or len(group) != 2 or covered(k):
            return True
        first, second = group
        return (abs(first.width - second.width) > 1e-6 or first.locked != second.locked
                or id(first) in frozen or id(second) in frozen)

    for track in tracks:
        if id(track) in used or id(track) in frozen:
            continue
        # walk to one end
        start, current = track, track
        point = key(track.a)
        while not is_anchor(point):
            nxt = [t for t in ends[point] if t is not current][0]
            if nxt is start or id(nxt) in frozen:
                break
            point = key(nxt.b) if key(nxt.a) == point else key(nxt.a)
            current = nxt
        # walk back collecting
        sequence, points = [], []
        here = point
        segment = current
        points.append(segment.a if key(segment.a) == here else segment.b)
        while True:
            if id(segment) in used:
                break
            used.add(id(segment))
            sequence.append(segment)
            far = segment.b if key(segment.a) == key(points[-1]) else segment.a
            points.append(far)
            k = key(far)
            if is_anchor(k):
                break
            options = [t for t in ends[k] if t is not segment and id(t) not in used]
            if not options:
                break
            segment = options[0]
        if len(sequence) >= 2 or len(sequence) == 1:
            chains.append((points, track.width, track.locked, sequence))
    return chains


class Seg:
    def __init__(self, item):
        self.item = item
        s, e = item.GetStart(), item.GetEnd()
        self.a = (mm(s.x), mm(s.y))
        self.b = (mm(e.x), mm(e.y))
        self.width = mm(item.GetWidth())
        self.locked = item.IsLocked()


def path_length(points):
    return sum(math.dist(p, q) for p, q in zip(points, points[1:]))


def tighten(points, clear):
    """Line-of-sight straightening plus corner cutting. `clear(a, b)` tests a
    candidate segment. Endpoints never move."""
    points = list(points)
    for _ in range(PASSES):
        before = path_length(points)
        # line of sight
        result, k = [points[0]], 0
        while k < len(points) - 1:
            far = len(points) - 1
            while far > k + 1 and not clear(points[k], points[far]):
                far -= 1
            result.append(points[far])
            k = far
        points = result
        # corner cutting
        changed = False
        k = 1
        while k < len(points) - 1:
            prev, corner, nxt = points[k - 1], points[k], points[k + 1]
            for fraction in CORNER_FRACTIONS:
                a = (corner[0] + fraction * (prev[0] - corner[0]), corner[1] + fraction * (prev[1] - corner[1]))
                b = (corner[0] + fraction * (nxt[0] - corner[0]), corner[1] + fraction * (nxt[1] - corner[1]))
                if math.dist(a, b) < 0.05:
                    continue
                if clear(a, b):
                    replacement = [a, b]
                    points[k:k + 1] = replacement
                    k += len(replacement)
                    changed = True
                    break
            else:
                k += 1
        # drop near-duplicate points
        cleaned = [points[0]]
        for p in points[1:]:
            if math.dist(p, cleaned[-1]) > 0.01:
                cleaned.append(p)
        if len(cleaned) == 1:
            cleaned.append(points[-1])
        cleaned[-1] = points[-1]
        points = cleaned
        if not changed and path_length(points) > before - 1e-3:
            break
    return points


# --------------------------------------------------------------- the pass

def anchor_points(board, net):
    anchors = set()
    for footprint in board.GetFootprints():
        for pad in footprint.Pads():
            if pad.GetNetCode() == net:
                p = pad.GetPosition()
                anchors.add(key((mm(p.x), mm(p.y))))
    for item in board.GetTracks():
        if item.GetClass() == "PCB_VIA" and item.GetNetCode() == net:
            p = item.GetPosition()
            anchors.add(key((mm(p.x), mm(p.y))))
    return anchors


def connection_testers(board, net, layer):
    testers = []
    for footprint in board.GetFootprints():
        for pad in footprint.Pads():
            if pad.GetNetCode() == net and pad.IsOnLayer(layer):
                testers.append(lambda point, pad=pad, layer=layer: pad.HitTest(point, 0, layer))
    for item in board.GetTracks():
        if item.GetClass() == "PCB_VIA" and item.GetNetCode() == net:
            testers.append(lambda point, via=item: via.HitTest(point))
    return testers


def optimize(board):
    obstacles = build_obstacles(board)
    by_net = {}
    for item in board.GetTracks():
        if item.GetClass() == "PCB_TRACK" and item.GetLayer() in OUTER:
            by_net.setdefault((item.GetNetCode(), item.GetLayer()), []).append(Seg(item))
    names = {code: n.GetNetname() for code, n in board.GetNetsByNetcode().items()}
    saved = 0.0
    total_before = 0.0
    rewritten = 0
    for (net, layer), tracks in sorted(by_net.items()):
        name = names.get(net, "")
        length = sum(math.dist(t.a, t.b) for t in tracks)
        total_before += length
        if name in SKIP_NETS or name.startswith(SKIP_PREFIXES):
            continue
        anchors = anchor_points(board, net)
        covers = connection_testers(board, net, layer)
        for points, width, locked, sequence in chains_for(board, net, layer, tracks, anchors, covers):
            if len(points) < 3:
                continue
            if any(in_area(board, "usb_field", p) for p in points):
                continue
            ignore = {uid(t.item) for t in sequence}
            half = width / 2

            def clear(a, b, layer=layer, half=half, net=net, ignore=ignore):
                return obstacles.clear(layer, a, b, half, net, ignore)

            new_points = tighten(points, clear)
            gain = path_length(points) - path_length(new_points)
            if gain < 0.05 and len(new_points) >= len(points):
                continue
            for t in sequence:
                board.Remove(t.item)
            for p, q in zip(new_points, new_points[1:]):
                track = pcbnew.PCB_TRACK(board)
                track.SetStart(vec(*p))
                track.SetEnd(vec(*q))
                track.SetLayer(layer)
                track.SetWidth(pcbnew.FromMM(width))
                track.SetNetCode(net)
                track.SetLocked(locked)
                board.Add(track)
                obstacles.add(layer, p, q, half, CLEARANCE, net, uid(track))
            obstacles.freeze()
            saved += gain
            rewritten += 1
    return rewritten, saved, total_before


def remove_redundant_vias(board):
    """A non-GND via whose net has copper on one layer only at that spot."""
    removed = 0
    names = {code: n.GetNetname() for code, n in board.GetNetsByNetcode().items()}
    tracks = [t for t in board.GetTracks() if t.GetClass() == "PCB_TRACK"]
    for via in [t for t in board.GetTracks() if t.GetClass() == "PCB_VIA"]:
        net = via.GetNetCode()
        if names.get(net, "") in SKIP_NETS:
            continue
        p = via.GetPosition()
        point = (mm(p.x), mm(p.y))
        radius = mm(via.GetWidth(pcbnew.F_Cu)) / 2
        layers = set()
        for t in tracks:
            if t.GetNetCode() != net:
                continue
            a = numpy.array([[mm(t.GetStart().x), mm(t.GetStart().y)]])
            b = numpy.array([[mm(t.GetEnd().x), mm(t.GetEnd().y)]])
            if point_seg_distance(numpy.array([point]), a, b)[0] <= radius + mm(t.GetWidth()) / 2:
                layers.add(t.GetLayer())
        for footprint in board.GetFootprints():
            for pad in footprint.Pads():
                if pad.GetNetCode() == net and pad.HitTest(p):
                    layers |= {l for l in (pcbnew.F_Cu, pcbnew.In1_Cu, pcbnew.In2_Cu, pcbnew.B_Cu)
                               if pad.IsOnLayer(l)}
        if len(layers) == 1:
            # merge happens through the remaining layer's tracks: safe to drop
            board.Remove(via)
            removed += 1
    return removed


def drc(path):
    report = os.path.join(HERE, "relay_board_drc.txt")
    subprocess.run([KICAD_CLI, "pcb", "drc", "--schematic-parity", "--severity-error",
                    "-o", report, path], capture_output=True)
    text = open(report, encoding="utf-8", errors="replace").read()
    import re
    violations = int(re.search(r"Found (\d+) DRC violations", text).group(1)) if "DRC violations" in text else -1
    unconnected = int(re.search(r"Found (\d+) unconnected", text).group(1)) if "unconnected" in text else -1
    return violations, unconnected, text


def total_length(board):
    return sum(mm(t.GetLength()) for t in board.GetTracks() if t.GetClass() == "PCB_TRACK")


def main():
    dry = "--dry-run" in sys.argv
    backup = BOARD.replace(".kicad_pcb", "_preopt.kicad_pcb")
    shutil.copyfile(BOARD, backup)
    before = drc(BOARD)
    board = pcbnew.LoadBoard(BOARD)
    length_before = total_length(board)
    vias_before = sum(1 for t in board.GetTracks() if t.GetClass() == "PCB_VIA")
    rewritten, saved, _ = optimize(board)
    vias = remove_redundant_vias(board)
    # a second optimization pass after via removal can merge chains
    rewritten2, saved2, _ = optimize(board)
    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(board.Zones())
    length_after = total_length(board)
    vias_after = sum(1 for t in board.GetTracks() if t.GetClass() == "PCB_VIA")
    print(f"chains rewritten: {rewritten + rewritten2}; track length {length_before:.0f} -> "
          f"{length_after:.0f} mm ({length_before - length_after:.0f} mm shorter); "
          f"vias {vias_before} -> {vias_after} ({vias} redundant removed)")
    if dry:
        print("dry run: board not saved")
        return 0
    board.Save(BOARD)
    after = drc(BOARD)
    print(f"DRC errors {before[0]} -> {after[0]}, unconnected {before[1]} -> {after[1]}")
    if after[0] > before[0] or after[1] > before[1]:
        shutil.copyfile(backup, BOARD)
        drc(BOARD)
        print("worse than before: original board restored (relay_board_preopt.kicad_pcb kept)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
