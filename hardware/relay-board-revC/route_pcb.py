"""
Route the rev C board: freerouting DSN/SES round trip, zone fill, DRC.

    "C:\\Program Files\\KiCad\\10.0\\bin\\python.exe" route_pcb.py

Programmatic trace-by-trace routing was abandoned in rev A (see
ROUTING_GUIDE.md there); the supported path is the Specctra round trip with
the freerouting jar vendored in tools/.  This script drives it end to end:

    relay_board.dsn -> freerouting -> relay_board.ses -> ImportSpecctraSES
    -> fill zones -> save -> kicad-cli pcb drc
"""

import os
import subprocess
import sys

import pcbnew

import generate_pcb as layout          # geometry + DSN plane-layer patch

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD_PATH = os.path.join(HERE, "relay_board.kicad_pcb")
DSN = os.path.join(HERE, "relay_board.dsn")
SES = os.path.join(HERE, "relay_board.ses")
JAR = os.path.join(HERE, "..", "..", "tools", "freerouting-1.9.0.jar")
KICAD_CLI = r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe"

# freerouting 1.9 needs Java 17+; a portable Adoptium JRE is vendored in
# tools/ next to the freerouting jar.
_TOOLS_JAVA = next(iter(sorted(__import__("glob").glob(
    os.path.join(HERE, "..", "..", "tools", "jdk-*-jre", "bin", "java.exe")))), "")
JAVA = _TOOLS_JAVA if os.path.exists(_TOOLS_JAVA) else "java"


def patch_dsn_power_layers():
    """Mark the plane layers (In1/In3/In4) as power in the DSN so freerouting
    never routes there (must run after EVERY ExportSpecctraDSN)."""
    count = layout.patch_dsn_power_layers(DSN)
    if count:
        print(f"  DSN: {count} plane layers set to power")


def in_plane_region(x, y):
    """True where the GND planes (and the strip's +5V plane) exist."""
    return (layout.BOARD_X0 + 1.0 < x < layout.BOARD_X1 - 1.0
            and layout.BOARD_Y0 + 1.0 < y < layout.BOARD_Y1 - 1.0
            and not layout.point_in_polygon(x, y, layout.VOID))


def run_freerouting(passes=12):
    if not os.path.exists(DSN):
        raise SystemExit("relay_board.dsn missing -- run generate_pcb.py first")
    print(f"freerouting: {os.path.basename(DSN)} -> {os.path.basename(SES)}")
    command = [JAVA, "-jar", JAR, "-de", DSN, "-do", SES,
               "-mp", str(passes), "-us", "greedy"]   # global: >30 min on the 6-layer board
    result = subprocess.run(command, capture_output=True, text=True,
                            cwd=HERE, timeout=7200)
    tail = (result.stdout or "").strip().splitlines()[-8:]
    for line in tail:
        print("   ", line)
    if not os.path.exists(SES):
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        raise SystemExit("freerouting produced no session file")


def mm(value):
    return pcbnew.FromMM(value)


def plane_fanout(board):
    """Deterministic via fanout for GND and +5V SMD pads.

    The DSN describes the zones as planes, so freerouting treats plane-net
    pads as already connected and skips them; the planes are on In1/In2/B.Cu
    while the pads are on F.Cu.  For every such pad without copper already
    arriving (a locked feed), drop a stub + via at the first collision-free
    candidate offset.
    """
    plane_nets = {"/GND": "/GND"}         # +5V is routed in rev C
    obstacles = []
    for track in board.GetTracks():
        obstacles.append(track)

    def collides(x, y, net_code):
        position = pcbnew.VECTOR2I(mm(x), mm(y))
        hit = pcbnew.PCB_VIA(board)
        hit.SetPosition(position)
        hit.SetDrill(mm(0.3))
        hit.SetWidth(mm(0.6))
        for item in obstacles:
            if item.GetNetCode() == net_code:
                continue
            if item.HitTest(position, mm(0.55)):
                return True
        for footprint in board.GetFootprints():
            for pad in footprint.Pads():
                if pad.GetNetCode() == net_code:
                    continue
                if pad.HitTest(position, mm(0.45)):
                    return True
        return False

    added, skipped = 0, 0
    candidates = ((1.3, 0), (-1.3, 0), (0, 1.3), (0, -1.3),
                  (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0))
    for footprint in board.GetFootprints():
        for pad in footprint.Pads():
            net_name = pad.GetNetname()
            if net_name not in plane_nets:
                continue
            if pad.GetAttribute() != pcbnew.PAD_ATTRIB_SMD:
                continue            # through-hole pads reach the planes
            position = pad.GetPosition()
            px, py = pcbnew.ToMM(position.x), pcbnew.ToMM(position.y)
            if net_name == "/+5V" and px < layout.STRIP_X0:
                continue            # relay coils: pre-routed In3 tree
            already = any(t.GetNetCode() == pad.GetNetCode()
                          and t.GetClass() == "PCB_VIA"
                          and abs(pcbnew.ToMM(t.GetPosition().x) - px) < 2.2
                          and abs(pcbnew.ToMM(t.GetPosition().y) - py) < 2.2
                          for t in board.GetTracks())
            if already:
                continue
            placed = False
            for dx, dy in candidates:
                x, y = px + dx, py + dy
                if not in_plane_region(x, y):
                    continue        # via must land inside the plane region
                if net_name == "/+5V" and x < layout.STRIP_X0 + 0.5:
                    continue        # +5V plane only in the control strip
                if any(collides(px + dx * f, py + dy * f, pad.GetNetCode())
                       for f in (0.4, 0.7, 1.0)):
                    continue
                track = pcbnew.PCB_TRACK(board)
                track.SetStart(position)
                track.SetEnd(pcbnew.VECTOR2I(mm(x), mm(y)))
                track.SetNet(pad.GetNet())
                track.SetLayer(pcbnew.F_Cu)
                track.SetWidth(mm(0.25))
                board.Add(track)
                via = pcbnew.PCB_VIA(board)
                via.SetPosition(pcbnew.VECTOR2I(mm(x), mm(y)))
                via.SetDrill(mm(0.3))
                via.SetWidth(mm(0.6))
                via.SetNet(pad.GetNet())
                via.SetViaType(pcbnew.VIATYPE_THROUGH)
                board.Add(via)
                obstacles.append(track)
                obstacles.append(via)
                added += 1
                placed = True
                break
            if not placed:
                skipped += 1
                print(f"  fanout: no room at {footprint.GetReference()}"
                      f".{pad.GetNumber()} ({px:.1f},{py:.1f})")
    print(f"plane fanout: {added} vias added, {skipped} pads without room")


def repair_locked_clearance(board, margin_mm=0.0):
    """Remove freerouted (unlocked) copper that lands too close to the locked
    pre-routes -- freerouting's clearance checks against fixed wires are
    unreliable. Returns the set of net names whose geometry was culled."""
    all_items = [t for t in board.GetTracks()]
    locked = [t for t in all_items if t.IsLocked()]
    fiducials = [pad.GetPosition() for footprint in board.GetFootprints()
                 if footprint.GetReference().startswith("FID")
                 for pad in footprint.Pads()]
    culled_nets = set()
    to_remove = []
    for item in all_items:
        if item.IsLocked():
            continue
        sx, sy = item.GetStart().x, item.GetStart().y
        ex, ey = item.GetEnd().x, item.GetEnd().y
        anchors = ((sx, sy), (ex, ey), ((sx + ex) // 2, (sy + ey) // 2))
        offender = False
        for fixed in locked:
            if fixed.GetNetCode() == item.GetNetCode():
                continue
            if item.GetClass() != "PCB_VIA" and fixed.GetClass() != "PCB_VIA"                     and fixed.GetLayer() != item.GetLayer():
                continue
            for ax, ay in anchors:
                if fixed.HitTest(pcbnew.VECTOR2I(int(ax), int(ay)),
                                 mm(margin_mm + 0.14)):
                    offender = True
                    break
            if offender:
                break
        if not offender:
            for fid in fiducials:
                if item.HitTest(fid, mm(1.25)):
                    offender = True
                    break
        if offender:
            to_remove.append(item)
            culled_nets.add(item.GetNetname())
    for item in to_remove:
        board.Delete(item)
    if to_remove:
        print(f"repair: culled {len(to_remove)} freerouted items too close to "
              f"locked copper (nets: {sorted(culled_nets)})")
    return culled_nets


def return_path_stitching(board, reach=1.5):
    """A GND via within `reach` of every signal via outside the measurement
    void. A signal moving F.Cu <-> B.Cu changes reference plane (In1 <-> In2);
    its return current crosses between the planes at the nearest GND via, so
    that via must be right there or the return loop opens up (Faraday's
    return-path rule; Ott ch. 16)."""
    gnd = board.FindNet("/GND").GetNetCode()
    vias = [t for t in board.GetTracks() if t.GetClass() == "PCB_VIA"]
    gnd_xy = [(pcbnew.ToMM(v.GetPosition().x), pcbnew.ToMM(v.GetPosition().y))
              for v in vias if v.GetNetCode() == gnd]
    for footprint in board.GetFootprints():          # THT GND pads count too
        for pad in footprint.Pads():
            if pad.GetNetCode() == gnd and pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                gnd_xy.append((pcbnew.ToMM(pad.GetPosition().x), pcbnew.ToMM(pad.GetPosition().y)))
    pads = [pad for fp in board.GetFootprints() for pad in fp.Pads()]

    def spot_free(x, y):
        position = pcbnew.VECTOR2I(mm(x), mm(y))
        for item in board.GetTracks():
            if item.GetClass() == "PCB_VIA":
                q = item.GetPosition()
                if ((pcbnew.ToMM(q.x) - x) ** 2 + (pcbnew.ToMM(q.y) - y) ** 2) ** 0.5 < 0.85:
                    return False
            if item.GetNetCode() != gnd and item.HitTest(position, mm(0.55)):
                return False
        for pad in pads:
            if pad.GetNetCode() != gnd and pad.HitTest(position, mm(0.55)):
                return False
        return True

    added, missing = 0, 0
    for via in vias:
        if via.GetNetCode() == gnd:
            continue
        x, y = pcbnew.ToMM(via.GetPosition().x), pcbnew.ToMM(via.GetPosition().y)
        if layout.point_in_polygon(x, y, layout.VOID):
            continue                                  # no planes there
        if any((gx - x) ** 2 + (gy - y) ** 2 < reach ** 2 for gx, gy in gnd_xy):
            continue
        for radius in (0.9, 1.2, 1.5):
            spots = [(x + radius * __import__("math").cos(a), y + radius * __import__("math").sin(a))
                     for a in [k * 3.14159265 / 6 for k in range(12)]]
            spot = next(((sx, sy) for sx, sy in spots
                         if not layout.point_in_polygon(sx, sy, layout.VOID) and spot_free(sx, sy)), None)
            if spot:
                stitch = pcbnew.PCB_VIA(board)
                stitch.SetPosition(pcbnew.VECTOR2I(mm(spot[0]), mm(spot[1])))
                stitch.SetDrill(mm(0.3))
                stitch.SetWidth(mm(0.6))
                stitch.SetNetCode(gnd)
                stitch.SetViaType(pcbnew.VIATYPE_THROUGH)
                stitch.SetLocked(True)
                board.Add(stitch)
                gnd_xy.append(spot)
                added += 1
                break
        else:
            missing += 1
    print(f"return-path stitching: {added} GND vias added"
          + (f", {missing} signal vias without room" if missing else ""))


def sweep_dangling(board):
    """Iteratively remove unlocked track segments with a floating end
    (geometric test: nothing else touches the endpoint)."""
    removed_total = 0
    for _ in range(12):
        tracks = [t for t in board.GetTracks()]
        pads = [pad for footprint in board.GetFootprints()
                for pad in footprint.Pads()]
        removed = []
        for item in tracks:
            if item.IsLocked() or item.GetClass() == "PCB_VIA":
                continue
            if item.GetLength() < mm(0.01):
                removed.append(item)
                continue
            for anchor in (item.GetStart(), item.GetEnd()):
                touched = False
                for other in tracks:
                    if other is item:
                        continue
                    if other.GetClass() != "PCB_VIA"                             and other.GetLayer() != item.GetLayer():
                        continue
                    if other.HitTest(anchor, mm(0.01)):
                        touched = True
                        break
                if not touched:
                    for pad in pads:
                        if pad.HitTest(anchor, mm(0.01)):
                            touched = True
                            break
                if not touched:
                    removed.append(item)
                    break
        if not removed:
            break
        for item in removed:
            board.Delete(item)
        removed_total += len(removed)
    if removed_total:
        print(f"swept {removed_total} dangling fragments")


def import_session():
    board = pcbnew.LoadBoard(BOARD_PATH)
    before = len([t for t in board.GetTracks()])
    if not pcbnew.ImportSpecctraSES(board, SES):
        raise SystemExit("ImportSpecctraSES failed")
    after = len([t for t in board.GetTracks()])
    print(f"session imported: {before} -> {after} tracks+vias")

    # purge unlocked inner-layer tracks: only locked pre-routes belong on
    # the planes (freerouting briefly routed there while the DSN power-type
    # patch was broken, and that junk otherwise survives every import)
    inner = (pcbnew.In1_Cu, pcbnew.In2_Cu)          # the plane layers
    junk = [t for t in board.GetTracks()
            if not t.IsLocked() and t.GetClass() != "PCB_VIA"
            and t.GetLayer() in inner]
    for item in junk:
        board.Delete(item)
    if junk:
        print(f"purged {len(junk)} unlocked inner-layer tracks")
    culled = repair_locked_clearance(board)
    sweep_dangling(board)
    plane_fanout(board)
    return_path_stitching(board)
    filler = pcbnew.ZONE_FILLER(board)
    filler.Fill(board.Zones())
    board.Save(BOARD_PATH)
    print("zones filled, board saved")
    return board, culled


def run_drc():
    report = os.path.join(HERE, "relay_board_drc.txt")
    result = subprocess.run([KICAD_CLI, "pcb", "drc", "--format", "report",
                             "--severity-all", "-o", report, BOARD_PATH],
                            capture_output=True, text=True)
    print(result.stdout.strip() or result.stderr.strip())
    counts = {}
    if os.path.exists(report):
        for line in open(report, encoding="utf-8", errors="replace"):
            if line.startswith("["):
                key = line.split("]")[0][1:]
                counts[key] = counts.get(key, 0) + 1
            if line.startswith("** Found"):
                print("  " + line.strip())
    for key, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {count:4d}  {key}")
    return counts


def finish_without_autorouter():
    """Everything is maze-routed (maze_route.py, coarse + --fine): no
    freerouting. Clean up, fan GND out, stitch return paths, fill, DRC."""
    board = pcbnew.LoadBoard(BOARD_PATH)
    sweep_dangling(board)
    plane_fanout(board)
    return_path_stitching(board)
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(BOARD_PATH)
    print("zones filled, board saved")
    counts = run_drc()
    sys.exit(1 if counts.get("unconnected_items") or counts.get("clearance")
             or counts.get("shorting_items") or counts.get("tracks_crossing") else 0)


if __name__ == "__main__":
    if "--no-freerouting" in sys.argv:
        finish_without_autorouter()
    if "--skip-route" not in sys.argv:
        run_freerouting()
    board, culled = import_session()
    if culled:
        # Second freerouting pass over the updated board: the culled nets
        # re-route against the now-committed remainder, which changes the
        # landscape enough to avoid the previous collision in practice.
        print("re-running freerouting for the culled nets...")
        board = pcbnew.LoadBoard(BOARD_PATH)
        pcbnew.ExportSpecctraDSN(board, DSN)
        patch_dsn_power_layers()
        run_freerouting(passes=16)   # different depth -> different landscape
        board, culled = import_session()
        if culled:
            print(f"WARNING: nets still conflicting after repair: {culled}")
    counts = run_drc()
    extra_round = 0
    while counts.get("unconnected_items") and extra_round < 12:
        extra_round += 1
        print(f"stragglers remain -- extra freerouting round {extra_round}...")
        board = pcbnew.LoadBoard(BOARD_PATH)
        pcbnew.ExportSpecctraDSN(board, DSN)
        patch_dsn_power_layers()
        run_freerouting(passes=12 + 4 * extra_round)
        import_session()
        counts = run_drc()
    sys.exit(1 if counts.get("unconnected_items") or counts.get("clearance")
             or counts.get("shorting_items") or counts.get("tracks_crossing")
             else 0)
