"""
Route the rev B board: freerouting DSN/SES round trip, zone fill, DRC.

    "C:\\Program Files\\KiCad\\9.0\\bin\\python.exe" route_pcb.py

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

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD_PATH = os.path.join(HERE, "relay_board.kicad_pcb")
DSN = os.path.join(HERE, "relay_board.dsn")
SES = os.path.join(HERE, "relay_board.ses")
JAR = os.path.join(HERE, "..", "..", "tools", "freerouting-1.9.0.jar")
KICAD_CLI = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"

# freerouting 1.9 needs Java 17+; a portable Adoptium JRE is vendored in
# tools/ next to the freerouting jar.
_TOOLS_JAVA = os.path.join(HERE, "..", "..", "tools",
                           'jdk-17.0.20.1+1-jre', "bin", "java.exe")
JAVA = _TOOLS_JAVA if os.path.exists(_TOOLS_JAVA) else "java"


def patch_dsn_power_layers():
    """Mark In1/In2 as power in the DSN so freerouting never routes there
    (must run after EVERY ExportSpecctraDSN)."""
    with open(DSN, encoding="utf-8") as handle:
        lines = handle.readlines()
    count = 0
    pending = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith("(layer In1.Cu")
                or stripped.startswith("(layer In2.Cu")):
            pending = index
        elif pending is not None and "(type signal)" in line:
            lines[index] = line.replace("(type signal)", "(type power)")
            count += 1
            pending = None
        elif stripped.startswith("(layer "):
            pending = None
    with open(DSN, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    if count:
        print(f"  DSN: {count} inner layers set to power")


def run_freerouting(passes=24):
    if not os.path.exists(DSN):
        raise SystemExit("relay_board.dsn missing -- run generate_pcb.py first")
    print(f"freerouting: {os.path.basename(DSN)} -> {os.path.basename(SES)}")
    command = [JAVA, "-jar", JAR, "-de", DSN, "-do", SES,
               "-mp", str(passes), "-us", "global"]
    result = subprocess.run(command, capture_output=True, text=True,
                            cwd=HERE, timeout=1800)
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
    plane_nets = {"/GND": "/GND", "/+5V": "/+5V"}
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
            if px < 150.0 and net_name == "/+5V":
                continue            # signal region +5V is pre-routed
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
                if not (152.3 < x < 188.5 and 41.5 < y < 128.5):
                    continue        # via must land inside the plane region
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
    inner = (pcbnew.In1_Cu, pcbnew.In2_Cu)
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


if __name__ == "__main__":
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
        run_freerouting(passes=30)   # different depth -> different landscape
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
        run_freerouting(passes=28 + 6 * extra_round)
        import_session()
        counts = run_drc()
    sys.exit(1 if counts.get("unconnected_items") or counts.get("clearance")
             or counts.get("shorting_items") or counts.get("tracks_crossing")
             else 0)
