"""Follow-up to fix_usb_orientation.py: resolve the 8 DRC hits.
- CC1 descent moved west (was crossing R2.1's pad column)
- GND row pads: north stubs + vias instead of peg-hugging shield links
- J1 silk nose trimmed inside the tab edge
"""

import shutil

import pcbnew

BOARD = "relay_board.kicad_pcb"
shutil.copy(BOARD, "relay_board_backup_usbfix2.kicad_pcb")
board = pcbnew.LoadBoard(BOARD)
mm = pcbnew.FromMM


def pt(x, y):
    return pcbnew.VECTOR2I(mm(x), mm(y))


def close(a, b):
    return abs(a - b) < mm(0.01)


j1 = next(fp for fp in board.GetFootprints() if fp.GetReference() == "J1")
nets = {}
for fp in board.GetFootprints():
    for pad in fp.Pads():
        nets.setdefault(pad.GetNetname(), pad.GetNet())
graveyard = []

DEL = {
    "/CC1": [((149.49, 119.0), (149.49, 126.8)),
             ((149.49, 126.8), (162.75, 126.8))],
    "/GND": [((160.75, 123.705), (160.75, 124.62)),
             ((160.75, 124.62), (160.18, 124.62)),
             ((167.25, 123.705), (167.25, 124.62)),
             ((167.25, 124.62), (167.82, 124.62))],
}
removed = 0
for track in list(board.GetTracks()):
    if track.Type() == pcbnew.PCB_VIA_T:
        continue
    s, e = track.GetStart(), track.GetEnd()
    for (a, b) in DEL.get(track.GetNetname(), ()):
        fwd = (close(s.x, mm(a[0])) and close(s.y, mm(a[1]))
               and close(e.x, mm(b[0])) and close(e.y, mm(b[1])))
        rev = (close(s.x, mm(b[0])) and close(s.y, mm(b[1]))
               and close(e.x, mm(a[0])) and close(e.y, mm(a[1])))
        if fwd or rev:
            board.Remove(track)
            graveyard.append(track)
            removed += 1
            break
assert removed == 6, f"expected 6 removals, got {removed}"


def seg(net, layer, width, a, b):
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(pt(*a)); t.SetEnd(pt(*b))
    t.SetWidth(mm(width)); t.SetLayer(layer)
    t.SetNet(nets[net]); t.SetLocked(True)
    board.Add(t)


def via(net, x, y):
    v = pcbnew.PCB_VIA(board)
    v.SetPosition(pt(x, y))
    v.SetWidth(mm(0.4)); v.SetDrill(mm(0.25))
    v.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    v.SetNet(nets[net]); v.SetLocked(True)
    board.Add(v)


F = pcbnew.F_Cu
# CC1: dodge west of the R1/R2 pad column, then the same lower lane
seg("/CC1", F, 0.2, (149.49, 119.0), (148.6, 119.5))
seg("/CC1", F, 0.2, (148.6, 119.5), (148.6, 126.8))
seg("/CC1", F, 0.2, (148.6, 126.8), (162.75, 126.8))
# GND row pads: short north stubs to plane vias, clear of the NPTH pegs
seg("/GND", F, 0.25, (160.75, 123.705), (160.75, 122.3))
via("/GND", 160.75, 122.3)
seg("/GND", F, 0.25, (167.25, 123.705), (167.25, 122.3))
via("/GND", 167.25, 122.3)

# J1 silk: trim the nose lines back inside the tab edge (y_abs < 131.4)
fixed_silk = 0
for item in j1.GraphicalItems():
    if item.GetClass() != "PCB_SHAPE" or item.GetLayer() != pcbnew.F_SilkS:
        continue
    s, e = item.GetStart(), item.GetEnd()
    ymax = mm(131.15)
    if s.y > ymax and e.y > ymax:          # fully outside: the front bar
        board_parent = item.GetParentFootprint()
        j1.Remove(item)
        fixed_silk += 1
    elif e.y > ymax:
        item.SetEnd(pcbnew.VECTOR2I(e.x, ymax)); fixed_silk += 1
    elif s.y > ymax:
        item.SetStart(pcbnew.VECTOR2I(s.x, ymax)); fixed_silk += 1
print("silk items fixed:", fixed_silk)
assert fixed_silk == 3

filler = pcbnew.ZONE_FILLER(board)
filler.Fill(board.Zones())
pcbnew.SaveBoard(BOARD, board)
print("saved")
