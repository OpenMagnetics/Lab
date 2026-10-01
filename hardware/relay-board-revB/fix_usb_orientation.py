"""One-shot surgical fix: J1 USB-C was placed rotated 180 deg (opening
facing the board interior). Rotate to 0, reposition so the mating face is
flush with the tab edge (y=131.4), rip the connector-side copper and
re-route to the mirrored pad positions. Runs on the fabrication master.

    "C:\\Program Files\\KiCad\\9.0\\bin\\python.exe" fix_usb_orientation.py
"""

import shutil

import pcbnew

BOARD = "relay_board.kicad_pcb"
shutil.copy(BOARD, "relay_board_backup_usbfix.kicad_pcb")

board = pcbnew.LoadBoard(BOARD)
mm = pcbnew.FromMM


def pt(x, y):
    return pcbnew.VECTOR2I(mm(x), mm(y))


def close(a, b):
    return abs(a - b) < mm(0.01)


# grab handles BEFORE any Remove(): removals invalidate swig wrappers
j1 = next(fp for fp in board.GetFootprints()
          if fp.GetReference() == "J1")
nets = {}
for fp in board.GetFootprints():
    for pad in fp.Pads():
        nets.setdefault(pad.GetNetname(), pad.GetNet())
graveyard = []   # keep removed tracks alive so GC never touches them

# ---- 1. delete the old connector-side copper -------------------------
DEL_SEGS = {
    "/CC1": [((149.49, 119), (149.49, 120.25)),
             ((149.49, 120.25), (165.25, 120.25)),
             ((165.25, 120.25), (165.25, 130.345))],
    "/CC2": [((149.49, 122), (149.49, 127)),
             ((149.49, 127), (162.25, 127)),
             ((162.25, 127), (162.25, 130.345))],
    "/USB_DP_CON": [((164.25, 128.75), (163.25, 128.75)),
                    ((163.25, 128.75), (163.25, 130.345)),
                    ((164.25, 130.345), (164.25, 128.75)),
                    ((163.25, 120.9), (163.25, 128.75))],
    "/USB_DM_CON": [((163.75, 130.345), (163.75, 129.3)),
                    ((164.75, 130.345), (164.75, 129.3)),
                    ((163.75, 129.3), (164.75, 129.3)),
                    ((164.75, 121.55), (164.75, 129.3)),
                    ((156.3, 121.55), (164.75, 121.55))],
    "/VBUS": [((161.55, 130.345), (161.55, 129.95)),
              ((161.55, 129.95), (161.5, 129.95)),
              ((166.45, 129.95), (166.5, 129.95)),
              ((161.5, 129.95), (161.5, 129.7)),
              ((166.5, 129.95), (166.5, 129.7)),
              ((166.45, 130.345), (166.45, 129.95)),
              ((162.0, 123.0), (156.4, 123.0)),
              ((162.0, 130.9), (162.0, 123.0)),
              ((166.5, 129.7), (166.5, 130.9)),
              ((161.5, 129.7), (161.5, 130.9)),
              ((161.5, 130.9), (166.5, 130.9))],
    "/GND": [((167.25, 130.345), (167.5, 130.75)),
             ((160.75, 130.345), (160.5, 130.75))],
}
DEL_VIAS = {
    "/USB_DM_CON": [(163.75, 129.3), (164.75, 129.3)],
    "/VBUS": [(161.5, 129.7), (166.5, 129.7)],
    "/GND": [(160.5, 130.75), (167.5, 130.75)],
}

removed = 0
for track in list(board.GetTracks()):
    net = track.GetNetname()
    if track.Type() == pcbnew.PCB_VIA_T:
        for (x, y) in DEL_VIAS.get(net, ()):
            p = track.GetPosition()
            if close(p.x, mm(x)) and close(p.y, mm(y)):
                board.Remove(track)
                graveyard.append(track)
                removed += 1
                break
    else:
        s, e = track.GetStart(), track.GetEnd()
        for (a, b) in DEL_SEGS.get(net, ()):
            fwd = (close(s.x, mm(a[0])) and close(s.y, mm(a[1]))
                   and close(e.x, mm(b[0])) and close(e.y, mm(b[1])))
            rev = (close(s.x, mm(b[0])) and close(s.y, mm(b[1]))
                   and close(e.x, mm(a[0])) and close(e.y, mm(a[1])))
            if fwd or rev:
                board.Remove(track)
                graveyard.append(track)
                removed += 1
                break
expected = sum(len(v) for v in DEL_SEGS.values()) + sum(
    len(v) for v in DEL_VIAS.values())
print(f"removed {removed}/{expected} old items")
assert removed == expected, "deletion list mismatch -- aborting"

# ---- 2. move the connector --------------------------------------------
j1.SetOrientationDegrees(0)
j1.SetPosition(pt(164.0, 127.75))
# sanity: CC1 pad A5 must land at (162.75, 123.705)
for pad in j1.Pads():
    if pad.GetName() == "A5":
        p = pad.GetPosition()
        assert close(p.x, mm(162.75)) and close(p.y, mm(123.705)), \
            f"A5 at ({p.x},{p.y}) -- orientation convention wrong"
print("J1 moved: rot 0 at (164, 127.75), nose flush with tab edge y=131.4")

def seg(net, layer, width, a, b, locked=True):
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(pt(*a)); t.SetEnd(pt(*b))
    t.SetWidth(mm(width))
    t.SetLayer(layer)
    t.SetNet(nets[net])
    t.SetLocked(locked)
    board.Add(t)


def via(net, x, y):
    v = pcbnew.PCB_VIA(board)
    v.SetPosition(pt(x, y))
    v.SetWidth(mm(0.4)); v.SetDrill(mm(0.25))
    v.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    v.SetNet(nets[net])
    v.SetLocked(True)
    board.Add(v)


F, B = pcbnew.F_Cu, pcbnew.B_Cu

# ---- 3. new routing ---------------------------------------------------
# CC1: R1 south, then east below the diff-pair area, up into pad A5
seg("/CC1", F, 0.2, (149.49, 119.0), (149.49, 126.8))
seg("/CC1", F, 0.2, (149.49, 126.8), (162.75, 126.8))
seg("/CC1", F, 0.2, (162.75, 126.8), (162.75, 123.705))
# CC2: R2 west-south, lower lane, up into pad B5
seg("/CC2", F, 0.2, (149.49, 122.0), (149.0, 122.51))
seg("/CC2", F, 0.2, (149.0, 122.51), (149.0, 127.4))
seg("/CC2", F, 0.2, (149.0, 127.4), (165.75, 127.4))
seg("/CC2", F, 0.2, (165.75, 127.4), (165.75, 123.705))
# DP_CON: extend the F lane to x=163.75, drop into A6, tie B6 southside
seg("/USB_DP_CON", F, 0.25, (163.25, 120.9), (163.75, 120.9))
seg("/USB_DP_CON", F, 0.2, (163.75, 120.9), (163.75, 123.705))
seg("/USB_DP_CON", F, 0.15, (163.75, 123.705), (163.75, 124.95))
via("/USB_DP_CON", 163.75, 124.95)
seg("/USB_DP_CON", B, 0.15, (163.75, 124.95), (164.75, 124.95))
via("/USB_DP_CON", 164.75, 124.95)
seg("/USB_DP_CON", F, 0.15, (164.75, 124.95), (164.75, 123.705))
# DM_CON: B lane to x=164.25, via, drop into A7, tie B7 southside
seg("/USB_DM_CON", B, 0.25, (156.3, 121.55), (164.25, 121.55))
via("/USB_DM_CON", 164.25, 121.55)
seg("/USB_DM_CON", F, 0.2, (164.25, 121.55), (164.25, 123.705))
seg("/USB_DM_CON", F, 0.15, (164.25, 123.705), (164.25, 125.75))
via("/USB_DM_CON", 164.25, 125.75)
seg("/USB_DM_CON", B, 0.15, (164.25, 125.75), (163.25, 125.75))
via("/USB_DM_CON", 163.25, 125.75)
seg("/USB_DM_CON", F, 0.15, (163.25, 125.75), (163.25, 123.705))
# VBUS: B lane along y=123 under the row, rise into both wide pads
seg("/VBUS", B, 0.25, (156.4, 123.0), (161.55, 123.0))
seg("/VBUS", B, 0.25, (161.55, 123.0), (161.55, 122.5))
via("/VBUS", 161.55, 122.5)
seg("/VBUS", F, 0.25, (161.55, 122.5), (161.55, 123.705))
seg("/VBUS", B, 0.25, (161.55, 123.0), (166.45, 123.0))
seg("/VBUS", B, 0.25, (166.45, 123.0), (166.45, 122.5))
via("/VBUS", 166.45, 122.5)
seg("/VBUS", F, 0.25, (166.45, 122.5), (166.45, 123.705))
# GND row pads tie straight to the (through-hole, plane-fed) shield pads
seg("/GND", F, 0.25, (160.75, 123.705), (160.75, 124.62))
seg("/GND", F, 0.25, (160.75, 124.62), (160.18, 124.62))
seg("/GND", F, 0.25, (167.25, 123.705), (167.25, 124.62))
seg("/GND", F, 0.25, (167.25, 124.62), (167.82, 124.62))
print("new routing added")

# ---- 4. refill zones, save -------------------------------------------
filler = pcbnew.ZONE_FILLER(board)
filler.Fill(board.Zones())
pcbnew.SaveBoard(BOARD, board)
print("saved", BOARD)
