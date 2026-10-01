"""Fix two netlist-level bugs on the fabrication master:

1. D2 USBLC6-2SC6 is a flow-through ESD array (pins 1&6 = I/O1, 3&4 =
   I/O2 internally tied). The old wiring shorted D+ to D- on both sides.
   New: 1=USB_DP_CON, 6=USB_DP, 3=USB_DM_CON, 4=USB_DM.
2. D1 LED was reverse-biased (LED_A on the cathode pad). New: pad 2
   (anode) = LED_A, pad 1 (cathode) = GND.

Matches the regenerated schematic/netlist. Backup saved first.
"""

import shutil

import pcbnew

BOARD = "relay_board.kicad_pcb"
shutil.copy(BOARD, "relay_board_backup_pinfix.kicad_pcb")
board = pcbnew.LoadBoard(BOARD)
mm = pcbnew.FromMM
F, B = pcbnew.F_Cu, pcbnew.B_Cu


def pt(x, y):
    return pcbnew.VECTOR2I(mm(x), mm(y))


def close(a, b):
    return abs(a - b) < mm(0.01)


nets = {}
pads = {}
for fp in board.GetFootprints():
    for pad in fp.Pads():
        nets.setdefault(pad.GetNetname(), pad.GetNet())
        pads[(fp.GetReference(), pad.GetName())] = pad
graveyard = []

# ---- pad net swaps ----------------------------------------------------
pads[("D2", "3")].SetNet(nets["/USB_DM_CON"])
pads[("D2", "6")].SetNet(nets["/USB_DP"])
pads[("D1", "1")].SetNet(nets["/GND"])
pads[("D1", "2")].SetNet(nets["/LED_A"])

# ---- deletions --------------------------------------------------------
DEL = {
    "/USB_DM_CON": [((155.137, 121.55), (156.3, 121.55)),
                    ((156.3, 121.55), (164.25, 121.55)),
                    ((164.25, 121.55), (164.25, 123.705))],
    "/USB_DP": [((152.863, 123.451), (153.71, 124.298)),
                ((153.71, 124.298), (153.71, 124.512)),
                ((153.71, 124.512), (152.075, 122.876)),
                ((152.075, 122.876), (152.075, 117.909))],
    "/LED_A": [((155.51, 51.703), (154.212, 53.0)),
               ((154.213, 53.0), (151.15, 53.0))],
}
DEL_VIAS = {
    "/USB_DM_CON": [(156.3, 121.55), (164.25, 121.55)],
    "/USB_DP": [(153.71, 124.512)],
}
removed = 0
for track in list(board.GetTracks()):
    net = track.GetNetname()
    if track.Type() == pcbnew.PCB_VIA_T:
        for (x, y) in DEL_VIAS.get(net, ()):
            p = track.GetPosition()
            if close(p.x, mm(x)) and close(p.y, mm(y)):
                board.Remove(track); graveyard.append(track); removed += 1
                break
    else:
        s, e = track.GetStart(), track.GetEnd()
        hit = False
        for (a, b) in DEL.get(net, ()):
            if ((close(s.x, mm(a[0])) and close(s.y, mm(a[1]))
                 and close(e.x, mm(b[0])) and close(e.y, mm(b[1]))) or
                (close(s.x, mm(b[0])) and close(s.y, mm(b[1]))
                 and close(e.x, mm(a[0])) and close(e.y, mm(a[1])))):
                hit = True
        # GND crumb stack + stub feeding old D1 pad 2
        if (net == "/GND" and track.GetLayer() == F
                and mm(155.7) < s.x < mm(155.9) and mm(52.9) < s.y < mm(54.5)
                and mm(155.7) < e.x < mm(155.9) and mm(52.9) < e.y < mm(54.5)):
            hit = True
        if hit:
            board.Remove(track); graveyard.append(track); removed += 1
print("removed", removed, "items")
assert removed >= 10, "too few removals"


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
    v.SetLayerPair(F, B)
    v.SetNet(nets[net]); v.SetLocked(True)
    board.Add(v)


# ---- D2 rework --------------------------------------------------------
# USB_DP: MCU meander now enters at pad 6
seg("/USB_DP", B, 0.2, (152.075, 117.909), (152.075, 121.55))
seg("/USB_DP", B, 0.2, (152.075, 121.55), (154.6, 121.55))
via("/USB_DP", 154.6, 121.55)
seg("/USB_DP", F, 0.2, (154.6, 121.55), (155.137, 121.55))
# USB_DM_CON: connector-bound line now leaves from pad 3, joins the
# southside tie via at J1 (163.25, 125.75)
seg("/USB_DM_CON", F, 0.2, (152.863, 123.45), (152.6, 123.9))
via("/USB_DM_CON", 152.6, 123.9)
seg("/USB_DM_CON", B, 0.2, (152.6, 123.9), (163.25, 123.9))
seg("/USB_DM_CON", B, 0.15, (163.25, 123.9), (163.25, 125.75))

# ---- D1 rework --------------------------------------------------------
# LED_A: R10 branch bends into pad 2 (anode); PC13 loop joins that branch
seg("/LED_A", F, 0.2, (155.51, 51.703), (155.788, 52.7))
seg("/LED_A", F, 0.2, (155.788, 52.7), (155.788, 53.0))
seg("/LED_A", F, 0.2, (151.15, 53.0), (151.9, 52.15))
seg("/LED_A", F, 0.2, (151.9, 52.15), (155.06, 52.15))
seg("/LED_A", F, 0.2, (155.06, 52.15), (155.51, 51.703))
# GND: pad 1 (cathode) to the kept plane via at (155.787, 54.4)
seg("/GND", F, 0.2, (154.212, 53.0), (154.212, 54.4))
seg("/GND", F, 0.2, (154.212, 54.4), (155.787, 54.4))

pcbnew.ZONE_FILLER(board).Fill(board.Zones())
pcbnew.SaveBoard(BOARD, board)
print("saved")
