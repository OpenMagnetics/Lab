import sys, json, pcbnew
b = pcbnew.LoadBoard(sys.argv[1]); mm = pcbnew.ToMM
d = {"tracks": [], "vias": [], "pads": [], "refs": [], "boxes": []}
for t in b.GetTracks():
    if t.GetClass() == "PCB_VIA":
        p = t.GetPosition(); d["vias"].append([mm(p.x), mm(p.y), mm(t.GetWidth(pcbnew.F_Cu)), t.GetNetname()])
    else:
        s, e = t.GetStart(), t.GetEnd()
        d["tracks"].append([mm(s.x), mm(s.y), mm(e.x), mm(e.y), mm(t.GetWidth()), b.GetLayerName(t.GetLayer()), t.GetNetname(), bool(t.IsLocked())])
for f in b.GetFootprints():
    p = f.GetPosition(); d["refs"].append([f.GetReference(), mm(p.x), mm(p.y)])
    try:
        cy = f.GetCourtyard(pcbnew.F_CrtYd).BBox()
        d["boxes"].append([mm(cy.GetLeft()), mm(cy.GetTop()), mm(cy.GetWidth()), mm(cy.GetHeight())])
    except Exception:
        pass
    for pad in f.Pads():
        bb = pad.GetBoundingBox()
        d["pads"].append([mm(bb.GetLeft()), mm(bb.GetTop()), mm(bb.GetWidth()), mm(bb.GetHeight()), pad.GetNetname(), pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH])
json.dump(d, open(sys.argv[2], "w"))
