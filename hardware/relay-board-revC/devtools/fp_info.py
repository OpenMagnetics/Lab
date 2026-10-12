import os
import sys, pcbnew
for fid in sys.argv[1:]:
    lib, name = fid.split(":")
    base = r"C:\Program Files\KiCad\10.0\share\kicad\footprints" if lib != "OpenMagnetics" else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    fp = pcbnew.FootprintLoad(base + "\\" + lib + ".pretty", name)
    cy = fp.GetCourtyard(pcbnew.F_CrtYd).BBox() if hasattr(fp,"GetCourtyard") else None
    bb = fp.GetBoundingBox(False)
    print(fid, "bbox x %.2f..%.2f y %.2f..%.2f" % tuple(pcbnew.ToMM(v) for v in (bb.GetLeft(), bb.GetRight(), bb.GetTop(), bb.GetBottom())))
    seen=set()
    for p in fp.Pads():
        q=p.GetPosition(); s=p.GetSize(pcbnew.F_Cu)
        k=(p.GetNumber(),round(pcbnew.ToMM(q.x),2),round(pcbnew.ToMM(q.y),2))
        if k in seen: continue
        seen.add(k); print("   pad", k, "size %.2fx%.2f" % (pcbnew.ToMM(s.x),pcbnew.ToMM(s.y)), "PTH" if p.GetAttribute()==pcbnew.PAD_ATTRIB_PTH else "")
