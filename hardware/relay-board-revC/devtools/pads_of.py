import sys, pcbnew
b = pcbnew.LoadBoard(sys.argv[1])
want = set(sys.argv[2:])
for f in b.GetFootprints():
    if f.GetReference() in want:
        for p in f.Pads():
            q = p.GetPosition(); bb = p.GetBoundingBox()
            print(f.GetReference(), p.GetNumber(), round(pcbnew.ToMM(q.x),3), round(pcbnew.ToMM(q.y),3),
                  "%.2fx%.2f" % (pcbnew.ToMM(bb.GetWidth()), pcbnew.ToMM(bb.GetHeight())), p.GetNetname())
