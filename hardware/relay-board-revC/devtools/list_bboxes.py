import sys, pcbnew
b = pcbnew.LoadBoard(sys.argv[1])
x0,x1,y0,y1 = map(float, sys.argv[2:6])
for f in b.GetFootprints():
    bb = f.GetBoundingBox(False)
    l,r,t,bo = [pcbnew.ToMM(v) for v in (bb.GetLeft(), bb.GetRight(), bb.GetTop(), bb.GetBottom())]
    if r > x0 and l < x1 and bo > y0 and t < y1:
        print(f"{f.GetReference():5s} x {l:6.1f}-{r:6.1f}  y {t:6.1f}-{bo:6.1f}")
