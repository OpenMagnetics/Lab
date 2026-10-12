import sys, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
d = json.load(open(sys.argv[1])); out = sys.argv[2]; x0, x1, y0, y1 = map(float, sys.argv[3:7])
layers = sys.argv[7].split(",") if len(sys.argv) > 7 else None
col = {"F.Cu": "#d03030", "B.Cu": "#3050d0", "In1.Cu": "#c0a000", "In2.Cu": "#20a020", "In3.Cu": "#e08000", "In4.Cu": "#a000a0"}
W = 18; fig, ax = plt.subplots(figsize=(W, W * (y1 - y0) / (x1 - x0)))
ppm = W * 72 / (x1 - x0)
for l, t, w, h in d.get("boxes", []):
    ax.add_patch(Rectangle((l, t), w, h, fill=False, ec="#999999", lw=0.6, ls="--"))
for l, t, w, h, net, pth in d["pads"]:
    ax.add_patch(Rectangle((l, t), w, h, color="#505050" if pth else "#909090", alpha=0.6))
for xa, ya, xb, yb, w, layer, net, locked in d["tracks"]:
    if layers and layer not in layers: continue
    ax.plot([xa, xb], [ya, yb], color=col.get(layer, "gray"), lw=max(0.4, w * ppm), alpha=0.55, solid_capstyle="round")
for x, y, w, net in d["vias"]:
    ax.add_patch(Circle((x, y), w / 2, color="#00a0a0" if net == "/GND" else "k", alpha=0.7))
for r, x, y in d["refs"]:
    if x0 <= x <= x1 and y0 <= y <= y1: ax.text(x, y, r, fontsize=8, ha="center", va="center", color="purple", weight="bold")
ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal")
ax.set_xticks(range(int(x0), int(x1) + 1, 5)); ax.set_yticks(range(int(y0), int(y1) + 1, 5)); ax.grid(alpha=0.2)
ax.tick_params(labelsize=7)
fig.savefig(out, dpi=70, bbox_inches="tight")
