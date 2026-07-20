import glob
import hashlib
import os
import statistics

import geopandas as gpd


def stats(d):
    hits = glob.glob(os.path.join(d, "**", "*.gpkg"), recursive=True)
    if not hits:
        return None
    g = gpd.read_file(max(hits, key=os.path.getsize), layer="stems",
                      engine="pyogrio")
    wkb = sorted(x.wkb_hex for x in g.geometry)
    vol = float(g["volume"].sum()) if "volume" in g.columns else None
    return {"stems": len(g),
            "vol_m3": round(vol, 2) if vol is not None else None,
            "md5": hashlib.md5("".join(wkb).encode()).hexdigest()[:12]}


times = {}
for line in open("/bench/timings.txt"):
    tag, n, rc, wall = line.split()
    times.setdefault(tag, []).append((int(n), int(rc), int(wall)))

print("=" * 64)
print("LEGACY (TensorFlow/.hdf5, upstream)  vs  NEW (ONNX/CUDA, fork)")
print("Spruce_Deadwood | 20220212_Barnekow_4.tiff | RTX 4080 SUPER")
print("=" * 64)

med = {}
for tag in ("legacy", "new"):
    runs = times.get(tag, [])
    if not runs:
        continue
    ok = [w for _, rc, w in runs if rc == 0]
    med[tag] = statistics.median(ok) if ok else None
    print("\n%s" % tag.upper())
    print("  wall (s) : %s   median %s"
          % ([w for _, _, w in runs], med[tag]))
    sigs = []
    for n, _, _ in runs:
        s = stats("/bench/%s_%d_out" % (tag, n))
        sigs.append(s)
        print("  run %d    : %s" % (n, s))
    good = [s for s in sigs if s]
    if len(good) >= 2:
        same = len({s["md5"] for s in good}) == 1
        print("  identical geometry across runs: %s"
              % ("YES" if same else "NO"))

if med.get("legacy") and med.get("new"):
    print("\nSPEEDUP: %.1fx  (%ss -> %ss)"
          % (med["legacy"] / med["new"], med["legacy"], med["new"]))
