"""Block-wise pixel comparison of two stem-map rasters.

    python raster_diff.py a.tif b.tif

Gate (ii) asks for a bit-identical raster. File checksums do not answer that --
two runs differed by 28 bytes of TIFF metadata while their pixels were nearly
identical -- so compare pixels, block by block, and report the count rather than
a boolean. The count is the interesting number: it distinguishes "no change"
from "a few hundred pixels near the 0.5 threshold flipped" from "broken".

Reads block-aligned so a 25-gigapixel raster never lands in RAM.
"""
import sys

import numpy as np
import rasterio


def main(a_path: str, b_path: str) -> int:
    with rasterio.open(a_path) as a, rasterio.open(b_path) as b:
        if (a.shape, a.count) != (b.shape, b.count):
            print(f"SHAPE MISMATCH {a.shape}x{a.count} vs {b.shape}x{b.count}")
            return 2
        differing = total = nonzero = 0
        for _, window in a.block_windows(1):
            x = a.read(window=window)
            y = b.read(window=window)
            differing += int((x != y).sum())
            total += x.size
            nonzero += int((x != 0).sum())
    print(f"shape {a.shape} bands {a.count}")
    print(f"differing px {differing:,} of {total:,} ({100 * differing / total:.6f}%)")
    print(f"nonzero px in A {nonzero:,}"
          + (f" -- differing = {100 * differing / nonzero:.4f}% of those" if nonzero else ""))
    return 0 if differing == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
