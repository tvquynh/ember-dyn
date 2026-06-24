"""Compress *.jsonl → *.jsonl.zst with Zstandard (level 19).

zstandard level 19 = max compression that's still reasonable speed.
Expected ratio: 3-5x for JSON data.

After successful compression, removes the source .jsonl.
"""

import time
from pathlib import Path

try:
    import zstandard as zstd
except ImportError:
    print("ERROR: install python-zstandard:  pip install zstandard")
    raise SystemExit(1)

ROOT = Path(".")
SPLITS = ["train", "val", "test", "challenge"]

cctx = zstd.ZstdCompressor(level=19)

total_in = total_out = 0
for split in SPLITS:
    src = ROOT / "dataset" / "static" / f"{split}.jsonl"
    dst = ROOT / "dataset" / "static" / f"{split}.jsonl.zst"

    if not src.exists():
        print(f"  [SKIP] {src} not found")
        continue

    t0 = time.time()
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        cctx.copy_stream(fin, fout)
    elapsed = time.time() - t0

    in_sz = src.stat().st_size
    out_sz = dst.stat().st_size
    ratio = in_sz / out_sz if out_sz > 0 else 0
    total_in += in_sz
    total_out += out_sz

    print(f"  {split:9s}: {in_sz / 1024 / 1024:>7.1f} MB -> "
          f"{out_sz / 1024 / 1024:>6.1f} MB  "
          f"({ratio:.2f}x, {elapsed:.1f}s)")

    src.unlink()  # delete uncompressed source

print()
print(f"  TOTAL:   {total_in / 1024 / 1024:.1f} MB -> "
      f"{total_out / 1024 / 1024:.1f} MB  "
      f"({total_in / total_out:.2f}x compression)")
print(f"  Saved:   {(total_in - total_out) / 1024 / 1024:.1f} MB")
