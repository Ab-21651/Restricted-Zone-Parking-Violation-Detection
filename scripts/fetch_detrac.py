"""Download a frame sample of selected UA-DETRAC sequences from Kaggle, file by file.

Kaggle's website cannot download single folders, but its API can download single
files. Consecutive frames are near-identical, so every Nth *annotated* frame is
enough for detection evaluation.

Needs Kaggle credentials (run `kaggle auth login` once, or put your API token in
%USERPROFILE%\\.kaggle\\access_token). Safe to re-run: existing files are skipped.

    python scripts/fetch_detrac.py                    # default 8 sequences, every 10th frame
    python scripts/fetch_detrac.py --step 20 --seqs MVI_39051 MVI_40773
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

DATASET = "bratjay/ua-detrac-orig"
REMOTE_DIR = "DETRAC-Images/DETRAC-Images"
DEFAULT_SEQS = [
    "MVI_39051", "MVI_39211",  # sunny
    "MVI_39501", "MVI_40711",  # cloudy
    "MVI_40773", "MVI_40774",  # night
    "MVI_40902", "MVI_40904",  # rainy
]
ANN_DIRS = [Path("data/detrac/annotations/DETRAC-Test-Annotations-XML"),
            Path("data/detrac/annotations/DETRAC-Train-Annotations-XML")]


def annotated_frames(seq: str) -> list[int]:
    for d in ANN_DIRS:
        f = d / f"{seq}.xml"
        if f.exists():
            return [int(fr.get("num")) for fr in ET.parse(f).getroot().iter("frame")]
    raise FileNotFoundError(f"No annotation XML for {seq}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqs", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--step", type=int, default=10, help="Keep every Nth annotated frame")
    ap.add_argument("--out", default="data/detrac/images")
    ap.add_argument("--retries", type=int, default=6)
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi  # import triggers auth check

    api = KaggleApi()
    api.authenticate()

    plan = [(s, n) for s in args.seqs for n in annotated_frames(s)[:: args.step]]
    print(f"{len(plan)} frames from {len(args.seqs)} sequences (every {args.step}th annotated frame)")
    done = failed = 0
    for i, (seq, num) in enumerate(plan, 1):
        name = f"img{num:05d}.jpg"
        dest_dir = Path(args.out) / seq
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            done += 1
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, args.retries + 1):
            try:
                api.dataset_download_file(DATASET, f"{REMOTE_DIR}/{seq}/{name}", path=str(dest_dir), quiet=True)
                z = dest_dir / f"{name}.zip"
                if z.exists():  # Kaggle sometimes wraps single files in a zip
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(dest_dir)
                    z.unlink()
                if dest.exists():
                    done += 1
                    break
                raise RuntimeError("file missing after download")
            except Exception as e:  # network resets are common; back off and retry
                if attempt == args.retries:
                    failed += 1
                    print(f"  FAILED {seq}/{name}: {e}")
                else:
                    time.sleep(min(60, 2 ** attempt))
        if i % 25 == 0 or i == len(plan):
            print(f"  {i}/{len(plan)}  ok={done}  failed={failed}", flush=True)
    print(f"Done: {done} ok, {failed} failed -> {args.out}  (re-run to retry failures)")


if __name__ == "__main__":
    main()
