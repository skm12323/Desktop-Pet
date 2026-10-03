"""Rebuild FINAL's six local turn packages from the existing approved video."""
from __future__ import annotations
import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import process_turn_clip as processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-root", type=Path, default=ROOT / "output/final_detail_fix_2026-10-02/turns")
    parser.add_argument("--heights", default="256,512,1024")
    args = parser.parse_args()
    pkg = ROOT / "assets/rig_final_walk_v1"
    raw = pkg / "clips/raw/wan3/wan3_1080_l240r200_s1.mp4"
    frames, fps = processor.decode(str(raw))
    print(f"Decoded {len(frames)} frames at {fps:.4f} fps", flush=True)
    original_decode = processor.decode
    original_argv = sys.argv
    processor.decode = lambda _: (frames, fps)
    try:
        for height in (int(v) for v in args.heights.split(",")):
            for reverse in (False, True):
                name = "turn_side_to_front" if reverse else "turn_front_to_side"
                if height != 512:
                    name += f"_h{height}"
                refs = [pkg / "references/front_rest.png", pkg / "references/side_rest.png"]
                if reverse:
                    refs.reverse()
                print(f"Building {name}", flush=True)
                sys.argv = ["process_turn_clip.py", str(raw),
                    "--endpoints", str(pkg / "clips/endpoints/front_to_side_l240r200"),
                    "--first-ref", str(refs[0]), "--last-ref", str(refs[1]),
                    "--out", str(pkg / "clips" / name), "--qa-dir", str(args.qa_root / name),
                    "--src-range", "0:84", "--duration", "0.6", "--motion-retime", "0.35",
                    "--morph-frames", "6", "--hold-frames", "3", "--height", str(height),
                    "--bidirectional-morph"]
                if reverse:
                    sys.argv.append("--reverse")
                processor.main()
                gc.collect()
    finally:
        processor.decode = original_decode
        sys.argv = original_argv


if __name__ == "__main__":
    main()
