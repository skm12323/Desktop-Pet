"""Rebuild only the front ADULT eye meshes with shared, sclera-covering blink zones."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mesh_generator import generate_layer_mesh

# Registered on the existing head texture. Bounds contain the complete sclera,
# including the left edge of the right eye omitted by the old x=517/r=24 zone.
ZONES = {"l": [404, 316, 363, 33], "r": [505, 310, 356, 34]}
EYE_LAYERS = {"head_base", "pupil_l", "pupil_r", "eyelid_l", "eyelid_r"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="assets/rig_adult")
    args = ap.parse_args()
    pkg = Path(args.package)
    spec = json.loads((pkg / "spec.json").read_text(encoding="utf-8"))
    mesh = json.loads((pkg / "mesh/mesh_data.json").read_text(encoding="utf-8"))
    size = spec["skeleton"]["source_reference"]["image_size_px"]
    changed = {}
    for layer in spec["layers"]:
        if layer["id"] not in EYE_LAYERS:
            continue
        lid = layer["id"]
        layer["blink_zones"] = [ZONES["l"], ZONES["r"]] if lid == "head_base" else [ZONES[lid[-1]]]
        changed[lid] = generate_layer_mesh(layer, spec["skeleton"], str(pkg / "layers" / f"{lid}.png"), tuple(size))
        assert changed[lid] is not None
    mesh["layers"] = [changed.get(layer["id"], layer) for layer in mesh["layers"]]
    (pkg / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (pkg / "mesh/mesh_data.json").write_text(json.dumps(mesh, ensure_ascii=False, indent=2), encoding="utf-8")
    print({key: len(value["vertices"]) for key, value in changed.items()})


if __name__ == "__main__":
    main()
