"""Build all cutoff-specific stage definitions and feature manifests."""

from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crop_yield_xai.extension_stage_features import (  # noqa: E402
    build_stage_frame,
    derive_stage_definitions,
)


OUT = ROOT / "data" / "derived" / "stage_features"
SCHEMA_VERSION = 2


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def completed_manifest(target: Path) -> dict[str, object] | None:
    manifest_path = target / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return None
    for name, metadata in manifest.get("files", {}).items():
        path = target / name
        if not path.exists() or digest(path) != metadata.get("sha256"):
            return None
    return manifest


def main() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "extension_factorial_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    cutoffs = sorted(
        {
            int(fold["train_end"])
            for fold in protocol["outer_folds"]
        }
        | {
            int(inner["train_end"])
            for fold in protocol["outer_folds"]
            for inner in fold["inner"]
        }
    )
    manifests = []
    for cutoff in cutoffs:
        target = OUT / f"cutoff_{cutoff}"
        existing = completed_manifest(target)
        if existing is not None:
            manifests.append(existing)
            print(f"cutoff {cutoff}: PASS (verified existing)")
            continue
        definitions = derive_stage_definitions(ROOT, cutoff)
        target.mkdir(parents=True, exist_ok=True)
        paths = {
            "calendar_definitions.csv": definitions.calendar,
            "gdd_definitions.csv": definitions.gdd,
            "source_coverage.csv": definitions.coverage,
            "stage_features.csv": build_stage_frame(ROOT, definitions),
        }
        for name, frame in paths.items():
            frame.to_csv(target / name, index=False)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "cutoff_year": cutoff,
            "status": "PASS",
            "files": {
                name: {
                    "sha256": digest(target / name),
                    "bytes": (target / name).stat().st_size,
                    "rows": len(frame),
                }
                for name, frame in paths.items()
            },
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        manifests.append(manifest)
        print(f"cutoff {cutoff}: PASS")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cutoff_registry.json").write_text(
        json.dumps(manifests, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cutoffs": cutoffs, "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
