"""Create and verify a hash manifest for the canonical audit evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files() -> list[Path]:
    required = [
        ROOT / "configs" / "fidelity_gate.yaml",
        ROOT / "artifacts" / "audit_records" / "fidelity_gate_components.csv",
        ROOT / "artifacts" / "audit_records" / "topk_null_audit.csv",
        ROOT / "artifacts" / "audit_records" / "rank_null_audit.csv",
        ROOT / "artifacts" / "audit_records" / "target_scale_sensitivity.csv",
        ROOT / "artifacts" / "audit_records" / "alternative_detrending_sensitivity.csv",
        ROOT / "artifacts" / "audit_records" / "temporal_and_capacity_audits.csv",
        ROOT / "artifacts" / "audit_records" / "expanded_model_baselines.csv",
        ROOT / "artifacts" / "audit_records" / "group_macro_metrics.csv",
        ROOT / "artifacts" / "audit" / "final_test" / "seed_aggregated_predictions.csv",
        ROOT / "artifacts" / "data" / "data_flow.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required audit evidence: " + ", ".join(missing))
    return required


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/audit_manifest.json")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    path = ROOT / args.manifest
    required = files()
    observed = {str(item.relative_to(ROOT)).replace("\\", "/"): sha256(item) for item in required}
    if args.verify_only:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["files"] != observed:
            raise AssertionError("Artifact hash manifest differs from current evidence")
        print("Artifact manifest verified")
        return
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    payload = {"schema": "ictai2026-audit-manifest-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "files": observed}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Artifact manifest written: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
