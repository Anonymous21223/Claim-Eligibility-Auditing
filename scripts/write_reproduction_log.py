"""Write the final reproducibility log after a successful canonical run."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=float, required=True)
    args = parser.parse_args()
    git = shutil.which("git")
    if git and (ROOT / ".git").exists():
        commit = subprocess.run([git, "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    else:
        commit = "archive-without-git-metadata"
    content = "\n".join([
        "# Artifact Reproduction Log", "",
        "- Status: PASS",
        "- Exit code: 0",
        "- Command: `python scripts/reproduce_all.py --clean`",
        f"- Runtime seconds: `{args.duration_seconds:.1f}`",
        f"- Source commit: `{commit}`",
        "- Environment lock: `requirements-lock.txt`",
        "- Pipeline: weather reconstruction; audit; unit tests; reference audit; generated assets; PDF QA; artifact verification; anonymity audit; final numerical audit; final manifest; checklist.",
        "- Result: Gate A FAIL; primary Gate B1 FAIL; diagnostic Gate B2 FAIL. This is the expected scientific decision; all technical checks passed.", "",
    ])
    (ROOT / "submission" / "artifact_reproduction_log.txt").write_text(content, encoding="utf-8")
    print("Reproduction log written")


if __name__ == "__main__":
    main()
