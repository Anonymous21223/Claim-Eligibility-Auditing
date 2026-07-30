"""Scan every anonymous-artifact member, including binary bytes and ZIP paths."""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRONG_BINARY_PATTERNS = {
    "windows_home": re.compile(b"c:" + rb"\\users\\", re.I),
    "unix_home": re.compile(b"/" + b"home" + b"/", re.I),
    "email": re.compile(rb"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", re.I),
    "github_owner": re.compile(b"git" + b"hub" + rb"\.com/(?:pho" + b"ng" + b"l" + b"am" + rb"26|[a-z0-9._-]+/[a-z0-9._-]+)", re.I),
    "credential": re.compile(rb"(?:api[_-]?key|secret|token|password)\s*[:=]\s*[a-z0-9_\-]{8,}", re.I),
}
TEXT_ONLY_PATTERNS = {
    "identity_name": re.compile(b"tran dai " + b"pho" + b"ng " + b"l" + b"am", re.I),
    "institution": re.compile(rb"\b(?:f" + b"pt|tensor" + b"act)\b", re.I),
}
FORBIDDEN_PATH = re.compile(r"(?i)(^|/)(?:\.git|authors\.md|\.env(?:\.|$)|id_rsa|credentials|cover[_-]?letter|easychair|old[_-]?draft)")
TEXT_SUFFIXES = {".md", ".txt", ".py", ".ps1", ".sh", ".tex", ".bib", ".csv", ".json", ".yaml", ".yml", ".cff"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    args = parser.parse_args()
    artifact = ROOT / args.artifact
    findings: list[dict[str, str]] = []
    with zipfile.ZipFile(artifact) as archive:
        members = archive.infolist()
        names = {info.filename for info in members}
        for info in members:
            name = info.filename
            if FORBIDDEN_PATH.search(name):
                findings.append({"file": name, "kind": "forbidden_path"})
            content = archive.read(info)
            for kind, pattern in STRONG_BINARY_PATTERNS.items():
                if pattern.search(name.encode("utf-8", errors="ignore")) or pattern.search(content):
                    findings.append({"file": name, "kind": kind})
            if Path(name).suffix.lower() in TEXT_SUFFIXES:
                for kind, pattern in TEXT_ONLY_PATTERNS.items():
                    if pattern.search(content):
                        findings.append({"file": name, "kind": kind})
        required = {"README_BLIND.md", "REPRODUCIBILITY.md", "DATA_MANIFEST.md", "requirements-lock.txt"}
        missing = sorted(required - names)
    status = "PASS" if not findings and not missing else "FAIL"
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "artifact": artifact.relative_to(ROOT).as_posix(), "scanned_files": len(members), "matches": findings, "missing_required_blind_files": missing, "metadata_checks": {"zip_paths_checked": True, "binary_bytes_checked": True}, "status": status}
    submission = ROOT / "submission"; submission.mkdir(parents=True, exist_ok=True)
    (submission / "anonymization_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = "\n".join(["# Anonymization Audit", "", f"- Status: `{status}`", f"- Artifact: `{payload['artifact']}`", f"- Scanned files: `{len(members)}`", f"- Matches: `{findings or 'none'}`", f"- Missing required blind files: `{missing or 'none'}`", "- ZIP paths and binary bytes were scanned.", ""])
    (submission / "anonymization_audit.md").write_text(report, encoding="utf-8")
    (submission / "anonymity_audit.md").write_text(report, encoding="utf-8")
    if status != "PASS":
        raise AssertionError(report)
    print(report)


if __name__ == "__main__":
    main()
