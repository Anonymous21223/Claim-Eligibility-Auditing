"""Scan the working tree and reachable Git history for credential-like strings."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "github_pat": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "generic_api_key": re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
}
EXCLUDED_PARTS = {".git", ".venv", "artifacts", "build", "dist", "tmp"}


def scan_text(label: str, text: str) -> list[dict[str, str]]:
    return [{"location": label, "pattern": name} for name, pattern in PATTERNS.items() if pattern.search(text)]


def main() -> None:
    findings: list[dict[str, str]] = []
    scanned = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() not in {".py", ".ps1", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".cff"}:
            continue
        scanned += 1
        findings.extend(scan_text(path.relative_to(ROOT).as_posix(), path.read_text(encoding="utf-8", errors="ignore")))
    history = subprocess.run(["git", "log", "-p", "--all", "--no-ext-diff"], cwd=ROOT, capture_output=True, text=True, errors="replace")
    if history.returncode == 0:
        findings.extend(scan_text("reachable_git_history", history.stdout))
    status = "PASS" if not findings else "FAIL"
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "scanned_text_files": scanned, "git_history_scanned": history.returncode == 0, "findings": findings, "status": status}
    reports = ROOT / "reports"; reports.mkdir(parents=True, exist_ok=True)
    (reports / "public_repo_secret_scan.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = "\n".join(["# Public Repository Secret Scan", "", f"- Status: `{status}`", f"- Text files scanned: `{scanned}`", f"- Reachable Git history scanned: `{history.returncode == 0}`", f"- Findings: `{findings or 'none'}`", ""])
    (reports / "public_repo_secret_scan.md").write_text(report, encoding="utf-8")
    if status != "PASS":
        raise AssertionError(report)
    print(report)


if __name__ == "__main__":
    main()
