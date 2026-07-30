"""Extract an explicit test report from a logged reproduction run."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="reports/reproduction_stderr.log")
    args = parser.parse_args()
    log = ROOT / args.log
    text = log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Ran (\d+) tests? in ([0-9.]+)s", text)
    passed = "OK" in text and "FAILED" not in text
    skipped = len(re.findall(r"\bskipped\b", text, re.I))
    reports = ROOT / "reports"; reports.mkdir(parents=True, exist_ok=True)
    (reports / "test_report.txt").write_text(text, encoding="utf-8")
    summary = "\n".join(["# Test Summary", "", f"- Status: `{'PASS' if passed else 'FAIL'}`", f"- Tests run: `{match.group(1) if match else 'unknown'}`", f"- Runtime seconds: `{match.group(2) if match else 'unknown'}`", f"- Skipped tests: `{skipped}`", f"- Source log: `{args.log}`", ""])
    (reports / "test_summary.md").write_text(summary, encoding="utf-8")
    if not passed or not match:
        raise AssertionError(summary)
    print(summary)


if __name__ == "__main__":
    main()
