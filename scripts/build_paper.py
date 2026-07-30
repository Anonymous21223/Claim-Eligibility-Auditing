"""Build the canonical blind manuscript and copy its immutable submission PDF."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "source"
FINAL = ROOT / "paper" / "final" / "ictai2026_paper_blind.pdf"
SUBMISSION = ROOT / "submission" / "ictai2026_paper_blind.pdf"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=SOURCE, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="ictai2026_blind", choices=("ictai2026_blind",))
    parser.parse_args()
    if not (SOURCE / "main.tex").exists():
        raise FileNotFoundError("Canonical source is missing: paper/source/main.tex")
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    run(["bibtex", "main"])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    pdf = SOURCE / "main.pdf"
    if not pdf.exists():
        raise AssertionError("LaTeX completed without producing main.pdf")
    (ROOT / "paper" / "build").mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, ROOT / "paper" / "build" / "ictai2026_paper_blind.pdf")
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, FINAL)
    shutil.copy2(pdf, SUBMISSION)
    digest = hashlib.sha256(FINAL.read_bytes()).hexdigest()
    (ROOT / "paper" / "final" / "ictai2026_paper_blind.sha256").write_text(digest + "\n", encoding="ascii")
    print(f"Built {SUBMISSION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
