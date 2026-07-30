"""Technical, layout, and anonymity QA for the canonical blind PDF."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATTERNS = [r"pho" + r"ng", r"l" + r"am", r"@", r"f" + r"pt", r"tensor" + r"act", r"git" + r"hub\.com", r"c:\\users", r"/" + r"home/", r"\b(?:todo|fixme)\b"]


def pdfinfo(pdf: Path) -> dict[str, str]:
    output = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    return {key.strip(): value.strip() for line in output.splitlines() if ":" in line for key, value in [line.split(":", 1)]}


def page_text(pdf: Path, page: int) -> str:
    return subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), str(pdf), "-"], check=True, capture_output=True).stdout.decode("utf-8", errors="replace")


def text_bounds(pdf: Path, width: float, height: float) -> dict[str, object]:
    raw = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"], check=True, capture_output=True).stdout
    root = ET.fromstring(raw)
    words = [node for node in root.iter() if node.tag.endswith("word")]
    outside = []
    for word in words:
        x0, y0, x1, y1 = (float(word.attrib[key]) for key in ("xMin", "yMin", "xMax", "yMax"))
        if x0 < -0.1 or y0 < -0.1 or x1 > width + 0.1 or y1 > height + 0.1:
            outside.append(word.text or "")
    return {"words_checked": len(words), "outside_page_bounds": outside[:20]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    args = parser.parse_args()
    pdf = ROOT / args.pdf
    info = pdfinfo(pdf)
    pages = int(info["Pages"])
    size_match = re.fullmatch(r"([0-9.]+) x ([0-9.]+) pts \(letter\)", info.get("Page size", ""), re.I)
    width, height = (float(size_match.group(1)), float(size_match.group(2))) if size_match else (0.0, 0.0)
    fonts = subprocess.run(["pdffonts", str(pdf)], check=True, capture_output=True, text=True).stdout.splitlines()
    font_rows = [line.split() for line in fonts[2:] if line.strip()]
    type3 = [row[0] for row in font_rows if len(row) >= 5 and " ".join(row[1:3]) == "Type 3"]
    not_embedded = [row[0] for row in font_rows if len(row) >= 5 and row[4].lower() != "yes"]
    all_text = "\n".join(page_text(pdf, page) for page in range(1, pages + 1))
    identity_matches = sorted({pattern for pattern in IDENTITY_PATTERNS if re.search(pattern, all_text, re.I)})
    metadata_identity = {field: info.get(field, "") for field in ("Author", "Creator", "Producer", "Title", "Subject", "Keywords") if re.search("|".join(IDENTITY_PATTERNS), info.get(field, ""), re.I)}
    source = (ROOT / "paper_versions" / "v3_method_benchmark" / "source" / "fidelity_gated_xai_method_benchmark_v3.tex").read_text(encoding="utf-8")
    bounds = text_bounds(pdf, width, height) if width and height else {"words_checked": 0, "outside_page_bounds": ["unable to parse Letter page size"]}
    final_reference_count = len(re.findall(r"\[(\d+)\]", page_text(pdf, pages)))
    checks = {
        "eight_or_fewer_pages": 1 <= pages <= 8,
        "us_letter": (width, height) == (612.0, 792.0),
        "ieee_two_column_source": "IEEEtran" in source and "conference" in source,
        "ten_point_source": "\\documentclass[conference]" in source,
        "all_fonts_embedded": not not_embedded,
        "type3_fonts_absent": not type3,
        "not_encrypted": info.get("Encrypted", "no").lower().startswith("no"),
        "author_metadata_empty": not info.get("Author", ""),
        "metadata_identity_absent": not metadata_identity,
        "text_identity_absent": not identity_matches,
        "text_inside_page_bounds": not bounds["outside_page_bounds"],
        "module_terminology": all(label in all_text for label in ("Module A", "Module B", "Module D")) and not any(label in all_text for label in ("Gate A", "Gate B1", "Gate B2")),
        "figure2_roles": "Required modules fail" in all_text and "E shows primary-tail RMSE" in all_text,
        "references_not_orphaned": final_reference_count >= 4,
        "undefined_references_absent": "??" not in all_text,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "pdf": pdf.relative_to(ROOT).as_posix(), "pdfinfo": info, "font_count": len(font_rows), "type3_fonts": type3, "not_embedded_fonts": not_embedded, "identity_text_patterns": identity_matches, "metadata_identity": metadata_identity, "layout_bounds": bounds, "last_page_reference_count": final_reference_count, "checks": checks, "status": status}
    submission = ROOT / "submission"; submission.mkdir(parents=True, exist_ok=True)
    (submission / "pdf_technical_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = ["# PDF Technical Audit", "", f"- Status: `{status}`", f"- PDF: `{pdf.relative_to(ROOT).as_posix()}`", f"- Pages: `{pages}`", f"- Paper size: `{info.get('Page size', 'missing')}`", f"- Fonts embedded: `{not not_embedded}`", f"- Type 3 fonts absent: `{not type3}`", f"- Encrypted: `{info.get('Encrypted', 'missing')}`", f"- Author metadata empty: `{not info.get('Author', '')}`", f"- Identity text patterns: `{identity_matches or 'none'}`", f"- Words outside page bounds: `{len(bounds['outside_page_bounds'])}`", f"- Final-page references: `{final_reference_count}`", ""]
    content = "\n".join(report)
    (submission / "pdf_technical_audit.md").write_text(content, encoding="utf-8")
    (submission / "pdf_technical_qa.md").write_text(content, encoding="utf-8")
    if status != "PASS":
        raise AssertionError(content)
    print(content)


if __name__ == "__main__":
    main()
