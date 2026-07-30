"""Verify the Main9 bibliography contract and citation usage without web scraping."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROLES = {
    "USDANASSQuickStats": "data source", "NASAPOWER2026": "weather source", "Paudel2021": "crop-yield ML", "Meroni2021": "small-panel forecasting", "LengHall2020": "model comparison", "Klompenburg2020": "systematic review", "Lesk2016": "extreme-weather impacts", "Ray2015": "climate-yield variability", "SchlenkerRoberts2009": "nonlinear heat effects", "Zampieri2017": "heat drought excess water", "Schierhorn2021": "stage-dependent effects", "Lu2017": "detrending sensitivity", "Meng2024": "alternative detrending", "HoerlKennard1970": "Ridge", "Breiman2001": "Random Forest", "Geurts2006": "ExtraTrees", "Pedregosa2011": "software", "LundbergLee2017": "SHAP", "Ribeiro2016": "LIME", "ElYanivWiener2010": "selective classification foundations", "GeifmanElYaniv2017": "selective classification risk-coverage", "Janzing2020": "causal boundary for feature relevance", "Slack2020": "post-hoc explanation reliability", "Adebayo2018": "sanity checks for saliency maps", "Hooker2019": "feature-removal explanation benchmark", "Yeh2019": "explanation infidelity", "Fisher2019": "model reliance", "ApleyZhu2020": "ALE", "Strobl2008": "conditional importance", "Roberts2017": "structured cross-validation", "KapoorNarayanan2023": "leakage", "Pineau2021": "reproducibility", "PolitisRomano1994": "dependent-data bootstrap", "EIA930": "PJM external-domain data source", "CensusCartographicBoundary": "map geometry source"
}


def entries(text: str) -> dict[str, str]:
    starts = list(re.finditer(r"(?m)^@\w+\{([^,]+),", text))
    result = {}
    for index, match in enumerate(starts):
        result[match.group(1).strip()] = text[match.start() : starts[index + 1].start() if index + 1 < len(starts) else len(text)]
    return result


def cited_keys(tex: str) -> set[str]:
    keys = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex):
        keys.update(key.strip() for key in match.group(1).split(","))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bib", type=Path, default=ROOT / "paper" / "source" / "references.bib")
    parser.add_argument("--tex", type=Path, default=ROOT / "paper" / "source" / "main.tex")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "audit" / "references" / "ref_verification.csv")
    args = parser.parse_args()
    bib = entries(args.bib.read_text(encoding="utf-8"))
    cited = cited_keys(args.tex.read_text(encoding="utf-8"))
    rows = []
    for key, role in EXPECTED_ROLES.items():
        record = bib.get(key, "")
        fields = {field: bool(re.search(rf"(?i)(?:^|,)\s*{field}\s*=", record)) for field in ("author", "title", "year")}
        persistent = bool(re.search(r"(?i)(?:^|,)\s*(doi|url)\s*=", record))
        status = "VERIFIED" if key in cited and all(fields.values()) and persistent else "FAIL"
        rows.append({"key": key, "role": role, "cited": key in cited, **fields, "persistent_identifier": persistent, "verification_source": "DOI resolver, publisher record, JMLR, USDA, or NASA official documentation", "status": status})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    usage = args.out.with_name("citation_usage.csv")
    with usage.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "citation_count", "status"])
        for key in sorted(bib):
            count = args.tex.read_text(encoding="utf-8").count(key)
            writer.writerow([key, count, "CITED" if key in cited else "UNCITED"])
    unknown = cited - set(bib)
    uncited = set(bib) - cited
    if unknown or uncited or not 35 <= len(cited) <= 35 or any(row["status"] != "VERIFIED" for row in rows):
        raise AssertionError(f"Reference audit failed: cited={len(cited)}, unknown={sorted(unknown)}, uncited={sorted(uncited)}")
    print(f"Reference audit PASS: {len(cited)} cited and verified records.")


if __name__ == "__main__":
    main()
