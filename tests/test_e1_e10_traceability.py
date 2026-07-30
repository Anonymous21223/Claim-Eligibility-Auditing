import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = ROOT / "artifacts" / "audit" / "e1_e10" / "e1_e10_traceability_manifest.csv"
MANIFEST_JSON = ROOT / "artifacts" / "audit" / "e1_e10" / "e1_e10_traceability_manifest.json"
MANIFEST_MD = ROOT / "artifacts" / "audit" / "e1_e10" / "e1_e10_traceability_manifest.md"
TRACEABILITY_TABLE = ROOT / "paper" / "generated" / "table_audit_traceability.tex"
NUMERIC_CSV = ROOT / "artifacts" / "audit_records" / "numeric_consistency_report.csv"
PDF = ROOT / "paper_versions" / "v3_method_benchmark" / "source" / "fidelity_gated_xai_method_benchmark_v3.pdf"
PDF_SHA_FILE = ROOT / "submission" / "final_pdf_sha256.txt"
REPRODUCTION_LOG = ROOT / "submission" / "v3_method_reproduction_log.txt"
RUNNER = ROOT / "scripts" / "reproduce_v3_method_release.py"
ARTIFACT_MANIFEST_JSON = ROOT / "submission" / "v3_method_anonymous_artifact_manifest.json"
ARTIFACT_MANIFEST_CSV = ROOT / "submission" / "v3_method_anonymous_artifact_manifest.csv"
VENUE_CHECKLIST = ROOT / "submission" / "venue_compliance_checklist.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class E1E10TraceabilityTest(unittest.TestCase):
    def test_manifest_has_complete_pass_records(self) -> None:
        with MANIFEST_CSV.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["evidence_id"] for row in rows], [f"E{i}" for i in range(1, 11)])
        for row in rows:
            self.assertEqual(row["status"], "PASS", row)
            self.assertTrue(row["paper_claim"].strip())
            self.assertTrue(row["primary_artifact"].strip())
            self.assertTrue(row["reproduction_command"].strip())
            self.assertTrue(row["input_hash"].strip())
            self.assertTrue(row["output_hash"].strip())
            self.assertTrue(row["primary_artifact_sha256"].strip())
            self.assertEqual(row["missing_artifacts"], "NONE")
            self.assertTrue((ROOT / row["primary_artifact"]).exists(), row["primary_artifact"])
            for support in row["support_artifacts"].split("; "):
                self.assertTrue((ROOT / support).exists(), support)

    def test_json_manifest_and_pdf_hash_match_current_pdf(self) -> None:
        payload = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        current_pdf_hash = sha256(PDF)
        self.assertEqual(payload["schema"], "e1-e10-traceability-v1")
        self.assertEqual(payload["numeric_crosscheck_status"], "PASS")
        self.assertEqual(payload["pdf_sha256"], current_pdf_hash)
        self.assertEqual(PDF_SHA_FILE.read_text(encoding="utf-8").strip(), current_pdf_hash)
        self.assertEqual(len(payload["records"]), 10)

    def test_numeric_consistency_report_is_all_pass(self) -> None:
        with NUMERIC_CSV.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 40)
        self.assertTrue(all(row["status"] == "PASS" for row in rows))

    def test_submission_facing_traceability_files_are_renderable(self) -> None:
        markdown = MANIFEST_MD.read_text(encoding="utf-8")
        table = TRACEABILITY_TABLE.read_text(encoding="utf-8")
        log = REPRODUCTION_LOG.read_text(encoding="utf-8")
        self.assertIn("| E1 |", markdown)
        self.assertIn("| E10 |", markdown)
        self.assertIn("E1 &", table)
        self.assertIn("E10 &", table)
        self.assertIn("CHECKED", table)
        self.assertNotIn("VERIFIED", table)
        self.assertIn("Status: PASS", log)
        self.assertIn("python scripts/reproduce_v3_method_release.py", log)

    def test_release_sidecars_exist(self) -> None:
        self.assertTrue(RUNNER.exists())
        manifest = json.loads(ARTIFACT_MANIFEST_JSON.read_text(encoding="utf-8"))
        with ARTIFACT_MANIFEST_CSV.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(manifest["file_count"], len(rows))
        self.assertTrue(all(row["sha256"] and row["bytes"] for row in rows))
        self.assertIn("PASS_PUBLIC_GUIDELINES", VENUE_CHECKLIST.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
