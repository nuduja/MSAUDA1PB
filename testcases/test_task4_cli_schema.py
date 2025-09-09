import json, os, sys, tempfile, subprocess, pathlib, unittest
from typing import List, Dict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SYS_PY = sys.executable

class TestTask4CLISchema(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self.tmpdir.name)

        # Tiny demo corpus
        self.docs_path = self.tmp / "docs.jsonl"
        self.queries_path = self.tmp / "queries.json"
        self.run_path = self.tmp / "run_task4_cli.json"

        # docs:
        # 10: climate change ... policy
        # 20: machine learning ...
        # 30: climate policy research climate science
        docs = [
            {"id": 10, "text": "climate change affects global policy"},
            {"id": 20, "text": "machine learning algorithms improve models"},
            {"id": 30, "text": "climate policy research climate science"},
        ]
        with self.docs_path.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")

        # Mix structured and NL queries
        queries = [
            {"qid": "Q1", "query": "climate AND change"},          # -> {10}
            {"qid": "Q2", "query": "\"machine learning\""},        # -> {20}
            {"qid": "Q3", "query": "climat*"},                     # -> {10,30}
            {"qid": "Q4", "query": "climate NEAR/2 policy"},       # -> {30}
            {"qid": "Q5", "query": "effects climate"},             # NL (OR) -> {10}
        ]
        self.expected = {
            "Q1": {10},
            "Q2": {20},
            "Q3": {10, 30},
            "Q4": {30},
            "Q5": {10},
        }
        self.doc_ids = {10, 20, 30}

        self.queries_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run_cli(self) -> subprocess.CompletedProcess:
        script = REPO_ROOT / "system" / "search_system.py"
        cmd = [SYS_PY, "-u", str(script), str(self.queries_path), str(self.docs_path), str(self.run_path)]
        return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180)

    def test_cli_runs_and_schema_is_valid(self):
        proc = self._run_cli()
        self.assertEqual(proc.returncode, 0, f"CLI failed:\nSTDERR:\n{proc.stderr}")

        self.assertTrue(self.run_path.exists(), "Run JSON not created")
        data = json.loads(self.run_path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 5)

        for obj in data:
            # Required fields and types
            self.assertIn("qid", obj)
            self.assertIn("doc_ids", obj)
            self.assertIsInstance(obj["qid"], str)
            self.assertIsInstance(obj["doc_ids"], list)
            self.assertLessEqual(len(obj["doc_ids"]), 10)

            # doc ids are ints and subset of input docs, deduped
            self.assertTrue(all(isinstance(d, int) for d in obj["doc_ids"]))
            self.assertEqual(len(obj["doc_ids"]), len(set(obj["doc_ids"])))
            self.assertTrue(set(obj["doc_ids"]).issubset(self.doc_ids))

            # scores optional but if present must align and be monotone non-increasing
            if "scores" in obj and obj["scores"] is not None:
                self.assertIsInstance(obj["scores"], list)
                self.assertEqual(len(obj["scores"]), len(obj["doc_ids"]))
                # monotone non-increasing
                for i in range(1, len(obj["scores"])):
                    self.assertGreaterEqual(obj["scores"][i-1], obj["scores"][i])

        # Spot-check some expected candidates per query
        by_qid = {o["qid"]: set(o["doc_ids"]) for o in data}
        self.assertEqual(by_qid["Q1"], self.expected["Q1"])
        self.assertEqual(by_qid["Q2"], self.expected["Q2"])
        self.assertTrue(self.expected["Q3"].issubset(by_qid["Q3"]))
        self.assertIn(30, by_qid["Q4"])
        self.assertNotIn(10, by_qid["Q4"])  # NEAR/2 should not match doc 10
        self.assertTrue(self.expected["Q5"].issubset(by_qid["Q5"]))

if __name__ == "__main__":
    unittest.main()
