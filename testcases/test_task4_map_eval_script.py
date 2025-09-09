import json, os, sys, tempfile, subprocess, pathlib, unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SYS_PY = sys.executable

class TestTask4MAPEval(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = pathlib.Path(self.tmpdir.name)

        # Minimal run file we will generate via the CLI and then move under ./runs/
        self.docs = self.tmp / "docs.jsonl"
        self.queries = self.tmp / "queries.json"
        self.run_local = self.tmp / "run_task4_map.json"
        self.run_under_repo = REPO_ROOT / "runs" / "run_task4_map.json"

        # Small corpus
        docs = [
            {"id": 10, "text": "climate change affects policy"},
            {"id": 20, "text": "machine learning algorithms"},
            {"id": 30, "text": "renewable energy policy"},
        ]
        with self.docs.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")

        # Two queries (won't necessarily match dev judge; we only check the script runs)
        queries = [
            {"qid": "QX1", "query": "climate AND change"},
            {"qid": "QX2", "query": "\"machine learning\""},
        ]
        self.queries.write_text(json.dumps(queries, indent=2), encoding="utf-8")

        # Ensure runs/ exists
        (REPO_ROOT / "runs").mkdir(exist_ok=True)

    def tearDown(self):
        try:
            if self.run_under_repo.exists():
                self.run_under_repo.unlink()
        finally:
            self.tmpdir.cleanup()

    def _run_cli(self):
        script = REPO_ROOT / "system" / "search_system.py"
        cmd = [SYS_PY, "-u", str(script), str(self.queries), str(self.docs), str(self.run_local)]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, f"CLI failed:\nSTDERR:\n{proc.stderr}")
        self.assertTrue(self.run_local.exists(), "CLI did not create a run file")

    def _run_map(self):
        # Move run to ./runs so metrics/eval_map.py will pick it up
        self.run_under_repo.write_text(self.run_local.read_text(encoding="utf-8"), encoding="utf-8")

        script = REPO_ROOT / "metrics" / "eval_map.py"
        proc = subprocess.run([SYS_PY, "-u", str(script)],
                              cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, f"MAP script failed:\nSTDERR:\n{proc.stderr}")
        return proc.stdout

    def test_eval_script_prints_table_and_contains_our_run(self):
        # Create a run and compute MAP
        self._run_cli()
        out = self._run_map()

        # Check basic table header and our file name present
        self.assertIn("Task 4 — MAP on dev set", out)
        self.assertIn("Run file", out)
        self.assertIn("MAP", out)
        self.assertIn("run_task4_map.json", out)

if __name__ == "__main__":
    unittest.main()
