import os, sys, subprocess, unittest
from pathlib import Path

class TestTask3DevCLI(unittest.TestCase):
    def test_cli_runs_and_prints_table(self):
        repo = Path(__file__).resolve().parents[1]
        script = repo / "ranking" / "rankers.py"
        self.assertTrue(script.exists(), "ranking/rankers.py missing")

        proc = subprocess.run(
            [sys.executable, "-u", str(script)],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=180
        )
        self.assertEqual(proc.returncode, 0, f"CLI failed:\n{proc.stderr}")
        out = proc.stdout
        self.assertIn("Task 3 — Dev Pearson Correlation", out)
        self.assertIn("Method", out)
        self.assertIn("Pearson r", out)

if __name__ == "__main__":
    unittest.main()
