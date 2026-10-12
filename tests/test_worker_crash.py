"""A worker that dies in native code leaves a readable record."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"


class WorkerCrashTests(unittest.TestCase):
    def run_worker(self, body: str) -> tuple[int, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        worker = Path(temporary.name) / "tab_001"
        worker.mkdir()
        manifest = worker / "run_manifest.json"
        manifest.write_text(json.dumps({"status": "pending"}))
        script = "\n".join([
            "import sys",
            f"sys.path.insert(0, {str(CORE)!r})",
            "import parallel_sweep",
            body,
            f"sys.exit(parallel_sweep.worker_main(parallel_sweep.Path({str(manifest)!r})))",
        ])
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        return completed.returncode, worker

    def test_native_crash_is_written_to_fatal_error_file(self):
        code, worker = self.run_worker(
            "import faulthandler\n"
            "parallel_sweep.run_worker = lambda path: faulthandler._sigsegv()")
        self.assertNotEqual(code, 0)
        record = (worker / "fatal_error.txt").read_text()
        self.assertIn("Fatal Python error", record)
        self.assertIn("worker_main", record)
        self.assertFalse((worker / "worker_error.txt").exists())

    def test_clean_exit_leaves_no_fatal_error_file(self):
        code, worker = self.run_worker("parallel_sweep.run_worker = lambda path: 0")
        self.assertEqual(code, 0)
        self.assertFalse((worker / "fatal_error.txt").exists())

    def test_python_error_is_recorded_before_the_status(self):
        # The manifest lacks every worker field, so run_worker fails in Python.
        code, worker = self.run_worker("")
        self.assertEqual(code, 1)
        self.assertIn("KeyError", (worker / "worker_error.txt").read_text())
        self.assertEqual(json.loads((worker / "run_manifest.json").read_text())["status"], "failed")
        self.assertFalse((worker / "fatal_error.txt").exists())


if __name__ == "__main__":
    unittest.main()
