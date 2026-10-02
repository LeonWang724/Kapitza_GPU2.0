"""Codex CUDA Port: unit tests for config and manifest provenance helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "phase_diagram" / "simulation_core"
sys.path.insert(0, str(CORE))

from cuda_workflow_common import (  # noqa: E402
    phase_dataset_label,
    read_config,
    sha256_file,
    update_config,
    write_json_atomic,
)


class ManifestHelperTests(unittest.TestCase):
    def test_phase_dataset_label_is_descriptive_and_filename_safe(self) -> None:
        self.assertEqual(
            phase_dataset_label(20.0, 40.0, -0.5),
            "LatticeDepth_20ER_InitialDepth_40ER_Phase_m0p5rad",
        )

    def test_config_update_preserves_unmodified_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpe1d.config"
            path.write_text("alpha=1\noutput_folder=old\nunknown=value\n", encoding="utf-8")
            update_config(path, {"output_folder": "new", "status_file": "status.csv"})
            self.assertEqual(
                read_config(path),
                {
                    "alpha": "1",
                    "output_folder": "new",
                    "unknown": "value",
                    "status_file": "status.csv",
                },
            )

    def test_json_is_written_atomically_and_hashable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_json_atomic(path, {"status": "completed", "runs": []})
            self.assertTrue(path.is_file())
            self.assertEqual(len(sha256_file(path)), 64)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
