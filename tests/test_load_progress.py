import io
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from src.pipeline import _open_prefix


class LoadProgressTest(unittest.TestCase):
    def test_open_prefix_reports_tensor_loading_progress(self):
        ticks = iter([100.0, 101.0, 102.0, 102.0])
        stream = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "sample.safetensors"
            save_file(
                {
                    "vae.first": torch.zeros(1),
                    "vae.second": torch.ones(1),
                    "model.skip": torch.full((1,), 2),
                },
                str(checkpoint),
            )

            tensors = _open_prefix(
                checkpoint,
                "vae.",
                progress_label="Loading VAE weights",
                progress_stream=stream,
                time_fn=lambda: next(ticks),
            )

        self.assertEqual(set(tensors), {"vae.first", "vae.second"})
        output = stream.getvalue()
        self.assertIn("Loading VAE weights", output)
        self.assertIn("2/2", output)
        self.assertIn("100.0%", output)
        self.assertTrue(output.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
