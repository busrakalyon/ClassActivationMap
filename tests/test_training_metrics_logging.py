import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed locally")
class TrainingMetricsLoggingTests(unittest.TestCase):
    def test_each_epoch_is_written_to_csv_json_and_checkpoint(self):
        import torch

        from edge_finer_research.models.training import train_experiment1

        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward_train(self, images, _labels):
                return self.linear(images)

            def forward_logits(self, images):
                return self.linear(images)

        samples = [
            (torch.tensor([[1.0, 0.0]]), torch.tensor([0]), ["real"]),
            (torch.tensor([[0.0, 1.0]]), torch.tensor([1]), ["fake"]),
        ]
        config = {
            "training": {
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "epochs": 2,
                "amp": False,
                "checkpoint_every": 1,
            },
            "scheduler": {"kind": "cosine", "minimum_learning_rate": 1e-5},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            train_experiment1(
                TinyClassifier(),
                samples,
                samples,
                config,
                output,
                torch.device("cpu"),
            )
            with (output / "training_metrics.csv").open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            report = json.loads((output / "training_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(csv_rows), 2)
            self.assertEqual(len(report["epochs"]), 2)
            self.assertEqual(
                set(csv_rows[0]),
                {
                    "epoch",
                    "train_loss",
                    "validation_loss",
                    "image_f1",
                    "auc",
                    "sensitivity",
                    "specificity",
                    "learning_rate",
                    "next_learning_rate",
                    "epoch_seconds",
                },
            )
            self.assertTrue((output / "checkpoints" / "epoch_001.pt").is_file())
            self.assertTrue((output / "checkpoints" / "epoch_002.pt").is_file())
            self.assertGreaterEqual(float(csv_rows[0]["epoch_seconds"]), 0.0)


if __name__ == "__main__":
    unittest.main()
