import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed locally")
class ArcFaceTests(unittest.TestCase):
    def test_label_free_logits_do_not_require_or_consume_labels(self):
        import torch

        from edge_finer_research.models.arcface import ArcMarginProduct

        head = ArcMarginProduct(4, 2, scale=50.0, margin=0.5)
        embeddings = torch.randn(3, 4)
        logits = head.forward_logits(embeddings)
        self.assertEqual(tuple(logits.shape), (3, 2))
        train_zero = head.forward_train(embeddings, torch.zeros(3, dtype=torch.long))
        train_one = head.forward_train(embeddings, torch.ones(3, dtype=torch.long))
        self.assertFalse(torch.equal(train_zero, train_one))


if __name__ == "__main__":
    unittest.main()
