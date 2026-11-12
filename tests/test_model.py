"""Model-level checks for the PAVE forecasting and distillation pipeline."""
import unittest
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import PAVE, build_model, sample_student_nodes, subset_relation_loss, prediction_losses
from model.utils import spatial_query_chunks


class PAVETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.args = dict(in_steps=4, out_steps=3, steps_per_day=24, model_dim=12,
                         feed_forward_dim=24, num_heads=3, num_layers=2, dropout=0.)
        self.raw = torch.zeros(2, 4, 7, 4)
        self.raw[..., 0] = torch.randn(2, 4, 7)
        self.raw[..., 1] = .25
        self.raw[..., 2] = 2
        self.relations = torch.randn(7, 7, 4)

    def test_forecasting_permutation_and_query_chunking(self):
        model = PAVE(**self.args).eval()
        permutation = torch.tensor([4, 0, 5, 2, 6, 1, 3])
        with torch.no_grad():
            pred, features, scores = model(self.raw, self.relations, return_relations=True)
            self.assertEqual(tuple(pred.shape), (2, 3, 7, 1))
            self.assertEqual(tuple(features.shape), (2, 4, 7, 12))
            self.assertEqual(tuple(scores.shape), (2, 3, 4, 7, 7))
            permuted = model(self.raw[:, :, permutation], self.relations[permutation][:, permutation])
            torch.testing.assert_close(permuted, pred[:, :, permutation], rtol=3e-5, atol=3e-6)
            for chunk in (1, 3, 10):
                with spatial_query_chunks(model, chunk):
                    actual = model(self.raw, self.relations)
                torch.testing.assert_close(actual, pred, rtol=2e-5, atol=2e-6)

    def test_context_and_full_to_partial_distillation(self):
        student, teacher = PAVE(**self.args), PAVE(**self.args).eval().requires_grad_(False)
        targets = torch.tensor([0, 4])
        nodes, local_targets = sample_student_nodes(7, targets, 5, 1., torch.Generator().manual_seed(8))
        self.assertTrue(torch.equal(nodes[local_targets], targets))
        partial_raw = self.raw.index_select(2, nodes)
        partial_relations = self.relations.index_select(0, nodes).index_select(1, nodes)
        prediction, _, student_scores = student(partial_raw, partial_relations, return_relations=True)
        with torch.no_grad():
            _, _, teacher_scores = teacher(self.raw, self.relations, return_relations=True)
        forecast_loss = prediction_losses(prediction, torch.zeros_like(prediction), local_targets, 1.)[0]
        relation_loss = subset_relation_loss(student_scores, teacher_scores, nodes, local_targets, 1.)
        (forecast_loss + relation_loss).backward()
        self.assertTrue(torch.isfinite(relation_loss))
        self.assertGreater(student.structure_learner.layers[-1].structure_generator.target_projection.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_model_args_configuration(self):
        config = {'model_args': dict(model_dim=12, num_heads=3, num_layers=2, feed_forward_dim=24, dropout=0.)}
        model = build_model(config, in_steps=4, out_steps=3, steps_per_day=24)
        self.assertEqual(tuple(model(self.raw, self.relations).shape), (2, 3, 7, 1))


if __name__ == '__main__':
    unittest.main()
