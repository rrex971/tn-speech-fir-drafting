from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from research.lexform.config import Config
from research.lexform.data import anchor_partition, audit, mapping_rows, section_id
from research.lexform.decomposition import validate_form
from research.lexform.metrics import calibrate, multilabel, select, selective_point
from research.lexform.model import FuzzyLogic, LexForm, TextEncoder, classification_loss, form_alignment


class NumericEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 16)

    def forward(self, values, statute=False):
        return self.embedding(torch.tensor([int(value) for value in values]))


class LogicTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_and_of_or_distinguishes_missing_group(self):
        membership = torch.tensor([[[1., 1., 0., 0.], [0., 0., 1., 1.]]])
        for norm in ('godel', 'product', 'lukasiewicz'):
            logic = FuzzyLogic(norm)
            score, groups = logic(torch.tensor([[1., 0., 0., 1.], [1., 0., 0., 0.]]), membership)
            self.assertGreater(float(score[0, 0].detach()), 0.9)
            self.assertLess(float(score[1, 0].detach()), 0.1)
            self.assertLess(float(groups[1].amin().detach()), 0.1)

    def test_exact_product_and_lukasiewicz(self):
        x = torch.tensor([[0.8, 0.7]])
        includes = torch.ones_like(x)
        self.assertAlmostEqual(float(FuzzyLogic('product').disjunction(x, includes)), 0.94, places=5)
        self.assertAlmostEqual(float(FuzzyLogic('product').conjunction(x)), 0.56, places=5)
        self.assertAlmostEqual(float(FuzzyLogic('lukasiewicz').disjunction(x, includes)), 1, places=5)
        self.assertAlmostEqual(float(FuzzyLogic('lukasiewicz').conjunction(x)), 0.5, places=5)

    def test_logic_bounded_monotonic_and_differentiable(self):
        for norm in ('godel', 'product', 'lukasiewicz'):
            logic = FuzzyLogic(norm)
            x = (torch.rand(3, 8) * 0.2 + 0.5).requires_grad_()
            membership = torch.rand(2, 4, 8)
            scores, _ = logic(x, membership)
            higher, _ = logic((x + 0.1).clamp(max=1), membership)
            self.assertTrue(torch.all((scores >= 0) & (scores <= 1)))
            self.assertTrue(torch.all(higher >= scores - 1e-6))
            scores.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_alignment_ignores_group_order(self):
        form = torch.rand(2, 4, 8)
        self.assertAlmostEqual(float(form_alignment(form, form[:, [2, 0, 3, 1]])), 0, places=6)

    def test_all_ablation_heads_have_finite_gradients(self):
        for variant in ('flat', 'conjunction', 'and_or', 'per_section', 'handwritten'):
            cfg = Config(bank_size=8, dimension=16, groups=2, variant=variant)
            model = LexForm(cfg, 2, NumericEncoder())
            fixed = None
            if variant == 'handwritten':
                fixed = torch.zeros(2, 2, 8)
                fixed[:, 0, 0] = 1
            output = model(['0', '1'], ['2', '3'], fixed)
            for kind in ('bce', 'pos_weight', 'focal', 'asymmetric'):
                loss = classification_loss(output['scores'], torch.eye(2), kind, torch.ones(2))
                self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_encoder_runs_offline_for_all_pooling_and_chunking(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, 'phone': 4, 'taken': 5}
            backend = Tokenizer(models.WordPiece(vocab=vocab, unk_token='[UNK]'))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            backend.post_processor = processors.TemplateProcessing(single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]')
            tokenizer.save_pretrained(path)
            BertModel(BertConfig(vocab_size=6, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)).save_pretrained(path)
            for pooling in ('cls', 'mean', 'attention'):
                for truncation in ('512', 'chunked'):
                    cfg = Config(encoder=str(path), dimension=16, pooling=pooling, truncation=truncation, max_length=80)
                    encoder = TextEncoder(cfg)
                    encoded = encoder(['phone taken ' * 90, 'phone'])
                    self.assertEqual(tuple(encoded.shape), (2, 16))
                    self.assertTrue(torch.isfinite(encoded).all())
                    encoded.sum().backward()


class ProtocolTests(unittest.TestCase):
    def test_training_checkpoint_roundtrip_on_explicit_synthetic_fixture(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
        from research.lexform.experiments import load_checkpoint, predict, train
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            encoder_path = root / 'initial-encoder'
            vocabulary = {'[PAD]': 0, '[UNK]': 1, 'phone': 2, 'taken': 3, 'fraud': 4}
            tokenizer_backend = Tokenizer(models.WordPiece(vocab=vocabulary, unk_token='[UNK]'))
            tokenizer_backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_backend, unk_token='[UNK]', pad_token='[PAD]')
            tokenizer.save_pretrained(encoder_path)
            BertModel(BertConfig(vocab_size=5, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)).save_pretrained(encoder_path)
            cfg = Config(encoder=str(encoder_path), bank_size=64, dimension=16, groups=4,
                         epochs=1, batch_size=2, accumulation=3, max_length=80, statute_length=80,
                         anchor_weight=0, freeze_encoder=True)
            examples = [{'id': str(i), 'text': text, 'labels': labels} for i, (text, labels) in enumerate([
                ('phone taken', ['379']), ('fraud', ['420']), ('phone fraud', ['379', '420']), ('taken phone', ['379'])])]
            splits = {'train': examples, 'dev': examples[:2], 'test': examples[2:]}
            context = ({'sha256': {'train': 'synthetic', 'dev': 'synthetic', 'test': 'synthetic'}},
                       splits, ['379', '420'], ['phone taken', 'fraud'],
                       [{'id': '379', 'code': 'IPC', 'text': 'phone taken'}, {'id': '420', 'code': 'IPC', 'text': 'fraud'}])
            output = root / 'synthetic-training'
            output.mkdir()
            with patch('research.lexform.experiments._context', return_value=context), patch('research.lexform.decomposition.cached_forms', return_value=[]), patch('research.lexform.features.output_dir', return_value=root / 'artifacts'):
                result = train(root, cfg, output)
            self.assertEqual(result['status'], 'completed')
            restored, restored_cfg, metadata, fixed = load_checkpoint(output)
            scores, weakest = predict(restored, splits['test'], metadata['statute_texts'], restored_cfg, fixed)
            saved = np.load(output / 'predictions.npz')
            np.testing.assert_allclose(scores, saved['test_scores'], atol=1e-6)
            np.testing.assert_allclose(weakest, saved['test_weakest'], atol=1e-6)

    def test_feature_cache_preserves_frozen_encoder_gradients(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend = Tokenizer(models.WordPiece(vocab={'[PAD]': 0, '[UNK]': 1, 'phone': 2}, unk_token='[UNK]'))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]').save_pretrained(root)
            BertModel(BertConfig(vocab_size=3, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)).save_pretrained(root)
            cfg = Config(encoder=str(root), freeze_encoder=True, dimension=8, max_length=80)
            encoder = TextEncoder(cfg)
            with patch('research.lexform.features.output_dir', return_value=root / 'artifacts'):
                encoder.enable_feature_cache()
            encoder.train()
            self.assertFalse(encoder.encoder.training)
            first = encoder(['phone'])
            with patch.object(encoder, 'pooled', side_effect=AssertionError('Cache miss')):
                second = encoder(['phone'])
            torch.testing.assert_close(first, second)
            (second[:, 0].sum()).backward()
            self.assertIsNotNone(encoder.projection.weight.grad)
            self.assertTrue(all(p.grad is None for p in encoder.encoder.parameters()))
            state = {key: value for key, value in encoder.state_dict().items() if not key.startswith('encoder.')}
            self.assertLess(sum(value.numel() for value in state.values()), 1000)

    def test_supplement_exclusions_are_audited_not_silently_lost(self):
        from research.lexform.data import validate_splits
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = {
                'train': [{'id': '1', 'text': 'a complaint', 'labels': ['379']}, {'id': '2', 'text': '', 'labels': ['379']}],
                'dev': [{'id': '3', 'text': 'a complaint', 'labels': ['379']}, {'id': '4', 'text': 'other facts', 'labels': ['420']}],
                'test': [{'id': '5', 'text': 'another case', 'labels': ['420']}],
            }
            for split, rows in fixtures.items():
                (root / f'{split}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            report, splits = validate_splits(root, 'supplement')
            self.assertEqual(report['original_counts']['train'], 2)
            self.assertEqual(report['excluded_invalid_documents']['train'][0]['id'], '2')
            self.assertEqual(report['excluded_overlap_documents']['dev'], 1)
            self.assertEqual(len(splits['dev']), 1)
            self.assertFalse(report['full_split'])
            with self.assertRaises(ValueError):
                validate_splits(root, 'canonical')

    def test_mapping_targets_include_composite_subsections(self):
        rows = mapping_rows()
        self.assertEqual(len(rows), 100)
        self.assertEqual(len([row for row in rows if row['target'] is None]), 7)
        mischief = next(row for row in rows if row['ipc_section'] == '427')
        self.assertEqual(mischief['target'], '324')
        self.assertTrue(mischief['eligible'])

    def test_anchor_partition_is_disjoint_by_target(self):
        train, heldout = anchor_partition(mapping_rows(), 42, 0.8)
        self.assertTrue(train)
        self.assertTrue(heldout)
        self.assertFalse({row['target'] for row in train} & {row['target'] for row in heldout})
        self.assertTrue(all(row['eligible'] for row in train + heldout))
        self.assertEqual((train, heldout), anchor_partition(mapping_rows(), 42, 0.8))

    def test_claim_audit_and_missing_corpus(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = audit(Path(temporary))
        self.assertEqual((value['handwritten_elements'], value['handwritten_sections'], value['literal_or_elements']), (62, 29, 29))
        self.assertFalse(value['data_ready'])

    def test_weakest_group_can_veto_high_score(self):
        prediction = select(np.array([[0.9, 0.8]]), np.array([[0.1, 0.7]]), 0.6, 0.5)
        self.assertEqual(prediction.tolist(), [[False, True]])

    def test_calibration_abstains_if_no_feasible_point(self):
        truth = np.array([[True, False]] * 40)
        scores = np.array([[0.1, 0.9]] * 40)
        operating_point = calibrate(truth, scores, scores, 0.01)
        self.assertFalse(operating_point['feasible'])
        self.assertGreater(operating_point['threshold'], 1)
        self.assertIsNone(operating_point['risk'])

    def test_no_coverage_is_not_zero_risk(self):
        truth = np.array([[True, False]])
        point = selective_point(truth, np.zeros_like(truth))
        self.assertEqual(point['coverage'], 0)
        self.assertIsNone(point['risk'])

    def test_decomposition_rejects_invented_evidence(self):
        statute = {'id': '1', 'code': 'IPC', 'text': 'A person takes movable property.'}
        with self.assertRaises(ValueError):
            validate_form({'groups': [{'alternatives': [{'text': 'knife', 'quote': 'uses a knife'}]}]}, statute)
        valid = validate_form({'groups': [{'alternatives': [{'text': 'movable property', 'quote': 'movable property'}]}]}, statute)
        self.assertFalse(valid['reviewed'])

    def test_section_labels(self):
        self.assertEqual(section_id('Section 304B in The Indian Penal Code'), '304B')
        self.assertEqual(section_id('379'), '379')
        self.assertNotEqual(section_id('Section 294(b) in The Indian Penal Code'), section_id('294'))
        self.assertNotEqual(section_id('Section 376(2) in The Indian Penal Code'), section_id('376'))

    def test_status_and_inference_are_not_fake_results(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from research.lexform.api import router
        app = FastAPI()
        app.include_router(router)
        with TestClient(app) as client, patch('research.lexform.api.runs', return_value=[]), patch('research.lexform.api.bundled_runs', return_value=[]):
            response = client.get('/v1/research/lexform/status')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['completed_experiments'], 0)
            self.assertFalse(response.json()['reference_target']['measured_locally'])
            missing = client.post('/v1/research/lexform/inspect', json={'text': 'A phone was taken.', 'checkpoint_id': 'missing', 'code': 'BNS'})
            self.assertEqual(missing.status_code, 409)
            catalog = client.get('/v1/research/lexform/statutes').json()
            self.assertEqual(sum(row['code'] == 'BNS' for row in catalog), 358)
            self.assertEqual(sum(row['code'] == 'IPC' for row in catalog), 29)
            traversal = client.post('/v1/research/lexform/inspect', json={'text': 'A phone was taken.', 'checkpoint_id': '../../bad', 'code': 'BNS'})
            self.assertEqual(traversal.status_code, 422)


if __name__ == '__main__':
    unittest.main()
