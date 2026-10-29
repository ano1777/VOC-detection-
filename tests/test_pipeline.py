"""Small synthetic tests: no VOC download, pretrained download or GPU needed."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from src.config import Config
from src.data import parse_annotation, VOCDataset, audit_voc, make_splits
from src.metrics import evaluate_detections, error_analysis
from src.model import build_model, save_checkpoint, load_checkpoint
from src.engine import train_experiment
from unittest.mock import patch


def target(boxes, labels, difficult=None):
    return {'boxes': np.array(boxes, dtype=float).reshape(-1,4), 'labels': np.array(labels),
            'difficult': np.array(difficult if difficult is not None else [False]*len(labels))}


def prediction(boxes, labels, scores):
    return {'boxes': np.array(boxes, dtype=float).reshape(-1,4),
            'labels': np.array(labels), 'scores': np.array(scores)}


class MetricTests(unittest.TestCase):
    def test_perfect_and_absent_classes(self):
        gt = target([[0,0,10,10]], [1])
        pred = prediction([[0,0,10,10]], [1], [.9])
        result = evaluate_detections([pred], [gt])
        self.assertEqual(result['map50'], 1.)
        self.assertEqual(result['map50_95_voc'], 1.)
        self.assertEqual(result['classes_with_gt'], 1)
        self.assertIsNone(result['per_class_ap50']['dog'])

    def test_difficult_is_ignored_and_duplicate_is_false_positive(self):
        gt = target([[0,0,10,10], [20,20,30,30], [40,40,50,50]], [1,1,1], [False,False,True])
        pred = prediction([[40,40,50,50], [0,0,10,10], [0,0,10,10], [20,20,30,30]],
                          [1,1,1,1], [.99,.9,.8,.7])
        result = evaluate_detections([pred], [gt])
        self.assertAlmostEqual(result['map50'], 5/6)
        counts = error_analysis([pred], [gt])[0]
        self.assertEqual(counts['ignored_difficult'], 1)
        self.assertEqual(counts['duplicate'], 1)
        self.assertEqual(counts['tp'], 2)
        self.assertEqual(counts['missed'], 0)

    def test_empty_prediction(self):
        self.assertEqual(evaluate_detections([prediction([],[],[])],
                         [target([[0,0,10,10]], [1])])['map50'], 0.)

    def test_false_positive_before_true_positive(self):
        result = evaluate_detections([prediction([[20,20,30,30],[0,0,10,10]], [1,1], [.9,.8])],
                                     [target([[0,0,10,10]], [1])])
        self.assertEqual(result['map50'], .5)

    def test_wrong_class_and_localization(self):
        gt = target([[0,0,10,10]], [1])
        pred = prediction([[0,0,10,10], [5,0,15,10]], [2,1], [.9,.8])
        errors = error_analysis([pred], [gt])[0]
        self.assertEqual(errors['wrong_class'], 1)
        self.assertEqual(errors['localization'], 1)
        self.assertEqual(errors['missed'], 1)


class DataTests(unittest.TestCase):
    def test_coordinate_conversion_clipping_and_empty_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp)/'a.xml'
            xml.write_text('''<annotation><object><name>cat</name><bndbox>
            <xmin>1</xmin><ymin>1</ymin><xmax>20</xmax><ymax>10</ymax>
            </bndbox></object><object><name>dog</name><bndbox>
            <xmin>15</xmin><ymin>5</ymin><xmax>8</xmax><ymax>2</ymax>
            </bndbox></object></annotation>''')
            objects, issues = parse_annotation(xml, 12, 10)
            self.assertEqual(objects[0]['box'], [0,0,12,10])
            self.assertIn('clipped_box', issues)
            self.assertIn('degenerate_box', issues)
            path = Path(tmp)/'im.png'
            Image.new('RGB',(12,10)).save(path)
            image, t = VOCDataset([{'path': str(path), 'objects': []}])[0]
            self.assertEqual(tuple(t['boxes'].shape), (0,4))
            self.assertEqual(tuple(t['labels'].shape), (0,))
            self.assertEqual(tuple(image.shape), (3,10,12))

    def test_flip_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'im.png'
            Image.new('RGB',(20,10)).save(path)
            record = {'path': str(path), 'objects': [{'box':[1,2,7,9],'label':1,'difficult':False}]}
            _, t = VOCDataset([record], flip_probability=1.)[0]
            torch.testing.assert_close(t['boxes'], torch.tensor([[13.,2.,19.,9.]]))

    def test_audit_deduplicates_and_keeps_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for part in ('JPEGImages','Annotations','ImageSets/Main'):
                (root/part).mkdir(parents=True)
            (root/'ImageSets/Main/val.txt').write_text('v\n')
            (root/'ImageSets/Main/train.txt').write_text('a\nb\nc\n')
            for name, color in [('v','red'),('a','red'),('b','blue'),('c','green')]:
                Image.new('RGB',(20,20),color).save(root/'JPEGImages'/f'{name}.jpg')
                (root/'Annotations'/f'{name}.xml').write_text('<annotation/>')
            clean, events = audit_voc(root, root/'out')
            self.assertEqual([r['id'] for r in clean['train']], ['b','c'])
            self.assertTrue(any(e['issue']=='duplicate_pixels' for e in events))
            splits = make_splits(clean, Config(), root/'out')
            self.assertEqual([len(v) for v in splits.values()], [1,1,1])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forward_backward_and_checkpoint_roundtrip(self):
        # Small resolution is a smoke test, not a useful training configuration.
        cfg = replace(Config(), min_size=64, max_size=96, freeze_backbone=True)
        model = build_model(cfg, pretrained=False)
        model.train()
        image = torch.rand(3,64,80)
        truth = {'boxes': torch.tensor([[5.,5.,35.,40.]]), 'labels': torch.tensor([1])}
        losses = model([image], [truth])
        self.assertTrue(torch.isfinite(sum(losses.values())))
        sum(losses.values()).backward()
        self.assertIsNotNone(model.roi_heads.box_predictor.cls_score.weight.grad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            save_checkpoint(path, model, cfg, epoch=1)
            loaded, _, _ = load_checkpoint(path)
            model.eval()
            with torch.inference_mode():
                before = model([image])[0]
                after = loaded([image])[0]
            for key in ('boxes','scores','labels'):
                torch.testing.assert_close(before[key], after[key])

class EngineTests(unittest.TestCase):
    def test_one_epoch_trial_saves_real_history(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'image.png'
            Image.new('RGB', (80,64), 'red').save(path)
            records = [{'path': str(path), 'objects': [
                {'box': [5,5,35,40], 'label': 1, 'difficult': False}]}]
            cfg = replace(Config(), epochs=1, batch_size=1, min_size=64,
                          max_size=96, freeze_backbone=True)
            with patch('src.engine.build_model', side_effect=lambda c: build_model(c, pretrained=False)):
                result = train_experiment(cfg, records, records, root/'trial', torch.device('cpu'))
            self.assertTrue(Path(result['checkpoint']).is_file())
            self.assertTrue((root/'trial/history.json').is_file())
            self.assertEqual(result['epochs_completed'], 1)

if __name__ == '__main__':
    unittest.main()
