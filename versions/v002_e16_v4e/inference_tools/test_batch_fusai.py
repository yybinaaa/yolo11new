"""Synthetic tests only; no competition image inference."""
import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import batch_fusai as b


class BatchTests(unittest.TestCase):
    def test_only_version_weights_and_path_safety(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'weights').mkdir(); (root/'code').mkdir()
            (root/'weights/best.pt').touch(); (root/'weights/epoch180.pt').touch(); (root/'code/pretrained.pt').touch()
            with patch.object(b,'VERSION',root):
                self.assertEqual([p.name for p in b.discover_weights()],['best.pt','epoch180.pt'])
                with self.assertRaises(ValueError):b.discover_weights(['../other.pt'])
                with self.assertRaises(ValueError):b.discover_weights(['best.pt','best.pt'])

    def test_duplicate_image_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'sub').mkdir(); (root/'a.jpg').touch(); (root/'sub/a.jpg').touch()
            with self.assertRaisesRegex(ValueError,'Duplicate'):b.discover_images(root)

    def test_export_filter_coordinates_and_empty_submission(self):
        detection=dict(class_name='qilie',class_id=5,bbox_xyxy=[1.2,2.2,10.8,20.1],score=.8)
        self.assertEqual(b.export([detection],False),[])
        self.assertEqual(b.export([detection],True)[0]['bbox_xyxy'],[1,2,11,20])
        with self.assertRaises(ValueError):b.export([dict(detection,score=float('nan'))],True)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); b.package(root,[dict(image_id='empty.jpg',detections=[])],{})
            with ZipFile(root/'submission.zip') as z:
                self.assertEqual(z.namelist(),['submission.json']); self.assertEqual(json.loads(z.read('submission.json')),[])
            self.assertEqual(json.loads((root/'summary.json').read_text())['images'],1)

    def test_resume_and_settings_isolation_with_fake_detector(self):
        b.bootstrap()
        import cv2
        import numpy as np
        import v4e_core
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'images'; source.mkdir(); weight=root/'best.pt'; weight.write_bytes(b'fake')
            for name in ('one.jpg','empty.jpg'):
                cv2.imencode('.jpg',np.zeros((32,32,3),dtype=np.uint8))[1].tofile(source/name)
            args=argparse.Namespace(check=False,source=source,device='cpu',visualize=False,keep_qilie=False)
            with patch.object(b,'VERSION',root), patch.object(b,'load_model',return_value=(object(),139)), patch.object(b,'source_digest',return_value='fake-code'), patch.object(v4e_core,'infer',return_value={'detections':[]}) as infer:
                b.worker(args,weight,b.discover_images(source)); self.assertEqual(infer.call_count,2)
                b.worker(args,weight,b.discover_images(source)); self.assertEqual(infer.call_count,2)
                args.keep_qilie=True
                b.worker(args,weight,b.discover_images(source)); self.assertEqual(infer.call_count,4)
                self.assertEqual(len(list((root/'inference_packages/fusai_auto/best').iterdir())),2)

    def test_core_matches_archived_functions(self):
        original=b.VERSION/'code/scripts/v4e_core.py'
        if not original.exists():self.skipTest('Legacy version has a different core filename')
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
        old=functions(original); new=functions(b.TOOLS/'v4e_core.py')
        for name,tree in old.items():self.assertEqual(tree,new[name],name)


if __name__=='__main__':unittest.main(verbosity=2)
