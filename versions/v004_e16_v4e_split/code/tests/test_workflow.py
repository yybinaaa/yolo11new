"""Synthetic checks only: no optimization on competition images or holdout inference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import _bootstrap
import argparse
import csv
import json
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from PIL import Image
import yaml
from common import CODE,VERSION,NAMES,settings,sha256,validate_split,write_json
from prepare_data import build,plan_tiles
from metrics import evaluate
from v4e_core import safe_add,infer

class DataTests(unittest.TestCase):
    def test_split_and_preparation_isolation(self):
        parent=CODE/'artifacts/tests';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root=Path(temp)/'source';rows=[]
            for index,split in enumerate(('train','test')):
                ip=root/'images'/split/f'{split}.jpg';lp=root/'labels'/split/f'{split}.txt'
                ip.parent.mkdir(parents=True);lp.parent.mkdir(parents=True)
                Image.fromarray(np.full((64,64),40+index*100,np.uint8)).save(ip)
                lp.write_text('0 0.5 0.5 0.25 0.25\n',encoding='utf-8')
                rows.append(dict(name=ip.name,split=split,group_id=str(index),source_group=str(index),image_sha256=sha256(ip),label_sha256=sha256(lp)))
            (root/'data.yaml').write_text(yaml.safe_dump(dict(names=NAMES)),encoding='utf-8')
            def manifest():
                with (root/'split_manifest.csv').open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            manifest();c=deepcopy(settings());c['dataset']=str(root);c['expected_split']=dict(train=1,test=1)
            c['preparation'].update(tile_size=32,keep_empty_ratio=1.0)
            _,buckets,audit=validate_split(c,True)
            out=Path(temp)/'prepared';built=build(c,root,buckets,audit,out)
            self.assertEqual(built['val_images'],1)
            samples=json.loads((out/'samples.json').read_text())
            self.assertTrue(all(e['source']=='train.jpg' for e in samples if e['kind']=='tile'))
            self.assertEqual(len([e for e in samples if e['source_split']=='test']),1)
            # Repeated preparation reuses the same complete cache deterministically.
            self.assertEqual(built,build(c,root,buckets,audit,out))
            rows[1]['group_id']=rows[0]['group_id'];manifest()
            with self.assertRaisesRegex(ValueError,'leakage'):validate_split(c)

    def test_long_box_visible_rule(self):
        opts=dict(tile_size=64,overlap=.2,min_visibility=.3,min_box_size=2,keep_empty_ratio=0,seed=42)
        result=plan_tiles('long.jpg',256,128,[(8,.5,.5,.9,.05)],opts)
        self.assertGreater(len(result),1)
        self.assertTrue(all(t['labels'] for t in result))

class MetricTests(unittest.TestCase):
    def test_empty_image_false_positive_duplicate_and_wrong_class(self):
        gt={'a.jpg':[dict(class_name='jiaza',bbox_xyxy=[0,0,10,10])],'empty.jpg':[]}
        d=lambda cls,score:dict(class_name=cls,bbox_xyxy=[0,0,10,10],score=score)
        rows=[dict(image_id='a.jpg',detections=[d('jiaza',.9),d('jiaza',.8),d('huashang',.7)]),
              dict(image_id='empty.jpg',detections=[d('jiaza',.8)])]
        r=evaluate(rows,gt);self.assertEqual((r['tp'],r['fp'],r['fn']),(1,3,0))
        self.assertEqual(r['precision'],.25)
        with self.assertRaisesRegex(ValueError,'every holdout'):evaluate(rows[:1],gt)

    def test_safe_add_preserves_anchors(self):
        a=dict(class_id=0,score=.2,bbox_xyxy=[0,0,10,10])
        duplicate=dict(a,score=.9);novel=dict(a,bbox_xyxy=[20,20,30,30])
        self.assertEqual(safe_add([a],[duplicate,novel]),[a,novel])

    def test_v4e_empty_model_complete_flow(self):
        class Boxes:
            class Data:
                def cpu(self):return self
                def tolist(self):return []
            data=Data()
        class Model:
            names=dict(enumerate(NAMES))
            def predict(self,*a,**k):return [argparse.Namespace(boxes=Boxes())]
        r=infer(np.zeros((64,64,3),np.uint8),Model(),'cpu')
        self.assertEqual(r['detections'],[]);self.assertEqual(r['added'],0)

class ModelTests(unittest.TestCase):
    def test_real_trainer_configuration_without_training(self):
        import train as module
        from ultralytics.models.yolo.detect import DetectionTrainer
        parent=CODE/'artifacts/tests';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root=Path(temp);code=root/'code';code.mkdir()
            opts=dict(settings()['train'],device='cpu',workers=0,model=str(CODE/'configs/yolo11m.yaml'),
                      pretrained=str(VERSION/settings()['pretrained']),data='synthetic.yaml',
                      project=str(code/'training_records'),name='train',exist_ok=True)
            data=dict(nc=9,names=dict(enumerate(NAMES)),channels=3,train='synthetic',val='synthetic')
            with patch.object(module,'VERSION',root),patch.object(module,'CODE',code),patch.object(DetectionTrainer,'get_dataset',return_value=data):
                trainer=module.SplitTrainer(overrides=opts,contract={'synthetic':True},kind=settings()['architecture'])
                self.assertEqual(trainer.last,root/'weights/last.pt')
                self.assertEqual(trainer.save_dir,code/'training_records/train')
                self.assertTrue(trainer.args.val)
                self.assertFalse((root/'weights/last.pt').exists())

    def test_architecture_initialization_and_resumable_checkpoint(self):
        import torch
        from ultralytics.nn.tasks import DetectionModel
        from train import build_model,SplitTrainer,architecture_check,optimizer_names
        torch.set_num_threads(4);torch.manual_seed(42)
        base=DetectionModel(str(CODE/'configs/yolo11m.yaml'),nc=9,verbose=False).eval()
        kind=settings()['architecture'];m=build_model(str(CODE/'configs/yolo11m.yaml'),base,kind).eval()
        x=torch.randn(1,3,128,128)
        with torch.no_grad():torch.testing.assert_close(base(x)[0],m(x)[0],rtol=0,atol=0)
        if kind=='e16':
            branch=m.model[16];f=torch.randn(1,branch.cv1.conv.in_channels,8,8)
            branch(f).square().mean().backward()
            self.assertGreater(branch.ms_branch.project.weight.grad.abs().sum().item(),0)
            m.zero_grad(set_to_none=True)
        # Isolated save/restore check with a synthetic AdamW state (no dataset/epoch loop).
        optimizer=torch.optim.AdamW(m.parameters(),lr=.001)
        first=next(m.parameters());optimizer.state[first]={'step':torch.tensor(1.),'exp_avg':torch.ones_like(first)*.01,'exp_avg_sq':torch.ones_like(first)*.001}
        trainer=object.__new__(SplitTrainer);trainer.model=m;trainer.optimizer=optimizer
        trainer.ema=argparse.Namespace(ema=deepcopy(m),updates=1)
        trainer.scaler=torch.amp.GradScaler('cpu',enabled=False)
        trainer.scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        trainer.args=argparse.Namespace(model='synthetic',epochs=2)
        trainer.epoch=0;trainer.epochs=2;trainer.save_period=1;trainer.metrics={'synthetic':1.};trainer.fitness=1.;trainer.best_fitness=1.
        trainer.contract={'synthetic':True};trainer.kind=kind;trainer.read_results_csv=lambda:{}
        parent=CODE/'artifacts/tests';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            trainer.wdir=Path(temp);trainer.last=Path(temp)/'last.pt';trainer.best=Path(temp)/'best.pt';trainer.save_dir=Path(temp)
            trainer.save_model();self.assertTrue((Path(temp)/'epoch1.pt').exists())
            ckpt=torch.load(trainer.last,weights_only=False,map_location='cpu')
            architecture_check(ckpt['ema'],kind)
            self.assertEqual(ckpt['split_experiment']['optimizer_names'],optimizer_names(optimizer,m))
            expected=next(m.parameters()).detach().clone()
            with torch.no_grad():next(m.parameters()).add_(1)
            trainer._load_checkpoint_state(ckpt)
            torch.testing.assert_close(next(m.parameters()),expected,rtol=0,atol=0)
            self.assertEqual(len(trainer.optimizer.state),1)
            wrong=dict(ckpt);wrong['split_experiment']=dict(ckpt['split_experiment'],contract={'wrong':1})
            with self.assertRaisesRegex(RuntimeError,'not from'):trainer._load_checkpoint_state(wrong)

if __name__=='__main__':unittest.main(verbosity=2)
