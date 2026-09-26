"""Portable inputs and fresh experiment creation; no training or real-image inference."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_batch_source_reference_preserves_explicit_external_input(tmp_path, monkeypatch):
    for version in sorted((ROOT/'versions').glob('v00[1-5]*')):
        path=version/'inference_tools/batch_fusai.py'
        spec=importlib.util.spec_from_file_location('batch_path_test',path)
        batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)
        monkeypatch.setattr(batch,'VERSION',tmp_path/'workspace/versions/example')
        assert batch.source_reference(tmp_path/'workspace/data/fusai') == 'data/fusai'
        # An explicitly supplied external directory stays usable, including another Windows drive.
        external=Path('Z:/custom_images') if sys.platform=='win32' else tmp_path/'external'
        assert batch.source_reference(external) == external.resolve().as_posix()


@pytest.mark.parametrize('version', ['v004_e16_v4e_split', 'v005_yolo11m_v4e_split'])
def test_prepared_data_relocates_without_original_directory(tmp_path, version):
    """Exercise the actual bundled loader after the source directory disappears."""
    code = ROOT / 'versions' / version / 'code'
    script = textwrap.dedent('''
        import csv, json, shutil, sys
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import yaml
        from PIL import Image
        sys.path.insert(0, sys.argv[1])
        import _bootstrap
        from common import settings, NAMES, sha256, validate_split
        from prepare_data import build
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.data.base import BaseDataset
        parent=Path(sys.argv[2]); source=parent/'source'; rows=[]
        for i,split in enumerate(('train','test')):
            image=source/'images'/split/(split+'.jpg'); label=source/'labels'/split/(split+'.txt')
            image.parent.mkdir(parents=True);label.parent.mkdir(parents=True)
            Image.new('RGB',(64,64),(i*90,80,80)).save(image)
            label.write_text('0 0.5 0.5 0.25 0.25\\n')
            rows.append(dict(name=image.name,split=split,group_id=str(i),source_group=str(i),
                             image_sha256=sha256(image),label_sha256=sha256(label)))
        with (source/'split_manifest.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        (source/'data.yaml').write_text(yaml.safe_dump(dict(names=NAMES)))
        config=settings();config['dataset']=str(source);config['expected_split']={'train':1,'test':1}
        config['expected_manifest_sha256']=sha256(source/'split_manifest.csv')
        config['preparation'].update(tile_size=32,keep_empty_ratio=1.0)
        _,buckets,audit=validate_split(config,True)
        original=parent/'generated';first=build(config,source,buckets,audit,original)
        moved=parent/'different machine'/'renamed cache';shutil.copytree(original,moved)
        original.rename(parent/'original_unavailable')
        source.rename(parent/'source_unavailable')
        with patch('ultralytics.data.utils.check_font'):
            dataset=check_det_dataset(str(moved/'data.yaml'),autodownload=False)
        for split in ('train','val'):
            images=BaseDataset.get_img_files(SimpleNamespace(prefix='',fraction=1.0),dataset[split])
            assert images and all(Path(p).is_file() and Path(p).is_relative_to(moved) for p in images)
        for filename in ('data.yaml','train.txt','val.txt','samples.json'):
            assert sha256(moved/filename)==first['files'][filename]
        for line in (moved/'train.txt').read_text().splitlines():assert line.startswith('./')
        print('Relocation passed: loader uses copied files and portable checksums are stable.')
    ''')
    result = subprocess.run([sys.executable, '-c', script, str(code/'scripts'), str(tmp_path)],
                            cwd=tmp_path, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr


def test_new_version_does_not_inherit_run_or_remove_ultralytics_data(tmp_path, monkeypatch):
    import new_version
    monkeypatch.setattr(new_version, 'WORKSPACE_ROOT', tmp_path)
    source=tmp_path/'versions/v005_example'
    files={
        'code/configs/experiment.yaml':yaml.safe_dump(dict(architecture='baseline',dataset='../../data/splits/d001_train_test_4to1',pretrained='../../pretrained/yolo11m.pt')),
        'code/ultralytics/data/loader.py':'# source must survive\n',
        'code/scripts/train.py':'# train\n',
        'code/data/old.jpg':'old cache',
        'code/training_records/train/contract.json':'{}',
        'code/pretrained/yolo11m.pt':'old duplicate',
        'weights/last.pt':'trained',
        'inference_packages/fusai/old.json':'{}',
        'inference_tools/artifacts/old.json':'{}',
    }
    for name,content in files.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    config=tmp_path/'experiment.yaml';config.write_text('versions: {}\n')
    destination=new_version.create_version('v007_example','v005_example','test',config)
    assert (destination/'code/ultralytics/data/loader.py').is_file()
    for name in ('code/data','code/pretrained','code/training_records','inference_tools/artifacts'):
        assert not (destination/name).exists()
    assert not list((destination/'weights').iterdir())
    assert not list((destination/'inference_packages').iterdir())
    assert yaml.safe_load((destination/'VERSION.yaml').read_text())['trained'] is False
    for name,entry in json.loads((destination/'manifest.json').read_text())['files'].items():
        assert (destination/name).stat().st_size == entry['bytes']
    with pytest.raises(FileExistsError):new_version.create_version('v007_example','v005_example','test',config)
    with pytest.raises(ValueError):new_version.create_version('../escape','v005_example','test',config)
