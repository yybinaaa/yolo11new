"""Evaluate cached local V4E outputs, or run V4E on the 640-image holdout first."""
import _bootstrap  # noqa
import argparse
import json
from pathlib import Path
from common import VERSION, settings, validate_split, write_json
from metrics import load_gt, evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epoch',type=int);p.add_argument('--weights')
    p.add_argument('--predictions',type=Path,help='Evaluate existing v4e_predictions.json without inference')
    p.add_argument('--device',default=settings()['inference']['device']);p.add_argument('--exclude-class',nargs='*',default=None)
    a=p.parse_args();c=settings();root,buckets,audit=validate_split(c)
    if a.exclude_class is None:a.exclude_class=c['inference']['exclude_classes']
    if a.weights and a.epoch is not None:raise ValueError('Choose --weights or --epoch, not both')
    if a.predictions:
        path=a.predictions.resolve();out=path.parent
    else:
        from predict import run
        out=run(argparse.Namespace(weights=a.weights,epoch=a.epoch,source=None,dataset_name=c['inference']['default_dataset_name'],device=a.device,
                                   exclude_class=a.exclude_class,visualize=False));path=out/'v4e_predictions.json'
    # Evidence must include the source/run contract, not just a JSON of boxes.
    contract=json.loads((out/'contract.json').read_text(encoding='utf-8'))
    if contract['split']!=audit['manifest_sha256']:raise ValueError('Prediction split mismatch')
    if set(contract['exclude_classes'])!=set(a.exclude_class):raise ValueError('Use the same exclusions as inference')
    expected={r['name']:r['image_sha256'] for r in buckets['test']}
    actual={r['name']:r['sha256'] for r in contract['images']}
    if actual!=expected:raise ValueError('Prediction images do not match the complete fixed holdout')
    rows=json.loads(path.read_text(encoding='utf-8'));gt=load_gt(root,buckets['test'])
    result=evaluate(rows,gt,c['evaluation']['iou'],a.exclude_class)
    result['data_role']='development_validation';result['split_sha256']=audit['manifest_sha256']
    result['by_filename_family']={}
    for family,prefix in [('C_prefix',True),('numeric_prefix',False)]:
        subset={k:v for k,v in gt.items() if k.startswith('C')==prefix}
        result['by_filename_family'][family]=evaluate([r for r in rows if r['image_id'] in subset],subset,c['evaluation']['iou'],a.exclude_class)
    write_json(out/'metrics.json',result)
    lines=['# 本地V4E验证结果','',f"图片：{result['images']}；TP/FP/FN：{result['tp']}/{result['fp']}/{result['fn']}",
           f"Precision={result['precision']:.4f}；Recall={result['recall']:.4f}；AP50={result['mAP50']:.4f}",'',
           '此为本地开发验证口径，不是官方平台成绩。AP以V4E导出框计算，不能等同于未截断置信度候选的检测器AP。','',
           '|类别|GT|TP|FP|FN|Recall|','|---|---:|---:|---:|---:|---:|']
    for cls,d in result['per_class'].items():lines.append(f"|{cls}|{d['gt']}|{d['tp']}|{d['fp']}|{d['fn']}|{d['recall'] if d['recall'] is not None else 'N/A'}|")
    (out/'evaluation_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    from archive import refresh
    refresh()
    print(json.dumps({k:result[k] for k in ('images','precision','recall','mAP50','tp','fp','fn')},ensure_ascii=False))
if __name__=='__main__':main()
