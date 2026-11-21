#!/usr/bin/env python3
"""Reproducible training-only grouped-CV ablations. NEVER reads Dev/Test.

Typical workflow: --action check -> --action run [--methods ...] -> --action aggregate.
Stage 1: seed 42 and small set of weights on the SAME grouped CV folds.
Stage 2: after stage1 analysis, expand to seeds 42, 52, 62 (must retrain matched baseline).
Full/deployment testing is intentionally NOT included, to avoid accidentally using
an already-consulted official Test set during model selection.
"""
import argparse, gc, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from training_utils_v2 import bin_labels, fold_weights, scored_metrics, training_loss

METHODS={
    'marked_l1':('l1',0.,False),
    'a_mse':('mse',0.,False),
    'aux_0.02':('mse',.02,False),
    'aux_0.05':('mse',.05,False),
    'aux_0.10':('mse',.10,False),
    'aux_0.20':('mse',.20,False),
    'a_huber':('huber',0.,False),
    'both_huber':('both_huber',0.,False),
    'a_reweight_l1':('l1',0.,True),
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_rows(project):
    sys.path.insert(0,str(project/'src'))
    from main import train
    rows,_=train()
    return rows

def verify_manifest(project,manifest):
    if not manifest.is_file():raise FileNotFoundError(f'Missing frozen CV manifest: {manifest}')
    m=pd.read_csv(manifest,encoding='utf-8-sig',keep_default_na=False)
    rows=source_rows(project)
    need=['row_index','id','text','aspect','V','A','group','fold']
    if not set(need)<=set(m):raise ValueError('Manifest lacks required columns')
    if len(m)!=len(rows) or not np.array_equal(m.row_index.to_numpy(),np.arange(len(rows))):
        raise ValueError('Manifest row count/order differs from source training data')
    for k in ('id','text','aspect'):
        if not np.array_equal(m[k].to_numpy(),rows[k].to_numpy()):raise ValueError(f'Manifest data mismatch: {k}')
    if not np.allclose(m[['V','A']].to_numpy(),rows[['V','A']].to_numpy(),atol=1e-6,rtol=1e-6):
        raise ValueError('Manifest labels differ from source training labels')
    if m.fold.isna().any() or m.group.isna().any():raise ValueError('Missing fold/group assignments')
    for group,g in m.groupby('group'):
        if g.fold.nunique()!=1:raise ValueError('A connected group spans multiple folds')
    if m.fold.nunique()<3:raise ValueError('Require at least 3 folds')
    review=m.id.str.split(':').str[0]
    for fold in sorted(m.fold.unique()):
        tr=m.fold!=fold;va=~tr
        if set(review[tr])&set(review[va]):raise ValueError('Cross-fold review leak')
        if set(m.text[tr])&set(m.text[va]):raise ValueError('Cross-fold text leak')
    return rows,m

def train_one(train_rows, eval_rows, *, method, seed, model_path, project, epochs, batch_size, device):
    # Lazy heavy imports: manifest audits and self-tests run on CPU without HF dependencies.
    from transformers import AutoModel, set_seed, get_linear_schedule_with_warmup
    from refine_experiment import tokenizer_for,encode,infer
    from aspect_pooling_model_v1 import AspectPoolingModel
    from bert_experiment import token_batch
    kind,aux_weight,use_weight=METHODS[method]
    set_seed(seed)
    tok=tokenizer_for('marked')
    tokens=encode(train_rows,'marked',tok)
    y=torch.tensor(train_rows[['V','A']].to_numpy(dtype=np.float32))
    labels,ce_weights,a_weights,counts=fold_weights(y,use_weight)
    encoder=AutoModel.from_pretrained(str(model_path),local_files_only=True)
    model=AspectPoolingModel(encoder,means=y.mean(0),start_id=tok.convert_tokens_to_ids('[unused1]'),end_id=tok.convert_tokens_to_ids('[unused2]')).to(device)
    steps=epochs*math.ceil(len(y)/batch_size)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,int(steps*.1),steps)
    amp=device.type=='cuda'
    dtype=torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=amp and dtype==torch.float16)
    generator=torch.Generator().manual_seed(seed)
    history=[]
    for epoch in range(1,epochs+1):
        model.train();start=time.perf_counter();losses=0.0
        for idx in torch.randperm(len(y),generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
                values,aux=model(return_aux=True,**token_batch(tokens,idx,device))
                loss=training_loss(values,y[idx].to(device),aux_logits=aux,aux_weight=aux_weight,
                                   kind=kind,sample_weights=a_weights[idx].to(device) if use_weight else None,
                                   aux_class_weights=ce_weights.to(device) if aux_weight else None)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            old=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=old:scheduler.step()
            losses+=float(loss.detach())*len(idx)
        row={'epoch':epoch,'train_loss':losses/len(y),'seconds':round(time.perf_counter()-start,2)}
        print('TRAIN',method,seed,row,flush=True);history.append(row)
    preds=infer(model,eval_rows,'marked',tok,device,batch_size=batch_size)
    if preds.shape!=(len(eval_rows),2):raise ValueError('Output has wrong shape')
    del model,optimizer,scheduler,tokens,tok
    gc.collect()
    if amp:torch.cuda.empty_cache()
    return preds,history,counts.tolist(),ce_weights.tolist()

def signature(args,rows,manifest):
    return {'architecture':'aspect_span_pool_v1','training_source_file_sha256':{p.name:sha(p) for p in sorted((args.project/'data').glob('*Training*.json'))},
            'manifest_sha256':sha(args.manifest),'rows':len(rows),'fold_count':int(manifest.fold.nunique()),
            'model_path':str(args.model_path.resolve()),'epochs':args.epochs,'batch_size':args.batch_size,
            'learning_rate':2e-5,'max_length':128,'input_variant':'marked',
            'aux_bins':[5.0,6.5], 'prediction_rounding':'clip 1..9 then round 2',
            'dev_test_access':False}

def run(args,rows,m):
    import importlib.util
    if importlib.util.find_spec('transformers') is None:
        raise RuntimeError('Transformers is missing. Install in your own NLP training environment first.')
    if not args.model_path.is_dir() or not (args.model_path/'config.json').exists():
        raise FileNotFoundError('Missing local pretrained BERT base (config.json and weights); supply --model-path')
    from refine_experiment import ROOT
    if not (ROOT/'demo_tokenizer'/'tokenizer.json').is_file():raise FileNotFoundError('Original tokenizer missing')
    dev=torch.device(args.device)
    if dev.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('No CUDA device. Use a GPU PC, or --device cpu (slow).')
    params=signature(args,rows,m)
    base=args.out/'training_v2';base.mkdir(parents=True,exist_ok=True)
    for method in args.methods:
        if method not in METHODS:raise ValueError(f'Unknown method: {method}')
        for seed in args.seeds:
            for fold in sorted(m.fold.unique()):
                fold=int(fold)
                folder=base/method/f'seed_{seed}'/f'fold_{fold}'
                folder.mkdir(parents=True,exist_ok=True)
                config={**params,'method':method,'method_config':METHODS[method], 'seed':seed,'fold':fold}
                confpath=folder/'config.json'
                predpath=folder/'heldout_predictions.csv'
                if confpath.exists() or predpath.exists():
                    if not (confpath.exists() and predpath.exists() and json.loads(confpath.read_text(encoding='utf-8'))==config):
                        raise RuntimeError(f'Resume config mismatch or partial run in {folder}; inspect before proceeding')
                    print('REUSE verified existing result:',folder,flush=True);continue
                tr=rows.loc[m.fold!=fold].reset_index(drop=True)
                held=rows.loc[m.fold==fold].reset_index(drop=True)
                preds,hist,counts,ceweights=train_one(tr,held,method=method,seed=seed,model_path=args.model_path,
                                                       project=args.project,epochs=args.epochs,batch_size=args.batch_size,
                                                       device=dev)
                out=held[['id','text','aspect','V','A']].copy()
                out[['pred_V','pred_A']]=np.round(np.clip(preds,1,9),2)
                out['fold']=fold
                out['group']=m.loc[m.fold==fold,'group'].to_numpy()
                out.to_csv(predpath,index=False,encoding='utf-8-sig')
                (folder/'history.json').write_text(json.dumps({'history':hist,'train_only_bin_counts':counts,
                    'train_only_aux_class_weights':ceweights,'heldout_metrics':scored_metrics(held[['V','A']].to_numpy(),preds)},
                    ensure_ascii=False,indent=2),encoding='utf-8')
                confpath.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
                print('SAVED',folder,flush=True)

def aggregate(args,rows,m):
    # Only summarize complete method×seed grids, never silently ignore missing folds.
    output=[];folds=[]
    target=args.out/'training_v2'
    for method in args.methods:
        for seed in args.seeds:
            buf=[]
            for fold in sorted(m.fold.unique()):
                path=target/method/f'seed_{seed}'/f'fold_{int(fold)}'/'heldout_predictions.csv'
                if not path.exists():raise FileNotFoundError(f'Incomplete training grid: {path}')
                d=pd.read_csv(path,encoding='utf-8-sig',keep_default_na=False)
                check=rows.loc[m.fold==fold].reset_index(drop=True)
                if not d[['id','text','aspect']].equals(check[['id','text','aspect']]):
                    raise ValueError(f'Results do not match held-out source rows: {path}')
                score=scored_metrics(d[['V','A']].to_numpy(),d[['pred_V','pred_A']].to_numpy())
                folds.append({'method':method,'seed':seed,'fold':int(fold),**score})
                buf.append(d)
            joined=pd.concat(buf,ignore_index=True)
            if len(joined)!=len(rows):raise ValueError('OOF size mismatch')
            output.append({'method':method,'seed':seed,**scored_metrics(joined[['V','A']].to_numpy(),joined[['pred_V','pred_A']].to_numpy())})
    summary=pd.DataFrame(output);target.mkdir(exist_ok=True)
    summary.to_csv(target/'oof_seed_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(folds).to_csv(target/'oof_fold_metrics.csv',index=False,encoding='utf-8-sig')
    metric_cols=['V_MAE','A_MAE','mean_MAE','high_A_MAE']
    table=summary.groupby('method')[metric_cols].agg(['mean','std'])
    table.to_csv(target/'seed_mean_std.csv',encoding='utf-8-sig')
    print(summary.to_string(index=False));print(table)
    (target/'aggregate_status.json').write_text(json.dumps({'training_source':'only local Training JSON',
       'official_dev_test_used_for_new_training':False,'status':'training runs aggregated; still need human error audit and independent development gate',
       'methods':args.methods,'seeds':args.seeds,'folds':int(m.fold.nunique()),
       'warning':'Never compare against old CV if fold assignments and training seeds differ; rerun matched baseline.'},
       ensure_ascii=False,indent=2),encoding='utf-8')

def self_test():
    torch.manual_seed(7)
    p=torch.nn.Parameter(torch.randn(12,2)+5)
    y=torch.ones(12,2)*5; y[:3,1]=7.7; y[3:8,1]=5.8
    logits=torch.nn.Parameter(torch.randn(12,3))
    _,w,sw,counts=fold_weights(y,True)
    for kind in ['l1','mse','huber','both_huber']:
        for aux in [0.,.02,.1]:
            a=training_loss(p,y,logits,aux,kind,sw,w)
            assert torch.isfinite(a)
            a.backward();assert torch.isfinite(p.grad).all()
            p.grad=None;logits.grad=None
    print(json.dumps({'self_test':'pass','training_fold_bin_counts':counts.tolist(),
                      'losses':['l1','mse','huber','both_huber'],'aux_weights_tested':[0.,.02,.1]},ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--action',choices=['check','run','aggregate','self-test'],default='check')
    ap.add_argument('--project',type=Path,default=Path('技术成果核对表'))
    ap.add_argument('--out',type=Path,default=Path('optimization_v2'))
    ap.add_argument('--manifest',type=Path,default=None)
    ap.add_argument('--model-path',type=Path,default=None)
    ap.add_argument('--methods',nargs='+',choices=list(METHODS),default=['marked_l1','aux_0.02','aux_0.05','aux_0.10','aux_0.20'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[42])
    ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=ap.parse_args()
    if args.action=='self-test':self_test();return
    args.project=args.project.resolve();args.out=args.out.resolve()
    if args.manifest is None:args.manifest=args.out/'validation'/'proposed_5fold_manifest.csv'
    if args.model_path is None:args.model_path=args.project.parent/'模型缓存'/'chinese-bert-wwm-ext'
    if args.epochs<1 or args.batch_size<1:ap.error('epochs and batch_size must be >=1')
    rows,m=verify_manifest(args.project,args.manifest)
    summary={'verified':'TRAIN only','project':str(args.project),'manifest':str(args.manifest),
             'model_path':str(args.model_path),'folds':int(m.fold.nunique()),'n':len(rows),
             'methods':args.methods,'seeds':args.seeds,
             'required_full_trainings':len(args.methods)*len(args.seeds)*int(m.fold.nunique()),
             'pretrained_base_present':args.model_path.is_dir() and (args.model_path/'config.json').is_file(),
             'training_mode':'fixed epochs, no heldout early stopping, no official dev/test'}
    if args.action=='check':
        print(json.dumps(summary,ensure_ascii=False,indent=2));return
    if args.action=='run':run(args,rows,m)
    else:aggregate(args,rows,m)

if __name__=='__main__':main()

