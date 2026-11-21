"""Pure-PyTorch regression and auxiliary-loss utilities; no Transformers dependency."""
import numpy as np
import torch
import torch.nn.functional as F

BINS=(5.0,6.5)

def bin_labels(y):
    return torch.bucketize(y[:,1].contiguous(),torch.tensor(BINS,device=y.device),right=True)

def fold_weights(y, reweight=False):
    """Derive all weights from the current TRAINING fold ONLY."""
    bins=bin_labels(y)
    counts=torch.bincount(bins,minlength=3).float()
    raw=(len(y)/(3*counts.clamp_min(1))).sqrt().clamp(.5,3)
    sample=raw[bins]
    if reweight:sample=sample/sample.mean()
    else:sample=torch.ones_like(sample)
    return bins,raw,sample,counts

def training_loss(pred,true,aux_logits=None,aux_weight=0.0,kind='l1',sample_weights=None,aux_class_weights=None):
    if pred.shape!=true.shape or pred.shape[1]!=2:
        raise ValueError('Regression heads must predict [N,2]')
    delta=pred.float()-true.float()
    v=delta[:,0].abs()
    a=delta[:,1]
    if kind=='l1':a=a.abs()
    elif kind=='mse':a=a.square()
    elif kind=='huber':a=F.huber_loss(pred[:,1].float(),true[:,1].float(),reduction='none',delta=.5)
    elif kind=='both_huber':
        v=F.huber_loss(pred[:,0].float(),true[:,0].float(),reduction='none',delta=.5)
        a=F.huber_loss(pred[:,1].float(),true[:,1].float(),reduction='none',delta=.5)
    else:raise ValueError(f'Unsupported regression loss: {kind}')
    if sample_weights is not None:
        a=(a*sample_weights).mean()
    else:a=a.mean()
    loss=(v.mean()+a)/2
    if aux_weight:
        if aux_logits is None:raise ValueError('Auxiliary logits required')
        # bins defined from true labels; only training fold labels reach this function
        loss=loss+float(aux_weight)*F.cross_entropy(aux_logits.float(),bin_labels(true),weight=aux_class_weights)
    return loss

def scored_metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.round(np.clip(np.asarray(p,dtype=float),1,9),2)
    if y.shape!=p.shape or y.ndim!=2 or y.shape[1]!=2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Invalid arrays for scoring')
    e=np.abs(y-p);mask=y[:,1]>=6.5
    d={'n':len(y),'V_MAE':float(e[:,0].mean()),'A_MAE':float(e[:,1].mean()),
       'mean_MAE':float(e.mean()),'high_A_n':int(mask.sum()),'high_A_MAE':float(e[mask,1].mean()) if mask.any() else None}
    for j,k in enumerate(['V_PCC','A_PCC']):
        d[k]=float(np.corrcoef(y[:,j],p[:,j])[0,1]) if np.std(y[:,j])>0 and np.std(p[:,j])>0 else None
    return d
