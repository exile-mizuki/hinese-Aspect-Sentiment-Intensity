#!/usr/bin/env python3
"""Create a standalone replacement web_demo.py for an INDEPENDENT project copy."""
import argparse, difflib
from pathlib import Path

def patch(original):
    s=original
    old="""        self.current_info=dict(CURRENT)
        self.candidate_info=dict(CANDIDATE)
"""
    new="""        self.current_info=dict(CURRENT)
        self.candidate_info=dict(CANDIDATE,available=False)
        self.candidate=None
"""
    assert s.count(old)==1;s=s.replace(old,new)
    old="""            self.current=load_model('marked',42,self.device)
            self.message='当前模型已载入，正在准备实验对照。'
            state=torch.load(ROOT/'checkpoints/round2/candidate.pt',map_location='cpu',weights_only=True)
            decision=json.loads((ROOT/'results/round2/deployment.json').read_text(encoding='utf-8'))
            assert state['method']==decision['candidate']
            self.candidate_info['id']=state['method']
            self.candidate_info['accepted']=decision['accepted']
            self.candidate=ArousalModel(AutoModel.from_config(AutoConfig.from_pretrained(ROOT/'demo_tokenizer',local_files_only=True)))
            self.candidate.load_state_dict(state['model'])
            self.current.eval();self.candidate.eval()
            self.message='本地模型已就绪'
            self.state='ready'
            print(self.message,flush=True)
"""
    new="""            self.current=load_model('marked',42,self.device)
            self.current.eval()
            # Candidate is OPTIONAL: its absence must not disable the approved model.
            # Never automatically deploy a candidate that has not passed the gate.
            candidate_path=ROOT/'checkpoints/round2/candidate.pt'
            decision_path=ROOT/'results/round2/deployment.json'
            if candidate_path.is_file() and decision_path.is_file():
                try:
                    state=torch.load(candidate_path,map_location='cpu',weights_only=True)
                    decision=json.loads(decision_path.read_text(encoding='utf-8'))
                    if state['method']!=decision['candidate']:
                        raise ValueError('候选模型与部署记录不一致')
                    trial=ArousalModel(AutoModel.from_config(AutoConfig.from_pretrained(ROOT/'demo_tokenizer',local_files_only=True)))
                    trial.load_state_dict(state['model'])
                    trial.eval()
                    self.candidate=trial
                    self.candidate_info.update(id=state['method'],accepted=bool(decision['accepted']),available=True)
                except Exception:
                    print('实验候选模型不可用，当前正式模型仍可单独演示。',flush=True)
                    traceback.print_exc()
            self.message='正式模型已就绪' + ('；实验候选可对照' if self.candidate is not None else '；候选缺失或不兼容，仅演示正式模型')
            self.state='ready'
            print(self.message,flush=True)
"""
    assert s.count(old)==1;s=s.replace(old,new)
    old="""        if not isinstance(compare,bool):raise ValueError('模型对比选项应为开或关。')
"""
    new="""        if not isinstance(compare,bool):raise ValueError('模型对比选项应为开或关。')
        if compare and self.candidate is None:
            raise ValueError('实验候选模型尚未加载；请取消对比选项，仍可使用正式模型。')
"""
    assert s.count(old)==1;s=s.replace(old,new)
    return s

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    src=a.project/'src'/'web_demo.py'
    s=src.read_text(encoding='utf-8')
    fixed=patch(s)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'web_demo.py').write_text(fixed,encoding='utf-8')
    diff=''.join(difflib.unified_diff(s.splitlines(keepends=True),fixed.splitlines(keepends=True),fromfile='original/src/web_demo.py',tofile='independent_copy/src/web_demo.py'))
    (a.out/'web_demo_v2.patch').write_text(diff,encoding='utf-8')
    print('Patch generated. No original files modified.')
