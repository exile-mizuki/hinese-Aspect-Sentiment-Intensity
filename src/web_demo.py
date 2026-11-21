"""Local-only Aspect Lab web UI backed by the project's actual saved models.

No new runtime dependencies: Python HTTP server + existing NLP environment.
"""
import argparse
import csv
import hashlib
import json
import os
import secrets
import threading
import time
import traceback
import urllib.request
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'web'
VERSION='aspect-web-1'
PROJECT_ID=hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
CURRENT=dict(id='marked_l1_seed42_epoch3',label='正式模型 · BERT 方面标记 · Epoch 3')
CANDIDATE=dict(id='arousal_mse_aux',label='实验候选 · A辅助分类',accepted=False)


def numeric_rows(name):
    with (ROOT/'results'/'round2'/name).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for k,v in row.items():
            if k not in ('method','split'): row[k]=float(v) if v else None
    return rows


class Engine:
    def __init__(self):
        self.state='loading'
        self.message='正在载入本地模型，首次打开请稍候。'
        self.gate=threading.Lock()
        self.token=secrets.token_urlsafe(32)
        self.current_info=dict(CURRENT)
        self.candidate_info=dict(CANDIDATE)

    def load(self):
        try:
            import torch
            from transformers import AutoConfig, AutoModel
            from refine_experiment import load_model, tokenizer_for
            from round2_experiment import ArousalModel
            torch.set_num_threads(3)
            self.device=torch.device('cpu')
            self.tokenizer=tokenizer_for('marked')
            # Release v1: official frozen full-training baseline.
            import hashlib
            folder=ROOT.parent/'optimization_v2/final_model/marked_l1/seed_42'
            checkpoint=folder/'model.pt'
            release_path=ROOT.parent/'optimization_v2/release_v1/baseline_manifest.json'
            release=json.loads(release_path.read_text(encoding='utf-8-sig'))

            digest=hashlib.sha256()
            with checkpoint.open('rb') as stream:
                for block in iter(lambda: stream.read(4*1024*1024), b''):
                    digest.update(block)
            actual_sha=digest.hexdigest()

            assert actual_sha.lower()==release['checkpoint_sha256'].lower(), 'Frozen checkpoint SHA256 mismatch'

            state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
            assert state['method']=='marked_l1'
            assert state['seed']==42
            assert state['epoch']==3
            assert state['train_rows']==7476
            assert state['input_variant']=='marked'
            assert state['max_length']==128

            config=AutoConfig.from_pretrained(folder/'encoder_config',local_files_only=True)
            self.current=ArousalModel(AutoModel.from_config(config))
            self.current.load_state_dict(state['model'],strict=True)
            self.current.eval()
            del state

            self.current_info.update({
                'id':'marked_l1_seed42_epoch3',
                'label':'正式模型 · BERT 方面标记 · Epoch 3',
                'epoch':3,
                'seed':42,
                'train_rows':7476,
                'release_method':'marked_l1',
                'release_version':'release_v1',
                'checkpoint_sha256':actual_sha
            })
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
        except Exception as exc:
            self.message=f'模型加载失败：{type(exc).__name__}。请在启动窗口查看具体原因；确保已解压完整模型和分词器。'
            self.state='error'
            traceback.print_exc()

    def metadata(self):
        return dict(token=self.token,current_model=self.current_info,candidate_model=self.candidate_info,
            cv_metrics=numeric_rows('cv_pooled_metrics.csv'),official_metrics=numeric_rows('official_metrics.csv'),
            guide=dict(score_min=1,score_max=9,max_aspects=8,max_tokens=128,
                task='指定方面的 V/A 回归预测，方面词由用户提供。',
                report_baseline='当前基线为完整训练集训练 3 轮的正式 marked_l1 模型；历史候选对比仅供探索性展示。',
                privacy='评论仅发给本机服务；服务不保存评论，也不发送到外部平台。',
                evaluation='既往查看过测试集，测试比较属于探索性结果。'))

    def predict(self,payload):
        import numpy as np
        import pandas as pd
        from refine_experiment import infer, marked_text
        from round2_errors import cues
        if not isinstance(payload,dict):raise ValueError('请求内容必须是一个对象。')
        text=payload.get('text');aspects=payload.get('aspects');compare=payload.get('compare',False)
        if not isinstance(text,str) or not text.strip():raise ValueError('请先填写一条餐饮评论。')
        text=text.strip()
        if len(text)>500:raise ValueError('评论最多 500 个字符；建议只输入一到两句话。')
        if not isinstance(compare,bool):raise ValueError('模型对比选项应为开或关。')
        if not isinstance(aspects,list) or not aspects or any(not isinstance(a,str) for a in aspects):
            raise ValueError('请至少指定一个方面词，例如“汤”或“服务”。')
        if len(aspects)>8:raise ValueError('一次最多分析 8 个方面。')
        aspects=list(dict.fromkeys(a.strip() for a in aspects))
        if any(not a or len(a)>60 for a in aspects):raise ValueError('方面词不能为空，且最多 60 个字符。')
        missing=[a for a in aspects if a!='NULL' and a not in text]
        if missing:raise ValueError('这些方面词没有出现在评论里：'+ '、'.join(missing)+'。请使用原文中的词。')
        special=['[unused1]','[unused2]','[CLS]','[SEP]','[PAD]','[MASK]','[UNK]']
        if any(s in text or any(s in a for a in aspects) for s in special):
            raise ValueError('评论和方面词不能包含模型的特殊标记。')
        lengths=[len(self.tokenizer(a,marked_text(text,a),truncation=False)['input_ids']) for a in aspects]
        if max(lengths)>128:raise ValueError(f'评论分词后最长为 {max(lengths)} 个词元，超过模型的 128 上限。请缩短评论，保留要分析的方面及相关评价。')
        started=time.perf_counter()
        rows=pd.DataFrame([dict(id='web:0',text=text,aspect=a) for a in aspects])
        values=infer(self.current,rows,'marked',self.tokenizer,self.device)
        additional=infer(self.candidate,rows,'marked',self.tokenizer,self.device) if compare else None
        if not np.isfinite(values).all() or (compare and not np.isfinite(additional).all()):
            raise RuntimeError('模型输出异常，请重试。')
        records=[]
        def score(x):return round(float(np.clip(x,1,9)),2)
        for i,a in enumerate(aspects):
            item=dict(aspect=a,V=score(values[i,0]),A=score(values[i,1]))
            if compare:
                item.update(candidate_V=score(additional[i,0]),candidate_A=score(additional[i,1]))
                item.update(delta_V=round(item['candidate_V']-item['V'],2),delta_A=round(item['candidate_A']-item['A'],2))
            records.append(item)
        _,evidence=cues(text,aspects[0],len(aspects)>1)
        warnings=['NULL 代表隐式方面；没有可高亮的字面词。'] if 'NULL' in aspects else []
        return dict(text=text,aspects=aspects,compare=compare,created_at=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=round((time.perf_counter()-started)*1000),
            models=dict(current=self.current_info,candidate=self.candidate_info),predictions=records,
            warnings=warnings,rule_cues=evidence,
            notes=['V/A 是 1–9 的回归分数，不是概率或置信度。','词语高亮只表示原文位置；规则线索不是模型解释。',
                   '演示句没有人工真值，不能据此评价准确率。'])


class Handler(BaseHTTPRequestHandler):
    server_version='AspectLab/1'
    def setup(self):
        super().setup()
        self.connection.settimeout(15)
    def log_message(self,*args):pass

    def send_payload(self,status,data,content_type='application/json; charset=utf-8'):
        body=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8') if not isinstance(data,bytes) else data
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass

    def same_host(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

    def do_GET(self):
        if not self.same_host():return self.send_payload(403,dict(error='仅允许从本机地址访问。'))
        route=self.path.split('?',1)[0]
        if route=='/api/status':
            e=self.server.engine
            return self.send_payload(200,dict(state=e.state,message=e.message,version=VERSION,project_id=PROJECT_ID))
        if route=='/api/meta':
            try:return self.send_payload(200,self.server.engine.metadata())
            except Exception:
                traceback.print_exc()
                return self.send_payload(500,dict(error='实验结果文件缺失，请使用完整项目。'))
        assets={'/':('index.html','text/html; charset=utf-8'),'/index.html':('index.html','text/html; charset=utf-8'),
                '/style.css':('style.css','text/css; charset=utf-8'),'/app.js':('app.js','application/javascript; charset=utf-8')}
        if route not in assets:return self.send_payload(404,dict(error='没有这个页面。'))
        filename,mime=assets[route]
        try:self.send_payload(200,(WEB/filename).read_bytes(),mime)
        except OSError:self.send_payload(500,dict(error='网页文件缺失，请确认 web 文件夹完整。'))

    def do_POST(self):
        if not self.same_host():return self.send_payload(403,dict(error='仅允许从本机地址访问。'))
        origin=self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
            return self.send_payload(403,dict(error='请求来源不匹配。请从本地演示页操作。'))
        if self.path!='/api/predict':return self.send_payload(404,dict(error='没有这个接口。'))
        engine=self.server.engine
        if not secrets.compare_digest(self.headers.get('X-Demo-Token',''),engine.token):
            return self.send_payload(403,dict(error='页面会话已变更，请刷新网页。'))
        if engine.state!='ready':return self.send_payload(503,dict(error=engine.message))
        length=self.headers.get('Content-Length','')
        if not length.isdecimal() or not 0<int(length)<=32768:
            return self.send_payload(400,dict(error='请求为空或过长，请缩短评论。'))
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':
            return self.send_payload(400,dict(error='请求格式应为 JSON。'))
        if not engine.gate.acquire(blocking=False):return self.send_payload(409,dict(error='正在处理另一条评论，请稍后再试。'))
        try:
            payload=json.loads(self.rfile.read(int(length)))
            result=engine.predict(payload)
            self.send_payload(200,result)
        except (ValueError,UnicodeDecodeError) as exc:self.send_payload(400,dict(error=str(exc)))
        except Exception:
            traceback.print_exc()
            self.send_payload(500,dict(error='本次分析未完成，请重试；若持续失败，请查看启动窗口。'))
        finally:engine.gate.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--strict-port',action='store_true')
    args=parser.parse_args()
    if not 1024<=args.port<=65525:parser.error('Port must be 1024..65525')
    for port in range(args.port,args.port+(1 if args.strict_port else 11)):
        url=f'http://127.0.0.1:{port}'
        try:
            with urllib.request.urlopen(url+'/api/status',timeout=.3) as response:existing=json.load(response)
            if existing.get('version')==VERSION and existing.get('project_id')==PROJECT_ID:
                print('演示已在运行：'+url,flush=True)
                if not args.no_browser:webbrowser.open(url)
                return
        except Exception:pass
        try:
            server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
            break
        except OSError:continue
    else:raise SystemExit('没有找到可用端口，请关闭旧的演示窗口后重试。')
    server.daemon_threads=True
    server.engine=Engine()
    print('观感 · 中文方面情感强度预测\n本地地址：'+url+'\n请保持此窗口开启；按 Ctrl+C 关闭服务。',flush=True)
    threading.Thread(target=server.engine.load,daemon=True).start()
    if not args.no_browser:threading.Timer(.7,lambda:webbrowser.open(url)).start()
    try:server.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:print('\n网页演示已关闭。',flush=True)
    finally:server.server_close()


if __name__=='__main__':main()
