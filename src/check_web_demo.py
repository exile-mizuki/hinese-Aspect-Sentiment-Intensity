"""Exercise the running local web service against saved model outputs and boundary cases."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    args=parser.parse_args()
    base=args.url
    def request(path,data=None,headers=None,raw=None):
        body=json.dumps(data).encode() if data is not None else raw
        req=urllib.request.Request(base+path,data=body,headers=headers or {})
        try:
            with urllib.request.urlopen(req,timeout=60) as r:return r.status,r.read(),r.headers
        except urllib.error.HTTPError as r:return r.code,r.read(),r.headers
    for attempt in range(60):
        status,body,_=request('/api/status')
        if json.loads(body).get('state')!='loading':break
        time.sleep(1)
    assert status==200 and json.loads(body)['state']=='ready',body
    status,body,_=request('/api/meta')
    meta=json.loads(body)
    assert meta['candidate_model']['accepted'] is False
    headers={'Content-Type':'application/json','X-Demo-Token':meta['token']}
    sample=dict(text='汤非常难喝，但是服务很好。',aspects=['汤','服务'],compare=True)
    status,body,_=request('/api/predict',sample,headers)
    assert status==200,body
    response=json.loads(body)
    expected=json.loads((Path(__file__).resolve().parents[1]/'results/round2/demo_example.json').read_text(encoding='utf-8'))
    for got,ref in zip(response['predictions'],expected['predictions']):
        assert got['aspect']==ref['aspect']
        for actual,saved in [('V','old_V'),('A','old_A'),('candidate_V','candidate_V'),('candidate_A','candidate_A')]:
            assert abs(got[actual]-ref[saved])<1e-4,(got,ref)
    checks=['web predictions match saved CPU CLI predictions']
    status,body,_=request('/api/predict',dict(sample,aspects=['汤','汤'],compare=False),headers)
    single=json.loads(body)
    assert status==200 and len(single['predictions'])==1 and 'candidate_V' not in single['predictions'][0]
    assert abs(single['predictions'][0]['V']-response['predictions'][0]['V'])<.011
    checks.append('repeat requests reuse models; duplicate aspects and compare-off work')
    cases=[{'text':'','aspects':['汤']},{'text':'汤不错','aspects':[]},
           {'text':'汤不错','aspects':['牛肉']},{'text':'汤不错','aspects':['汤']*9},
           {'text':'汤'+('很好喝'*100),'aspects':['汤']},
           {'text':'[unused1]汤很好','aspects':['汤']},{'text':123,'aspects':['汤']},
           {'text':'汤很好','aspects':['汤'],'compare':'true'}]
    for payload in cases:
        code,data,_=request('/api/predict',payload,headers)
        assert code==400 and json.loads(data)['error'],(payload,data)
    checks.append('empty/missing/malformed/long-token/special-token inputs return clear errors')
    for payload in [dict(text='总体不错',aspects=['NULL']),dict(text='🙂汤很棒，服务不错。',aspects=['汤','服务']),
                    dict(text='<script>alert(1)</script>这碗汤不错。',aspects=['汤'])]:
        code,data,_=request('/api/predict',payload,headers)
        assert code==200,data
        got=json.loads(data)
        assert all(got['text'][e['start']:e['end']]==e['text'] for e in got['rule_cues'])
    checks.append('NULL, emoji and literal HTML remain data; rule offsets are valid')
    assert request('/api/predict',sample,{'Content-Type':'application/json'})[0]==403
    assert request('/api/predict',sample,dict(headers,Origin='https://example.org'))[0]==403
    assert request('/api/status',headers={'Host':'example.org'})[0]==403
    assert request('/checkpoints/round2/candidate.pt')[0]==404
    assert request('/../README.md')[0]==404
    assert request('/api/predict',headers=headers,raw=b'{ broken')[0]==400
    checks.append('session/origin/host and static-path guards passed')
    out=Path(__file__).resolve().parents[1]/'results/web_demo'
    out.mkdir(exist_ok=True)
    (out/'api_verification.json').write_text(json.dumps(dict(passed=True,checks=checks,
         sample_response=response,web_test_time=time.strftime('%Y-%m-%dT%H:%M:%S')),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Web API checks passed:',len(checks))


if __name__=='__main__':main()
