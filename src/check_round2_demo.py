"""Run actual Windows launchers, including the interactive demo, and record results."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'round2'


def main():
    env=dict(os.environ,PYTHONUTF8='1',HF_HUB_OFFLINE='1')
    results=[]
    for name,stdin,expected in [
        ('三项改进对比演示.cmd','汤非常难喝，但是服务很好。\n汤,服务\n\n\n',['汤  旧模型 V=','服务  旧模型 V=','未全部通过']),
        ('复核三项改进.cmd','\n',['全部复核通过。'])]:
        data=(ROOT/name).read_bytes()
        assert b'\r\n' in data and b'\n' not in data.replace(b'\r\n',b'')
        proc=subprocess.run(['cmd.exe','/d','/c',str(ROOT/name)],input=stdin,
                            encoding='utf-8',errors='replace',capture_output=True,
                            cwd=ROOT,env=env,timeout=180)
        (OUT/(Path(name).stem+'_运行检查.txt')).write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
        assert proc.returncode==0,(name,proc.returncode,proc.stdout,proc.stderr)
        assert all(s in proc.stdout for s in expected),(name,proc.stdout)
        assert '>>>' not in proc.stdout,'Launcher entered Python REPL'
        results.append(dict(launcher=name,returncode=proc.returncode,actual_cmd_execution=True,expected_output_found=True))
    proc=subprocess.run([sys.executable,str(ROOT/'src'/'demo_round2.py'),'--text','汤非常难喝，但是服务很好。',
                         '--aspects','汤','服务','--json'],env=env,capture_output=True,encoding='utf-8',timeout=180)
    assert proc.returncode==0,proc.stderr
    output=json.loads(proc.stdout)
    assert [p['aspect'] for p in output['predictions']]==['汤','服务']
    for p in output['predictions']:
        assert all(1<=p[k]<=9 for k in ['old_V','old_A','candidate_V','candidate_A'])
    (OUT/'demo_example.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'launcher_check.json').write_text(json.dumps(dict(passed=True,launchers=results,
        cli_json_check='passed',sample_is_unlabelled=True),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Actual CMD launchers and JSON demo passed.')


if __name__=='__main__':main()
