"""Reproduce data, metric and case analysis from frozen predictions, optionally test CPU inference."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true', help='also load the packaged model and run CPU examples')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'results' / 'technical_review'
    out.mkdir(parents=True, exist_ok=True)
    scripts = ['data_audit.py', 'experiment_analysis.py', 'case_analysis.py']
    if args.demo:
        scripts.append('check_demo.py')
    env = {**os.environ, 'PYTHONUTF8': '1', 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '3'}
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    runs = []
    for script in scripts:
        print(f'\n正在复核：{script}', flush=True)
        started = time.perf_counter()
        result = subprocess.run([sys.executable, str(root / 'src' / script)], cwd=root,
                                env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (out / (script + '.log')).write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
        runs.append(dict(script=script, returncode=result.returncode, seconds=round(time.perf_counter()-started, 2)))
        (out / 'run_log.json').write_text(json.dumps(dict(python=sys.executable, demo=args.demo, runs=runs),
                                                     ensure_ascii=False, indent=2), encoding='utf-8')
        if result.returncode:
            print(result.stdout, result.stderr)
            raise SystemExit(result.returncode)
        print(f'通过，运行日志已保存。耗时 {runs[-1]["seconds"]} 秒。', flush=True)
    print(f'\n全部复核通过：{out}')
    print('CSV、PNG 和分析文字已更新；Excel 核对表为此次归档数据的整理版本。')


if __name__ == '__main__':
    main()
