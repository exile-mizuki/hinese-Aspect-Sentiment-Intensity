"""Build a self-contained comparison demo package without redundant prediction copies."""
import json
import zipfile
from pathlib import Path
from package_technical import digest


def main():
    root=Path(__file__).resolve().parents[1]
    out=root.parent/'中文方面情感强度预测-方面标记改进版.zip'
    temp=out.with_suffix('.zip.tmp')
    selection=json.loads((root/'results/refinement/final_selection.json').read_text(encoding='utf-8'))
    assert selection['selected_system']=='candidate_single' and selection['selected_variant']=='marked'
    names=['README.md','改进版使用说明.md','技术任务完成说明.md','技术成果核对表.xlsx',
           '四位组员任务安排.md','requirements.txt','requirements-bert.txt','requirements-analysis.txt',
           '运行演示.cmd','复核技术成果.cmd','改进演示.cmd',
           'checkpoints/bert_aspect/best.pt','checkpoints/refinement/marked/seed_42/best.pt']
    skip={f'results/{m}_{s}.txt' for m in ['baseline','improved','sentence_only'] for s in ['dev','test']}
    skip.update(['results/bert_console.txt','results/repeats/seed_17-console.txt','results/repeats/seed_2026-console.txt'])
    files=[root/name for name in names]
    for group in ['src','data','demo_tokenizer','results']:
        files.extend(p for p in sorted((root/group).rglob('*')) if p.is_file()
                     and '__pycache__' not in p.parts and p.name not in ['package_log.json','refinement_package_log.json']
                     and p.relative_to(root).as_posix() not in skip)
    manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files}
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in files:z.write(p,root.name+'/'+p.relative_to(root).as_posix())
        z.writestr(root.name+'/改进包文件校验.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(files)+1
        for name in names:assert root.name+'/'+name in z.namelist()
    temp.replace(out)
    log=dict(archive=str(out),bytes=out.stat().st_size,files=len(files)+1,sha256=digest(out),zip_crc_check='passed',
             demo_models=['original seed42','marked seed42'],other_training_checkpoints='retained locally; not packaged')
    (root/'results/refinement/refinement_package_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(log,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
