"""Self-contained web demo edition, preserving previous technical results."""
import json
import zipfile
from pathlib import Path
from package_technical import digest


def main():
    root=Path(__file__).resolve().parents[1]
    results=root/'results/web_demo'
    for check in ['api_verification.json','browser_verification.json']:
        assert json.loads((results/check).read_text(encoding='utf-8'))['passed']
    names=['README.md','网页演示使用说明.md','网页演示参考与设计.md','启动网页演示.cmd',
        '三项改进版使用说明.md','本轮方法来源.md','改进版使用说明.md',
        '技术任务完成说明.md','技术成果核对表.xlsx','四位组员任务安排.md',
        'requirements.txt','requirements-bert.txt','requirements-analysis.txt',
        '运行演示.cmd','复核技术成果.cmd','改进演示.cmd','三项改进对比演示.cmd','复核三项改进.cmd',
        'checkpoints/bert_aspect/best.pt','checkpoints/refinement/marked/seed_42/best.pt','checkpoints/round2/candidate.pt']
    files=[root/n for n in names]
    skip={f'results/{m}_{s}.txt' for m in ['baseline','improved','sentence_only'] for s in ['dev','test']}
    for folder in ['src','web','data','demo_tokenizer','results']:
        files.extend(p for p in sorted((root/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts
                     and 'console' not in p.name.lower() and not p.name.endswith('.inspect.ndjson')
                     and 'package_log' not in p.name and p.relative_to(root).as_posix() not in skip)
    assert len(files)==len(set(files))
    manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files}
    archive=root.parent/'中文方面情感强度预测-网页演示完整版.zip'
    temp=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in files:z.write(p,root.name+'/'+p.relative_to(root).as_posix())
        z.writestr(root.name+'/网页包文件校验.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(files)+1
        for name in names:assert root.name+'/'+name in z.namelist()
    temp.replace(archive)
    report=dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,
        sha256=digest(archive),zip_crc_check='passed',requires_existing_python_dependencies=True,
        includes_models=True,includes_pretrained_base=False)
    (results/'web_package_log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
