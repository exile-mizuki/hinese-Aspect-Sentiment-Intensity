"""Package the completed round with the three checkpoints required by bundled launchers."""
import json
import zipfile
from pathlib import Path
from package_technical import digest


def main():
    root=Path(__file__).resolve().parents[1]
    report=root/'results/round2'
    assert json.loads((report/'verification.json').read_text(encoding='utf-8'))['passed']
    assert json.loads((report/'launcher_check.json').read_text(encoding='utf-8'))['passed']
    archive=root.parent/'中文方面情感强度预测-三项深化完整版.zip'
    temp=archive.with_suffix('.zip.tmp')
    names=['README.md','三项改进版使用说明.md','本轮方法来源.md','改进版使用说明.md',
           '技术任务完成说明.md','技术成果核对表.xlsx','四位组员任务安排.md',
           'requirements.txt','requirements-bert.txt','requirements-analysis.txt',
           '运行演示.cmd','复核技术成果.cmd','改进演示.cmd','三项改进对比演示.cmd','复核三项改进.cmd',
           'checkpoints/bert_aspect/best.pt','checkpoints/refinement/marked/seed_42/best.pt',
           'checkpoints/round2/candidate.pt']
    skip={f'results/{m}_{s}.txt' for m in ['baseline','improved','sentence_only'] for s in ['dev','test']}
    files=[root/name for name in names]
    for folder in ['src','data','demo_tokenizer','results']:
        files.extend(p for p in sorted((root/folder).rglob('*')) if p.is_file()
                     and '__pycache__' not in p.parts and 'console' not in p.name.lower()
                     and not p.name.endswith('.inspect.ndjson')
                     and p.name not in ['package_log.json','refinement_package_log.json','round2_package_log.json']
                     and p.relative_to(root).as_posix() not in skip)
    assert len(files)==len(set(files))
    manifest={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files}
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in files:z.write(p,root.name+'/'+p.relative_to(root).as_posix())
        z.writestr(root.name+'/三项深化包文件校验.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(files)+1
        assert not any(n.endswith('.pptx') for n in z.namelist())
        for name,details in manifest.items(): assert z.getinfo(root.name+'/'+name).file_size==details['bytes']
    temp.replace(archive)
    result=dict(archive=str(archive),bytes=archive.stat().st_size,files=len(files)+1,
                sha256=digest(archive),zip_crc_check='passed',
                includes='three demo checkpoints, tokenizer, data, sources, frozen results, reports, launchers',
                excludes='PPT, Python environment, pretrained base model, other training checkpoints, caches and console logs')
    (report/'round2_package_log.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
