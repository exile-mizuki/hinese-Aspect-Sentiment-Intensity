"""Package technical sources, frozen predictions and the seed-42 offline demo model."""
import hashlib
import json
import zipfile
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    archive = root.parent / '中文方面情感强度预测-技术成果包.zip'
    temp = archive.with_suffix('.zip.tmp')
    selected = ['README.md', '技术任务完成说明.md', '技术成果核对表.xlsx', '四位组员任务安排.md',
                'requirements.txt', 'requirements-bert.txt', 'requirements-analysis.txt',
                '运行演示.cmd', '复核技术成果.cmd', 'checkpoints/bert_aspect/best.pt']
    files = [root / name for name in selected]
    for group in ('src', 'data', 'demo_tokenizer', 'results'):
        files.extend(p for p in sorted((root / group).rglob('*'))
                     if p.is_file() and '__pycache__' not in p.parts
                     and not p.name.startswith('~$') and p.name != 'package_log.json')
    manifest = {str(p.relative_to(root)).replace('\\', '/'): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in files}
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=2) as out:
        for p in files:
            out.write(p, (Path(root.name) / p.relative_to(root)).as_posix())
        out.writestr(root.name + '/技术包文件校验.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(temp) as check:
        assert check.testzip() is None
        assert len(check.namelist()) == len(set(check.namelist())) == len(files) + 1
        assert not any(name.endswith('.pptx') for name in check.namelist())
        for name, details in manifest.items():
            assert check.getinfo(root.name + '/' + name).file_size == details['bytes']
    temp.replace(archive)
    result = dict(archive=str(archive), bytes=archive.stat().st_size, files=len(files)+1,
                  sha256=digest(archive), zip_crc_check='passed',
                  includes='code, data, metrics, figures, cases, review workbook, seed-42 aspect demo checkpoint',
                  excludes='slides, member contribution sheet, base pretrained model, additional training checkpoints')
    (root / 'results/technical_review/package_log.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
