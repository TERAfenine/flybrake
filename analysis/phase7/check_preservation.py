"""Read-only checks of all pre-Phase7 files; record result only under Phase7."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.audit import sha256


def main():
    out=ROOT/'analysis/phase7'
    before=json.loads((out/'protected_before.json').read_text(encoding='utf-8'))
    missing=[p for p in before if not (ROOT/p).is_file()]
    changed=[p for p,d in before.items() if (ROOT/p).is_file() and sha256(ROOT/p)!=d]
    unexpected=[]
    for folder in ['data','results','analysis/phase6']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            if p.is_relative_to(ROOT/'results/phase7_20261105'):continue
            if p.relative_to(ROOT).as_posix() not in before:unexpected.append(p.relative_to(ROOT).as_posix())
    result=dict(protected=len(before),unchanged=len(before)-len(missing)-len(changed),missing=missing,changed=changed,
        unexpected_new_research_files=unexpected,allowed_new_result_bundle='results/phase7_20261105',passed=not(missing or changed or unexpected))
    (out/'preservation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    assert result['passed']


if __name__=='__main__':main()
