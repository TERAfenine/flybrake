"""Compare original protected files to the Phase 6 pre-analysis SHA256 manifest."""
from pathlib import Path
import hashlib
import json

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def main():
    before=json.loads((OUT/'protected_before.json').read_text(encoding='utf-8-sig'))
    missing=[];changed=[]
    for name,expected in before.items():
        p=ROOT/name
        if not p.is_file():missing.append(name)
        elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:changed.append(name)
    current={p.relative_to(ROOT).as_posix() for d in ['data','results'] for p in (ROOT/d).rglob('*') if p.is_file()}
    added=sorted(current-set(before))
    result={'protected_files':len(before),'unchanged':len(before)-len(missing)-len(changed),
            'missing':missing,'changed':changed,'new_files_under_data_or_results':added,
            'passed':not (missing or changed or added)}
    (OUT/'preservation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    assert result['passed'], 'Original research files differ from pre-analysis manifest'


if __name__=='__main__':main()
