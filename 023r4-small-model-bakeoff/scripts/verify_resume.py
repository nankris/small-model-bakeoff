"""Verify frozen inputs and grade coverage; reconstruct blinded judge inputs for audit."""
import hashlib
import json
from pathlib import Path
from score import normalize
r = Path(__file__).resolve().parents[1]
p = json.loads((r/'results/round4_provenance.json').read_text())
for rec in p['reused_files']:
    assert hashlib.sha256((r/rec['file']).read_bytes()).hexdigest() == rec['sha256']
items = {i['id']: i for i in json.loads((r/'items.json').read_text())}
answers = {m: {a['id']:a for a in map(json.loads,(r/f'results/{m}.jsonl').read_text().splitlines())} for m in ['grok','qwen']}
grades = list(map(json.loads,(r/'results/judge.jsonl').read_text().splitlines()))
keys = [(g['model'],g['id']) for g in grades]
expected = {(m,i) for m in answers for i,it in items.items() if it['task'] in ['summary','tweet']}
assert len(keys)==len(set(keys))==240 and set(keys)==expected
for m in answers:
    assert set(items)<=set(answers[m])
mapping = json.loads((r/'results/judge_map.json').read_text())
payload=[]
for alias,info in mapping.items():
    it=items[info['id']]
    src=it['source'] if it['task']=='summary' else '\n'.join(it['source'])
    payload.append(dict(alias=alias, task=it['task'], source=src, candidate=normalize(answers[info['model']][it['id']]['text'])))
(r/'results/judge_inputs_reconstructed.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
print('PASS: frozen input hashes, 600 answers, 240 unique grades, exact coverage; blinded inputs reconstructed.')
