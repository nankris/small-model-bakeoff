import pathlib,json,re,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=pathlib.Path(__file__).resolve().parents[1]
items={x['id']:x for x in json.load(open(D/'items_r1.json'))}
answers=[json.loads(l) for l in (D/'results/grok_r1.jsonl').read_text().splitlines()]
judges={}
for p in (D/'results').glob('judge_r1_*.json'):
 if '_input' in p.name:continue
 s=re.sub(r'^```(?:json)?\s*|\s*```$','',p.read_text().strip())
 for j in json.loads(s):judges[j['alias'].rsplit('_',1)[0]]=j
rows=[];scored=[]
for task in ['triage','summary','tweet','regex','json']:
 vals=[]
 for a in answers:
  it=items[a['id']];s=a['text'].strip()
  if it['task']!=task:continue
  if task=='triage':v=s==it['gold']
  elif task=='json':
   try:v=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',s))==it['gold']
   except Exception:v=False
  elif task=='regex':
   try:v=all(re.fullmatch(s,x) is not None for x in it['pos']) and all(re.fullmatch(s,x) is None for x in it['neg'])
   except re.error:v=False
  else:v=judges[a['id']]['pass']
  vals.append(int(v));scored.append(dict(id=a['id'],model='grok',success=int(v)))
 n=len(vals);k=sum(vals);p=k/n;z=1.95996398454;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 rows.append(dict(task=task,n=n,success=k,score=p,ci=[c-h,c+h]))
(D/'results/partial_scores.json').write_text(json.dumps(scored,indent=2))
(D/'results/partial_summary.json').write_text(json.dumps(dict(verdict='STILL INCONCLUSIVE',complete_rounds=0,paired_ci=None,rows=rows),indent=2))
lines=['# 023r bakeoff v2 — blocked rerun','','Verdict: **STILL INCONCLUSIVE**. Colab refused L4 allocation on all three attempts (precondition failed). No local model ran, so the preregistered paired comparison is unavailable. Zero evaluation rounds completed; this is an infrastructure failure, not a two-round null finding.','','## Partial diagnostic results','','150 synthetic items were prepared (30/task). Only '+str(len(answers))+' Grok calls completed before cancellation. These partial results do not satisfy the requested >=25 scored items/task and must not be interpreted as a model comparison.','','| Task | Grok successes / completed | Accuracy, Wilson 95% CI | SmolLM2-360M | Qwen3-0.6B | Qwen3-4B Q4 |','|---|---:|---|---|---|---|']
for q in rows:lines.append(f"| {q['task']} | {q['success']}/{q['n']} | {q['score']*100:.1f}% [{q['ci'][0]*100:.1f}, {q['ci'][1]*100:.1f}] | Not run | Not run | Not run |")
lines+=['','Primary Grok minus Qwen3-4B difference and 95% CI: **not estimable**. No scores have been imputed for missing arms. Summary/tweet grades come from independent blind Codex exec, not Grok. Raw anonymous judge inputs, grades and identity mapping are retained. The table describes only completed calls and is not a random subsample analysis.','','## Compute and artifacts','','Requested GPU: L4; actual GPU: none. CU used: **0.00** (account balance 179.84 before / 179.58 at final cleanup; no experiment runtime was allocated). Cleanup called `colab stop -s tig023r-v2`; session not found. Other active sessions left untouched. See results/compute.json and README.md for failure details and resume commands.','','The original folder was preserved. New scripts include deterministic synthetic item generation, resumable Grok calls, planned 4-bit NF4 Qwen3-4B and FP16 tiny-model L4 inference, blind Codex judging, per-task Wilson CIs, and paired stratified bootstrap analysis. Remote inference is unvalidated because allocation failed. Figure explicitly marks partial sample sizes and missing arms. No API dollar cost estimate was collected.','','## Limits','','Template-based synthetic items are not representative of all daily tasks; template correlation makes item-level CIs optimistic for broader generalization. Deterministic generations estimate item variation only. Strict binary rubric is not numerically comparable to the original partial-credit scoring. Independent judging remains subjective. No inference about quantization, speed, RAM, model size or superiority is supported by this incomplete run.']
(D/'writeup.md').write_text('\n'.join(lines)+'\n')
fig,ax=plt.subplots(figsize=(10,5))
x=list(range(5));y=[q['score']*100 for q in rows];lo=[(q['score']-q['ci'][0])*100 for q in rows];hi=[(q['ci'][1]-q['score'])*100 for q in rows]
ax.bar(x,y,color='#697f91',yerr=[lo,hi],capsize=5)
ax.set_xticks(x,[q['task']+f"\nn={q['n']} / 30" for q in rows]);ax.set_ylim(0,115);ax.set_ylabel('Partial Grok success %, Wilson 95% CI');ax.set_title('BLOCKED: no L4 allocation; three local arms missing')
fig.text(.5,.01,'Diagnostic partial data only — no model comparison or completed round',ha='center',fontsize=10);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(D/'chart.png',dpi=160)
print('Completed Grok calls:',len(answers));print(rows)
