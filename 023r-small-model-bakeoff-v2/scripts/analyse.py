import json,pathlib,re,math,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=pathlib.Path(__file__).resolve().parents[1]
models=['smollm2_360m','qwen3_0.6b','qwen3_4b_q4','grok'];tasks=['triage','summary','tweet','regex','json']
R=int(sys.argv[1]) if len(sys.argv)>1 else 1
items={};answers={};judges={}
for r in range(1,R+1):
 items.update({x['id']:x for x in json.load(open(D/f'items_r{r}.json'))})
 for kind in ['grok','local']:
  p=D/f'results/{kind}_r{r}.jsonl'
  if p.exists():
   for line in p.read_text().splitlines():
    a=json.loads(line);answers[a['model'],a['id']]=a
 mapping=json.load(open(D/f'results/judge_mapping_r{r}.json'))
 for p in sorted((D/'results').glob(f'judge_r{r}_*.json')):
  if '_input' in p.name:continue
  s=p.read_text().strip();s=re.sub(r'^```(?:json)?\s*|\s*```$','',s)
  for x in json.loads(s):
   k=mapping[x['alias']];judges[k['model'],k['id']]=x
scores={}; detailed=[]
for m in models:
 for iid,it in items.items():
  if (m,iid) not in answers:raise ValueError(f'Missing {m} {iid}')
  a=answers[m,iid];s=a['text'].strip();t=it['task'];why=''
  if a.get('exit',0)!=0:raise ValueError(f'Call failed {m} {iid}')
  if t=='triage':v=s==it['gold']
  elif t=='json':
   try:v=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',s))==it['gold']
   except Exception:v=False
  elif t=='regex':
   try:v=all(re.fullmatch(s,x) is not None for x in it['pos']) and all(re.fullmatch(s,x) is None for x in it['neg'])
   except re.error:v=False
  else:
   j=judges[m,iid];v=j['pass'];why=j['reason']
   if t=='summary':v=v and len(s.split())<=25
   else:v=v and len(s)<=280 and len(re.findall(r'#[A-Za-z][A-Za-z0-9_]*',s))==1
  scores[m,iid]=int(v);detailed.append(dict(model=m,id=iid,success=int(v),reason=why))
def wilson(k,n):
 z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-h,c+h]
rng=np.random.default_rng(23023);boot={m:np.zeros(20000) for m in models};table={}
for t in tasks:
 ids=[i for i,x in items.items() if x['task']==t];n=len(ids);ix=rng.integers(n,size=(20000,n))
 for m in models:
  a=np.array([scores[m,i] for i in ids]);table[m,t]={'n':n,'success':int(a.sum()),'score':float(a.mean()),'ci':wilson(a.sum(),n)};boot[m]+=a[ix].mean(axis=1)/5
pairs={}
for a,b in [('grok','qwen3_4b_q4'),('grok','qwen3_0.6b'),('grok','smollm2_360m'),('qwen3_4b_q4','qwen3_0.6b'),('qwen3_4b_q4','smollm2_360m')]:
 dif=boot[a]-boot[b];pairs[a+' minus '+b]={'difference':sum(table[a,t]['score']-table[b,t]['score'] for t in tasks)/5,'ci':np.quantile(dif,[.025,.975]).tolist()}
p=pairs['grok minus qwen3_4b_q4'];verdict='PROVEN' if p['ci'][0]>0 else 'DISPROVEN' if p['ci'][1]<0 else 'STILL INCONCLUSIVE'
res=dict(rounds=R,verdict=verdict,per_task={m:{t:table[m,t] for t in tasks} for m in models},comparisons=pairs,overall={m:dict(score=sum(table[m,t]['score'] for t in tasks)/5,ci=np.quantile(boot[m],[.025,.975]).tolist()) for m in models})
(D/'results/scores.json').write_text(json.dumps(detailed,indent=2));(D/'results/summary.json').write_text(json.dumps(res,indent=2))
fig,ax=plt.subplots(figsize=(12,5));x=np.arange(5);w=.2
for j,m in enumerate(models):
 vals=np.array([table[m,t]['score'] for t in tasks]);lo=np.array([table[m,t]['ci'][0] for t in tasks]);hi=np.array([table[m,t]['ci'][1] for t in tasks]);ax.bar(x+(j-1.5)*w,vals*100,w,label=m,yerr=np.array([vals-lo,hi-vals])*100,capsize=2)
ax.set_xticks(x,tasks);ax.set_ylim(0,110);ax.set_ylabel('Success %, Wilson 95% CI');ax.legend(fontsize=8);ax.set_title(f'023r: {len(items)//5} items per task; blind Codex judge for free text');fig.tight_layout();fig.savefig(D/'chart.png',dpi=170)
lines=['# 023r results','',f'Verdict: **{verdict}** for the preregistered hypothesis that Grok beats Qwen3-4B Q4.','', '| Task | '+' | '.join(models)+' |','|---|'+'---|'*4]
for t in tasks:
 cells=[]
 for m in models:
  q=table[m,t];cells.append(f"{q['success']}/{q['n']} = {q['score']*100:.1f}% [{q['ci'][0]*100:.1f}, {q['ci'][1]*100:.1f}]")
 lines.append('| '+t+' | '+' | '.join(cells)+' |')
lines+=['','Per-task brackets are Wilson 95% CIs. Overall and differences use 20,000 paired task-stratified bootstrap draws (seed 23023).','', '| Comparison | Difference, percentage points | 95% CI |','|---|---:|---|']
for k,v in pairs.items():lines.append(f"| {k} | {v['difference']*100:.1f} | [{v['ci'][0]*100:.1f}, {v['ci'][1]*100:.1f}] |")
lines+=['','## Limits','Synthetic generated template families, not a representative daily-work sample. Item CIs do not account for correlation within template families. One deterministic output per item. Codex judgment is independent of contestants but subjective; raw blinded inputs, judgments and mapping retained. Strict all-or-nothing scores differ from the original partial-credit scores. Hardware differs from the original CPU test; no timing or RAM advantage is inferred. No causal claim about parameter count versus quantization. Secondary comparisons are descriptive.']
(D/'writeup.md').write_text('\n'.join(lines)+'\n');print(json.dumps(res,indent=2))
