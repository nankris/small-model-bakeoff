import subprocess,json,pathlib,concurrent.futures,time,tempfile,sys
D=pathlib.Path(__file__).resolve().parents[1];r=sys.argv[1] if len(sys.argv)>1 else '1'
items=json.load(open(D/f'items_r{r}.json')); out=D/f'results/grok_r{r}.jsonl'
done={json.loads(l)['id'] for l in out.read_text().splitlines()} if out.exists() else set()
def call(it):
 t=time.time()
 with tempfile.TemporaryDirectory() as wd:
  p=subprocess.run(['grok','--no-auto-update','-p',it['prompt'],'--disable-web-search','--no-subagents','--tools','','--max-turns','1','--reasoning-effort','low','--system-prompt-override','You are a helpful assistant. Follow the output format exactly.'],cwd=wd,capture_output=True,text=True,timeout=180)
 return dict(model='grok',id=it['id'],text=p.stdout.strip(),exit=p.returncode,stderr=p.stderr,wall_s=time.time()-t)
with concurrent.futures.ThreadPoolExecutor(5) as ex:
 for rec in ex.map(call,[it for it in items if it['id'] not in done]):
  with out.open('a') as f:f.write(json.dumps(rec)+'\n')
  print(rec['id'],rec['exit'],flush=True)
