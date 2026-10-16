import subprocess,sys,json,time,gc,os
subprocess.run([sys.executable,'-m','pip','-q','install','transformers','accelerate','bitsandbytes'],check=True)
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
items=json.load(open('/content/items.json'))
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv']).decode(),flush=True)
for key,repo in [('smollm2_360m','HuggingFaceTB/SmolLM2-360M-Instruct'),('qwen3_0.6b','Qwen/Qwen3-0.6B'),('qwen3_4b_q4','Qwen/Qwen3-4B')]:
 kw={'quantization_config':BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16)} if '4b_' in key else {'torch_dtype':torch.float16}
 tok=AutoTokenizer.from_pretrained(repo); model=AutoModelForCausalLM.from_pretrained(repo,device_map='auto',**kw).eval()
 for it in items:
  args={'enable_thinking':False} if key.startswith('qwen') else {}
  p=tok.apply_chat_template([{'role':'user','content':it['prompt']}],tokenize=False,add_generation_prompt=True,**args)
  enc=tok(p,return_tensors='pt').to(model.device); t=time.time()
  with torch.inference_mode(): gen=model.generate(**enc,max_new_tokens=160,do_sample=False,pad_token_id=tok.eos_token_id)
  text=tok.decode(gen[0,enc.input_ids.shape[1]:],skip_special_tokens=True).strip()
  with open('/content/answers.jsonl','a') as f:f.write(json.dumps(dict(model=key,id=it['id'],text=text,wall_s=time.time()-t))+'\n')
 print(key,'complete',flush=True)
 del model,tok;gc.collect();torch.cuda.empty_cache()
