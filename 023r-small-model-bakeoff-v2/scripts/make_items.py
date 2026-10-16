import json,random,pathlib
D=pathlib.Path(__file__).resolve().parents[1]
def make(n=30,start=0):
 out=[]
 for i in range(start,start+n):
  label=['AI research','Job alert','Promotion','Product update'][i%4]
  topic=['translation','robotics','forecasting','search','coding','speech'][i%6]
  texts=[f'Paper {i}: A controlled study of {topic} reports a {i+7}% improvement over its baseline. Read the methods and ablations; no product is being released.',f'Vacancy {i}: We are recruiting a {topic} engineer in team {i}. Submit a CV by Friday; this is a hiring notice, not a software release.',f'Offer {i}: Save {i+15}% on the {topic} subscription this weekend using code DEAL{i}. Existing features unchanged.',f'Release {i}: Version {i+2}.1 of our {topic} tool now supports offline use and export. Free update for existing customers, no discount offer.']
  out.append(dict(id=f'triage{i}',task='triage',gold=label,prompt='Classify into exactly one category: AI research, Job alert, Promotion, Product update. Reply with category only.\n'+texts[i%4]))
  place=['Alder','Birch','Cedar','Dover','Elm','Fern'][i%6]+f' district {i}'
  source=f'{place} tested a new {topic} service for {i+12} weeks. The pilot cut processing time by {i+18} percent, but only on weekdays; weekend performance did not improve. Officials will extend the pilot to {i+3} additional sites in November rather than launch it citywide. The first review meeting was held on Tuesday.'
  out.append(dict(id=f'summary{i}',task='summary',source=source,prompt='Summarize in one sentence of at most 25 words. Preserve the main finding and its limitation. Reply with sentence only.\n'+source))
  notes=[f'Tested {topic} model on {100+i*10} examples',f'accuracy rose from {50+i}% to {60+i}% with retrieval',f'latency increased from {i+2} to {i+4} seconds']
  out.append(dict(id=f'tweet{i}',task='tweet',source=notes,prompt='Write one tech-blog tweet, <=280 characters, exactly one hashtag. Preserve all numerical findings, invent nothing. Reply with tweet only.\n'+'\n'.join(notes)))
  k=2+i%5; prefix=['AB','xy','QZ','rt','LM'][i%5]; sep=['-',':','_'][i%3]
  desc=f'the literal prefix {prefix}, then {sep}, then exactly {k} ASCII digits (0-9), including leading zeros; nothing else'
  pos=[prefix+sep+'0'*k,prefix+sep+'7'*k,prefix+sep+('123456'[:k])]
  neg=[prefix+sep+'1'*(k-1),prefix+sep+'1'*(k+1),prefix.lower()+sep+'1'*k,prefix+sep+'a'*k,' '+pos[0],pos[0]+'\n',prefix+sep+'١'*k]
  if prefix==prefix.lower(): neg[2]=prefix.upper()+sep+'1'*k
  out.append(dict(id=f'regex{i}',task='regex',pos=pos,neg=neg,prompt='Write a single Python regular expression that fully matches '+desc+'. Reply with regex only, no quotes/fences/explanation.'))
  g=dict(company=f'{place} Labs',role=['Data Scientist','ML Engineer','Research Engineer'][i%3],city=['Pune','Delhi','Chennai','Bengaluru','Hyderabad'][i%5],min_years=i%9)
  source=f"{g['company']} seeks a {g['role']} based in {g['city']}. At least {g['min_years']} years of experience required. Salary: {30+i} lakh. Our headquarters are in Mumbai; this role is not based there. Apply within {i+5} days."
  out.append(dict(id=f'json{i}',task='json',gold=g,prompt='Extract JSON with exactly company (string), role (string), city (string), min_years (integer). Reply only with JSON.\n'+source))
 return out
if __name__=='__main__':
 import sys
 r=int(sys.argv[1]) if len(sys.argv)>1 else 1
 data=make(30,(r-1)*30)
 (D/f'items_r{r}.json').write_text(json.dumps(data,indent=2))
