"""Count-first, secret-safe USDA NASS Quick Stats county-yield downloader."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]; API='https://quickstats.nass.usda.gov/api/'
def request(endpoint:str, params:dict[str,str])->requests.Response:
 for attempt in range(4):
  try:
   response=requests.get(API+endpoint,params=params,timeout=60)
   if response.status_code==429 and attempt<3: time.sleep(2**attempt); continue
   response.raise_for_status(); return response
  except requests.RequestException:
   if attempt==3: raise
   time.sleep(2**attempt)
 raise RuntimeError('unreachable')
def main()->None:
 p=argparse.ArgumentParser(); p.add_argument('--crop',default='WHEAT'); p.add_argument('--year-ge',default='2020'); p.add_argument('--state'); args=p.parse_args()
 key=os.getenv('NASS_API_KEY')
 if not key: raise SystemExit('BLOCKED_NASS_API_KEY')
 params={'key':key,'source_desc':'SURVEY','sector_desc':'CROPS','group_desc':'FIELD CROPS','agg_level_desc':'COUNTY','statisticcat_desc':'YIELD','commodity_desc':args.crop.upper(),'year__GE':str(args.year_ge),'format':'JSON'}
 if args.state: params['state_name']=args.state.upper()
 count=request('get_counts/',{k:v for k,v in params.items() if k!='format'}).json().get('count')
 if not isinstance(count,int) or count>=50000: raise SystemExit(f'BLOCKED_REQUEST_COUNT_{count}')
 response=request('api_GET/',params); raw=response.content; stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out=ROOT/'data'/'v2_county'/'raw'/'nass'; out.mkdir(parents=True,exist_ok=True)
 state_slug=(args.state or 'all_states').lower().replace(' ','_'); target=out/f'{args.crop.lower()}_{state_slug}_{args.year_ge}_{stamp}.json'; target.write_bytes(raw); digest=hashlib.sha256(raw).hexdigest()
 manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'source_mode':'NASS_API','endpoint':'api_GET','request':{k:('<REDACTED>' if k=='key' else v) for k,v in params.items()},'count_before_download':count,'http_status':response.status_code,'raw_path':str(target.relative_to(ROOT)).replace('\\','/'),'sha256':digest,'records':len(response.json().get('data',[]))}
 mdir=ROOT/'data'/'v2_county'/'manifests'; mdir.mkdir(parents=True,exist_ok=True); (mdir/f'nass_request_{stamp}.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'status':'PASS','count':count,'records':manifest['records'],'sha256':digest,'raw_path':manifest['raw_path']}))
if __name__=='__main__':main()
