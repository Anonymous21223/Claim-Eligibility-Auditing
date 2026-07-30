"""Audit immutable NASS county-yield raw responses before V2 population selection."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def main()->None:
 files=sorted((ROOT/'data'/'v2_county'/'raw'/'nass').glob('*.json'))
 if not files: raise SystemExit('BLOCKED_NO_NASS_RAW')
 rows=[]
 for f in files: rows.extend(json.loads(f.read_text())['data'])
 df=pd.DataFrame(rows); df['year']=pd.to_numeric(df.year,errors='coerce'); df['value_numeric']=pd.to_numeric(df.Value.astype(str).str.replace(',','',regex=False),errors='coerce'); df['suppressed']=df.Value.astype(str).str.contains(r'\(D\)|\(Z\)',regex=True,na=False); df['missing_county_ansi']=~df.county_ansi.astype(str).str.strip().str.fullmatch(r'\d{3}') | df.county_ansi.astype(str).str.strip().eq('000')
 keys=['commodity_desc','state_ansi','county_ansi','year']; dup=df.duplicated(keys,keep=False); conflicting=df[dup].groupby(keys).value_numeric.nunique().gt(1).sum()
 out=[]
 for (crop,year),g in df.groupby(['commodity_desc','year'],dropna=False):
  hist=df[(df.commodity_desc==crop)&~df.missing_county_ansi&df.value_numeric.notna()].groupby(['state_ansi','county_ansi']).year.nunique(); gconf=g[g.duplicated(keys,keep=False)].groupby(keys).value_numeric.nunique().gt(1).sum()
  out.append({'crop':crop,'year':int(year),'states':g.state_ansi.nunique(),'counties':g.loc[~g.missing_county_ansi,['state_ansi','county_ansi']].drop_duplicates().shape[0],'observations':len(g),'suppressed_values':int(g.suppressed.sum()),'missing_values':int(g.value_numeric.isna().sum()),'missing_county_ansi':int(g.missing_county_ansi.sum()),'series_history_ge_3':int((hist>=3).sum()),'series_history_ge_5':int((hist>=5).sum()),'series_history_ge_8':int((hist>=8).sum()),'series_history_ge_10':int((hist>=10).sum()),'unit':'; '.join(sorted(g.unit_desc.dropna().unique())),'source':'; '.join(sorted(g.source_desc.dropna().unique())),'duplicate_rows':int(g.duplicated(keys[1:],keep=False).sum()),'conflicting_estimate_groups':int(gconf)})
 coverage=pd.DataFrame(out); reports=ROOT/'reports'/'v2'; reports.mkdir(parents=True,exist_ok=True); coverage.to_csv(reports/'county_yield_coverage.csv',index=False)
 valid=df[~df.missing_county_ansi & df.value_numeric.notna()]; history=valid.groupby(['commodity_desc','state_ansi','county_ansi']).year.nunique(); proposals={'maximum_coverage_population':int(valid[['commodity_desc','state_ansi','county_ansi','year']].drop_duplicates().shape[0]),'balanced_crop_county_panel':int((history>=5).sum()),'long_history_population':int((history>=3).sum()),'selection_rule':'Choose by pre-specified coverage/history quality rules, never final model performance.'}
 summary={'status':'PASS','raw_files':[f.relative_to(ROOT).as_posix() for f in files],'rows':len(df),'earliest_year':int(df.year.min()),'latest_year':int(df.year.max()),'proposals':proposals,'coverage_csv':'reports/v2/county_yield_coverage.csv'}; (reports/'county_yield_coverage.json').write_text(json.dumps(summary,indent=2)+'\n'); (reports/'county_yield_coverage.md').write_text('# County Yield Coverage Audit\n\n'+''.join(f'- {k}: `{v}`\n' for k,v in summary.items())+'\nCoverage is raw-source quality evidence, not a model-selection result. The current raw extract intentionally retains multiple class and production-practice strata; do not train a panel until a pre-specified canonical stratum has been selected and conflict counts are resolved.\n')
 print(json.dumps(summary))
if __name__=='__main__':main()
