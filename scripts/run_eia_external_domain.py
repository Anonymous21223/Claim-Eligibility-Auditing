"""Run a reproducible EIA PJM daily-demand external-domain gate experiment."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

ROOT=Path(__file__).resolve().parents[1]
def main() -> None:
 params={'api_key':'DEMO_KEY','frequency':'daily','data[0]':'value','facets[respondent][]':'PJM','start':'2024-01-01','end':'2024-12-31','length':'5000'}
 url='https://api.eia.gov/v2/electricity/rto/daily-region-data/data/?'+urlencode(params); payload=json.load(urlopen(url,timeout=60)); raw=pd.DataFrame(payload['response']['data']); raw=raw[(raw.timezone=='Central') & raw.type.isin(['D','DF'])]
 frame=raw.pivot(index='period',columns='type',values='value').rename(columns={'D':'demand','DF':'forecast'}).astype(float).reset_index(); frame['date']=pd.to_datetime(frame.period); frame=frame.sort_values('date'); frame['dow']=frame.date.dt.dayofweek; frame['doy']=frame.date.dt.dayofyear; frame['sin_doy']=np.sin(2*np.pi*frame.doy/366); frame['cos_doy']=np.cos(2*np.pi*frame.doy/366); frame['target']=frame.demand
 train=frame.date<'2024-10-01'; test=~train; calendar=['dow','sin_doy','cos_doy']; full=calendar+['forecast']; out=ROOT/'artifacts'/'experiments'/'external-domain-eia'; out.mkdir(parents=True,exist_ok=True)
 def fit(cols):
  m=ExtraTreesRegressor(n_estimators=300,min_samples_leaf=3,random_state=23,n_jobs=-1).fit(frame.loc[train,cols],frame.loc[train,'target']); return m,m.predict(frame.loc[test,cols])
 mc,pc=fit(calendar); mf,pf=fit(full); y=frame.loc[test,'target'].to_numpy(); naive=np.repeat(float(frame.loc[train,'target'].mean()),len(y)); rmse_n=float(mean_squared_error(y,naive)**.5); rmse_c=float(mean_squared_error(y,pc)**.5); rmse_f=float(mean_squared_error(y,pf)**.5); delta=rmse_f-rmse_c; gate_a_delta=rmse_f-rmse_n
 rng=np.random.default_rng(23); gate_b1_draws=np.array([mean_squared_error(y[idx],pf[idx])**.5-mean_squared_error(y[idx],pc[idx])**.5 for idx in (rng.integers(0,len(y),len(y)) for _ in range(2000))]); gate_b1_ci_low,gate_b1_ci_high=np.quantile(gate_b1_draws,[.025,.975]); gate_b1=gate_b1_ci_high<0
 rng=np.random.default_rng(2301); gate_a_draws=np.array([mean_squared_error(y[idx],pf[idx])**.5-mean_squared_error(y[idx],naive[idx])**.5 for idx in (rng.integers(0,len(y),len(y)) for _ in range(2000))]); gate_a_ci_low,gate_a_ci_high=np.quantile(gate_a_draws,[.025,.975]); gate_a=gate_a_ci_high<0
 imp=permutation_importance(mf,frame.loc[test,full],y,n_repeats=50,random_state=23,scoring='neg_root_mean_squared_error'); pd.DataFrame({'feature':full,'permutation_rmse_increase':imp.importances_mean,'permutation_sd':imp.importances_std}).to_csv(out/'gated_feature_importance.csv',index=False)
 locked=pd.DataFrame({'date':frame.loc[test,'date'],'observed_demand':y,'naive_train_mean_prediction':naive,'calendar_prediction':pc,'calendar_forecast_prediction':pf})
 frame.to_csv(out/'pjm_daily_2024.csv',index=False); locked.to_csv(out/'locked_predictions.csv',index=False)
 boot=pd.concat([pd.DataFrame({'gate':'Gate A','comparison':'full vs train-mean naive demand baseline','draw':np.arange(len(gate_a_draws)),'delta_rmse_left_minus_right':gate_a_draws}),pd.DataFrame({'gate':'Gate B1','comparison':'full vs calendar-only','draw':np.arange(len(gate_b1_draws)),'delta_rmse_left_minus_right':gate_b1_draws})],ignore_index=True)
 boot.to_csv(out/'pjm_bootstrap_draws.csv',index=False)
 def digest(cols): return hashlib.sha256(locked[cols].to_csv(index=False,float_format='%.17g').encode('utf-8')).hexdigest()
 gate_decisions={'schema':'pjm-gate-decisions-v1','selection_rule':'ExtraTrees configuration and naive baseline are fixed before the October-December 2024 locked holdout; no locked-period model or baseline reselection.','xai_release_requires':['Gate A','Gate B1'],'gates':[{'gate':'Gate A','role':'predictive_fidelity','comparison':'full vs train-mean naive demand baseline','metric':'rmse_mwh','left':'calendar_plus_day_ahead_forecast_full_model','right':'train_period_mean_demand_naive_baseline','point_delta_rmse':float(gate_a_delta),'ci95_low':float(gate_a_ci_low),'ci95_high':float(gate_a_ci_high),'draws':2000,'status':'PASS' if gate_a else 'FAIL','left_prediction_sha256':digest(['date','calendar_forecast_prediction']),'right_prediction_sha256':digest(['date','naive_train_mean_prediction'])},{'gate':'Gate B1','role':'incremental_feature_group_value','comparison':'full vs calendar-only','metric':'rmse_mwh','left':'calendar_plus_day_ahead_forecast_full_model','right':'calendar_only_model','point_delta_rmse':float(delta),'ci95_low':float(gate_b1_ci_low),'ci95_high':float(gate_b1_ci_high),'draws':2000,'status':'PASS' if gate_b1 else 'FAIL','left_prediction_sha256':digest(['date','calendar_forecast_prediction']),'right_prediction_sha256':digest(['date','calendar_prediction'])}],'xai_release':'INTERPRET' if gate_a and gate_b1 else 'ABSTAIN'}
 (out/'pjm_gate_decisions.json').write_text(json.dumps(gate_decisions,indent=2)+'\n')
 result={'status':'PASS' if gate_a and gate_b1 else 'ABSTAIN','domain':'EIA PJM daily electricity demand, Form EIA-930','source_url':url,'rows':len(frame),'train_rows':int(train.sum()),'locked_rows':int(test.sum()),'naive_rmse':rmse_n,'calendar_r2':float(r2_score(y,pc)),'full_r2':float(r2_score(y,pf)),'calendar_rmse':rmse_c,'full_rmse':rmse_f,'gate_a_delta_rmse':gate_a_delta,'gate_a_paired_bootstrap_ci95':[float(gate_a_ci_low),float(gate_a_ci_high)],'gate_a_status':'PASS' if gate_a else 'FAIL','feature_group_delta_rmse':delta,'paired_bootstrap_ci95':[float(gate_b1_ci_low),float(gate_b1_ci_high)],'gate_b1_status':'PASS' if gate_b1 else 'FAIL','feature_group_gate':'PASS' if gate_b1 else 'FAIL','explanation_availability':'INTERPRET' if gate_a and gate_b1 else 'ABSTAIN','claim':'gated external-domain predictive feature-group attribution; not causal or agricultural validation' if gate_a and gate_b1 else 'feature-group contrast demonstration only; not a full INTERPRET case'}
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); reports=ROOT/'reports'/'experiments'; reports.mkdir(parents=True,exist_ok=True); (reports/'external-domain-eia.md').write_text('# External Domain: EIA Electricity Demand\n\n'+''.join(f'- {k}: `{v}`\n' for k,v in result.items())); (reports/'external-domain-eia.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
if __name__=='__main__': main()
