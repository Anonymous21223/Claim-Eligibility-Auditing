"""Initialize the County-Level Agricultural Risk Panel with explicit source contracts."""
from __future__ import annotations
import json, os
from datetime import date
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'v2_county'
SOURCES=[
 ('usda_nass_county_yield','USDA NASS Quick Stats','https://quickstats.nass.usda.gov/api/api_GET/; bulk fallback: https://www.nass.usda.gov/datasets/qs.crops_20260718.txt.gz','NASS_API_KEY','API key or official USDA NASS bulk terms','not included','API batches or NASS_OFFICIAL_BULK after disk preflight'),
 ('nasa_power_weather','NASA POWER daily API','https://power.larc.nasa.gov/api/temporal/daily/point','none','NASA attribution/notification terms','not included','documented county coordinate downloader'),
 ('cropland_data_layer','USDA NASS Cropland Data Layer','https://nassgeodata.gmu.edu/CropScape/','none','source-specific terms','not included','CropScape download/reconstruction'),
 ('us_drought_monitor','U.S. Drought Monitor','https://droughtmonitor.unl.edu/','none','source attribution','not included','official weekly data download'),
 ('soilgrids','ISRIC SoilGrids','https://rest.isric.org/soilgrids/v2.0/','none','CC BY 4.0; verify before redistribution','not included','SoilGrids API request'),
]
def main() -> None:
 for part in ('raw','interim','processed','manifests','schemas'): (DATA/part).mkdir(parents=True,exist_ok=True)
 pd.DataFrame(SOURCES,columns=['dataset','official_source','url','required_credential','license_or_terms','included_in_repo','reconstruction_command']).assign(access_date=str(date.today()),temporal_availability='must be checked before feature use',checksum='pending download',status='PENDING').to_csv(DATA/'DATA_MANIFEST.csv',index=False)
 schema={'dataset':'Dataset V2 - County-Level Agricultural Risk Panel','required_columns':['county_fips','state_fips','crop','year','yield','weather_feature_date_max','target_available_date','split_role'],'rules':['All features must have weather_feature_date_max before target_available_date.','Temporal/geographic holdout is immutable after registration.','Raw files are never silently imputed or redistributed without terms review.']}
 (DATA/'schemas'/'county_panel_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
 report=ROOT/'reports'/'experiments'; report.mkdir(parents=True,exist_ok=True); state='READY_FOR_DOWNLOAD' if os.getenv('NASS_API_KEY') else 'BLOCKED_NASS_API_KEY'; (report/'county-panel-v2.json').write_text(json.dumps({'status':state,'data_root':str(DATA.relative_to(ROOT)),'required_source_count':len(SOURCES),'next_action':'Set NASS_API_KEY and run approved downloader; no experiment has run.'},indent=2)+'\n'); print(state)
if __name__=='__main__': main()
