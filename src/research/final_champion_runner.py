"""ID-safe final pipelines selected from strict chronological local research."""
from __future__ import annotations
import argparse, os
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from framework import SUBMISSION_DIR, add_strict_state_history, load_competition_data, prepare_safe_features, write_submission_by_id

CATS=["state","county","prior_period_ipc_phase"]
CORE=["population","start_year","start_month","month_sin","month_cos","production_per_capita","gap_per_capita","gap_to_production","prior_ipc_numeric","prior_ipc_crisis_plus","prior_ipc_emergency_plus","prior_period_phase3plus_pct","prior_year_cereal_production_tonnes","prior_year_cereal_gap_tonnes","ipc_phase_x_phase3pct","ipc_phase_x_gap_pc","phase3pct_x_gap_pc"]
def unique(x): return list(dict.fromkeys(x))
def model(features):
 return Pipeline([("preprocess",ColumnTransformer([("num",SimpleImputer(strategy="median"),features),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),CATS)])),("model",HistGradientBoostingClassifier(learning_rate=.03,max_iter=450,max_leaf_nodes=12,min_samples_leaf=12,l2_regularization=2.5,random_state=42))])
def features_for(name,tr):
 county=[c for c in tr if c.startswith(("county_history_","hist_")) or c.endswith(("_lag1","_lag2","_lag3","_lag4","_lag5","_change_1","_change_2","_acceleration","_recent_vs_history","_period"))]
 elapsed=[c for c in tr if c.endswith("_change_per_prior_month")]+["months_since_previous_period","prior_history_interval_months"]
 state=[c for c in tr if c.startswith("state_")]
 if name=="state_horizon": return unique(CORE+county+state)
 if name=="interval_trend": return unique(CORE+county+elapsed)
 raise ValueError(name)
def main(name,output=None):
 train,test,sample=load_competition_data(); tr,te=prepare_safe_features(train,test)
 both=pd.concat([tr,te],ignore_index=True,sort=False); both=add_strict_state_history(both)
 tr=both.loc[both._is_train.eq(1)].copy(); te=both.loc[both._is_train.eq(0)].copy(); features=features_for(name,tr)
 assert all(c in tr for c in features) and all(c in te for c in features)
 fitted=model(features).fit(tr[CATS+features],tr.food_insecurity_risk.astype(int))
 predictions=np.clip(fitted.predict_proba(te[CATS+features])[:,1],1e-6,1-1e-6)
 by_id=pd.Series(predictions,index=te.ID); assert by_id.index.is_unique and set(by_id.index)==set(sample.ID)
 path=output or SUBMISSION_DIR/f"{name}.csv"
 if output is None and path.exists(): raise FileExistsError(f"Refusing to overwrite final candidate: {path}")
 write_submission_by_id(by_id,path,sample); print(path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',choices=('state_horizon','interval_trend'),required=True);p.add_argument('--output',type=Path);a=p.parse_args();main(a.name,a.output)
