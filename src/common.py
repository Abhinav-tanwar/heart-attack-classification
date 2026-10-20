from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

def preprocessor(numeric,categorical,scale=True):
    steps=[('impute',SimpleImputer(strategy='median'))]
    if scale:steps.append(('scale',StandardScaler()))
    return ColumnTransformer([('numeric',Pipeline(steps),numeric),('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)],remainder='drop')

def save_report(output,report):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report
