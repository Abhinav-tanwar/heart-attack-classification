"""Classify self-reported prior heart attack; not future-risk prediction."""
import argparse,copy
from common import *
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans
from imblearn.pipeline import Pipeline as SamplingPipeline
from imblearn.over_sampling import SMOTENC
TARGET='HadHeartAttack'
NUMERIC=['PhysicalHealthDays','MentalHealthDays','SleepHours','BMI']
DROP=['LastCheckupTime','RemovedTeeth','HadAngina','HadStroke','HadAsthma','HadCOPD','HadDepressiveDisorder','HadArthritis','CovidPos','HIVTesting','FluVaxLast12','PneumoVaxEver','TetanusLast10Tdap','State','HeightInMeters','WeightInKilograms','ChestScan','DeafOrHardOfHearing','BlindOrVisionDifficulty']

def prepare_data(raw):
    d=raw.drop(columns=DROP,errors='ignore').copy()
    if TARGET not in d:raise ValueError('Missing HadHeartAttack outcome')
    known=d[TARGET].isin(['Yes','No']);missing_target=int((~known).sum())
    d=d.loc[known].drop_duplicates().copy();y=d.pop(TARGET).map({'No':0,'Yes':1}).astype(int)
    for c in NUMERIC:d[c]=pd.to_numeric(d[c],errors='coerce')
    categorical=[c for c in d if c not in NUMERIC]
    return d,y,categorical,missing_target

def sampling_preprocessor(categorical):
    return ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMERIC),('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]),categorical)])

def build_model(categorical,forest=False,oversample=False):
    classifier=RandomForestClassifier(n_estimators=100,min_samples_leaf=2,random_state=42,n_jobs=1) if forest else LogisticRegression(max_iter=2500,random_state=42)
    if not oversample:return Pipeline([('preprocess',preprocessor(NUMERIC,categorical)),('model',classifier)])
    indices=list(range(len(NUMERIC),len(NUMERIC)+len(categorical)))
    # SMOTENC respects categorical values; ordinary SMOTE on encoded labels does not.
    encode=ColumnTransformer([('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),indices)],remainder='passthrough')
    return SamplingPipeline([('preprocess',sampling_preprocessor(categorical)),('sample',SMOTENC(categorical_features=indices,random_state=42)),('onehot',encode),('model',classifier)])

def evaluate(model,x,y):
    pred=model.predict(x);prob=model.predict_proba(x)[:,1]
    return {'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'roc_auc':float(roc_auc_score(y,prob)),'average_precision':float(average_precision_score(y,prob)),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def run(input_path,output='outputs',cv=True):
    raw=pd.read_csv(input_path);x,y,categorical,missing_target=prepare_data(raw)
    xt,xv,yt,yv=train_test_split(x,y,test_size=.3,random_state=42,stratify=y)
    assert set(xt.index).isdisjoint(xv.index) and TARGET not in xt and 'Cluster' not in xt
    report={'input_rows':len(raw),'rows_missing_outcome_excluded':missing_target,'unique_labeled_rows':len(x),'training_rows':len(xt),'test_rows':len(xv),'test_positive_cases':int(yv.sum()),'test_prevalence':float(yv.mean()),'target':'self-reported history of heart attack, not a prospective event','models':{}}
    dummy=DummyClassifier(strategy='prior').fit(xt,yt);report['models']['majority_baseline']=evaluate(dummy,xv,yv)
    folds=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    for forest in [False,True]:
        for oversample in [False,True]:
            name=('random_forest' if forest else 'logistic_regression')+('_smotenc' if oversample else '')
            model=build_model(categorical,forest,oversample)
            if cv:
                scores=cross_val_score(model,xt,yt,cv=folds,scoring='average_precision',n_jobs=1);cv_score=float(scores.mean())
            else:cv_score=None
            model.fit(xt,yt);metrics=evaluate(model,xv,yv)
            if cv_score is not None:metrics['training_cv_average_precision']=cv_score
            report['models'][name]=metrics
    # Descriptive segmentation fitted to training predictors only; outcome excluded.
    cluster_prep=preprocessor(NUMERIC,categorical);z=cluster_prep.fit_transform(xt)
    clusters=KMeans(n_clusters=3,n_init=10,random_state=42).fit(z)
    report['training_cluster_counts']={str(k):int(v) for k,v in pd.Series(clusters.labels_).value_counts().items()}
    save_report(output,report)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(report['models']);fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');ax.barh(names,[report['models'][n]['average_precision'] for n in names],color='#244af4');ax.axvline(yv.mean(),color='black',ls='--',label='Held-out prevalence');ax.set(xlabel='Average precision (held-out test)',title='Heart-attack history classification');ax.legend();fig.savefig(Path(output)/'model-comparison.svg');plt.close(fig)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='data/heart_10k_sample.csv');p.add_argument('--output',default='outputs');p.add_argument('--skip-cv',action='store_true');a=p.parse_args();print(json.dumps(run(a.input,a.output,cv=not a.skip_cv),indent=2))
