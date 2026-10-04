from pathlib import Path
import json, hashlib, shutil, sys, time, warnings
import numpy as np
import pandas as pd
import scipy, scipy.stats as st
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'rerun_results'
for sub in ['models','tables','figures']:
    (OUT/sub).mkdir(parents=True,exist_ok=True)
csv=BASE/'data/deficiency_anemia.csv'
xlsx=BASE/'spec/variables.xlsx'
d=pd.read_csv(csv); v=pd.read_excel(xlsx)
schema=json.loads((BASE/'config/audit_schema.json').read_text())
features=schema['model_features']; labs=[f for f in features if f not in ['age_years','sex']]
original=json.loads((BASE/'models/metrics.json').read_text()); classes=original['class_order']
X=d[features].copy(); X['sex']=X.sex.map({'F':0.,'M':1.}); y=d.anemia_class
def clean(obj):
    if isinstance(obj,dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [clean(v) for v in obj]
    if isinstance(obj,np.generic): return clean(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj): return None
    return obj
def dump(path,obj):
    (OUT/path).write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,default=str,allow_nan=False))
def table(name,frame):
    frame.to_csv(OUT/'tables'/f'{name}.csv',index=False)
    return frame
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
shutil.copy2(BASE/'models/model_metadata.json',OUT/'models/model_metadata.json')

# Union-find: exact laboratory profiles, no target or demographic fields in signature.
parent=list(range(len(d)))
def root(i):
    while parent[i]!=i:
        parent[i]=parent[parent[i]]; i=parent[i]
    return i
def union(a,b): parent[root(b)]=root(a)
seen_id={}; seen_lab={}
for i,row in d.iterrows():
    pid=str(row.patient_id).strip() if pd.notna(row.patient_id) else ''
    sig=tuple(None if pd.isna(z) else float(z) for z in row[labs])
    if pid:
        if pid in seen_id: union(i,seen_id[pid])
        else: seen_id[pid]=i
    if sig in seen_lab: union(i,seen_lab[sig])
    else: seen_lab[sig]=i
groups=np.array([root(i) for i in range(len(d))])
groupcounts=pd.DataFrame({'class':y,'group':groups}).groupby('class').group.nunique().reindex(classes)
assert groupcounts.min()>=5
dups={'repeated_id_values':int((d.patient_id.value_counts()>1).sum()),'extra_id_rows':int(d.patient_id.duplicated().sum()),'extra_exact_lab_profiles':int(d[labs].duplicated().sum()),'extra_full_37_profiles':int(X.duplicated().sum()),'extra_full_rows':int(d.duplicated().sum()),'independent_groups':len(set(groups)),'groups_by_class':groupcounts.to_dict()}
dump('models/groups.json',dups)
demo=[]
for c in classes:
    s=d[y==c]; f=int((s.sex=='F').sum()); m=int((s.sex=='M').sum())
    demo.append({'class':c,'N':len(s),'age_median':s.age_years.median(),'age_p25':s.age_years.quantile(.25),'age_p75':s.age_years.quantile(.75),'age_min':s.age_years.min(),'age_max':s.age_years.max(),'F':f,'M':m,'F_pct':100*f/len(s),'F_M_ratio':f/m if m else None,'groups':int(groupcounts[c])})
demo=table('demographics',pd.DataFrame(demo))
chi,p,df,expected=st.chi2_contingency(pd.crosstab(y,d.sex)); cram=np.sqrt(chi/(len(d)*min(11,1)))
kw=st.kruskal(*[d.loc[y==c,'age_years'] for c in classes]); eps=max(0,(kw.statistic-12+1)/(len(d)-12))
demo_tests={'sex_chi2':chi,'sex_p':p,'sex_cramers_v':cram,'sex_min_expected':expected.min(),'age_kruskal_H':kw.statistic,'age_p':kw.pvalue,'age_epsilon2':eps}
dump('models/demographic_tests.json',demo_tests)
desc=d[labs].describe(percentiles=[.05,.25,.5,.75,.95]).T.rename(columns={'count':'n','5%':'p05','25%':'p25','50%':'p50','75%':'p75','95%':'p95'})
desc.insert(0,'unit',[v.set_index('variable').loc[f,'unit'] for f in labs]); desc['missing_pct']=100*d[labs].isna().mean(); desc=table('descriptive_statistics',desc.rename_axis('feature').reset_index())
numeric_features=[f for f in features if f!='sex']
profile={'rows':len(d),'age_min':float(d.age_years.min()),'age_max':float(d.age_years.max()),'class_counts':y.value_counts().to_dict(),'features':{},'class_medians':{},'class_coverage':{}}
for f in numeric_features:
    s=d[f]
    profile['features'][f]={'coverage':float(s.notna().mean()),'p05':float(s.quantile(.05)),'p50':float(s.quantile(.5)),'p95':float(s.quantile(.95))}
for c in classes:
    sub=d[y==c]
    profile['class_medians'][c]={f:None if pd.isna(sub[f].median()) else float(sub[f].median()) for f in numeric_features}
    profile['class_coverage'][c]={f:float(sub[f].notna().mean()) for f in numeric_features}
for row in desc.to_dict('records'):
    profile['features'][row['feature']].update({k:val for k,val in row.items() if k!='feature'})
profile['research_note']='Descriptive statistics use observed values only; std ddof=1; quantiles linear. Not clinical reference intervals.'
dump('models/data_profile.json',profile)
ranges=table('technical_ranges',pd.DataFrame([{'feature':f,'min':0,'max':schema['fields'][f]['max'],'canonical_unit':schema['fields'][f]['unit'],'observed_min':d[f].min(),'observed_max':d[f].max(),'violations':int(((d[f]<0)|(d[f]>schema['fields'][f]['max'])).sum())} for f in labs]))
coverage=table('coverage_by_class',d.groupby('anemia_class')[labs].agg(lambda s:s.notna().mean()).reset_index())
anova=[]
for f in labs:
    samples=[d.loc[y==c,f].dropna().values for c in classes]
    valid=[s for s in samples if len(s)>=2]
    F,pv=st.f_oneway(*valid)
    n=sum(map(len,valid)); k=len(valid)
    anova.append({'feature':f,'ANOVA_F':F,'p_value':pv,'observed_N':n,'classes_with_n_ge2':k,'min_observed_per_class':min(map(len,samples)),'eta_squared':F*(k-1)/(F*(k-1)+n-k)})
anova=pd.DataFrame(anova).sort_values('p_value'); m=len(anova); anova['q_BH']=np.minimum.accumulate((anova.p_value.to_numpy()*m/np.arange(1,m+1))[::-1])[::-1].clip(0,1)
anova=table('anova_features',anova.sort_values('ANOVA_F',ascending=False))
corr=d[labs].corr(method='spearman',min_periods=30)
corr.to_csv(OUT/'tables/correlation_spearman.csv')
pairn=d[labs].notna().astype(int).T.dot(d[labs].notna().astype(int)); pairn.to_csv(OUT/'tables/correlation_pair_n.csv')
pairs=pd.DataFrame([{'a':a,'b':b,'rho':corr.loc[a,b],'N':int(pairn.loc[a,b])} for i,a in enumerate(labs) for b in labs[i+1:] if pd.notna(corr.loc[a,b])])
pairs['abs_rho']=pairs.rho.abs(); pairs=table('correlation_pairs',pairs.sort_values('abs_rho',ascending=False))
top=pairs.head(15)
fig,ax=plt.subplots(figsize=(8,8)); im=ax.imshow(top[['rho']].values,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto'); ax.set_yticks(range(15),[f'{r.a} / {r.b}  (n={r.N})' for r in top.itertuples()]); ax.set_xticks([0],['Spearman rho']);
for i,r in enumerate(top.itertuples()): ax.text(0,i,f'{r.rho:.3f}',ha='center',va='center',color='white' if abs(r.rho)>.65 else 'black')
ax.set_title('Top 15 absolute correlations • pairwise observed values'); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(OUT/'figures/top15_correlations.png',dpi=160); plt.close(fig)
fig,ax=plt.subplots(figsize=(14,12)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r'); ax.set_xticks(range(35),labs,rotation=90,fontsize=8); ax.set_yticks(range(35),labs,fontsize=8); fig.colorbar(im,ax=ax); ax.set_title('Spearman correlation • min 30 observed pairs'); fig.tight_layout(); fig.savefig(OUT/'figures/correlation_matrix.png',dpi=150); plt.close(fig)

splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(X,y,groups)); foldids=np.zeros(len(d),int)
for f,(tr,va) in enumerate(splits,1):
    assert not set(groups[tr])&set(groups[va]); foldids[va]=f
table('fold_assignments',pd.DataFrame({'row_index':np.arange(len(d)),'group':groups,'fold':foldids,'class':y}))
def metrics(yt,probs):
    yi=np.array([classes.index(c) for c in yt]); pred=np.array(classes)[probs.argmax(1)]; conf=probs.max(1); correct=(pred==np.asarray(yt)); bins=np.minimum((conf*10).astype(int),9)
    ece=sum(np.mean(bins==b)*abs(conf[bins==b].mean()-correct[bins==b].mean()) for b in range(10) if (bins==b).any())
    return {'accuracy':accuracy_score(yt,pred),'balanced_accuracy':balanced_accuracy_score(yt,pred),'macro_f1':f1_score(yt,pred,labels=classes,average='macro',zero_division=0),'multiclass_brier':np.mean(np.sum((probs-np.eye(12)[yi])**2,axis=1)),'top_score_ece_10bins':ece}
results=[]; allprobs={}; params={}; folds=[]
for algo in ['LogisticRegression','DecisionTree','RandomForest','GradientBoosting','DemographicsOnly','MissingnessOnly','LabsOnly_RF']:
    start=time.time(); probs=np.zeros((len(d),12))
    xx=X[['age_years','sex']] if algo=='DemographicsOnly' else X[labs].isna().astype(int) if algo=='MissingnessOnly' else X[labs] if algo=='LabsOnly_RF' else X
    for fold,(tr,va) in enumerate(splits,1):
        if algo=='LogisticRegression': est=LogisticRegression(C=1,class_weight='balanced',max_iter=4000,random_state=4200+fold)
        elif algo=='DecisionTree': est=DecisionTreeClassifier(min_samples_leaf=2,class_weight='balanced',random_state=4200+fold)
        elif algo=='GradientBoosting': est=GradientBoostingClassifier(n_estimators=100,learning_rate=.1,max_depth=3,random_state=4200+fold)
        else: est=RandomForestClassifier(n_estimators=500,min_samples_leaf=2,class_weight='balanced',random_state=4200+fold,n_jobs=-1)
        steps=[SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)]
        if algo=='LogisticRegression': steps.append(StandardScaler())
        model=make_pipeline(*steps,est)
        fitargs={'gradientboostingclassifier__sample_weight':compute_sample_weight('balanced',y.iloc[tr])} if algo=='GradientBoosting' else {}
        model.fit(xx.iloc[tr],y.iloc[tr],**fitargs)
        pr=model.predict_proba(xx.iloc[va]); probs[va]=pr[:,[list(model.classes_).index(c) for c in classes]]
        folds.append({'algorithm':algo,'fold':fold,'N':len(va),**metrics(y.iloc[va],probs[va])})
    params[algo]=model.get_params(deep=True)
    allprobs[algo]=probs; result={'algorithm':algo,**metrics(y,probs),'seconds':time.time()-start};results.append(result)
    print(json.dumps(result),flush=True)
benchmark=table('algorithm_comparison',pd.DataFrame(results)); table('fold_metrics',pd.DataFrame(folds))
dump('models/benchmark.json',{'protocol':'fixed 5-fold SGKF seed 42, median+missing indicators within train folds; raw scores, no clinical constraints; no tuning','results':results,'XGBoost':{'status':'not_run','reason':'xgboost not installed in available Python runtimes; no invented score'},'parameters':params})
probs=allprobs['RandomForest']; pred=np.array(classes)[probs.argmax(1)]
met={**metrics(y,probs),'rows':len(d),'class_order':classes,'classification_report':classification_report(y,pred,labels=classes,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y,pred,labels=classes).tolist(),'independent_groups':len(set(groups)),'validation':'reproduced 5-fold grouped OOF, raw RF scores, 500 trees, leaf=2, class_weight=balanced, seeds 4201..4205','original_metric_deltas':{k:float(metrics(y,probs)[k]-original[k]) for k in metrics(y,probs)}}
dump('models/metrics.json',met)
report=table('classification_report',pd.DataFrame(met['classification_report']).T.loc[classes].rename_axis('class').reset_index())
cm=np.array(met['confusion_matrix']); confused=table('confused_pairs',pd.DataFrame([{'actual':a,'predicted':b,'N':int(cm[i,j]),'pct_of_actual':100*cm[i,j]/sum(cm[i])} for i,a in enumerate(classes) for j,b in enumerate(classes) if i!=j and cm[i,j]>0]).sort_values(['N','actual'],ascending=[False,True]))
oof=pd.DataFrame({'row_index':np.arange(len(d)),'fold':foldids,'group':groups,'actual':y,'predicted':pred,'sex':d.sex,'age_years':d.age_years})
for i,c in enumerate(classes): oof['score_'+c]=probs[:,i]
table('oof_predictions',oof)
# Cluster bootstrap using fixed OOF predictions. Does not include model retraining uncertainty.
rng=np.random.default_rng(20261004)
def acc_ci(mask):
    ids=np.flatnonzero(mask); gs=np.unique(groups[ids]); vals=[]
    for _ in range(1500):
        chosen=rng.choice(gs,size=len(gs),replace=True); ix=np.concatenate([ids[groups[ids]==g] for g in chosen]); vals.append(np.mean(pred[ix]==y.iloc[ix].to_numpy()))
    return np.quantile(vals,[.025,.975])
fair=[]; ageband=pd.cut(d.age_years,[17,39,59,74,120],labels=['18–39','40–59','60–74','75+'])
for kind,series in [('sex',d.sex),('age',ageband)]:
    for val in series.dropna().unique():
        mask=(series==val).to_numpy(); lo,hi=acc_ci(mask)
        fair.append({'dimension':kind,'group':str(val),'N':int(mask.sum()),'represented_classes':int(y[mask].nunique()),**metrics(y[mask],probs[mask]),'acc_ci_low':lo,'acc_ci_high':hi})
fair=table('fairness',pd.DataFrame(fair)); dump('models/fairness.json',{'rows':fair.to_dict('records'),'ci_method':'1500 cluster-bootstrap resamples of fixed OOF predictions; seed 20261004; accuracy only; not retraining uncertainty','macro_f1_definition':'fixed 12 labels; absent classes zero; balanced_accuracy averages represented truth classes'})
sub=[]
for dim,series in [('sex',d.sex),('age',ageband)]:
    for val in series.dropna().unique():
        for c in classes:
            mask=((series==val)&(y==c)).to_numpy(); n=mask.sum()
            sub.append({'dimension':dim,'group':str(val),'class':c,'N':n,'recall':np.mean(pred[mask]==c) if n else np.nan})
table('subgroup_class_recall',pd.DataFrame(sub))
dump('models/research_metadata.json',{'date':'2026-10-04','data_sha256':sha(csv),'variables_sha256':sha(xlsx),'sklearn_version':sklearn.__version__,'pandas_version':pd.__version__,'scipy_version':scipy.__version__,'numpy_version':np.__version__,'contract_names_order_match':list(d.columns)==v.variable.tolist(),'features':features,'labs':labs,'xgboost_status':'not installed; not evaluated','model_metadata_note':'models/model_metadata.json copied from supplied project; describes original full-fit model, not newly fitted audit model','code_sha256':sha(Path(__file__))})
print('DONE',flush=True)
