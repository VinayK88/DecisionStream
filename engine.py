from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "amount_log", "account_age_signal", "device_risk", "geo_velocity",
    "stream_velocity_10m", "batch_velocity_24h", "device_account_fanout", "prior_review_rate"
]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def generate_events(seed=42, n=12000, start="2026-08-01"):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="min")
    accounts = rng.integers(1, 1800, n)
    devices = rng.integers(1, 950, n)
    amount = np.exp(rng.normal(3.3, 0.9, n))
    account_age = rng.exponential(160, n)
    device_risk = rng.beta(1.8, 5.0, n)
    geo_velocity = rng.beta(1.5, 7.0, n)
    base = pd.DataFrame({
        "event_id":[f"evt_{i:06d}" for i in range(n)],
        "timestamp":ts, "account_id":accounts, "device_id":devices,
        "amount":amount, "account_age_days":account_age,
        "device_risk":device_risk, "geo_velocity":geo_velocity,
    })
    grp = base.groupby("account_id").cumcount()
    base["stream_velocity_10m"] = np.minimum(grp % 12, 10) / 10.0
    base["batch_velocity_24h"] = base.groupby("account_id")["amount"].transform("count") / 18.0
    fanout = base.groupby("device_id")["account_id"].transform("nunique")
    base["device_account_fanout"] = np.clip(fanout / 10.0, 0, 1.5)
    base["prior_review_rate"] = rng.beta(1.2, 12.0, n)
    base["amount_log"] = np.log1p(base["amount"])
    base["account_age_signal"] = np.exp(-base["account_age_days"]/85.0)

    logit = (-5.2 + 0.42*base["amount_log"] + 1.9*base["device_risk"] + 1.35*base["geo_velocity"]
             + 1.15*base["stream_velocity_10m"] + 0.9*base["batch_velocity_24h"]
             + 1.15*base["device_account_fanout"] + 1.1*base["prior_review_rate"]
             + .85*base["account_age_signal"] + rng.normal(0,.3,n))
    base["label"] = rng.binomial(1, sigmoid(logit))
    return base

def train_and_score(df):
    cut = int(len(df)*0.72)
    train, test = df.iloc[:cut], df.iloc[cut:].copy()
    scaler=StandardScaler()
    Xtr=scaler.fit_transform(train[FEATURES])
    Xte=scaler.transform(test[FEATURES])
    champion=LogisticRegression(max_iter=500,class_weight="balanced").fit(Xtr,train.label)
    cprob=champion.predict_proba(Xte)[:,1]
    challenger=HistGradientBoostingClassifier(max_iter=120,learning_rate=.07,max_leaf_nodes=15,random_state=7).fit(train[FEATURES],train.label)
    hprob=challenger.predict_proba(test[FEATURES])[:,1]
    test["champion_score"]=cprob
    test["challenger_score"]=hprob
    test["shadow_disagreement"]=abs(cprob-hprob)
    return train,test

def policy(score):
    score=np.asarray(score)
    return np.select([score>=.82, score>=.64, score>=.38],["BLOCK","STEP_UP","REVIEW"],default="ALLOW")

def run_pipeline(seed=42,n=12000):
    df=generate_events(seed,n)
    train,test=train_and_score(df)
    test["decision"]=policy(test.champion_score)
    rng=np.random.default_rng(seed+99)
    base_latency=np.where(test.decision.eq("ALLOW"),18,np.where(test.decision.eq("REVIEW"),31,np.where(test.decision.eq("STEP_UP"),43,26)))
    test["decision_latency_ms"]=np.maximum(5,base_latency+rng.normal(0,5,len(test)))
    test["decision_cost_micros"]=np.where(test.decision.eq("ALLOW"),7,np.where(test.decision.eq("REVIEW"),19,np.where(test.decision.eq("STEP_UP"),31,24)))
    test["expected_loss"]=test.amount*(.25+.75*test.label)
    test["prevented_loss"]=np.where(test.decision.isin(["STEP_UP","BLOCK"]) & test.label.eq(1),test.expected_loss*.82,0)
    pred=(test.champion_score>=.64).astype(int)
    metrics={
        "Events scored":int(len(test)),"Synthetic event rate":float(test.label.mean()),
        "Champion PR-AUC":float(average_precision_score(test.label,test.champion_score)),
        "Champion ROC-AUC":float(roc_auc_score(test.label,test.champion_score)),
        "Precision @ intervention":float(precision_score(test.label,pred,zero_division=0)),
        "Recall @ intervention":float(recall_score(test.label,pred,zero_division=0)),
        "Brier score":float(brier_score_loss(test.label,test.champion_score)),
        "Challenger PR-AUC":float(average_precision_score(test.label,test.challenger_score)),
        "Shadow disagreement":float(test.shadow_disagreement.mean()),
        "P50 latency ms":float(test.decision_latency_ms.median()),"P95 latency ms":float(test.decision_latency_ms.quantile(.95)),
        "Allow rate":float((test.decision=="ALLOW").mean()),"Review rate":float((test.decision=="REVIEW").mean()),
        "Step-up rate":float((test.decision=="STEP_UP").mean()),"Block rate":float((test.decision=="BLOCK").mean()),
        "Mean decision cost μs":float(test.decision_cost_micros.mean()),"Prevented synthetic loss":float(test.prevented_loss.sum()),
        "Feature parity checks":len(FEATURES),"Replay rows":int(len(test)),"Live production data":"No",
    }
    return df,train,test,metrics

def main():
    p=argparse.ArgumentParser(description="DecisionStream synthetic real-time + batch decisioning replay")
    p.add_argument("--out",default="artifacts");p.add_argument("--rows",type=int,default=12000);p.add_argument("--seed",type=int,default=42)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df,train,test,metrics=run_pipeline(args.seed,args.rows)
    test.to_csv(out/"decision_replay.csv",index=False);pd.Series(metrics).to_json(out/"metrics.json",indent=2)
    print(pd.Series(metrics).to_string())

if __name__=="__main__": main()
