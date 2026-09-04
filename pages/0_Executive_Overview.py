from __future__ import annotations
import numbers
import streamlit as st
import pandas as pd
import plotly.express as px
from engine import run_pipeline

st.set_page_config(page_title="DecisionStream · Executive Overview", page_icon="◉", layout="wide")
CSS="""
<style>
:root{--ink:#1d1d1f;--muted:#6e6e73;--soft:#f5f5f7;--line:#e8e8ed;--blue:#0071e3}
html,body,[class*="css"],.stApp,p,li,div,span,button,input,label{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Helvetica Neue",Helvetica,Arial,sans-serif!important;-webkit-font-smoothing:antialiased;color:var(--ink);font-size:16px}.stApp{background:linear-gradient(180deg,#fff,#fbfbfd)}.block-container{max-width:1500px;padding:2.3rem 2.2rem 5rem}#MainMenu,footer,header{visibility:hidden}[data-testid="stSidebar"]{background:#f5f5f7;border-right:1px solid var(--line)}[data-testid="stSidebar"] *{font-size:15px!important}.hero{background:radial-gradient(circle at 8% 0%,rgba(0,113,227,.16),transparent 32%),linear-gradient(155deg,#fff,#f5f5f7);border:1px solid var(--line);border-radius:36px;padding:54px 56px 48px;box-shadow:0 18px 48px rgba(0,0,0,.045)}.hero h1{font-size:3.65rem;line-height:.98;letter-spacing:-.062em;margin:.55rem 0 .9rem;font-weight:720;max-width:1020px}.hero p{color:var(--muted);font-size:1.12rem;line-height:1.6;max-width:940px;margin:0}.eyebrow{color:var(--blue);font-weight:760;font-size:.78rem;letter-spacing:.13em;text-transform:uppercase}.pills{margin-top:22px;display:flex;flex-wrap:wrap;gap:9px}.pill{background:#fff;border:1px solid var(--line);border-radius:999px;padding:8px 13px;color:#515154;font-size:.8rem}.section{font-size:1.72rem;letter-spacing:-.04em;font-weight:720;margin:32px 0 6px}.sub{color:var(--muted);font-size:.96rem;margin-bottom:16px}.kpi{background:#fff;border:1px solid var(--line);border-radius:26px;padding:20px;min-height:122px;box-shadow:0 8px 24px rgba(0,0,0,.028)}.kpi-label{color:var(--muted);font-size:.70rem;font-weight:760;letter-spacing:.08em;text-transform:uppercase}.kpi-value{font-size:1.62rem;line-height:1.03;font-weight:720;letter-spacing:-.045em;margin-top:12px}.callout{background:#fff;border:1px solid var(--line);border-radius:28px;padding:24px;min-height:162px;box-shadow:0 8px 24px rgba(0,0,0,.025)}.callout .cap{color:var(--blue);font-size:.70rem;font-weight:760;letter-spacing:.1em;text-transform:uppercase}.callout h3{font-size:1.25rem;letter-spacing:-.03em;margin:.55rem 0 .45rem}.callout p{color:var(--muted);font-size:.92rem;line-height:1.55;margin:0}</style>
"""
st.markdown(CSS,unsafe_allow_html=True)

def fmt(v):
    if isinstance(v,bool): return "Yes" if v else "No"
    if isinstance(v,numbers.Integral): return f"{int(v):,}"
    if isinstance(v,numbers.Real):
        v=float(v)
        if -1<=v<=1:return f"{v:.3f}"
        if abs(v)>=1000:return f"{v:,.0f}"
        return f"{v:,.2f}"
    return str(v).replace("_"," ")

def cards(items,cols=5):
    for s in range(0,len(items),cols):
        row=st.columns(cols)
        for i,(k,v) in enumerate(items[s:s+cols]):row[i].markdown(f'<div class="kpi"><div class="kpi-label">{k}</div><div class="kpi-value">{fmt(v)}</div></div>',unsafe_allow_html=True)

def style(fig,h=420):
    fig.update_layout(template="plotly_white",height=h,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family='-apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial',size=14,color="#1d1d1f"),title_font=dict(size=20),margin=dict(l=12,r=12,t=60,b=12),legend_title_text="")
    fig.update_xaxes(gridcolor="#ececf0");fig.update_yaxes(gridcolor="#ececf0");return fig

with st.sidebar:
    st.markdown("### DecisionStream")
    seed=st.number_input("Synthetic seed",1,9999,42)
    n=st.slider("Event volume",6000,30000,12000,1000)
    st.caption("Executive view uses the same replay engine as the main decision dashboard.")

df,train,test,base=run_pipeline(seed,n)
intervention=test.decision.isin(["REVIEW","STEP_UP","BLOCK"])
high_impact=test.decision.isin(["STEP_UP","BLOCK"])
champ_pred=(test.champion_score>=.64).astype(int)
false_pos=((champ_pred==1)&(test.label==0)).mean();false_neg=((champ_pred==0)&(test.label==1)).mean()
q90=test.champion_score.quantile(.90);top=test[test.champion_score>=q90]
chal_lift=float(base["Challenger PR-AUC"]-base["Champion PR-AUC"])

kpis=dict(base)
kpis.update({
    "Train rows":int(len(train)),"Scoring rows":int(len(test)),"Replay coverage":float(len(test)/len(df)),"Unique accounts":int(test.account_id.nunique()),"Unique devices":int(test.device_id.nunique()),
    "Adverse cases":int(test.label.sum()),"Non-adverse cases":int((1-test.label).sum()),"Intervention rate":float(intervention.mean()),"High-impact action rate":float(high_impact.mean()),
    "False-positive rate":float(false_pos),"False-negative rate":float(false_neg),"Top-decile event capture":float(top.label.sum()/max(test.label.sum(),1)),"Mean champion score":float(test.champion_score.mean()),"P95 champion score":float(test.champion_score.quantile(.95)),
    "Mean challenger score":float(test.challenger_score.mean()),"Challenger lift":chal_lift,"Mean shadow disagreement":float(test.shadow_disagreement.mean()),"P95 shadow disagreement":float(test.shadow_disagreement.quantile(.95)),
    "P99 latency ms":float(test.decision_latency_ms.quantile(.99)),"Mean latency ms":float(test.decision_latency_ms.mean()),"Mean decision cost μs":float(test.decision_cost_micros.mean()),"P95 decision cost μs":float(test.decision_cost_micros.quantile(.95)),
    "Avg event amount":float(test.amount.mean()),"P95 event amount":float(test.amount.quantile(.95)),"High-value share":float((test.amount>=test.amount.quantile(.90)).mean()),"Prevented loss / intervention":float(test.prevented_loss.sum()/max(intervention.sum(),1)),
    "Prevented loss / 1K events":float(test.prevented_loss.sum()/max(len(test),1)*1000),"Allow count":int((test.decision=="ALLOW").sum()),"Review count":int((test.decision=="REVIEW").sum()),"Step-up count":int((test.decision=="STEP_UP").sum()),"Block count":int((test.decision=="BLOCK").sum()),
    "Replay start":str(test.timestamp.min())[:16],"Replay end":str(test.timestamp.max())[:16],"Policy bands":4,"Tracked features":8,"Live production data":"No"
})

st.markdown('''<div class="hero"><div class="eyebrow">DecisionStream · Executive decision intelligence</div><h1>Make every decision observable.</h1><p>A product-level view of real-time + batch feature health, model quality, policy actions, latency, shadow behavior, and synthetic business impact.</p><div class="pills"><span class="pill">45+ KPIs</span><span class="pill">Streaming + batch</span><span class="pill">Policy bands</span><span class="pill">Shadow model</span><span class="pill">Replay</span></div></div>''',unsafe_allow_html=True)
st.markdown('<div class="section">Executive scorecard</div><div class="sub">Model quality, operational health, policy behavior, and synthetic business outcomes in one view.</div>',unsafe_allow_html=True)
cards(list(kpis.items()),5)

st.markdown('<div class="section">Three things to notice</div>',unsafe_allow_html=True)
c=st.columns(3)
c[0].markdown(f'<div class="callout"><div class="cap">Policy load</div><h3>{intervention.mean():.1%} intervention</h3><p>Review, step-up, and block actions consume operational capacity. Policy quality must be judged with workload and outcomes together.</p></div>',unsafe_allow_html=True)
c[1].markdown(f'<div class="callout"><div class="cap">Shadow behavior</div><h3>P95 disagreement {test.shadow_disagreement.quantile(.95):.3f}</h3><p>Large champion/challenger gaps are where replay analysis is most valuable before model promotion.</p></div>',unsafe_allow_html=True)
c[2].markdown(f'<div class="callout"><div class="cap">Latency</div><h3>P95 {test.decision_latency_ms.quantile(.95):.1f} ms</h3><p>Decision quality is only useful if the action arrives inside the operational latency budget.</p></div>',unsafe_allow_html=True)

c1,c2=st.columns(2)
with c1:
    d=test.decision.value_counts().rename_axis("decision").reset_index(name="events");fig=px.bar(d,x="decision",y="events",title="Policy action distribution");st.plotly_chart(style(fig,430),use_container_width=True)
with c2:
    test2=test.copy();test2["bucket"]=pd.qcut(test2.champion_score,10,duplicates="drop");cal=test2.groupby("bucket",observed=True).agg(champion=("champion_score","mean"),challenger=("challenger_score","mean"),actual=("label","mean")).reset_index(drop=True);fig=px.line(cal,y=["champion","challenger","actual"],markers=True,title="Champion vs challenger calibration");st.plotly_chart(style(fig,430),use_container_width=True)

st.caption("Synthetic replay only. Kafka/Spark-style integration patterns are architectural references, not claims of a live production deployment.")
