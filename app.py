from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from engine import run_pipeline

st.set_page_config(page_title="DecisionStream", page_icon="◉", layout="wide")
CSS="""<style>html, body, [class*="css"], .stApp {font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Helvetica Neue",Helvetica,Arial,sans-serif;color:#1d1d1f}.stApp{background:#fff}.block-container{max-width:1440px;padding-top:2.2rem;padding-bottom:4rem}#MainMenu,footer,header{visibility:hidden}.hero{background:radial-gradient(circle at 14% 8%,rgba(0,113,227,.12),transparent 30%),linear-gradient(180deg,#fff,#f5f5f7);border:1px solid #e8e8ed;border-radius:32px;padding:44px 48px 40px;margin-bottom:22px;box-shadow:0 12px 34px rgba(0,0,0,.035)}.eyebrow{color:#0071e3;font-weight:700;font-size:.78rem;letter-spacing:.11em;text-transform:uppercase}.hero h1{font-size:3.15rem;letter-spacing:-.055em;line-height:1.02;margin:.45rem 0 .7rem;font-weight:700}.hero p{color:#6e6e73;max-width:920px;font-size:1.06rem;line-height:1.55;margin:0}.pills{margin-top:18px;display:flex;gap:8px;flex-wrap:wrap}.pill{background:#fff;color:#515154;border:1px solid #e8e8ed;padding:7px 11px;border-radius:999px;font-size:.79rem}.kpi{background:#f5f5f7;border:1px solid #ececf0;border-radius:24px;padding:18px;min-height:116px;box-shadow:0 8px 22px rgba(0,0,0,.025)}.kpi-label{color:#6e6e73;text-transform:uppercase;letter-spacing:.075em;font-size:.68rem;font-weight:700}.kpi-value{font-size:1.48rem;font-weight:700;letter-spacing:-.035em;margin-top:9px}.section-title{font-size:1.5rem;letter-spacing:-.03em;margin:26px 0 12px;font-weight:700}.status{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;background:#f5f5f7;border-radius:999px;font-size:.78rem;color:#515154}.dot{width:8px;height:8px;border-radius:50%;background:#34c759;display:inline-block}.note{color:#6e6e73;font-size:.85rem}</style>"""
st.markdown(CSS,unsafe_allow_html=True)

def fmt(v):
    if isinstance(v,int): return f"{v:,}"
    if isinstance(v,float):
        if 0<=v<=1:return f"{v:.3f}"
        if abs(v)>=1000:return f"{v:,.0f}"
        return f"{v:,.2f}"
    return str(v).replace("_"," ")

def cards(items,cols=5):
    for s in range(0,len(items),cols):
        cs=st.columns(cols)
        for i,(k,v) in enumerate(items[s:s+cols]):
            cs[i].markdown(f'<div class="kpi"><div class="kpi-label">{k}</div><div class="kpi-value">{fmt(v)}</div></div>',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load(seed,n): return run_pipeline(seed,n)

with st.sidebar:
    st.markdown("### DecisionStream")
    seed=st.number_input("Synthetic seed",1,9999,42)
    n=st.slider("Event volume",6000,30000,12000,1000)
    st.caption("Reference replay. Kafka/Spark-style concepts are simulated locally with deterministic synthetic events.")

df,train,test,metrics=load(seed,n)
st.markdown("""<div class="hero"><div class="eyebrow">Security decisioning · Real-time + batch</div><h1>One decision surface. Two data speeds.</h1><p>DecisionStream demonstrates a production-minded decision pipeline that joins streaming behavioral signals with batch context, scores risk, applies policy, records latency/cost, and replays champion-vs-challenger behavior.</p><div class="pills"><span class="pill">Streaming features</span><span class="pill">Batch aggregates</span><span class="pill">Policy engine</span><span class="pill">Shadow model</span><span class="pill">Replay</span><span class="pill">SQL reference</span></div></div><div class="status"><span class="dot"></span> Synthetic replay healthy · live integrations disabled</div>""",unsafe_allow_html=True)
st.markdown('<div class="section-title">Decision health</div>',unsafe_allow_html=True);cards(list(metrics.items()),5)
tabs=st.tabs(["Decision flow","Latency & throughput","Champion / challenger","Policy outcomes","Replay data"])
with tabs[0]:
    st.markdown("#### Reference architecture");st.code("""Events → stream features ┐
                           ├→ feature join → champion score → policy → ALLOW / REVIEW / STEP_UP / BLOCK
Batch aggregates → SQL ───┘                         │
                                                   ├→ decision log
Shadow challenger ─────────────────────────────────┘
                                                   ↓
                                                replay""",language="text")
    d=test["decision"].value_counts().rename_axis("decision").reset_index(name="count")
    fig=px.bar(d,x="decision",y="count",title="Decision distribution");fig.update_layout(template="plotly_white",height=390);st.plotly_chart(fig,use_container_width=True)
with tabs[1]:
    c1,c2=st.columns(2)
    with c1:
        fig=px.histogram(test,x="decision_latency_ms",nbins=40,title="Decision latency distribution");fig.update_layout(template="plotly_white",height=390);st.plotly_chart(fig,use_container_width=True)
    with c2:
        by_action=test.groupby("decision",as_index=False).agg(avg_latency_ms=("decision_latency_ms","mean"),events=("event_id","count"))
        fig=px.scatter(by_action,x="avg_latency_ms",y="events",text="decision",size="events",title="Latency vs event volume");fig.update_traces(textposition="top center");fig.update_layout(template="plotly_white",height=390);st.plotly_chart(fig,use_container_width=True)
with tabs[2]:
    test2=test.copy();test2["bucket"]=pd.qcut(test2["champion_score"],10,duplicates="drop")
    cal=test2.groupby("bucket",observed=True).agg(champion=("champion_score","mean"),challenger=("challenger_score","mean"),actual=("label","mean")).reset_index(drop=True)
    fig=px.line(cal,y=["champion","challenger","actual"],markers=True,title="Score calibration by champion decile");fig.update_layout(template="plotly_white",height=430,legend_title_text="");st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Largest shadow disagreements");st.dataframe(test.nlargest(25,"shadow_disagreement")[["event_id","champion_score","challenger_score","shadow_disagreement","decision","label"]],hide_index=True,use_container_width=True)
with tabs[3]:
    pol=test.groupby("decision",as_index=False).agg(events=("event_id","count"),event_rate=("label","mean"),avg_amount=("amount","mean"),prevented_loss=("prevented_loss","sum"),p95_latency=("decision_latency_ms",lambda s:s.quantile(.95)))
    st.dataframe(pol,hide_index=True,use_container_width=True);fig=px.bar(pol,x="decision",y="event_rate",title="Synthetic adverse-event rate by action");fig.update_layout(template="plotly_white",height=380);st.plotly_chart(fig,use_container_width=True)
with tabs[4]:
    sample=test.sample(min(700,len(test)),random_state=7);st.dataframe(sample,hide_index=True,use_container_width=True);st.download_button("Export replay sample",test.to_csv(index=False),"decisionstream_replay.csv","text/csv")
st.markdown('<p class="note">Synthetic reference implementation. It demonstrates decisioning mechanics and observability; it is not connected to Apple, Microsoft, customer data, Kafka, or a production enforcement plane.</p>',unsafe_allow_html=True)
