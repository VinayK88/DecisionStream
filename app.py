from __future__ import annotations
import numbers
import streamlit as st
import pandas as pd
import plotly.express as px
from engine import run_pipeline

st.set_page_config(page_title="DecisionStream", page_icon="◉", layout="wide")

CSS="""
<style>
html,body,[class*="css"],.stApp{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Helvetica Neue",Helvetica,Arial,sans-serif;color:#1d1d1f}
.stApp{background:linear-gradient(180deg,#ffffff 0%,#fbfbfd 100%)}
.block-container{max-width:1480px;padding-top:2rem;padding-bottom:5rem}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stSidebar"]{background:#f5f5f7;border-right:1px solid #e8e8ed}
.hero{position:relative;overflow:hidden;background:radial-gradient(circle at 9% 0%,rgba(0,113,227,.15),transparent 31%),linear-gradient(155deg,#fff,#f5f5f7);border:1px solid #e8e8ed;border-radius:34px;padding:50px 52px 44px;margin-bottom:18px;box-shadow:0 18px 48px rgba(0,0,0,.045)}
.hero:after{content:"";position:absolute;width:280px;height:280px;right:-85px;top:-100px;border-radius:50%;background:rgba(0,113,227,.05)}
.eyebrow{color:#0071e3;font-weight:750;font-size:.76rem;letter-spacing:.12em;text-transform:uppercase}
.hero h1{font-size:3.45rem;letter-spacing:-.06em;line-height:.98;margin:.55rem 0 .8rem;font-weight:730;max-width:940px}
.hero p{color:#6e6e73;max-width:960px;font-size:1.08rem;line-height:1.58;margin:0}
.pills{margin-top:20px;display:flex;gap:8px;flex-wrap:wrap}.pill{background:rgba(255,255,255,.9);color:#515154;border:1px solid #e8e8ed;padding:7px 12px;border-radius:999px;font-size:.79rem}
.status-row{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 24px}.status{display:inline-flex;align-items:center;gap:8px;padding:9px 13px;background:#fff;border:1px solid #e8e8ed;border-radius:999px;font-size:.79rem;color:#515154;box-shadow:0 6px 18px rgba(0,0,0,.025)}.dot{width:8px;height:8px;border-radius:50%;background:#34c759;display:inline-block}
.section-title{font-size:1.55rem;letter-spacing:-.035em;margin:30px 0 13px;font-weight:720}.section-sub{color:#6e6e73;font-size:.91rem;margin-top:-6px;margin-bottom:15px}
.kpi{background:rgba(255,255,255,.97);border:1px solid #e8e8ed;border-radius:25px;padding:18px;min-height:116px;box-shadow:0 8px 24px rgba(0,0,0,.028)}.kpi:hover{box-shadow:0 12px 30px rgba(0,0,0,.045);transform:translateY(-1px);transition:.16s ease}.kpi-label{color:#6e6e73;text-transform:uppercase;letter-spacing:.075em;font-size:.67rem;font-weight:760}.kpi-value{font-size:1.48rem;font-weight:730;letter-spacing:-.04em;margin-top:10px;line-height:1.05}
.insight{background:#fff;border:1px solid #e8e8ed;border-radius:26px;padding:22px 23px;min-height:148px;box-shadow:0 8px 24px rgba(0,0,0,.025)}.insight .cap{color:#0071e3;text-transform:uppercase;letter-spacing:.09em;font-size:.68rem;font-weight:760}.insight h3{font-size:1.15rem;letter-spacing:-.025em;margin:.55rem 0 .45rem}.insight p{color:#6e6e73;font-size:.88rem;line-height:1.48;margin:0}.note{color:#6e6e73;font-size:.84rem}
div[data-baseweb="tab-list"]{gap:8px}button[data-baseweb="tab"]{border-radius:999px;padding-left:14px;padding-right:14px}
</style>
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
        cs=st.columns(cols)
        for i,(k,v) in enumerate(items[s:s+cols]):
            cs[i].markdown(f'<div class="kpi"><div class="kpi-label">{k}</div><div class="kpi-value">{fmt(v)}</div></div>',unsafe_allow_html=True)


def style_fig(fig,height=430):
    fig.update_layout(template="plotly_white",height=height,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family='-apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial',color="#1d1d1f"),title_font=dict(size=18),margin=dict(l=12,r=12,t=58,b=12),legend_title_text="")
    fig.update_xaxes(gridcolor="#eeeeF2",zerolinecolor="#eeeeF2")
    fig.update_yaxes(gridcolor="#eeeeF2",zerolinecolor="#eeeeF2")
    return fig


@st.cache_data(show_spinner=False)
def load(seed,n): return run_pipeline(seed,n)


with st.sidebar:
    st.markdown("### DecisionStream")
    st.caption("Real-time + batch decisioning")
    seed=st.number_input("Synthetic seed",1,9999,42)
    n=st.slider("Event volume",6000,30000,12000,1000)
    st.divider()
    st.caption("Reference replay. Kafka/Spark-style concepts are simulated locally while preserving a clear feature contract and decision log.")


df,train,test,metrics=load(seed,n)

intervention=test.champion_score>=.64
pred=intervention.astype(int)
false_positive_rate=float(((pred==1)&(test.label==0)).sum()/max((test.label==0).sum(),1))
false_negative_rate=float(((pred==0)&(test.label==1)).sum()/max((test.label==1).sum(),1))
high_impact=test.decision.isin(["STEP_UP","BLOCK"])
challenger_lift=metrics["Challenger PR-AUC"]-metrics["Champion PR-AUC"]
shadow_p95=float(test.shadow_disagreement.quantile(.95))
avg_cost=float(test.decision_cost_micros.mean())
p95_cost=float(test.decision_cost_micros.quantile(.95))
interventions=max(int(intervention.sum()),1)
prevented_per_intervention=float(test.prevented_loss.sum()/interventions)
prevented_per_1k=float(test.prevented_loss.sum()/max(len(test),1)*1000)
avg_amount=float(test.amount.mean())
high_value_share=float((test.amount>=test.amount.quantile(.90)).mean())
score_p95=float(test.champion_score.quantile(.95))
replay_hours=float((test.timestamp.max()-test.timestamp.min()).total_seconds()/3600) if len(test)>1 else 0.0
adverse_cases=int(test.label.sum())
top_decile=test.nlargest(max(1,int(len(test)*.10)),"champion_score")
top_decile_capture=float(top_decile.label.sum()/max(test.label.sum(),1))

extended=dict(metrics)
extended.update({
    "Training rows":int(len(train)),
    "Adverse cases scored":adverse_cases,
    "Intervention rate":float(intervention.mean()),
    "High-impact action rate":float(high_impact.mean()),
    "False-positive rate":false_positive_rate,
    "False-negative rate":false_negative_rate,
    "Top-decile event capture":top_decile_capture,
    "Champion score P95":score_p95,
    "Shadow disagreement P95":shadow_p95,
    "Challenger lift":float(challenger_lift),
    "P95 decision cost μs":p95_cost,
    "Prevented loss / intervention":prevented_per_intervention,
    "Prevented loss / 1K":prevented_per_1k,
    "Average event amount":avg_amount,
    "High-value event share":high_value_share,
    "Replay window h":replay_hours,
    "Policy bands":4,
    "Streaming + batch features":8,
})

st.markdown("""
<div class="hero"><div class="eyebrow">Security decisioning · Real-time + batch</div><h1>One decision surface. Two data speeds.</h1><p>DecisionStream joins fresh behavioral signals with durable historical context, scores each event with a champion model, applies a transparent policy, shadows a challenger, and records the operational evidence needed for replay, regression analysis, and business-health monitoring.</p><div class="pills"><span class="pill">Streaming features</span><span class="pill">Batch aggregates</span><span class="pill">Policy engine</span><span class="pill">Shadow model</span><span class="pill">Latency + cost</span><span class="pill">Replay</span><span class="pill">SQL contract</span></div></div>
<div class="status-row"><div class="status"><span class="dot"></span>Synthetic replay healthy</div><div class="status">Champion controls policy</div><div class="status">Challenger remains shadow-only</div><div class="status">Live integrations disabled</div></div>
""",unsafe_allow_html=True)

st.markdown('<div class="section-title">Decision health</div>',unsafe_allow_html=True)
st.markdown('<div class="section-sub">30+ quality, latency, policy, cost, and business KPIs around the same decision point.</div>',unsafe_allow_html=True)
cards(list(extended.items()),5)

st.markdown('<div class="section-title">Executive readout</div>',unsafe_allow_html=True)
ins=st.columns(3)
ins[0].markdown(f'<div class="insight"><div class="cap">Quality</div><h3>{metrics["Champion PR-AUC"]:.3f} PR-AUC</h3><p>Champion quality is paired with intervention precision, recall, calibration, and top-decile event capture—not reported as an isolated score.</p></div>',unsafe_allow_html=True)
ins[1].markdown(f'<div class="insight"><div class="cap">Efficiency</div><h3>{metrics["P95 latency ms"]:.1f} ms P95</h3><p>Operational health includes latency, synthetic decision cost, replay volume, and the policy mix across allow, review, step-up, and block.</p></div>',unsafe_allow_html=True)
ins[2].markdown(f'<div class="insight"><div class="cap">Shadow</div><h3>Δ {challenger_lift:+.3f}</h3><p>Challenger lift is visible alongside disagreement. Better aggregate PR-AUC alone does not trigger promotion.</p></div>',unsafe_allow_html=True)


tabs=st.tabs(["Decision flow","Latency & throughput","Champion / challenger","Policy outcomes","Replay data"])

with tabs[0]:
    st.markdown("#### Reference architecture")
    st.code("""Events → streaming features ┐
                             ├→ feature join → champion score → policy → ALLOW / REVIEW / STEP_UP / BLOCK
Batch history → SQL ────────┘                         │
                                                     ├→ decision telemetry
Shadow challenger ───────────────────────────────────┘
                                                     ↓
                                                   replay""",language="text")
    d=test["decision"].value_counts().rename_axis("decision").reset_index(name="count")
    fig=px.bar(d,x="decision",y="count",title="Decision distribution")
    st.plotly_chart(style_fig(fig,400),use_container_width=True)

with tabs[1]:
    c1,c2=st.columns(2)
    with c1:
        fig=px.histogram(test,x="decision_latency_ms",nbins=40,title="Decision latency distribution")
        st.plotly_chart(style_fig(fig,405),use_container_width=True)
    with c2:
        by_action=test.groupby("decision",as_index=False).agg(avg_latency_ms=("decision_latency_ms","mean"),p95_latency_ms=("decision_latency_ms",lambda s:s.quantile(.95)),events=("event_id","count"),avg_cost=("decision_cost_micros","mean"))
        fig=px.scatter(by_action,x="avg_latency_ms",y="events",text="decision",size="avg_cost",title="Latency vs event volume")
        fig.update_traces(textposition="top center")
        st.plotly_chart(style_fig(fig,405),use_container_width=True)
    st.dataframe(by_action,hide_index=True,use_container_width=True)

with tabs[2]:
    test2=test.copy();test2["bucket"]=pd.qcut(test2["champion_score"],10,duplicates="drop")
    cal=test2.groupby("bucket",observed=True).agg(champion=("champion_score","mean"),challenger=("challenger_score","mean"),actual=("label","mean"),events=("event_id","count")).reset_index(drop=True)
    fig=px.line(cal,y=["champion","challenger","actual"],markers=True,title="Score calibration by champion decile")
    st.plotly_chart(style_fig(fig,445),use_container_width=True)
    st.markdown("#### Largest shadow disagreements")
    st.dataframe(test.nlargest(30,"shadow_disagreement")[["event_id","champion_score","challenger_score","shadow_disagreement","decision","amount","label"]],hide_index=True,use_container_width=True)

with tabs[3]:
    pol=test.groupby("decision",as_index=False).agg(events=("event_id","count"),event_rate=("label","mean"),avg_score=("champion_score","mean"),avg_amount=("amount","mean"),prevented_loss=("prevented_loss","sum"),avg_latency=("decision_latency_ms","mean"),p95_latency=("decision_latency_ms",lambda s:s.quantile(.95)))
    st.dataframe(pol,hide_index=True,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(pol,x="decision",y="event_rate",title="Synthetic adverse-event rate by action")
        st.plotly_chart(style_fig(fig,390),use_container_width=True)
    with c2:
        fig=px.bar(pol,x="decision",y="prevented_loss",title="Prevented synthetic loss by action")
        st.plotly_chart(style_fig(fig,390),use_container_width=True)
    st.info("Policy thresholds are intentionally explicit: ALLOW < .38, REVIEW .38–.64, STEP_UP .64–.82, BLOCK ≥ .82. They are synthetic demo thresholds, not production recommendations.")

with tabs[4]:
    st.markdown("#### Replayable decision log")
    sample=test.sample(min(900,len(test)),random_state=7)
    st.dataframe(sample,hide_index=True,use_container_width=True)
    st.download_button("Export replay sample",test.to_csv(index=False),"decisionstream_replay.csv","text/csv")

st.markdown('<p class="note">Synthetic reference implementation. It demonstrates decisioning mechanics and observability; it is not connected to Apple, Microsoft, customer data, Kafka, Spark, or a production enforcement plane.</p>',unsafe_allow_html=True)
