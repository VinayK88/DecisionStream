from __future__ import annotations
import streamlit as st
import pandas as pd
from engine import run_pipeline

st.set_page_config(page_title="DecisionStream · Sample I/O", page_icon="↔", layout="wide")

st.markdown("""
<style>
html,body,[class*="css"],.stApp{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Helvetica Neue",Helvetica,Arial,sans-serif;color:#1d1d1f}.stApp{background:linear-gradient(180deg,#fff 0%,#fbfbfd 100%)}.block-container{max-width:1380px;padding-top:2.2rem;padding-bottom:5rem}#MainMenu,footer,header{visibility:hidden}.hero{background:radial-gradient(circle at 8% 0%,rgba(0,113,227,.14),transparent 32%),linear-gradient(155deg,#fff,#f5f5f7);border:1px solid #e8e8ed;border-radius:34px;padding:44px 48px;margin-bottom:22px;box-shadow:0 18px 48px rgba(0,0,0,.04)}.eyebrow{color:#0071e3;font-size:.75rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase}.hero h1{font-size:3rem;letter-spacing:-.055em;line-height:1;margin:.55rem 0 .7rem}.hero p{color:#6e6e73;max-width:920px;font-size:1.04rem;line-height:1.55}.card{background:#fff;border:1px solid #e8e8ed;border-radius:26px;padding:22px;box-shadow:0 8px 24px rgba(0,0,0,.025)}.step{color:#0071e3;font-size:.69rem;font-weight:760;letter-spacing:.1em;text-transform:uppercase}.big{font-size:1.35rem;font-weight:720;letter-spacing:-.03em;margin:.35rem 0}.muted{color:#6e6e73;font-size:.9rem;line-height:1.5}.arrow{text-align:center;font-size:2rem;color:#0071e3;padding-top:40px}.badge{display:inline-block;background:#f5f5f7;border:1px solid #e8e8ed;border-radius:999px;padding:7px 10px;margin:0 6px 6px 0;color:#515154;font-size:.78rem}</style>
<div class="hero"><div class="eyebrow">DecisionStream · Sample Input → Output</div><h1>Follow one event through the decision surface.</h1><p>This example uses the same synthetic replay as the main dashboard and shows how a raw event becomes streaming/batch features, champion and challenger scores, a policy action, and decision telemetry.</p></div>
""",unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load(): return run_pipeline(seed=42,n=12000)

df,train,test,metrics=load()
idx=st.slider("Replay example",0,min(50,len(test)-1),0)
row=test.iloc[idx]

st.markdown("### Example flow")
c1,c2,c3=st.columns([1,0.12,1])
with c1:
    st.markdown('<div class="card"><div class="step">1 · Sample input</div><div class="big">Real-time event + joined context</div><div class="muted">In production, these fields could come from Kafka/Event Hubs/Kinesis plus warehouse or feature-store aggregates.</div></div>',unsafe_allow_html=True)
    input_cols=["event_id","timestamp","account_id","device_id","amount","device_risk","geo_velocity","stream_velocity_10m","batch_velocity_24h","device_account_fanout","prior_review_rate","account_age_signal"]
    st.dataframe(pd.DataFrame([{k:row[k] for k in input_cols}]),hide_index=True,use_container_width=True)
with c2: st.markdown('<div class="arrow">→</div>',unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card"><div class="step">2 · Decision output</div><div class="big">Score + policy + telemetry</div><div class="muted">The champion controls the synthetic policy; the challenger remains shadow-only.</div></div>',unsafe_allow_html=True)
    output_cols=["champion_score","challenger_score","shadow_disagreement","decision","decision_latency_ms","decision_cost_micros","expected_loss","prevented_loss","label"]
    st.dataframe(pd.DataFrame([{k:row[k] for k in output_cols}]),hide_index=True,use_container_width=True)

st.markdown("### What the pipeline did")
st.markdown('<span class="badge">stream feature extraction</span><span class="badge">batch feature join</span><span class="badge">champion scoring</span><span class="badge">shadow challenger</span><span class="badge">policy thresholds</span><span class="badge">decision logging</span>',unsafe_allow_html=True)

st.markdown("### Policy interpretation")
score=float(row.champion_score)
if row.decision=="ALLOW": msg="Risk stayed below the review boundary, so the event is allowed in this reference policy."
elif row.decision=="REVIEW": msg="The event enters a human-review band rather than being blocked automatically."
elif row.decision=="STEP_UP": msg="The event crosses the step-up threshold, representing an extra verification or friction step."
else: msg="The event crosses the synthetic block threshold. A real system would require policy, legal, and operational validation before using such an action."
st.info(f"**Champion score {score:.3f} → {row.decision}**. {msg}")

st.caption("All events, outcomes, latency, cost, and prevented-loss values are synthetic. This page demonstrates the input/output contract of the decision service.")
