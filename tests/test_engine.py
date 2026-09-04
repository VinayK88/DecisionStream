from engine import generate_events, run_pipeline, policy

def test_generator_shape():
    df=generate_events(seed=3,n=1200)
    assert len(df)==1200
    assert {"stream_velocity_10m","batch_velocity_24h","label"}.issubset(df.columns)

def test_policy_orders_risk():
    out=policy([.1,.5,.7,.9]).tolist()
    assert out==["ALLOW","REVIEW","STEP_UP","BLOCK"]

def test_pipeline_metrics():
    _,_,test,metrics=run_pipeline(seed=4,n=5000)
    assert len(test)>1000
    assert 0<=metrics["Champion PR-AUC"]<=1
    assert metrics["P95 latency ms"]>=metrics["P50 latency ms"]
    assert metrics["Live production data"]=="No"
