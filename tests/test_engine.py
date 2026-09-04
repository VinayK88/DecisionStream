import pandas as pd
import pytest

from engine import (
    FEATURES,
    add_temporal_features,
    generate_events,
    policy,
    run_pipeline,
    validate_feature_contract,
)


def _small_events():
    return pd.DataFrame([
        {"event_id": "e1", "timestamp": "2026-01-01T00:00:00Z", "account_id": 1, "device_id": 8, "amount": 10, "account_age_days": 20, "device_risk": .1, "geo_velocity": .1, "prior_review_rate": .1},
        {"event_id": "e2", "timestamp": "2026-01-01T00:05:00Z", "account_id": 1, "device_id": 8, "amount": 20, "account_age_days": 20, "device_risk": .2, "geo_velocity": .2, "prior_review_rate": .1},
        {"event_id": "e3", "timestamp": "2026-01-01T01:00:00Z", "account_id": 1, "device_id": 8, "amount": 30, "account_age_days": 20, "device_risk": .3, "geo_velocity": .3, "prior_review_rate": .1},
    ])


def test_temporal_features_use_real_windows():
    output = add_temporal_features(_small_events()).set_index("event_id")
    assert output.loc["e1", "stream_velocity_10m"] == 0
    assert output.loc["e2", "stream_velocity_10m"] > 0
    assert output.loc["e3", "stream_velocity_10m"] == 0
    assert output.loc["e3", "batch_velocity_24h"] > output.loc["e2", "batch_velocity_24h"]


def test_future_event_does_not_change_prior_features():
    first_two = add_temporal_features(_small_events().iloc[:2]).set_index("event_id")
    all_three = add_temporal_features(_small_events()).set_index("event_id")
    pd.testing.assert_frame_equal(first_two[FEATURES], all_three.loc[["e1", "e2"], FEATURES])


def test_feature_contract_detects_missing_data():
    output = add_temporal_features(_small_events())
    assert validate_feature_contract(output)["valid"]
    assert not validate_feature_contract(output.drop(columns=["device_risk"]))["valid"]
    with pytest.raises(ValueError):
        add_temporal_features(_small_events().drop(columns=["amount"]))


def test_policy_orders_risk():
    assert policy([.1, .5, .7, .9]).tolist() == ["ALLOW", "REVIEW", "STEP_UP", "BLOCK"]


def test_pipeline_is_time_ordered_and_hashed():
    _, train, test, metrics = run_pipeline(seed=4, n=2500)
    assert train.timestamp.max() < test.timestamp.min()
    assert metrics["Feature contract valid"] is True
    assert len(metrics["Replay SHA-256"]) == 64
    assert metrics["P95 latency ms"] >= metrics["P50 latency ms"]
    assert metrics["Live production data"] == "No"
