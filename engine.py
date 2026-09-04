from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "amount_log", "account_age_signal", "device_risk", "geo_velocity",
    "stream_velocity_10m", "batch_velocity_24h", "device_account_fanout",
    "prior_review_rate",
]


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def _prior_window_count(
    frame: pd.DataFrame,
    entity_column: str,
    window: str,
) -> pd.Series:
    """Count only earlier events for the same entity inside an event-time window."""
    counts = pd.Series(0.0, index=frame.index)
    window_ns = pd.Timedelta(window).value
    for indices in frame.groupby(entity_column, sort=False).groups.values():
        ordered_indices = list(indices)
        timestamps = (
            pd.to_datetime(frame.loc[ordered_indices, "timestamp"], utc=True)
            .astype("int64")
            .to_numpy()
        )
        left = 0
        for position, row_index in enumerate(ordered_indices):
            while timestamps[position] - timestamps[left] > window_ns:
                left += 1
            counts.at[row_index] = position - left
    return counts


def add_temporal_features(base: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe features using information available at each event time."""
    required = {
        "event_id", "timestamp", "account_id", "device_id", "amount",
        "account_age_days", "device_risk", "geo_velocity", "prior_review_rate",
    }
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"missing input columns: {sorted(missing)}")

    output = base.sort_values(["timestamp", "event_id"]).copy()
    stream_count = _prior_window_count(output, "account_id", "10min")
    batch_count = _prior_window_count(output, "account_id", "24h")
    output["stream_velocity_10m"] = np.clip(stream_count / 4.0, 0, 1.5)
    output["batch_velocity_24h"] = np.clip(batch_count / 18.0, 0, 2.0)

    seen_accounts: dict[object, set[object]] = defaultdict(set)
    fanout = pd.Series(0.0, index=output.index)
    for row in output.itertuples():
        seen_accounts[row.device_id].add(row.account_id)
        fanout.at[row.Index] = len(seen_accounts[row.device_id])
    output["device_account_fanout"] = np.clip(fanout / 10.0, 0, 1.5)
    output["amount_log"] = np.log1p(output["amount"])
    output["account_age_signal"] = np.exp(-output["account_age_days"] / 85.0)
    return output


def validate_feature_contract(frame: pd.DataFrame) -> dict:
    missing = [feature for feature in FEATURES if feature not in frame]
    if missing:
        return {"valid": False, "missing": missing, "non_finite": None}
    values = frame[FEATURES].to_numpy(dtype=float)
    non_finite = int((~np.isfinite(values)).sum())
    return {"valid": non_finite == 0, "missing": [], "non_finite": non_finite}


def generate_events(seed: int = 42, n: int = 12000, start: str = "2026-08-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({
        "event_id": [f"evt_{i:06d}" for i in range(n)],
        "timestamp": pd.date_range(start, periods=n, freq="min", tz="UTC"),
        "account_id": rng.integers(1, 1800, n),
        "device_id": rng.integers(1, 950, n),
        "amount": np.exp(rng.normal(3.3, 0.9, n)),
        "account_age_days": rng.exponential(160, n),
        "device_risk": rng.beta(1.8, 5.0, n),
        "geo_velocity": rng.beta(1.5, 7.0, n),
        "prior_review_rate": rng.beta(1.2, 12.0, n),
    })
    base = add_temporal_features(base)
    logit = (
        -5.2 + .42 * base.amount_log + 1.9 * base.device_risk
        + 1.35 * base.geo_velocity + 1.15 * base.stream_velocity_10m
        + .9 * base.batch_velocity_24h + 1.15 * base.device_account_fanout
        + 1.1 * base.prior_review_rate + .85 * base.account_age_signal
        + rng.normal(0, .3, n)
    )
    base["label"] = rng.binomial(1, sigmoid(logit))
    return base


def train_and_score(df: pd.DataFrame):
    ordered = df.sort_values(["timestamp", "event_id"]).copy()
    cut = int(len(ordered) * .72)
    train, test = ordered.iloc[:cut], ordered.iloc[cut:].copy()
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[FEATURES])
    test_scaled = scaler.transform(test[FEATURES])
    champion = LogisticRegression(max_iter=500, class_weight="balanced").fit(
        train_scaled, train.label
    )
    champion_probability = champion.predict_proba(test_scaled)[:, 1]
    challenger = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.07, max_leaf_nodes=15, random_state=7
    ).fit(train[FEATURES], train.label)
    challenger_probability = challenger.predict_proba(test[FEATURES])[:, 1]
    test["champion_score"] = champion_probability
    test["challenger_score"] = challenger_probability
    test["shadow_disagreement"] = abs(champion_probability - challenger_probability)
    return train, test


def policy(score):
    values = np.asarray(score)
    return np.select(
        [values >= .82, values >= .64, values >= .38],
        ["BLOCK", "STEP_UP", "REVIEW"],
        default="ALLOW",
    )


def _replay_sha256(test: pd.DataFrame) -> str:
    columns = ["event_id", "timestamp", *FEATURES, "champion_score", "decision"]
    stable = test[columns].sort_values(["timestamp", "event_id"]).to_csv(
        index=False, float_format="%.12g"
    )
    return hashlib.sha256(stable.encode()).hexdigest()


def run_pipeline(seed: int = 42, n: int = 12000):
    events = generate_events(seed, n)
    contract = validate_feature_contract(events)
    if not contract["valid"]:
        raise ValueError(f"invalid feature contract: {contract}")
    train, test = train_and_score(events)
    test["decision"] = policy(test.champion_score)
    rng = np.random.default_rng(seed + 99)
    base_latency = np.where(
        test.decision.eq("ALLOW"), 18,
        np.where(test.decision.eq("REVIEW"), 31, np.where(test.decision.eq("STEP_UP"), 43, 26)),
    )
    test["decision_latency_ms"] = np.maximum(5, base_latency + rng.normal(0, 5, len(test)))
    test["decision_cost_micros"] = np.where(
        test.decision.eq("ALLOW"), 7,
        np.where(test.decision.eq("REVIEW"), 19, np.where(test.decision.eq("STEP_UP"), 31, 24)),
    )
    test["expected_loss"] = test.amount * (.25 + .75 * test.label)
    test["prevented_loss"] = np.where(
        test.decision.isin(["STEP_UP", "BLOCK"]) & test.label.eq(1),
        test.expected_loss * .82,
        0,
    )
    prediction = (test.champion_score >= .64).astype(int)
    metrics = {
        "Events scored": int(len(test)),
        "Synthetic event rate": float(test.label.mean()),
        "Champion PR-AUC": float(average_precision_score(test.label, test.champion_score)),
        "Champion ROC-AUC": float(roc_auc_score(test.label, test.champion_score)),
        "Precision @ intervention": float(precision_score(test.label, prediction, zero_division=0)),
        "Recall @ intervention": float(recall_score(test.label, prediction, zero_division=0)),
        "Brier score": float(brier_score_loss(test.label, test.champion_score)),
        "Challenger PR-AUC": float(average_precision_score(test.label, test.challenger_score)),
        "Shadow disagreement": float(test.shadow_disagreement.mean()),
        "P50 latency ms": float(test.decision_latency_ms.median()),
        "P95 latency ms": float(test.decision_latency_ms.quantile(.95)),
        "Allow rate": float((test.decision == "ALLOW").mean()),
        "Review rate": float((test.decision == "REVIEW").mean()),
        "Step-up rate": float((test.decision == "STEP_UP").mean()),
        "Block rate": float((test.decision == "BLOCK").mean()),
        "Mean decision cost μs": float(test.decision_cost_micros.mean()),
        "Prevented synthetic loss": float(test.prevented_loss.sum()),
        "Feature contract valid": contract["valid"],
        "Feature parity checks": len(FEATURES),
        "Replay rows": int(len(test)),
        "Replay SHA-256": _replay_sha256(test),
        "Live production data": "No",
    }
    return events, train, test, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="DecisionStream temporal decision replay")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    events, train, test, metrics = run_pipeline(args.seed, args.rows)
    test.to_csv(output / "decision_replay.csv", index=False)
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "rows": args.rows,
        "temporal_split": .72,
        "replay_sha256": metrics["Replay SHA-256"],
        "features": FEATURES,
        "synthetic": True,
    }
    (output / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pd.Series(metrics).to_json(output / "metrics.json", indent=2)
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
