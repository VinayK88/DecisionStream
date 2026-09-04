# Evaluation notes

DecisionStream evaluates more than predictive quality. A decision service also needs operational and policy health.

The replay reports:

- PR-AUC / ROC-AUC and calibration
- precision and recall at the intervention boundary
- champion-vs-challenger disagreement
- action mix
- P50 / P95 decision latency
- synthetic decision cost
- prevented synthetic loss
- feature-contract parity count

The recommended production extension is to replace the local stream simulator with Kafka-compatible ingestion and the local batch aggregation with Spark/SQL while preserving the same feature contract and replay log.

All current metrics are synthetic and should be interpreted only as reference behavior.
