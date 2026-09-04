-- DecisionStream batch feature contract (reference SQL)
-- Replace dialect-specific timestamp syntax for the target warehouse.

WITH history AS (
  SELECT
    account_id,
    event_id,
    event_ts,
    decision,
    COUNT(*) OVER (
      PARTITION BY account_id
      ORDER BY event_ts
      RANGE BETWEEN INTERVAL '24' HOUR PRECEDING AND CURRENT ROW
    ) AS events_24h,
    AVG(CASE WHEN decision = 'REVIEW' THEN 1.0 ELSE 0.0 END) OVER (
      PARTITION BY account_id
      ORDER BY event_ts
      ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING
    ) AS prior_review_rate
  FROM decision_events
)
SELECT
  event_id,
  account_id,
  events_24h / 18.0 AS batch_velocity_24h,
  COALESCE(prior_review_rate, 0.0) AS prior_review_rate
FROM history;
