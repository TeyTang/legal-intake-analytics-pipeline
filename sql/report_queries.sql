-- Open matters by practice area
SELECT
    practice_area,
    COUNT(*) AS open_matters
FROM legal_matters_cleaned
WHERE status = 'open'
GROUP BY practice_area
ORDER BY open_matters DESC;

-- Highest-risk matters for review
SELECT
    matter_id,
    practice_area,
    jurisdiction,
    risk_score,
    estimated_value_usd,
    priority_flag
FROM legal_matters_cleaned
ORDER BY risk_score DESC, estimated_value_usd DESC
LIMIT 10;

-- Summary table for stakeholder reporting
SELECT
    practice_area,
    matter_count,
    ROUND(avg_estimated_value_usd, 2) AS avg_estimated_value_usd,
    ROUND(avg_risk_score, 2) AS avg_risk_score,
    ROUND(avg_days_to_assign, 2) AS avg_days_to_assign,
    priority_matters
FROM practice_area_summary
ORDER BY matter_count DESC, avg_estimated_value_usd DESC;

-- Priority matters by intake channel
SELECT
    intake_channel,
    COUNT(*) AS total_matters,
    SUM(CASE WHEN priority_flag = 'priority' THEN 1 ELSE 0 END) AS priority_matters
FROM legal_matters_cleaned
GROUP BY intake_channel
ORDER BY priority_matters DESC, total_matters DESC;

-- Delayed assignment risk by practice area
SELECT
    practice_area,
    COUNT(*) AS total_matters,
    SUM(CASE WHEN assignment_delay_flag = 'delayed' THEN 1 ELSE 0 END) AS delayed_matters,
    ROUND(
        100.0 * SUM(CASE WHEN assignment_delay_flag = 'delayed' THEN 1 ELSE 0 END) / COUNT(*),
        1
    ) AS delayed_assignment_rate_pct
FROM legal_matters_cleaned
GROUP BY practice_area
ORDER BY delayed_assignment_rate_pct DESC, delayed_matters DESC;
