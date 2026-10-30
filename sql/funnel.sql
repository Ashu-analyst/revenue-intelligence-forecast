-- Quarterly full-funnel conversion by channel (DuckDB SQL, portable to Snowflake/BigQuery/Postgres).
-- Cohort logic: every stage is attributed to the quarter the LEAD was created,
-- so conversion rates compare like with like.
WITH lead_cohort AS (
    SELECT
        l.lead_id,
        l.channel,
        date_trunc('quarter', CAST(l.created_date AS DATE))      AS cohort_quarter,
        l.mql_date IS NOT NULL                                   AS is_mql,
        l.sql_date IS NOT NULL                                   AS is_sql,
        l.opportunity_id IS NOT NULL                             AS is_opp,
        COALESCE(o.is_won, FALSE)                                AS is_won,
        CASE WHEN o.is_won THEN o.amount_eur ELSE 0 END          AS won_amount
    FROM leads l
    LEFT JOIN opportunities o USING (opportunity_id)
)
SELECT
    cohort_quarter,
    channel,
    COUNT(*)                                             AS leads,
    SUM(is_mql::INT)                                     AS mqls,
    SUM(is_sql::INT)                                     AS sqls,
    SUM(is_opp::INT)                                     AS opps,
    SUM(is_won::INT)                                     AS won,
    ROUND(SUM(is_mql::INT) / COUNT(*), 3)                AS lead_to_mql,
    ROUND(SUM(is_sql::INT) / NULLIF(SUM(is_mql::INT), 0), 3) AS mql_to_sql,
    ROUND(SUM(is_won::INT) / NULLIF(SUM(is_opp::INT), 0), 3) AS opp_win_rate,
    ROUND(SUM(won_amount))                               AS won_eur
FROM lead_cohort
GROUP BY ALL
ORDER BY cohort_quarter, channel;
