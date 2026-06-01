-- ============================================================
-- Customer Call Pattern Analysis — MySQL Schema & Queries
-- ============================================================

-- ──────────────────────────────────────────────────────────
-- 1. CREATE DATABASE
-- ──────────────────────────────────────────────────────────
CREATE DATABASE IF NOT EXISTS call_analytics;
USE call_analytics;

-- ──────────────────────────────────────────────────────────
-- 2. CREATE TABLE
-- ──────────────────────────────────────────────────────────
DROP TABLE IF EXISTS call_records;

CREATE TABLE call_records (
    call_id            VARCHAR(20)  NOT NULL PRIMARY KEY,
    customer_id        VARCHAR(15)  NOT NULL,
    call_date          DATE         NOT NULL,
    call_time          TIME         NOT NULL,
    call_duration      INT          COMMENT 'Duration in seconds',
    call_status        ENUM('Completed','Failed','Dropped','Transferred','Voicemail') NOT NULL,
    sip_response_code  SMALLINT     NOT NULL,
    city               VARCHAR(50)  NOT NULL,
    state              VARCHAR(50)  NOT NULL,
    agent_id           VARCHAR(15)  DEFAULT 'NO_AGENT',
    voicebot_flow      VARCHAR(50),
    call_type          ENUM('Inbound','Outbound','Callback') NOT NULL,
    language           VARCHAR(20),
    call_disposition   VARCHAR(50),
    created_at         TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_customer    (customer_id),
    INDEX idx_date        (call_date),
    INDEX idx_status      (call_status),
    INDEX idx_city        (city),
    INDEX idx_agent       (agent_id),
    INDEX idx_date_status (call_date, call_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Voicebot call records';


-- ──────────────────────────────────────────────────────────
-- 3. LOAD DATA FROM CSV  (run after Python generates CSV)
-- ──────────────────────────────────────────────────────────
LOAD DATA INFILE '/var/lib/mysql-files/call_records.csv'
INTO TABLE call_records
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(call_id, customer_id, call_date, call_time, call_duration, call_status,
 sip_response_code, city, state, agent_id, voicebot_flow, call_type,
 language, call_disposition);


-- ──────────────────────────────────────────────────────────
-- 4. ANALYTICAL QUERIES
-- ──────────────────────────────────────────────────────────

-- ── Q1: Peak Calling Hours ──────────────────────────────
SELECT
    HOUR(call_time)                              AS hour_of_day,
    COUNT(*)                                     AS total_calls,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_day,
    SUM(CASE WHEN call_status = 'Completed' THEN 1 ELSE 0 END) AS completed,
    SUM(CASE WHEN call_status = 'Failed'    THEN 1 ELSE 0 END) AS failed,
    ROUND(AVG(call_duration), 1)                 AS avg_duration_sec
FROM call_records
GROUP BY HOUR(call_time)
ORDER BY total_calls DESC;


-- ── Q2: Daily Call Volume (Last 90 Days) ────────────────
SELECT
    call_date,
    DAYNAME(call_date)                           AS day_name,
    COUNT(*)                                     AS total_calls,
    SUM(CASE WHEN call_status = 'Completed'  THEN 1 ELSE 0 END) AS completed,
    SUM(CASE WHEN call_status = 'Failed'     THEN 1 ELSE 0 END) AS failed,
    ROUND(AVG(call_duration), 1)                 AS avg_duration
FROM call_records
WHERE call_date >= CURDATE() - INTERVAL 90 DAY
GROUP BY call_date
ORDER BY call_date;


-- ── Q3: Average Call Duration by Status & Flow ──────────
SELECT
    call_status,
    voicebot_flow,
    COUNT(*)                                     AS total_calls,
    ROUND(AVG(call_duration), 1)                 AS avg_sec,
    ROUND(MIN(call_duration), 1)                 AS min_sec,
    ROUND(MAX(call_duration), 1)                 AS max_sec,
    ROUND(STDDEV(call_duration), 1)              AS std_dev
FROM call_records
WHERE call_duration IS NOT NULL
GROUP BY call_status, voicebot_flow
ORDER BY call_status, avg_sec DESC;


-- ── Q4: Repeat Callers ──────────────────────────────────
SELECT
    customer_id,
    COUNT(*)                                     AS total_calls,
    COUNT(DISTINCT call_date)                    AS unique_days,
    MIN(call_date)                               AS first_call,
    MAX(call_date)                               AS last_call,
    DATEDIFF(MAX(call_date), MIN(call_date))     AS customer_lifespan_days,
    ROUND(AVG(call_duration), 1)                 AS avg_duration,
    SUM(CASE WHEN call_status = 'Completed' THEN 1 ELSE 0 END) AS resolved_calls,
    COUNT(DISTINCT voicebot_flow)                AS flows_used
FROM call_records
GROUP BY customer_id
HAVING total_calls > 3
ORDER BY total_calls DESC
LIMIT 50;


-- ── Q5: Failed Calls Deep Dive ──────────────────────────
SELECT
    sip_response_code,
    call_disposition,
    call_type,
    city,
    COUNT(*)                                     AS failed_calls,
    ROUND(COUNT(*) * 100.0 /
          (SELECT COUNT(*) FROM call_records WHERE call_status = 'Failed'), 2) AS pct
FROM call_records
WHERE call_status = 'Failed'
GROUP BY sip_response_code, call_disposition, call_type, city
ORDER BY failed_calls DESC
LIMIT 20;


-- ── Q6: City-Wise Analysis ──────────────────────────────
SELECT
    city,
    state,
    COUNT(*)                                     AS total_calls,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS share_pct,
    SUM(CASE WHEN call_status = 'Completed'  THEN 1 ELSE 0 END) AS completed,
    SUM(CASE WHEN call_status = 'Failed'     THEN 1 ELSE 0 END) AS failed,
    ROUND(SUM(CASE WHEN call_status = 'Failed' THEN 1.0 ELSE 0 END)
          / COUNT(*) * 100, 2)                   AS failure_rate_pct,
    ROUND(AVG(call_duration), 1)                 AS avg_duration_sec,
    COUNT(DISTINCT customer_id)                  AS unique_customers
FROM call_records
GROUP BY city, state
ORDER BY total_calls DESC;


-- ── Q7: Monthly Trends ──────────────────────────────────
SELECT
    DATE_FORMAT(call_date, '%Y-%m')              AS month,
    COUNT(*)                                     AS total_calls,
    SUM(CASE WHEN call_status = 'Completed'   THEN 1 ELSE 0 END) AS completed,
    SUM(CASE WHEN call_status = 'Failed'      THEN 1 ELSE 0 END) AS failed,
    SUM(CASE WHEN call_status = 'Dropped'     THEN 1 ELSE 0 END) AS dropped,
    SUM(CASE WHEN call_status = 'Transferred' THEN 1 ELSE 0 END) AS transferred,
    ROUND(SUM(CASE WHEN call_status = 'Completed' THEN 1.0 ELSE 0 END)
          / COUNT(*) * 100, 2)                   AS success_rate_pct,
    ROUND(AVG(call_duration), 1)                 AS avg_duration_sec,
    COUNT(DISTINCT customer_id)                  AS unique_callers,
    COUNT(DISTINCT agent_id)                     AS agents_active
FROM call_records
GROUP BY DATE_FORMAT(call_date, '%Y-%m')
ORDER BY month;


-- ── Q8: Voicebot Flow Performance ───────────────────────
SELECT
    voicebot_flow,
    COUNT(*)                                     AS total_calls,
    ROUND(AVG(call_duration), 1)                 AS avg_sec,
    SUM(CASE WHEN call_status = 'Completed' THEN 1 ELSE 0 END) AS completed,
    ROUND(SUM(CASE WHEN call_status = 'Completed' THEN 1.0 ELSE 0 END)
          / COUNT(*) * 100, 2)                   AS completion_rate,
    SUM(CASE WHEN call_disposition = 'Self_Served' THEN 1 ELSE 0 END) AS self_served,
    SUM(CASE WHEN call_disposition = 'Escalated_to_Agent' THEN 1 ELSE 0 END) AS escalated
FROM call_records
WHERE voicebot_flow IS NOT NULL
GROUP BY voicebot_flow
ORDER BY total_calls DESC;


-- ── Q9: Agent Workload & Performance ────────────────────
SELECT
    agent_id,
    COUNT(*)                                     AS total_handled,
    ROUND(AVG(call_duration), 1)                 AS avg_sec,
    ROUND(SUM(CASE WHEN call_status = 'Completed' THEN 1.0 ELSE 0 END)
          / COUNT(*) * 100, 2)                   AS success_rate,
    COUNT(DISTINCT customer_id)                  AS unique_customers,
    COUNT(DISTINCT call_date)                    AS active_days
FROM call_records
WHERE agent_id != 'NO_AGENT'
GROUP BY agent_id
ORDER BY total_handled DESC
LIMIT 20;


-- ── Q10: Language-wise Breakdown ────────────────────────
SELECT
    language,
    COUNT(*)                                     AS total_calls,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS share_pct,
    ROUND(AVG(call_duration), 1)                 AS avg_duration,
    ROUND(SUM(CASE WHEN call_status = 'Completed' THEN 1.0 ELSE 0 END)
          / COUNT(*) * 100, 2)                   AS success_rate
FROM call_records
WHERE language IS NOT NULL
GROUP BY language
ORDER BY total_calls DESC;


-- ── Q11: Day-of-Week Patterns ───────────────────────────
SELECT
    DAYOFWEEK(call_date)                         AS dow_num,
    DAYNAME(call_date)                           AS day_name,
    COUNT(*)                                     AS total_calls,
    ROUND(AVG(call_duration), 1)                 AS avg_duration,
    ROUND(SUM(CASE WHEN call_status = 'Completed' THEN 1.0 ELSE 0 END)
          / COUNT(*) * 100, 2)                   AS success_rate
FROM call_records
GROUP BY DAYOFWEEK(call_date), DAYNAME(call_date)
ORDER BY dow_num;


-- ── Q12: SIP Response Code Distribution ─────────────────
SELECT
    sip_response_code,
    CASE sip_response_code
        WHEN 200 THEN 'OK / Success'
        WHEN 302 THEN 'Redirect'
        WHEN 408 THEN 'Request Timeout'
        WHEN 480 THEN 'Temporarily Unavailable'
        WHEN 486 THEN 'Busy Here'
        WHEN 487 THEN 'Request Terminated'
        WHEN 500 THEN 'Server Internal Error'
        WHEN 503 THEN 'Service Unavailable'
        ELSE 'Other'
    END                                          AS description,
    COUNT(*)                                     AS occurrences,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM call_records
GROUP BY sip_response_code
ORDER BY occurrences DESC;
