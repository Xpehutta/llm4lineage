-- Golden fixtures for PL/pgSQL lineage extraction.
-- Three routines covering the cases called out in CURSOR_TASKS.md Phase A3:
-- a temp table, IF branches, and dynamic EXECUTE.

-- 1) Temp table staged between statements: lineage must chain
--    sales.orders -> tmp_daily_orders -> analytics.daily_summary.
CREATE OR REPLACE FUNCTION analytics.build_daily_summary(p_day date)
RETURNS void AS $$
DECLARE
    v_rows integer := 0;
BEGIN
    CREATE TEMP TABLE tmp_daily_orders AS
    SELECT o.customer_id,
           o.amount,
           o.order_date
    FROM sales.orders o
    WHERE o.order_date = p_day;

    INSERT INTO analytics.daily_summary (customer_id, total_amount, day)
    SELECT t.customer_id,
           SUM(t.amount) AS total_amount,
           t.order_date
    FROM tmp_daily_orders t
    GROUP BY t.customer_id, t.order_date;

    SELECT count(*) INTO v_rows FROM analytics.daily_summary;
    RAISE NOTICE 'inserted % rows', v_rows;
END;
$$ LANGUAGE plpgsql;


-- 2) Branching: every arm of the IF/ELSIF/ELSE must appear in the graph.
CREATE OR REPLACE FUNCTION analytics.route_customer(p_mode text)
RETURNS void AS $$
BEGIN
    IF p_mode = 'new' THEN
        INSERT INTO analytics.new_customers (id, name)
        SELECT c.id, c.name FROM crm.customers c WHERE c.created_at > now();
    ELSIF p_mode = 'churn' THEN
        INSERT INTO analytics.churned_customers (id, name)
        SELECT c.id, c.name FROM crm.customers c WHERE c.closed_at IS NOT NULL;
    ELSE
        UPDATE analytics.customer_state s
        SET status = c.status
        FROM crm.customers c
        WHERE s.id = c.id;
    END IF;

    FOR rec IN SELECT id FROM crm.blacklist LOOP
        DELETE FROM analytics.new_customers WHERE id = rec.id;
    END LOOP;
END;
$$ LANGUAGE plpgsql;


-- 3) Dynamic SQL: the target is only known at runtime and must be reported
--    as unresolved, while the statically visible source is still recovered.
CREATE OR REPLACE FUNCTION staging.load_partition(p_table text)
RETURNS void AS $$
DECLARE
    v_sql text;
BEGIN
    EXECUTE format('INSERT INTO %I SELECT id, payload FROM staging.raw_events', p_table);

    v_sql := 'SELECT 1';
    EXECUTE v_sql;

    EXECUTE 'INSERT INTO staging.audit (note) SELECT ''loaded'' FROM staging.raw_events';
END;
$$ LANGUAGE plpgsql;
