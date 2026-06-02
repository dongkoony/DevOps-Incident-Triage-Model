# Database Connection Saturation Wiki

## Wiki Metadata

- Wiki ID: `wiki-database-connection-saturation`
- Source type: `runbook`
- Domain label: `database_state`
- Service: `database-platform`
- Severity: `medium`
- Owner: `data-platform-team`
- Last reviewed: `2026-06-02`
- Confidence level: `preview`

## Scope

Portfolio example for database connection pool saturation, lock contention, and query
timeout incidents. This is not a real production incident record.

## Common Symptoms

- Application requests time out while database CPU is not fully saturated.
- Connection pool usage reaches the configured maximum.
- Writes or migrations wait on long-running locks.

## First Checks

- Check active connections and pool usage by application instance.
- Inspect long-running queries and lock waits.
- Compare the incident with recent deploys, migrations, or traffic spikes.

## Useful Commands

```bash
psql -c "select state, count(*) from pg_stat_activity group by state;"
psql -c "select pid, wait_event_type, wait_event, query from pg_stat_activity where wait_event is not null;"
```

## Escalation Notes

Escalate to the database owner before terminating sessions, changing pool sizes, or
rolling back migrations in production.
