# Database State Runbook

## Scope

Placeholder runbook for database connectivity, lock, replication, migration, and storage-state incidents. This is a portfolio example, not a record of real production incidents.

## Common Symptoms

- Application writes time out or fail with connection errors.
- Schema migrations hang or create lock contention.
- Replication lag increases.
- Disk, IOPS, or connection pool limits are reached.

## First Checks

- Check active connections, lock waits, and slow queries.
- Identify recent migrations, failovers, or configuration changes.
- Review replication health and storage utilization.
- Confirm whether the issue affects reads, writes, or both.

## Useful Commands

```sql
SELECT pid, state, wait_event_type, wait_event, query FROM pg_stat_activity;
SELECT * FROM pg_locks WHERE NOT granted;
SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;
```

```bash
psql "$DATABASE_URL" -c "SELECT count(*) FROM pg_stat_activity;"
```

## Escalation Notes

Escalate to database ownership when lock contention affects production writes, replication lag threatens recovery objectives, or storage limits are close to exhaustion.

## RAG Metadata

- Domain label: `database_state`
- Retrieval keywords: `database`, `postgres`, `lock`, `migration`, `replication`, `connection pool`, `storage`
- Suggested citations: lock checks, replication health, active connection review

