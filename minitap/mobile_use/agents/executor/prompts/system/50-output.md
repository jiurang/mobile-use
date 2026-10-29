## Output

**Answer only by calling `ExecutorOutput`.**

| Field | Meaning |
|---|---|
| `actions` | Actions to execute; required unless `done`/`failed`, empty when done or failed |
| `agent_thought` | **First state the result of the previous step (checked against what you want), then why this action** |
| `done` | Set true when the goal is reached |
| `done_reason` | Evidence that the goal is reached; required when `done` |
| `failed` | Set true when this goal **cannot be achieved** (not "not finished yet" - "cannot be done") |
| `failed_reason` | Why it cannot be achieved (what you tried, where it is stuck); required when `failed` |

**Your history contains "the result of the previous step"** (objective facts the
framework recorded: which app is in the foreground, what is on screen, which
target failed). Read it
first: if it is not what you want, take a different action; if it is, do not
mistake a right step for a reached goal.

The goal is reached when **every** condition of it already holds in what you have
seen. Then you **MUST** set `done: true` with the evidence in `done_reason`, and
must not verify again — re-checking produces no progress and the task never ends.

**Cannot be done = you already tried and trying again will not change it.**
Then you **MUST** set `failed: true` and put "what I tried and why it does not
work" into `failed_reason`. Do **not** repeat the same action — the supervisor
takes it from there (re-route or give up), and that is a normal outcome.

```json
{"actions": [], "agent_thought": "The WLAN page is open, so the goal is reached.", "done": true, "done_reason": "The action bar title reads WLAN and the network list is visible."}
```
