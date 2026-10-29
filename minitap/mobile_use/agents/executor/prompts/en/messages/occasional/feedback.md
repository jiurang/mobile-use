## Feedback

{%- if block == "dispatch_rejection" %}

Your previous action was NOT executed: {{ reason }}. Decide again — read the Tools table and the current screen description.
{%- elif block == "retry" %}

Your last answer could not be used: {{ reason }}. You must call the ExecutorOutput tool with the fields from the Output section. Do not write the output as plain text — a plain-text answer is discarded.
{%- elif block == "extra_actions" %}

Only the first of your {{ count }} actions was executed; the rest were discarded and must be decided again next round.
{%- elif block == "stall_notice" %}

[no progress] {% if kind == "screen" %}You came back to the same page {{ repeats }} times in the last {{ window }} steps (screen fingerprint {{ fingerprint }}) — nothing on that page changed.{% else %}You repeated the same action {{ repeats }} times in the last {{ window }} steps ({{ fingerprint }}) — the result did not change.{% endif %}
Do not repeat the action you just took. Work out these two things:
1. Can this actually be done? If not, set `failed: true` (say what you tried and why it does not work) - that is a normal outcome, not a failure on your part.
2. If it can, take a **different** route than the one you just took (another entry point, another order, another page). Do not go round in circles.
{% if vision_enabled %}
If you are still circling the same target or the same area, call `scan_visual_subjects` first to look at the current screen before choosing another route. Switching to another index is not a new route.{% endif %}
{%- endif %}
