## Previous attempt

Your previous attempt on this goal ended as **{{ outcome }}**.
{% if summary %}Summary: {{ summary }}
{% endif %}{% if reason %}Why it ended: {{ reason }}
{% endif %}{% if failed_targets %}Targets that already failed — do not reuse them, pick a different element: {{ failed_targets }}
{% endif %}
This attempt starts with a fresh history; only this summary is carried over. Do not repeat a route that already failed — take a different one.
