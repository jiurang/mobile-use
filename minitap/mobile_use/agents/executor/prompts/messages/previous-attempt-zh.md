## 上一轮尝试

上一轮在这个目标上的结果是 **{{ outcome }}**。
{% if summary %}摘要：{{ summary }}
{% endif %}{% if reason %}结束原因：{{ reason }}
{% endif %}{% if failed_targets %}已经失败过的目标 —— 不要再用，换一个元素：{{ failed_targets }}
{% endif %}
本轮历史是从零开始的，只带过来这条摘要。不要重复上一轮已经失败过的路线，换一条走。
