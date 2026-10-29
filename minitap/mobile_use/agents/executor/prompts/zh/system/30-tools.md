## 工具

| 工具 | 参数 | 说明 |
|------|-----------|-------|
{{ executor_tools_table }}

- 位置感知

原点 `(0,0)` 在屏幕左上角：x 向右增大，y 向下增大，右下角是 `(xmax,ymax)`。元素的 `bounds=(x,y,W,H)` 中，`(x,y)` 是左上角，`W×H` 是宽高；元素中心是 `(x+W/2, y+H/2)`。

`swipe_percentages` 的 0% 在左/上边缘，100% 在右/下边缘。

- 滑动

```json
[{"action": "swipe_percentages", "start_x_percent": 50, "start_y_percent": 80, "end_x_percent": 50, "end_y_percent": 20}]
```

- 点击

```json
{"actions": [{"action": "tap", "target": {"index": 7}}], "agent_thought": "点击搜索框让它获得焦点。"}
```
