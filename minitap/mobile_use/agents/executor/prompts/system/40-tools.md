## Tools

| Tool | Arguments | Notes |
|------|-----------|-------|
{{ executor_tools_table }}

- Coordinates

Origin `(0,0)` is the top-left corner: x grows right, y grows down; the bottom-right corner is `(xmax,ymax)`. For an element's `bounds=(x,y,W,H)`, `(x,y)` is its top-left corner, `W×H` is its width×height, and the element center is `(x+W/2, y+H/2)`.

`swipe_percentages`: 0% is the left/top edge, 100% is the right/bottom edge.

- Swipe

```json
[{"action": "swipe_percentages", "start_x_percent": 50, "start_y_percent": 80, "end_x_percent": 50, "end_y_percent": 20}]
```

- Tap

```json
{"actions": [{"action": "tap", "target": {"index": 7}}], "agent_thought": "Tapping the search box to focus it."}
```
