## 工具

可用工具：{{ executor_tools_list }}

| 工具 | 参数 | 说明 |
|------|-----------|-------|
| `launch_app` | `app_name` | 打开应用 |
| `open_link` | `url` | 能正确处理深链 |
| `tap` | `target` | 点击 |
| `long_press_on` | `target`，可选 `duration_ms` | 长按 |
| `focus_and_input_text` | `target`，`text` | 先聚焦再输入，如果反馈为空要核实 |
| `focus_and_clear_text` | `target` | 如果失败：长按 -> 全选 -> `erase_one_char` |
| `swipe_percentages` | `start_x_percent`、`start_y_percent`、`end_x_percent`、`end_y_percent`（0-100），可选 `duration` | **优先用它**；跨屏幕尺寸都正确 |
| `swipe_coordinates` | `start_x`、`start_y`、`end_x`、`end_y`，可选 `duration` | 只用于需要精确控制的位置，比如滑块 |
| `stop_app` | 可选 `package_name` | 关闭应用 |
| `press_key` | `key` | `Enter` 确认/搜索、`Home` 回到桌面、`Back` 返回 |
| `wait_for_delay` | `time_in_ms` | 等待整数毫秒 |
| `back`、`erase_one_char` | *(无)* | — |
| `wake_screen` | *(无)* | 点亮屏幕 |
| `start_video_recording` | *(无)* | 开始后台录屏：最长 3 分钟（Android）/ 15 分钟（iOS），不录音频；一直录到调用 `stop_video_recording` |
| `stop_video_recording` | 可选 `prompt` | 停止录制，然后分析视频；`prompt` 说明要提取什么（例如「每 10 秒发生了什么」） |

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
