## Tools

Available tools: {{ executor_tools_list }}

| Tool | Arguments | Notes |
|------|-----------|-------|
| `launch_app` | `app_name` | Open an app |
| `open_link` | `url` | Handles deep links correctly |
| `tap` | `target` | Tap |
| `long_press_on` | `target`, optional `duration_ms` | Long press |
| `focus_and_input_text` | `target`, `text` | Focus then type; verify if feedback is empty |
| `focus_and_clear_text` | `target` | If it fails: long press -> select all -> `erase_one_char` |
| `swipe_percentages` | `start_x_percent`, `start_y_percent`, `end_x_percent`, `end_y_percent` (0-100), optional `duration` | **Prefer this**; correct across screen sizes |
| `swipe_coordinates` | `start_x`, `start_y`, `end_x`, `end_y`, optional `duration` | Only for controls needing precision, e.g. sliders |
| `stop_app` | optional `package_name` | Close the app |
| `press_key` | `key` | `Enter` confirm/search, `Home` go home, `Back` go back |
| `wait_for_delay` | `time_in_ms` | Wait that many milliseconds |
| `back`, `erase_one_char` | *(none)* | — |
| `wake_screen` | *(none)* | Turn the screen on |
| `start_video_recording` | *(none)* | Starts a background screen recording: max 3 min (Android) / 15 min (iOS), no audio; runs until `stop_video_recording` |
| `stop_video_recording` | optional `prompt` | Stops the recording, then analyzes the video; `prompt` says what to extract (e.g. "what happens after each 10s") |

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
