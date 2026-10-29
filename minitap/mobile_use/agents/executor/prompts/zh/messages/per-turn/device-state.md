{%- set lines = [] -%}
{%- if device_date %}{% set lines = lines + ["设备日期：" ~ device_date] %}{% endif -%}
{%- if focused_app_info %}{% set lines = lines + ["前台应用：" ~ focused_app_info] %}{% endif -%}
{%- if screen_off %}{% set lines = lines + ["屏幕是关的（设备已息屏）。元素表里只剩 systemui 残留，没有可操作的东西。先调用 wake_screen，下一轮再重新读屏。"] %}{% endif -%}
{%- if no_perception %}{% set lines = lines + ["没有任何感知通道可用：UI 元素表、OCR 文本和视觉扫描都关着。你完全看不到屏幕描述 —— 不要凭空想象屏幕上有什么。在 agent_thought 里如实说明：没有屏幕描述就无法行动。"] %}{% endif -%}
{%- if lines -%}
## 设备状态

{{ lines | join("\n") }}
{%- endif -%}
