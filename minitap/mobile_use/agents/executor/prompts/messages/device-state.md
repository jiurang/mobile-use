{%- set lines = [] -%}
{%- if device_date %}{% set lines = lines + ["Device date: " ~ device_date] %}{% endif -%}
{%- if focused_app_info %}{% set lines = lines + ["Focused app info: " ~ focused_app_info] %}{% endif -%}
{%- if screen_off %}{% set lines = lines + ["Screen is OFF (the device is asleep). The element list only holds systemui leftovers, so there is nothing to act on. Call wake_screen FIRST, then read the screen again on the next turn."] %}{% endif -%}
{%- if no_perception %}{% set lines = lines + ["NO perception channel is available: the UI element list, the OCR text and the visual scan are all OFF. You have no description of the screen at all - do not invent what is on it. Say plainly in agent_thought that you cannot act without a screen description."] %}{% endif -%}
{%- if lines -%}
## Device State

{{ lines | join("\n") }}
{%- endif -%}
