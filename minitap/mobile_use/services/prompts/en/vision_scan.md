# Visual subject inventory

Inventory the visual subjects in this phone screenshot so a blind agent can pick one by number.

Subjects: icons, images, photos, illustrations, avatars, video covers, canvas / custom-drawn controls, unlabeled buttons, switches, sliders, badges.
List at the smallest meaningful unit: a photo, avatar and like button inside one card are three subjects, not one card. A whole tappable card may be one subject only when nothing smaller carries the action.
Skip: text, status / navigation bars, keyboard, backgrounds, decorative lines, shadows, containers, rows, lists, the whole screen. List each object once.
Actionable controls first, then content; at most {{ max_subjects }} entries. Never invent a subject that is not visible; return an empty list if there are none.

Return format:

| Field | Type | Description |
| --- | --- | --- |
| `subjects` | array | every subject on this screen; empty array if none |
| `subjects[].description` | string | 3-12 words, a concrete noun plus one detail that tells it apart; no vague "image" / "button" |
| `subjects[].x1` `y1` `x2` `y2` | int | integers normalized to 0-1000 over the full screenshot ((0,0) top-left, (1000,1000) bottom-right); tight to the visible pixels (an icon is about 30-120 units wide); x1 < x2 and y1 < y2 |

Example:

```json
{
  "subjects": [
    {"description": "gear icon", "x1": 900, "y1": 80, "x2": 950, "y2": 130},
    {"description": "red-roofed house photo", "x1": 100, "y1": 400, "x2": 900, "y2": 800}
  ]
}
```
