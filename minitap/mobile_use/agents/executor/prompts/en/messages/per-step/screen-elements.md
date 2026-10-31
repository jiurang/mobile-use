# Perception

Screen elements, one per line. UI rows and OCR rows share one numbering; `[i]` is 1-based,
copy it verbatim into the `index` field of `target`.

- UI rows (accessibility tree): [i] "text" | id=<id tail> | Class | (x,y,WxH) | flags
  flags: C=clickable L=long-press F=focusable K=checkbox S=selected R=scrollable
  id= is the tail of the resource-id (package omitted), shown only when the text is
  missing or repeats.
- OCR text rows (read from the screenshot): [i] "text" (x,y,WxH) confidence
  confidence is 0.00-1.00; verify any row below 0.6 before using it.

Screen = notch status bar + content area:
- Notch status bar: the thin strip at the top; it carries only app notifications and
  the clock. It is system chrome, not app content — tap it only when the task targets
  the notification shade itself.
- Content area: changes with the app and page, and may be empty. Some apps expose no
  UI at all (their controls are unreadable); then there are no UI rows and the table
  may hold nothing but a few status-bar text rows.
{{ rows }}
