## Perception

Screen elements, one per line: [i] "text" | id=<id tail> | Class | (x,y,WxH) | flags
flags: C=clickable L=long-press F=focusable K=checkbox S=selected R=scrollable
`[i]` is 1-based: copy it verbatim into the `index` field of `target`.
One numbering for all rows. Rows without id= are OCR text. id= is the tail of the
resource-id (package omitted), shown only when the text is missing or repeats.
OCR rows also end with their OCR confidence (0.00-1.00); verify any confidence
below 0.6 before using that row.
{{ rows }}
