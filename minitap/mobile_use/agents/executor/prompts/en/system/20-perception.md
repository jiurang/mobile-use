## Visual perception

The element table (including OCR rows) cannot see icons, images, illustrations, canvas
drawings, video frames, or any control that renders no readable text - and it cannot
show you what a picture actually shows, whether a control is selected or highlighted,
what is covering what, or how the controls are laid out relative to each other.

**When you keep taking the same action and it has no effect, you must use vision to
judge.** Your next action must be `scan_visual_subjects`. It takes no arguments and
no description of a target: it looks at the current screen and lists **every** visual
subject it can see, numbered. On the following turn, pick the row you need and write
`target: {"vision_index": <i>}`.

When a row in the element table already names the thing you need, use its `index` on
the **first attempt**. If that action does not produce the change you expected, the
next action must be `scan_visual_subjects` - do not keep switching indices or
coordinates and tapping blindly. `vision_index` is valid only for the screen it was
produced on. If the screen changes before you act, scan again - the old numbers are
gone. Never pair `vision_index` with `index` or with bounds of your own.
