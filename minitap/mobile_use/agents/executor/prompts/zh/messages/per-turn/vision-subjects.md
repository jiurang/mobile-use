## 感知

当前屏幕的视觉主体（图标 / 图片 / 画布 / 无文字控件），每行一条：[i] "描述" (x,y,WxH)

这张表来自你上一次 `scan_visual_subjects` 调用，只对当前这一屏有效。要操作其中
一项，`target` 写 `{"vision_index": i}` —— 不要自己给 bounds。动作之前屏幕变了，
就重新扫描。
{{ rows }}
