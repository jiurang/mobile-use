# 备份：移除「截图」相关的提示词与工具

**日期**：2026-10-27
**原因**：把 Executor 面向 LLM 的截图能力**整条摘掉** —— 提示词里的截图契约、截图工具、图通道、相关状态字段全部清干净。
**性质**：**可回滚**。原件在这里，按下面的命令复制回去即可完全恢复。

## 这次一共动了 15 个文件

| 备份文件 | 对应仓库路径 | 改了什么 |
|---|---|---|
| `minitap/mobile_use/tools/mobile/take_screenshot.py` | 同名 | **整个文件删除**（工具实现本体） |
| `minitap/mobile_use/tools/index.py` | 同名 | 删掉 `take_screenshot_wrapper` 的 import + `EXECUTOR_WRAPPERS_TOOLS` 里的注册行；`wake_screen` 那行注释改写（不再说「息屏时截图全黑」） |
| `minitap/mobile_use/agents/executor/executor.md` | 同名 | 删掉工具表里 `take_screenshot` 那行；删掉「A picture is a supplement…」整段；`no_perception` 那句去掉「You can still call take_screenshot…」 |
| `minitap/mobile_use/agents/executor/executor-zh.md` | 同名 | 同上（中文版，三处逐条对应） |
| `minitap/mobile_use/agents/executor/prompt.py` | 同名 | `build_executor_messages` 去掉 `screenshot_b64` 参数与末尾那条「图片感知」消息；模块文档更新；去掉 `build_screenshot_message_content` 的 import |
| `minitap/mobile_use/agents/executor/executor.py` | 同名 | 不再传 `screenshot_b64=state.executor_screenshot` |
| `minitap/mobile_use/agents/cortex/cortex.py` | 同名 | 删掉「换子目标时清空截图」那行 + `EXECUTOR_SCREENSHOT_KEY` 的 import |
| `minitap/mobile_use/utils/message_content.py` | 同名 | 删掉 `SCREENSHOT_CHANNEL_LABEL_EN` / `_ZH`、`screenshot_channel_label`、`build_screenshot_blocks`、`build_screenshot_message_content` |
| `minitap/mobile_use/constants.py` | 同名 | 删掉 `EXECUTOR_SCREENSHOT_KEY` |
| `minitap/mobile_use/graph/state.py` | 同名 | 删掉 `executor_screenshot` 字段 |
| `minitap/mobile_use/context.py` | 同名 | `PerceptionConfig` 文档更新（图片通道整条没了） |
| `minitap/mobile_use/tools/mobile/wake_screen.py` | 同名 | 模块文档第 1 条改写：「截图是全黑的」→「画面全黑，OCR 认不出字」 |
| `tests/mobile_use/test_take_screenshot_tool.py` | 同名 | **整个文件删除**（工具单测） |
| `tests/test_prompt_templates.py` | 同名 | `test_executor_prompt_has_no_trace_of_the_picture_channel` 改成断言「一个截图痕迹都不许有」（中英两份都查） |
| `tests/mobile_use/test_executor_autonomy.py` | 同名 | T18 系列 6 条 → 3 条新用例；T13b 重写；去掉不再用的 `PromptLanguage` import 与 fixture 里过期的 `latest_screenshot` 键 |

删除的两个文件在**回收站**里（`mavis-trash`），也一并备份在这里 —— 复制回去即可。

## 怎么恢复

```powershell
$root = 'E:\MyCoding\mobile\mobile-use'
$bk   = Join-Path $root 'backup\20261027-remove-screenshot-tool'

# 1) 恢复被删的两个文件
New-Item -ItemType Directory -Force -Path (Join-Path $root 'minitap\mobile_use\tools\mobile') | Out-Null
Copy-Item (Join-Path $bk 'minitap\mobile_use\tools\mobile\take_screenshot.py') `
          (Join-Path $root 'minitap\mobile_use\tools\mobile\take_screenshot.py') -Force
New-Item -ItemType Directory -Force -Path (Join-Path $root 'tests\mobile_use') | Out-Null
Copy-Item (Join-Path $bk 'tests\mobile_use\test_take_screenshot_tool.py') `
          (Join-Path $root 'tests\mobile_use\test_take_screenshot_tool.py') -Force

# 2) 恢复其余 13 个文件（路径与仓库一致，逐个覆盖）
#    小工具：把 $bk 下的内容整体拷回 $root
robocopy $bk $root /E /XF README.md /NFL /NDL /NJH /NJS /NP
```

恢复后 Executor 的工具数从 **13 回到 14**（开录屏 15 → 16）。

## ⚠️ 连带后果（移除期间的真实影响）

1. **模型少了一条感知通道。**
   元素表和 OCR 都是**文字**通道 —— 图标、插图、颜色、布局、「这个按钮到底长什么样」
   这类问题它们给不了答案。以前 `take_screenshot` 能补这一块，现在补不了了。
2. **两条通道都关掉时，模型彻底看不见屏幕。**
   `perception.ui_hierarchy=false` + `perception.ocr=false` ⇒ 以前还能靠截图救场，
   现在只能如实说「没有屏幕描述就无法行动」。
3. **模型凭先验喊 `take_screenshot` 会被派发层拒掉**（未知 action），白白多烧一轮。
4. **好处**：每次拍图约 1,200 input token（真机实测）不再产生；prompt 末尾少一条消息。

## 没动的东西（故意留着）

- **OCR 通道本身** —— 截图仍用于 OCR（`agents/contextor/contextor.py` 的
  `on_screenshot` 回调），只是 OCR 结果是**文字**。截图是取数手段，不是给模型看的。
- **坐标校验用的截图** —— `tools/bounds_validation.py::_recognize_ocr` 仍现截一张图跑 OCR。
- `minitap/mobile_use/utils/image_compress.py` —— 压缩函数还在（controllers / recorder 用）。
- `minitap/mobile_use/controllers/unified_controller.py::take_screenshot()` ——
  设备取图的**控制器 API**，不是给模型调的工具。
- `minitap/mobile_use/sdk/agent.py::get_screenshot()` —— 对外 SDK API（Manager 侧用）。
- `minitap/mobile_use/utils/message_content.py` 的 `content_to_text` /
  `has_image_block` / `with_text_replaced` —— **通用多模态处理**，保留。
  它们防的是「任何带图的消息」，哪天某个工具又塞了图片块，这些坑不该再踩一遍。
- `minitap/mobile_use/tools/mobile/wake_screen.py` —— **唤醒工具保留**。
  息屏时两条文字通道都读不到东西，「先唤醒」比以前更该保住。
- `start_video_recording` / `stop_video_recording` —— 录屏不是截图通道。
- `minitap/mobile_use/agents/executor/types.py` —— 输出 schema 早就不含
  `need_screenshot`，这次不用动。

## 恢复后要跑的验证

```powershell
cd E:\MyCoding\mobile\mobile-use
.\.venv\Scripts\python.exe -m pytest tests -q
```

⚠️ 注意：本次开工前，仓库里的 **`scripts/` 整个目录已被删除（未提交，108 个文件）**，
`scripts/check_prompt_templates.py` 不在了 ⇒ `tests/test_prompt_templates.py` 里有 5 条
测试会因 `FileNotFoundError` 直接红（**与本次改动无关**）。
模板渲染校验可以用从 git 取回的那份脚本代替：

```powershell
git show HEAD:scripts/check_prompt_templates.py > "$env:TEMP\check_prompt_templates_probe.py"
# 再把脚本里的 ROOT 改成绝对路径，然后：
.\.venv\Scripts\python.exe "$env:TEMP\check_prompt_templates_probe.py"
```

移除状态下这份脚本跑出来是 **48/48 全部通过**。
