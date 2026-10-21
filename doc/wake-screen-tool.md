# 唤醒屏幕工具（wake_screen）

> 状态：**已实现，已在真机端到端验证通过**（2026-10-21，Redmi Turbo 4 / Android 16）
> 相关：`doc/cache-hit-rate-audit-20261020.md`（真机 A/B 时发现的「息屏 → 步数暴涨」现象）

---

## 1. 要解决的问题

真机跑长任务时，**手机会自己息屏**（默认 30 分钟超时）。屏幕一黑：

- 截图全黑；
- 层级树里只剩 systemui 的残留元素；
- agent 于是**看不到任何业务内容**，会在黑屏上反复尝试、原地打转。

2026-10-21 真机 A/B 实测的原始症状：

| | 步数 | 结果 |
|---|---|---|
| 屏幕常亮 | 30 步 | 正常完成 |
| 跑到一半息屏 | **51 步** | 还没跑完 |

**全程没有任何报错**，症状是「步数暴涨」，极易被误判成代码回归。

本功能的做法：**harness 读出屏幕灭没灭 → 告诉模型 → 模型调 `wake_screen` 点亮**。

---

## 2. 实测事实（这些是本功能的地基，全部来自真机）

### 2.1 熄屏时层级树**不是空的** —— 「树空 ⇒ 熄屏」这个判据被证伪

最初的设计是「层级树为空时才查电源状态」，理由是「屏幕亮着时层级树一定有东西，
正常路径一次额外 adb 调用都不花」。

**实测推翻了它**（`probe_out/probe_screen_off_elements.py`）：

| | 元素数 | 元素是什么 | 截图 |
|---|---|---|---|
| 亮屏 | 197 | 业务 app + systemui | 正常 |
| 熄屏 | **19** | **全是** `com.android.systemui` / `miui.systemui.plugin`（通知面板容器、状态栏、`keyguard_message_area_container`…） | **全黑** |

⇒ 那个判据**永远不会触发**，唤醒功能等于没接上，而且**不报任何错**。
必须改成**无条件**读电源状态（`dumpsys power` 的 `mWakefulness`）。

### 2.2 `uiautomator2` 冷连接会唤醒屏幕，热连接不会

这一条差点把整个验证带进沟里（`probe_out/probe_who_wakes_device.py`、
`probe_out/probe_capture_wakes_screen.py`）：

| 动作 | 熄屏设备是否被唤醒 |
|---|---|
| `adb shell wm size` / `dumpsys power` / `pm list packages` | 否 |
| `adb shell am start` | 否 |
| `adb shell uiautomator dump` | 否 |
| `am instrument` 重启 u2 server | 否 |
| **`UIAutomatorClient.get_screen_data()`（u2 冷连接）** | **是** |

⇒ 「启动前就把屏幕熄掉，然后跑 agent」这个测法**测不到 off 状态** ——
agent 启动阶段的 u2 冷连接就把屏幕点亮了，contextor 读到的永远是 `on`。
正确测法是**任务跑到一半熄屏**（对应真实的 30 分钟超时）。

### 2.3 `svc power stayon true` 会让设备自己醒过来

测试机上设过 `svc power stayon true`（长跑要锁常亮），而设备在充电
（`mStayOn=true` / `mStayOnWhilePluggedInSetting=15`），
于是 **`KEYCODE_SLEEP` 之后设备自己又亮了**。测「熄屏」必须先
`svc power stayon false`（真实用户手机默认就是 false）。

### 2.4 成本：`dumpsys power` ~100ms，可以被并发藏掉

| 方案 | 成本 | 结论 |
|---|---|---|
| 全量 `dumpsys power` | 中位 **98ms**（min 85 / max 241） | 基准 |
| `dumpsys power \| grep mWakefulness` | 中位 107ms | **不省**（成本在设备端生成，不在传输） |
| PNG 解码判「截图全黑」 | 亮屏 **78.8ms** / 熄屏 11.3ms | 比查电源状态还贵，且不权威 |
| 「树里没有非 systemui 元素」 | — | **失效**（熄屏时仍剩 1 个非 systemui 元素） |
| **和抓屏并发** | 串行 698ms → 并发 **589ms** | **省下 109ms/轮** |

⇒ 选「无条件读电源状态 + 和抓屏并发」。真机实测 8 轮并发，元素数稳定
（120/120/…），**没有连接池被抽干**。

---

## 3. 实现

### 3.1 读状态 + 点亮（控制器层）

| 文件 | 内容 |
|---|---|
| `controllers/android_controller.py` | `get_screen_power_state()` → `"on"` / `"off"` / `"unknown"`（解析 `dumpsys power` 的 `mWakefulness`）；`wake_up()` → `input keyevent 224`（KEYCODE_WAKEUP）+ `wm dismiss-keyguard` |
| `controllers/cloud_device_controller.py` | 同上（Android 系） |
| `controllers/ios_controller.py` | `get_screen_power_state()` → `"unknown"`；`wake_up()` → `False` + warn（**iOS 没有可用接口**） |
| `controllers/device_controller.py` | 协议：**不抽象**，给诚实默认值（unknown / False），不破坏既有测试替身 |
| `controllers/unified_controller.py` | 两个方法的透传 |

### 3.2 contextor：无条件 + 并发（`agents/contextor/contextor.py`）

```python
power_task = asyncio.create_task(device_controller.get_screen_power_state())
await asyncio.sleep(0)          # create_task 只是排进日程表，要显式让一次
try:
    device_data = await device_controller.get_screen_data(on_screenshot=_run_ocr_inline)
finally:
    screen_state = await power_task   # 抓屏炸了也要收回来，不留悬空 task
```

三个细节：

1. **无条件**读 —— 见 §2.1。
2. **并发** + 显式 `await asyncio.sleep(0)` —— `create_task` **不会**立刻开跑，
   不让一次控制权它就退化成串行 +100ms。
3. **旧控制器兼容**：用 `getattr(device_controller, "get_screen_power_state", None)`
   取方法，duck-typed 的自定义控制器没跟上新接口时**不炸**，`screen_state` 留 `None`。

### 3.3 告诉模型（`agents/cortex/cortex.py` + prompt）

`Device state` 块里，`screen_state == "off"` 时追加一行：

```
Screen is OFF (the device is asleep). The screenshot is black and the element list
only holds systemui leftovers, so there is nothing to act on. Call wake_screen
FIRST, then read the screen again on the next turn.
```

prompt（`cortex.md` / `cortex.zh.md`）两处：

- Perception 段：「**The device may be asleep.** … Call **wake_screen** …」
- Tools 表新增一行 `| wake_screen | *(none)* | Turn the screen on; use it only when the device state line says the screen is OFF |`

> Tools 表在 system prompt 里，属于**缓存前缀**，加一行不影响命中率。

### 3.4 工具（`tools/mobile/wake_screen.py`）

- 唤醒后**再读一次**状态确认：
  - `"on"` → 成功：「The screen is on now. Read the screen again before acting.」
  - `"off"` → **如实报失败**（有 PIN 锁屏划不掉，只能靠人解锁）
  - `"unknown"` → 如实报「状态读不出来」
- 注册进 `tools/index.py` 的 `EXECUTOR_WRAPPERS_TOOLS`（直连派发按名字查表 → 自动生效）。

### 3.5 State

`graph/state.py` 新增 `screen_state: str | None = None`（**必须有默认值** ——
`State` 是 pydantic model，缺默认值会让所有直接 `State(...)` 构造的测试炸
`Field required`，实测 15 个测试因此失败）。

---

## 4. 真机端到端验证

探针：`probe_out/probe_wake_screen_e2e.py`（第 2 轮 contextor 之前**物理熄屏**）

任务：`打开系统设置应用，告诉我当前电量百分比`（Redmi Turbo 4 / Android 16）

```
[probe] >>> 第 2 轮：物理熄屏（模拟任务跑到一半手机自动息屏）
[probe] >>> 熄屏后 mWakefulness = Asleep
[probe] >>> 第 2 轮 contextor 读到的 screen_state = 'off'
[probe] >>> prompt 里出现了 'Screen is OFF' 提示行
[probe] >>> 模型决策里出现了 wake_screen（全部动作: ['wake_screen']）
[probe] >>> wake_up() 被调用了（调用前 mWakefulness=Asleep）
[probe] >>> wake_up() 返回 True（调用后 mWakefulness=Awake）
[probe] >>> 第 3 轮 contextor 读到的 screen_state = 'on'
```

判定：

| 检查项 | 结果 |
|---|---|
| contextor 读出 `off` | ✅（第 2 轮） |
| prompt 出现 "Screen is OFF" | ✅ |
| 模型决策包含 `wake_screen` | ✅ |
| 屏幕真的从 Asleep 变 Awake | ✅ |
| 任务完成 | ✅ `Automation is success`，读到「剩余电量 78%」 |

总耗时 114s。

---

## 5. 边界与已知限制

- **iOS 不支持**：`wake_up()` 返回 `False` 并 warn。iOS 没有公开的唤醒接口
  （越狱/私有 API 不在考虑范围）。
- **有 PIN 锁屏划不掉**：`wm dismiss-keyguard` 只能划掉**无密码**锁屏。
  工具会如实回报失败，让模型知道「得靠人解锁」，不会假装成功。
- **`Dozing` 算灭屏**：`mWakefulness=Dozing`（AOD 息屏）也返回 `"off"`。
- **休眠中唤醒后可能需要解锁**：唤醒点亮的是锁屏，agent 之后要处理锁屏界面。
- **不是自动唤醒**：harness 只**告诉**模型，唤醒这一步由模型决定要不要做。
  自动唤醒会掩盖「屏幕为什么灭了」这个信息。

---

## 6. 相关单测

`tests/mobile_use/test_wake_screen.py`（16 条）

- 状态解析：`Awake` → on / `Asleep` → off / `Dozing` → off / 认不出 → unknown /
  bytes 输出 / shell 失败 → unknown
- 唤醒命令：`input keyevent 224` + `wm dismiss-keyguard` / 失败返回 False
- 工具注册 + 无必填参数
- 工具如实回报：亮 → 成功；还灭着 → 报 PIN 失败
- **contextor 回归**（钉死 §2.1 那个被证伪的判据）：
  - 树**非空**时也读电源状态
  - 电源状态在抓屏**还在跑**的时候就读到了（并发没退化成串行）
  - 控制器没有该方法时整轮不炸
  - 抓屏抛异常时电源状态仍被收回（不留悬空 task）
