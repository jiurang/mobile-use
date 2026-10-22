"""宏（脚本）—— 把做过的一串动作存下来，下次直接回放。

为什么是这个包，而不是某个具体进程里的一段代码
------------------------------------------------
宏这件事有三方，各自有各的稳定边界：

* **数据**（脚本库）在 manager-server —— 跨任务、跨设备共享、可人工审阅/禁用。
* **算法**（编译、降级链、判稳）在**这里** —— 纯逻辑，不依赖任何设备通道。
* **执行**（跑起来）在设备侧的执行面（bridge / runner）—— 那是唯一碰设备的地方。

所以本包只做算法，不碰设备：它给出一个「执行环境」的接口
（:class:`MacroEnv`），由调用方把「怎么执行一个工具调用」和「怎么读当屏的树」
接进来。这样 bridge 挪到哪儿、runner 怎么改，这个包都不用动。

宏长什么样
----------
**一串工具调用**（和设备侧 ``POST /call`` 发的是同一种东西）::

    {
      "name": "ks_like_and_comment",
      "when_to_use": "在快手极速版给当前视频点赞并评论",
      "app_package": "com.kuaishou.nebula",
      "device_serial": "89R8ZXNBGQIRJRLF",
      "steps": [
        {"tool": "launch_app", "args": {"package": "com.kuaishou.nebula"}},
        {"tool": "tap", "args": {"target": {"resource_id": "com.ks:id/like_button"}},
         "fallback": {"context": {"t": "like", "i": 2, "w": ["collect", "like", "comment"]}}}
      ]
    }

回放时的四条硬规矩（每一条都是真机踩出来的）
------------------------------------------
1. **只按 ``resource_id`` 定位，不用 ``index``** ——
   ``index`` 只在「那一次建表」上成立，跨运行必失效。
2. **定位不到就等，等不到就停** ——
   真机实测：``launch_app`` 返回时 app 还没到前台；顶栏是动态的（视频播放时收起）。
   所以「元素不在屏上」要**轮询重试**，而不是立刻判死；但重试到上限就**停**，
   报清「第几步、哪个工具、为什么」，**绝不瞎点**。
3. **降级要上报** ——
   脚本要的是 ``resource_id``，工具却用别的路径命中了（index / text / 位置），
   这就是**降级**，必须记下来告诉调用方：这个宏可能已经过期了。
4. **前置条件** —— 设备对不对、前台 app 对不对，跑之前先确认。
"""

from __future__ import annotations

from minitap.mobile_use.macros.model import Macro, MacroStep, MacroRunResult
from minitap.mobile_use.macros.runner import MacroEnv, run_macro

__all__ = ["Macro", "MacroStep", "MacroRunResult", "MacroEnv", "run_macro"]
