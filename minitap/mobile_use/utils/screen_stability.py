"""屏幕**稳定性**判定 —— 「这一屏是不是已经定下来了」。

为什么需要它
------------

沉降等待（``PerceptionConfig.settle_delay_ms``）原来是一个**纯时间**的下限：
「距上次动屏不足 N 毫秒就补等到 N 毫秒」。

2026-10-22 真机实测（Redmi Turbo 4 / Android 16）把这个数证伪了：

* ``dump_hierarchy`` **自带 waitForIdle** —— 屏幕还在动的时候它会自己等。
  实测滚动场景里 dump#0 发起于动作后 496ms、**耗时 609ms**，
  那 609ms 就是它在里面等屏幕静下来。
* ��以 3 秒下限**从来没有让树更完整**，它只是让每次抓屏晚 3 秒才发起。
  返回键场景里树一次就稳（141 个元素，5 次 dump 全同），3 秒纯属白等。

但「静」不等于「全」
------------------

**idle 等待只保证「不再动了」，不保证「该挂的都挂上了」。**

页面导航场景实测（点进 WLAN）::

    dump#0  235ms   49 个元素   ← 过渡态
    dump#1   70ms   91 个元素
    dump#2  134ms   91 个元素
    dump#3  360ms  228 个元素   ← 新页面完整内容
    dump#4   87ms  228 个元素

新页面的控件是**懒加载**出来的：dump#1 和 dump#2 的内容已经一样（91 个元素），
按「两次相同 ⇒ 稳定」判会判成稳定 —— 但 228 才是完整的。

⇒ 所以「稳定」这个判据**有已知误判面**：它会漏掉「静默期内才挂上的控件」。
本模块提供的是**最便宜可得的稳定性信号**，它把「还在动」这种最常见的情况
（滚动、动画、过渡态）挡掉，但挡不住懒加载。

**验收必须看「步数」和「失败目标数」，不是「点错率」**
（表和屏是同一次抓的，索引契约自洽；真正的代价是「表里缺控件 ⇒ 模型以为
按钮不存在 ⇒ 绕路多走几步」）。

指纹怎么算
----------

``resource-id`` + ``text`` + ``class`` + ``bounds`` 四元组的**排序集合**：

* **要 bounds** —— 滚动时元素集合可能不变、只有位置在变，那也是「没稳定」。
* **要排序** —— 树的遍历顺序在结构相同时是确定的，但排序能挡住
  「同一屏、顺序微变」这种无意义的差异（少一次无谓的重抓）。
* **不要只数元素个数** —— 实测滚动场景 292→267→240→239 里有两次数相同
  但内容不同（bounds 变了），只数个数会漏判。
"""

from __future__ import annotations

import hashlib

# 单元分隔符。用不可见字符，避免和用户可见文本里的分隔符撞上。
_SEP = "\x1f"
_REC = "\x1e"


def tree_fingerprint(elements: list[dict] | None) -> str:
    """算一棵树的内容指纹。

    同样的内容 ⇒ 同样的指纹。内容变了（元素增减、文字变、位置变）⇒ 指纹变。

    **滚动条必须排除**（2026-10-22 真机实测）
    ----------------------------------------
    MIUI 的滚动条（``com.android.settings:id/main_settings_scrollBar``）
    会**淡入淡出**：即使屏幕完全静止，它也会在两次采样之间出现/消失。
    实测（设置首页静止不动，142 个元素）::

        第一次采样: 142 个（含 scrollBar）
        第二次采样: 141 个（scrollBar 淡出了）
        ⇒ 指纹不同 ⇒ 误判「屏还在动」

    ⇒ 滚动条是**纯视觉装饰**，和「屏定下来没有」无关，必须排除。
    判据用 resource-id 里含 ``scrollbar``（不区分大小写）。

    Args:
        elements: ``_parse_hierarchy_xml_to_elements`` 的输出（或同形状的 dict 列表）。

    Returns:
        十六进制 sha1。空树返回空串的哈希（仍是个稳定值）。
    """
    if not elements:
        return hashlib.sha1(b"").hexdigest()

    parts = [
        _SEP.join(
            (
                str(e.get("resource-id") or ""),
                str(e.get("text") or ""),
                str(e.get("class") or ""),
                str(e.get("bounds") or ""),
            )
        )
        for e in elements
        if not _is_scrollbar(e)
    ]
    parts.sort()
    return hashlib.sha1(_REC.join(parts).encode("utf-8", errors="replace")).hexdigest()


def _is_scrollbar(element: dict) -> bool:
    """这个元素是不是滚动条。

    滚动条会淡入淡出 ⇒ 它出现/消失**不代表**屏幕在动。
    """
    resource_id = str(element.get("resource-id") or "").lower()
    return "scrollbar" in resource_id


def is_stable(previous: str | None, current: str | None) -> bool:
    """两次采样是否算「同一屏」。

    没有上一次采样（第一次）⇒ **不算稳定** —— 我们还不知道它稳没稳，
    保守地让调用方再采一次。
    """
    if previous is None or current is None:
        return False
    return previous == current
