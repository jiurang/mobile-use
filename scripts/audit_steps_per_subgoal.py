"""按 `steps_per_subgoal` 的真实分布，给出**单子目标步数上限**的建议值。

这是方案 §12.3 的 C3：「按 A/B 结果定步数上限（`steps_per_subgoal` 的 p90）」。

为什么必须靠这个脚本，而不是拍一个数
------------------------------------
`MAX_STEPS_PER_SUBGOAL` 现在写死 12（方案 §4.3），依据是**一次**运行的
4.0~5.3 步/子目标。实测已经证明这个依据不牢：

* 第四轮：3 个子目标 / 21 步 executor ≈ **1.67** 步/子目标
* 第五轮：3 个子目标 / 17 步 executor ≈ **5.7** 步/子目标

同一个任务、同一个模型，差了 3.4 倍。**上限定低了会把正常子目标腰斩**
（子目标其实做成了，但被步数上限打断，报成 failed）；**定高了会让跑飞的
子目标继续烧钱**。所以这个数只能从**多次运行**的分布里取。

数据从哪来
----------
`scripts/probe_executor_e2e.py` 会拦截 `telemetry.capture_subgoal_progress`
和 `telemetry.capture_replan`，把每次真机运行的子目标步数写进
`probe_out/executor_e2e.json`。本脚本读那些文件（可以一次读多个运行），
算分布并给出建议上限。

跑法：
    # 先跑若干次真机（每次覆盖 probe_out/executor_e2e.json，所以要先把旧的挪走）
    uv run python scripts/probe_executor_e2e.py --goal "打开设置，进入 WLAN 页面"
    Copy-Item probe_out/executor_e2e.json probe_out/run1.json
    # ... 再跑几次，每次存成 run2.json / run3.json ...

    uv run python scripts/audit_steps_per_subgoal.py probe_out/run*.json

⚠️ **样本量不够时它会明说**，不会给一个看起来很确定的数。
（2026-10-21 实测：两次运行的 p90 差 3.4 倍 ⇒ 单次运行不足以定上限。）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minitap"))

# 建议上限 = 观测到的 p90，再留一点余量。
#
# 余量取 1 步：p90 本身是「90% 的子目标在这个步数内跑完」，
# 再加 1 步给那些「刚好卡在 p90 附近」的子目标留一次机会。
#
# ⚠️ **下限 12**：方案 §4.3 的初始值。样本不足时宁可保持原值 ——
# 拿 2 个样本改一个熔断阈值，比不改更危险。
MIN_LIMIT = 12


def current_limit() -> int | None:
    """从**代码里**读当前上限（单一事实源，不在这里复写一个数）。

    返回 None = **默认不限**（2026-10-22 起 `MAX_STEPS_PER_SUBGOAL = None`）。
    """
    try:
        from minitap.mobile_use.agents.executor.executor import MAX_STEPS_PER_SUBGOAL

        return MAX_STEPS_PER_SUBGOAL
    except Exception:  # noqa: BLE001 - 读不到就退回「不限」
        return None


def _percentile(values: list[int], q: float) -> float:
    """线性插值分位数（不用 numpy）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def collect(
    paths: list[Path],
) -> tuple[list[int], list[dict], int, dict[str, list[tuple[int, str]]]]:
    """读若干次运行的 json。

    返回 (步数列表, replan 事件, 运行次数, 每个子目标的历次汇报序列)。
    最后一项用来判断「撞上限的那次尝试，是**瞎转**还是**真需要那么多步**」。
    """
    steps: list[int] = []
    replans: list[dict] = []
    runs = 0
    per_subgoal: dict[str, list[tuple[int, str]]] = {}
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"⚠ 跳过 {path.name}: {e}", file=sys.stderr)
            continue
        progress = data.get("subgoal_progress") or []
        if not progress:
            print(f"⚠ {path.name} 没有 subgoal_progress 字段（探针版本太老？）", file=sys.stderr)
            continue
        runs += 1
        for p in progress:
            steps.append(int(p.get("steps", 0)))
            per_subgoal.setdefault(p.get("subgoal_id", "unknown"), []).append(
                (int(p.get("steps", 0)), p.get("outcome", "unknown"))
            )
        replans.extend(data.get("replans") or [])
    return steps, replans, runs, per_subgoal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="*",
        default=[str(ROOT / "probe_out" / "executor_e2e.json")],
        help="若干次真机运行的 executor_e2e.json",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=5,
        help="至少几次运行才给建议（默认 5）",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.files]
    steps, replans, runs, per_subgoal = collect(paths)

    print("=" * 68)
    print("steps_per_subgoal 分布 —— 单子目标步数上限的校准（方案 §12.3 C3）")
    print("=" * 68)
    if not steps:
        print("\n没有数据。先跑 `uv run python scripts/probe_executor_e2e.py --goal ...`，")
        print("并把 probe_out/executor_e2e.json 复制成多个 run*.json 再喂进来。")
        return 1

    p50 = _percentile(steps, 0.50)
    p90 = _percentile(steps, 0.90)
    limit = current_limit()
    # 「打到上限」= 样本恰好落在上限上（说明那次被截断了，真实值看不到）。
    # **默认不限（limit is None）⇒ 没有删失这回事**，分布是完整的。
    censored = [s for s in steps if limit is not None and s >= limit]

    print(f"\n运行次数: {runs}   子目标样本: {len(steps)}")
    print(f"步数分布: min={min(steps)} p50={p50:.1f} p90={p90:.1f} max={max(steps)}")
    print(f"逐个: {sorted(steps)}")
    print(
        f"当前 MAX_STEPS_PER_SUBGOAL = {limit if limit is not None else '不限（None）'}"
    )
    if replans:
        print(f"replan 次数: {len(replans)}（>0 = 看门狗真的被触发过，R16 生效）")
    else:
        print("replan 次数: 0（本次样本里没有计划级失败 ⇒ 看门狗没被触发过）")

    # ---- 右删失检查：这一步决定这份数据能不能用来定上限 ----
    #
    # 如果有样本**正好打在上限上**，那它不是「这个子目标用了 N 步」，
    # 而是「这个子目标**至少**用了 N 步，但被上限截断了」。真实值可能是
    # 13、20、50 —— 观测不到。
    #
    # ⇒ 这时候算出来的 max / p90 是**被截断的**，不能拿来定上限。
    #
    # ⚠️ **但「删失」本身不指示上限该往哪调** —— 两种解释都成立：
    #
    #   (a) 上限**偏小**：这个子目标真需要 13+ 步，被腰斩了
    #       ⇒ 该**上调**（否则正常子目标做不完）
    #   (b) 那次尝试在**瞎转**（执行器在原地打转），
    #       上限正好把它切了，重试很快就做完了
    #       ⇒ 该**下调**（每次瞎转少烧几步）或者保持
    #
    # **怎么分辨 (a) 和 (b)**：看这个子目标**后来**的汇报。
    # 如果被截断之后**同一子目标**又跑了一次、而且**步数更少就完成了**
    # ⇒ 那次被截断的尝试是瞎转（b），上限**救了我们**。
    # 如果后来没有汇报，或者后来还是用了很多步 ⇒ (a) 偏可能。
    if censored:
        print()
        print(f"🔴 **有 {len(censored)} 个样本打到了上限（{limit} 步）** ⇒ 分布是**右删失**的：")
        print(f"   这些子目标不是「用了 {limit} 步」，是「**至少**用了 {limit} 步还被截断」。")
        print("   ⇒ 上面算出的 max / p90 **不能**用来定上限（真实值更大，看不到）。")
        print()

        # 分辨 (a) 上限偏小 / (b) 那次是瞎转
        # （到这里 `limit` 一定不是 None —— `censored` 非空就意味着配了上限）
        assert limit is not None
        flailing, unknown = [], []
        for sid, seq in per_subgoal.items():
            hit_idx = next((i for i, (s, _) in enumerate(seq) if s >= limit), None)
            if hit_idx is None:
                continue
            later = seq[hit_idx + 1 :]
            if not later:
                unknown.append((sid, seq))
            elif later[-1][1] == "completed" and later[-1][0] < limit:
                flailing.append((sid, seq))
            else:
                unknown.append((sid, seq))

        if flailing:
            print(f"   ✅ 其中 **{len(flailing)} 个**是**瞎转**（不是真需要那么多步）：")
            for sid, seq in flailing:
                after = seq[seq.index(next(x for x in seq if x[0] >= limit)) + 1 :]
                print(f"      子目标 {sid}: {seq} → 撞上限后重试 {after[0][0]} 步就完成")
            print("      ⇒ 上限**救了钱**（切掉了原地打转的尝试，重试很快就成）。")
            print(f"      ⇒ 这类情况的**成本**是「{limit} 步 + 一次重发周期」；")
            print(f"        调低上限能省每次瞎转的这 {limit} 步，调高只会更贵。")
            print(f"      ⇒ **建议：可以把上限从 {limit} 往下试**（例如 8），")
            print("        用同样的脚本对比「每次瞎转省下的步数」和「正常子目标被腰斩的次数」。")
        if unknown:
            print()
            print(f"   ⚠ {len(unknown)} 个**分不清**（撞上限后没有更短的成功重试）：")
            for sid, seq in unknown:
                print(f"      子目标 {sid}: {seq}")
            print("      ⇒ 这些可能是「上限偏小、真需要那么多步」—— 暂时**不要下调**。")
        if not flailing and not unknown:
            print("   （没有撞上限的子目标记录）")
        return 0

    suggested = max(MIN_LIMIT, int(p90) + 1)

    print()
    if limit is None:
        print("⇒ **当前默认不限步数**（`MAX_STEPS_PER_SUBGOAL = None`）。")
        print(f"  观测到的步数分布是**完整**的（没有被截断的样本）：p90={p90:.1f} max={max(steps)}")
        print("  ⇒ 这个分布可以用来回答「一个子目标通常要几步」，")
        print("     但**不该**再据此配一个上限（2026-10-22 实测：12 会腰斩正常子目标）。")
        print("  ⇒ 兜底只剩**全局动作步预算**（`constants.DEFAULT_MAX_STEPS`）。")
        return 0

    if runs < args.min_runs:
        print(f"⚠ **样本不足**（{runs} < {args.min_runs} 次运行）⇒ 不改步数上限。")
        print("  实测已证明单次运行不可靠：两次运行的每子目标步数差了 3.4 倍。")
        print(f"  建议保持 MAX_STEPS_PER_SUBGOAL = {limit} 不变。")
        return 0

    print(f"⇒ 建议 MAX_STEPS_PER_SUBGOAL = {suggested}")
    print(f"  依据：p90({p90:.1f}) + 1 步余量，且不低于下限 {MIN_LIMIT}。")
    print("  改的地方：minitap/mobile_use/agents/executor/executor.py")
    print("            MAX_STEPS_PER_SUBGOAL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
